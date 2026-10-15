# isaacsim_spot_nav2

Boston Dynamics Spot in Isaac Sim 6.1, walking under NVIDIA's bundled Isaac Lab locomotion policy,
navigating with Nav2 (ROS 2 Jazzy). Sensors: Ouster OS0 (3D LiDAR) and a front RealSense D455 stereo pair.

## Run

```bash
# 1. Simulator (Isaac Sim's own Python; add --headless to run without the GUI)
ros2 launch isaacsim_spot_nav2 spot_sim.launch.xml
#    or directly:  ~/isaacsim/python.sh <pkg>/isaac/spot_ros2_sim.py --help
#    (then also export FASTRTPS_DEFAULT_PROFILES_FILE=<pkg>/params/fastdds_large_messages.xml)

# 2. Nav2 (+ RViz)
ros2 launch isaacsim_spot_nav2 spot_navigation.launch.xml \
    map:=$HOME/IsaacSim-ros_workspaces/jazzy_ws/scenes/dd100_office_opened_navigation.yaml
```

Spot spawns at (0, 0, 0 deg) in the office by default and AMCL starts there, so you can send a
goal straight away. If you spawn elsewhere (`--spawn X Y YAW_DEG`), pass the same pose to the
navigation launch (`initial_x`, `initial_y`, `initial_yaw` in radians).

Both sides use `ROS_DOMAIN_ID` from the environment; keep them the same.

## ROS 2 interface (namespace `spot`)

| Topic | Type | Notes |
|---|---|---|
| `/spot/cmd_vel` (sub) | Twist | body frame: linear.x, linear.y, angular.z |
| `/clock` | Clock | sim time |
| `/spot/odom` | Odometry | odom -> base_footprint, ground truth |
| `/spot/tf`, `/spot/tf_static` | TF | see frames below |
| `/spot/OS0/point_cloud` | PointCloud2 | frame `OS0` |
| `/spot/front_stereo/{left,right}/image_raw`, `.../camera_info` | Image, CameraInfo | 640x480, 0.095 m baseline |
| `/spot/front_stereo/depth/image_rect_raw` | Image 32FC1 | metres, aligned with the left camera |

Frames: `odom` and `base_footprint` lie on the floor; `base_link` is Spot's body (it bobs and tilts
while walking). Nav2 uses `base_footprint`, so every costmap height and scan slice is measured from
the floor. Sensor frames (`OS0`, `front_stereo_{left,right}_optical_frame`) are children of `base_link`.

## Things that are specific to this robot

These were measured, not assumed; the reasoning is in comments in `spot_ros2_sim.py` and `spot_nav2.yaml`.

- **Command watchdog.** Isaac's Twist subscriber keeps the last message forever, so the simulator
  stops Spot if no `cmd_vel` arrives for 0.5 s of sim time.
- **Deadband compensation.** From a settled stand the policy ignores small commands (up to ~0.2 m/s
  forward, ~0.15 m/s sideways, ~0.2-0.25 rad/s turning). A small non-zero command is scaled up,
  keeping its direction, only when Spot is standing. `--no-deadband-comp` turns it off.
- **Heading hold.** Each time Spot comes to rest the policy turns it ~40 deg while it settles. With a
  zero command the simulator steers back to the heading it stopped at (held within ~3 deg).
- **Controller.** Regulated Pure Pursuit rather than MPPI: MPPI overshot goals by ~1 m on this robot.
- **Physics rate and device.** The policy needs 500 Hz physics; the simulator sets it, and the
  `--device` choice (GPU dynamics and broadphase on or off), on every physics scene. Isaac only updates
  scenes it registered itself, so the office scene otherwise kept 60 Hz (Spot collapses) and GPU
  dynamics even with `--device cpu`.
- **Actuators.** Spot's actuator model (PD with a 4-step command delay, 45 Nm hip limit,
  angle-dependent knee limit) is computed for all 12 joints in one NumPy pass. Isaac's
  `ArticulationActuators` steps each joint as a separate Newton actuator, ~60 Warp launches before
  every physics step, which took ~3 ms per 2 ms step. The parameters are read from Isaac's own
  actuator objects, and the torques match Isaac's to 7e-6 Nm (12,677 steps of varied driving,
  torques up to 67 Nm). `--actuators isaac` switches back.
- **D455 asset.** NVIDIA's D455 is a rigid body with no collider; mounted on a robot it falls away.
  The simulator makes it visual-only.
- **LiDAR mount.** The OS0 sits on a 10 cm mast (visual only: no collider, no mass) above Spot's back.
  The self-filter ranges in `spot_navigation.launch.xml` were measured at this height; if you change
  `--lidar-mount`, re-measure them. The mast and plate are below the OS0's 45 deg field of view.
- **LiDAR profile.** The OS0 runs its 128x512 / 10 Hz profile (`--lidar-config` selects another);
  the asset's default 128x1024 is ~110k points / 1.3 MB per scan.
- **Point clouds and DDS.** With FastDDS's defaults a 0.6-1.3 MB cloud is split into many fragments,
  and a best-effort subscriber drops the whole cloud when one fragment is lost. Nav2's costmap layers
  subscribe best-effort (not configurable): they got 40% of 1.3 MB clouds and 79% of 640 KB ones
  (measured with a plain subscriber; a reliable one got all). `spot_sim.launch.xml` gives Isaac Sim
  `params/fastdds_large_messages.xml` (a large shared-memory segment), and then 100% arrive. Only the
  publisher needs the profile. It covers processes on the same machine. If Nav2 runs on another
  machine, or you start the simulator without the launch file, use
  `spot_navigation.launch.xml cloud_relay:=true`: `scripts/cloud_relay.py` subscribes reliably, drops
  Isaac's (0,0,0) no-return points, thins to 10 cm voxels and republishes on `/spot/OS0/points_nav`.
  Point-cloud compression (`point_cloud_transport`) would not help: Nav2 cannot subscribe to it
  without a decompressing republisher in between.

## Verified (headless, office scene)

- Five-goal tour through the office (~52 m, doorways included), last re-run after the performance and
  DDS changes: 5/5 succeeded, final position within
  the 0.30 m tolerance, AMCL within 7 cm on average (15 cm max) and 0.8 deg (3.6 deg max) of ground truth.
- Unmapped obstacle (0.6 m box across the route): Spot went around it both ways; closest gap between its
  footprint and the box 0.40 m; no falls.
- Point clouds, measured at the verifier's best-effort subscriber during the tour: 10.0 per sim second,
  no gap over 0.1 s (before the DDS profile: 5.7 per sim second, 12 gaps over 0.5 s).
- Final heading lands near the 0.35 rad (20 deg) tolerance, because Regulated Pure Pursuit stops turning
  as soon as it is inside it. It can be tightened (`yaw_goal_tolerance`), but turning in place also shifts
  Spot's body 0.1-0.25 m, so a much tighter value risks dithering at the goal (not yet tested with RPP).

## Performance

Real-time factor (sim seconds per wall second), office scene, headless, Spot standing, Ryzen 9 9900X3D
and RTX 5090:

| | before the fixes | now |
|---|---|---|
| All sensors | 0.30 | 0.72 |
| No cameras (`--no-camera`) | 0.39 | 1.03 |
| No sensors | - | 1.38 |
| All sensors, `--device cuda` | 0.28 | 0.46 |

Where the time went, from Kit's CPU profiler: before, ~80% of the main thread was Isaac's per-joint
actuator step (see Actuators) and the office ran GPU physics with CPU read-back (see Physics rate and
device). The GPU sat at 6-50%. Now the rest is mostly rendering: the RTX LiDAR, the two cameras
(~8 ms per frame) and the viewport. Hence:

- Nav2 does not use the cameras: `--no-camera` is the biggest remaining speed-up.
- `--camera-rate HZ` renders the cameras only at that rate (Isaac's `omni:sensor:tickRate`); at 15 Hz
  the RTF was ~0.80 instead of ~0.70. `--camera-skip` only skips publishing, not rendering, and does
  not give an exact rate (it is deprecated in Isaac Sim 6.1).
- CPU physics is the right choice for one robot; CUDA pays for GPU-CPU transfers every step.

The sim logs its real-time factor every 10 s. The GUI was ~1.0x the headless rate before the
fixes; it has not been re-measured.

## Limits

- **Stairwells look like floor to Nav2.** The 2D map is a height slice, so both office stair cores are
  free space and plannable, including the flight down to the basement. Add a keepout zone over them.
- **Flat ground only.** The bundled policy is trained on flat/cobblestone terrain. It cannot climb
  stairs (it stalls at the first 15 cm step and can flip over) and descends them unsafely.
- The simulator runs at ~0.7x real time with all sensors on and ~1x without the cameras. Everything
  runs on sim time, so this only affects how long things take to watch.
