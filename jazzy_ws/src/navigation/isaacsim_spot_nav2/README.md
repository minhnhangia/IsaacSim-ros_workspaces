# isaacsim_spot_nav2

Boston Dynamics Spot in Isaac Sim 6.1, walking under NVIDIA's bundled Isaac Lab locomotion policy and
navigating with Nav2 (ROS 2 Jazzy). Sensors: an Ouster OS0 3D LiDAR and a front RealSense D455 stereo pair.

## Run

```bash
# 1. Simulator. Options go in sim_args, e.g. sim_args:="--headless --no-camera"
ros2 launch isaacsim_spot_nav2 spot_sim.launch.xml

# 2. Nav2 and RViz
ros2 launch isaacsim_spot_nav2 spot_navigation.launch.xml \
    map:=$HOME/IsaacSim-ros_workspaces/jazzy_ws/scenes/dd100_office_opened_navigation.yaml
```

Spot spawns at (0, 0, 0°) in the office, where AMCL starts, so you can send a goal straight away.
If you spawn elsewhere (`--spawn X Y YAW_DEG`), pass the same pose to the navigation launch
(`initial_x`, `initial_y`, `initial_yaw` in radians). Both launches use `ROS_DOMAIN_ID` from the environment.

All simulator options: `~/isaacsim/python.sh isaac/spot_ros2_sim.py --help`. The most useful:

| Option | Effect |
|---|---|
| `--headless` | no GUI |
| `--no-camera` | no stereo cameras (Nav2 does not use them; the biggest speed-up) |
| `--camera-rate HZ` | render and publish the cameras at this rate instead of every frame |
| `--spawn X Y YAW_DEG` | spawn pose |
| `--scene office\|grid\|FILE.usd` | environment |
| `--view-offset FWD LEFT UP` | initial GUI camera relative to Spot |

## ROS 2 interface (namespace `spot`)

| Topic | Type | Notes |
|---|---|---|
| `/spot/cmd_vel` (sub) | Twist | body frame: linear.x, linear.y, angular.z |
| `/clock` | Clock | |
| `/spot/odom` | Odometry | odom → base_footprint, ground truth |
| `/spot/tf`, `/spot/tf_static` | TF | odom → base_footprint → base_link → sensor frames |
| `/spot/OS0/point_cloud` | PointCloud2 | frame `OS0` |
| `/spot/front_stereo/{left,right}/image_raw`, `.../camera_info` | Image, CameraInfo | 640x480, 0.095 m baseline |
| `/spot/front_stereo/depth/image_rect_raw` | Image 32FC1 | metres, aligned with the left camera |

`odom` and `base_footprint` lie on the floor and `base_link` is the body, which bobs and tilts while
walking, so every height Nav2 uses is measured from the floor.

## Robot-specific behaviour

Each of these fixes a measured problem; the details are in the docstrings of the modules named.

- **Commands** (`commands.py`): a watchdog stops Spot 0.5 s after the last `cmd_vel`; small commands
  are scaled up from a stand, because the policy ignores them; with a zero command Spot holds the
  heading it stopped at, instead of the ~40° turn the policy makes while settling.
- **Actuators** (`actuators.py`): Spot's actuator model runs for all joints in one NumPy pass. Isaac's
  per-joint implementation took ~80% of the frame time; the torques are identical (to float32
  rounding). `--actuators isaac` switches back.
- **Physics** (`scene.py`): 500 Hz and the `--device` choice are set on every physics scene, because
  Isaac leaves a scene loaded from a USD file at its own settings (the office: 60 Hz and GPU dynamics).
- **Sensors** (`sensors.py`): the OS0 is on a 10 cm mast; Nav2's self-filters were measured at that
  height, so re-measure them if you change `--lidar-mount`. NVIDIA's D455 asset is made visual-only,
  since as a rigid body it falls off the robot.
- **Point clouds and DDS** (`params/fastdds_large_messages.xml`): Nav2's costmaps subscribe best-effort,
  and with FastDDS's defaults they lost 20-60% of the clouds. `spot_sim.launch.xml` gives the simulator
  this profile, after which all arrive. It works when Nav2 runs on the same machine; across a network
  consider `rmw_zenoh`. If you run `spot_ros2_sim.py` without the launch file, export
  `FASTRTPS_DEFAULT_PROFILES_FILE` to the profile yourself.
- **Controller** (`params/spot_nav2.yaml`): Regulated Pure Pursuit; MPPI overshot goals by ~1 m.

## Code layout

`isaac/spot_ros2_sim.py` parses the options, starts Isaac Sim and runs `spot_sim.simulation.SpotSimulation`.

| `isaac/spot_sim/` | |
|---|---|
| `config.py` | command-line options |
| `simulation.py` | builds everything below and runs the frame loop |
| `scene.py` | environment, obstacle boxes, physics settings |
| `sensors.py` | OS0 and D455 mounting |
| `ros_graph.py` | the OmniGraph that publishes and subscribes the topics above |
| `odometry.py`, `commands.py`, `actuators.py` | floor-aligned odometry, cmd_vel handling, actuator model |
| `viewport.py` | initial GUI view and debugging snapshots (`--snapshot-dir`) |

`config`, `geometry`, `commands`, `odometry` and `actuators` do not import Isaac Sim and are covered by
unit tests: `colcon test --packages-select isaacsim_spot_nav2`, or `python3 -m pytest test`.

## Performance

Real-time factor in the office (Ryzen 9 9900X3D, RTX 5090, headless): 0.72 with all sensors, 1.03 with
`--no-camera`, 1.38 without sensors. The remaining cost is mostly rendering. CPU physics beats
`--device cuda` (0.46) for a single robot. The simulator logs its real-time factor every 10 s.

## Verified

Office scene, headless, through both launch files: a five-goal tour (~52 m, through doorways) succeeded
5/5, ending 0.13-0.30 m from each goal (the goal tolerance, 0.30 m, applies to AMCL's estimate, which
was within 7 cm of ground truth on average). With an unmapped 0.6 m box across the route Spot went
around it both ways, at least 0.38 m clear, without falling. Final headings land near the 20° yaw
tolerance, because Regulated Pure Pursuit stops turning as soon as it is inside it.

## Limitations

- **Stairwells look like floor to Nav2.** The 2D map is a height slice, so both office stair cores are
  free space, including the flight down to the basement. Add a keepout zone over them.
- **Flat ground only.** The bundled policy cannot climb stairs (it stalls at the first 15 cm step and can
  flip over) and descends them unsafely.
