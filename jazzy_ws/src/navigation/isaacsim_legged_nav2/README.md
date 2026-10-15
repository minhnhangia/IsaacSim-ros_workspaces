# isaacsim_legged_nav2

Legged robots in Isaac Sim 6.1, walking under Isaac Lab locomotion policies and navigating with Nav2
(ROS 2 Jazzy). Sensors: an Ouster OS0 3D LiDAR and a front RealSense D455 stereo pair.

| Robot | Policy | Stairs | Notes |
|---|---|---|---|
| `spot` (default) | NVIDIA's bundled flat-terrain policy | no | |
| `anymal_d` | Isaac Lab's pretrained rough-terrain policy | **yes, up and down** | needs a one-time policy export |

## Run

```bash
# once, for ANYmal-D (needs a Python with torch: pip install torch)
ros2 run isaacsim_legged_nav2 export_policies.py anymal_d

# 1. Simulator. Other options go in sim_args, e.g. sim_args:="--headless --no-camera"
ros2 launch isaacsim_legged_nav2 sim.launch.xml robot:=anymal_d

# 2. Nav2 and RViz
ros2 launch isaacsim_legged_nav2 navigation.launch.py robot:=anymal_d \
    map:=$HOME/IsaacSim-ros_workspaces/jazzy_ws/scenes/dd100_office_opened_navigation.yaml
```

The robot spawns at (0, 0, 0°) in the office, where AMCL starts, so you can send a goal straight away.
If you spawn elsewhere (`--spawn X Y YAW_DEG`), pass the same pose to the navigation launch
(`initial_x`, `initial_y`, `initial_yaw` in radians). Both launches use `ROS_DOMAIN_ID` from the environment.

All simulator options: `~/isaacsim/python.sh isaac/legged_ros2_sim.py --help`. The most useful:

| Option | Effect |
|---|---|
| `--headless` | no GUI |
| `--no-camera` | no stereo cameras (Nav2 does not use them; the biggest speed-up) |
| `--camera-rate HZ` | render and publish the cameras at this rate instead of every frame |
| `--spawn X Y YAW_DEG` | spawn pose |
| `--scene office\|grid\|FILE.usd` | environment |
| `--[no-]open-stair-doors` | open the office's stairwell doors (default: for robots that climb stairs) |
| `--view-offset FWD LEFT UP` | initial GUI camera relative to the robot |

## ROS 2 interface (namespace = robot name)

| Topic | Type | Notes |
|---|---|---|
| `cmd_vel` (sub) | Twist | body frame: linear.x, linear.y, angular.z |
| `/clock` | Clock | |
| `odom` | Odometry | odom → base_footprint, ground truth |
| `tf`, `tf_static` | TF | odom → base_footprint → base_link → sensor frames |
| `OS0/point_cloud` | PointCloud2 | frame `OS0` |
| `front_stereo/{left,right}/image_raw`, `.../camera_info` | Image, CameraInfo | 640x480, 0.095 m baseline |
| `front_stereo/depth/image_rect_raw` | Image 32FC1 | metres, aligned with the left camera |

`odom` and `base_footprint` lie on the floor and `base_link` is the body, which bobs and tilts while
walking, so every height Nav2 uses is measured from the floor.

## Robot-specific behaviour

Each of these fixes a measured problem; the details are in the docstrings of the modules named.

- **ANYmal-D** (`robots/anymal_d.py`, `rough_policy.py`, `height_scanner.py`): Isaac Lab's
  `Isaac-Velocity-Rough-Anymal-D-v0` policy, reproducing its training setup (200 Hz physics, 50 Hz
  policy, the ANYdrive LSTM actuator network, a 17x11 terrain height scan). The height scan comes from
  PhysX raycasts against the scene, i.e. ground-truth geometry standing in for an elevation map.
- **Spot** (`robots/spot.py`, `commands.py`, `actuators.py`): small commands are scaled up from a stand
  because its policy ignores them; with a zero command it holds the heading it stopped at; its
  actuator model runs in one NumPy pass instead of Isaac's per-joint implementation (same torques).
  ANYmal-D needs neither command fix.
- **Commands** (`commands.py`): a watchdog stops the robot 0.5 s after the last `cmd_vel`; commands
  are clamped to each robot's trained range.
- **Scene** (`scene.py`): physics rate and `--device` are set on every physics scene (a scene loaded
  from a file otherwise keeps its own: the office's 60 Hz and GPU dynamics). The office's surfaces get
  a physics material (friction 1.0, combined with the feet's), since without one PhysX uses 0.5 and
  the policies slip. Its stairwell doors are opened for ANYmal-D.
- **Sensors** (`sensors.py`): the OS0 sits on a 10 cm mast. Nav2's self-filters were measured at that
  height per robot (`params/robot_profiles.yaml`); re-measure them if you change `--lidar-mount`.
  NVIDIA's D455 asset is made visual-only, since as a rigid body it falls off the robot.
- **Point clouds and DDS** (`params/fastdds_large_messages.xml`): Nav2's costmaps subscribe best-effort,
  and with FastDDS's defaults they lost 20-60% of the clouds. `sim.launch.xml` gives the simulator this
  profile, after which all arrive. It works when Nav2 runs on the same machine; across a network
  consider `rmw_zenoh`. If you run `legged_ros2_sim.py` without the launch file, export
  `FASTRTPS_DEFAULT_PROFILES_FILE` to the profile yourself.
- **Nav2** (`params/nav2.yaml`): one configuration for both robots. Regulated Pure Pursuit, because
  MPPI overshot goals by ~1 m on Spot. The footprint and self-filters come from `robot_profiles.yaml`.

## Code layout

`isaac/legged_ros2_sim.py` parses the options, starts Isaac Sim and runs `legged_sim.simulation.LeggedSimulation`.

| `isaac/legged_sim/` | |
|---|---|
| `config.py` | command-line options; robot-dependent defaults come from the robot |
| `simulation.py` | builds everything below and runs the frame loop |
| `robots/` | `base.py` (the interface), `spot.py`, `anymal_d.py` |
| `scene.py`, `physics_materials.py` | environment, doors, friction, physics settings |
| `sensors.py`, `ros_graph.py` | OS0 and D455 mounting; the OmniGraph behind the topics above |
| `odometry.py`, `commands.py` | floor-aligned odometry; cmd_vel handling |
| `rough_policy.py`, `networks.py`, `height_scanner.py` | rough-terrain observation and policy, NumPy networks, height scan |
| `policy_assets.py` | exported weights (`~/.cache/isaacsim_legged_nav2/policies`), checked against torch on load |
| `actuators.py` | Spot's actuator model |
| `viewport.py` | initial GUI view and debugging snapshots (`--snapshot-dir`) |

Launch: `sim.launch.xml` (simulator), `navigation.launch.py` (picks the robot profile) →
`nav2_stack.launch.xml` (the Nav2 stack). `scripts/export_policies.py` downloads NVIDIA's checkpoints
and writes NumPy weights with torch reference outputs and SHA-256 hashes.

The modules that do not import Isaac Sim are covered by unit tests:
`colcon test --packages-select isaacsim_legged_nav2`, or `python3 -m pytest test`.

## Performance

Real-time factor in the office (Ryzen 9 9900X3D, RTX 5090, headless, all sensors): Spot 0.72, ANYmal-D
0.8-0.9. Without cameras: Spot 1.03, ANYmal-D ~1.4 on the empty grid. CPU physics (the default) beats
`--device cuda` for a single robot. The simulator logs its real-time factor every 10 s.

## Verified

VERIFIED_PLACEHOLDER

## Limitations

- **Nav2 does not plan across floors.** The 2D map is one height slice: stairs are free space, and the
  flight down to the basement looks like floor. ANYmal-D can walk the stairs when commanded, but Nav2
  has no notion of floors yet; keep Spot out of stairwells with a keepout zone.
- **ANYmal-D's actuator network is ANYmal-C's** (NVIDIA publishes no ANYmal-D network); fine in
  simulation, a sim-to-real gap on hardware.
- **Stairwells are narrow** (1.3 m in the office). ANYmal-D jammed against a wall when walked 0.15 m
  off the centreline; stairs must be approached centred.
