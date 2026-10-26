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

The robot spawns at (0, 0, 0°) in the office; send a goal straight away. Both launches use
`ROS_DOMAIN_ID` from the environment.

**Localization** (`navigation.launch.py localization:=...`):

| | |
|---|---|
| `ground_truth` (default) | `map -> odom` is identity: the simulator's odometry is exact and in world coordinates. Works on every floor of the building, from any spawn pose. |
| `amcl` | AMCL against the map, which covers the main floor only. If you spawn elsewhere (`--spawn X Y YAW_DEG`), pass the same pose as `initial_x`, `initial_y`, `initial_yaw` (radians). |

All simulator options: `~/isaacsim/python.sh isaac/legged_ros2_sim.py --help`. The most useful:

| Option | Effect |
|---|---|
| `--headless` | no GUI |
| `--no-camera` | no stereo cameras (Nav2 does not use them; the biggest speed-up) |
| `--camera-rate HZ` | render and publish the cameras at this rate instead of every frame |
| `--spawn X Y YAW_DEG` | spawn pose |
| `--scene office\|grid\|FILE.usd` | environment |
| `--[no-]stair-access` | make the office's stairwells reachable: open their doors, and give the real floors colliders in place of the ground plane that seals the flights down (default: for robots that climb stairs) |
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

`odom` is the world frame. `base_footprint` is the body projected onto the floor directly below it
(found by a raycast that ignores the robot), so it follows the robot from storey to storey, and
`base_link` is the body, which bobs and tilts while walking. Every height Nav2 uses is measured from
the floor the robot stands on.

## Robot-specific behaviour

Each of these fixes a measured problem; the details are in the docstrings of the modules named.

- **ANYmal-D** (`robots/anymal_d.py`, `rough_policy.py`, `height_scanner.py`): Isaac Lab's
  `Isaac-Velocity-Rough-Anymal-D-v0` policy, reproducing its training setup (200 Hz physics, 50 Hz
  policy, the ANYdrive LSTM actuator network, a 17x11 terrain height scan). The height scan comes from
  PhysX raycasts against the scene, i.e. ground-truth geometry standing in for an elevation map.
- **Spot** (`robots/spot.py`, `commands.py`, `actuators.py`): small commands are scaled up from a stand
  because its policy ignores them (up to ~0.2 m/s, ~0.3 rad/s); with a zero command it holds the
  heading it stopped at (it turns ~40° while settling); its actuator model runs in one NumPy pass
  instead of Isaac's per-joint implementation (same torques).
- **ANYmal-D's commands** (`robots/anymal_d.py`): it follows small linear commands and drifts only
  1-7° after stopping, so it has no heading hold. It turns at only ~50-60% of 0.05-0.15 rad/s from a
  stand, which stalled Nav2's turns in place, so small turn commands are scaled up to 0.2 rad/s.
- **Commands** (`commands.py`): a watchdog stops the robot 0.5 s after the last `cmd_vel`; commands
  are clamped to each robot's trained range.
- **Scene** (`scene.py`): physics rate and `--device` are set on every physics scene (a scene loaded
  from a file otherwise keeps its own: the office's 60 Hz and GPU dynamics). The office's surfaces get
  a physics material (friction 1.0, combined with the feet's), since without one PhysX uses 0.5 and
  the policies slip. For ANYmal-D the stairwells are opened (`--stair-access`): their doors, and the
  office's only ground-floor collider, an invisible infinite plane at z = 0 that also covers the flights
  down to the basement, is replaced by colliders on the visible floor tiles (which have the stair openings).
- **First floor** (`scenes/office_floor1.usda`, built by `scripts/build_office_floor.py`): NVIDIA's office
  has only a ground floor and a basement; its stair cores continue upward but the main one was sealed by
  a wall (`Cube24`). The office scene loads this layer, which adds the ground floor's interior (rooms,
  walls, furniture, floors with colliders, ceilings, lights) one storey up as a reference to
  `/World/office`, removes the roof slabs under it, and starts `Cube24` one storey higher, so the main
  staircase (`SM_StairsMain2`) arrives on the first floor. Neither `office.usd` nor
  `dd100_office_navigation.usd` is modified; rebuild the layer with
  `~/isaacsim/python.sh scripts/build_office_floor.py` after changing the scene.
- **Sensors** (`sensors.py`, `mesh_probe.py`): mounts are placed by ray casts against the body's visual
  mesh, not its bounding box (on ANYmal-D the box top is the rear LiDAR housing, 15 cm above the shell
  under the mast): the OS0's mast stands on the highest point under its plate (10 cm on Spot, 15 cm on
  ANYmal-D to clear that housing), and the D455 sits flush against the most protruding point of the
  front face it covers. Nav2's self-filters were measured per robot with these mounts
  (`params/robot_profiles.yaml`); re-measure them if you change `--lidar-mount`. NVIDIA's D455 asset is
  made visual-only, since as a rigid body it falls off the robot.
- **Point clouds and DDS** (`params/fastdds_large_messages.xml`): Nav2's costmaps subscribe best-effort,
  and with FastDDS's defaults they lost 20-60% of the clouds. `sim.launch.xml` gives the simulator this
  profile, after which all arrive. It works when Nav2 runs on the same machine; across a network
  consider `rmw_zenoh`. If you run `legged_ros2_sim.py` without the launch file, export
  `FASTRTPS_DEFAULT_PROFILES_FILE` to the profile yourself.
- **Nav2** (`params/nav2.yaml`): one configuration for both robots. Regulated Pure Pursuit, because
  MPPI overshot goals by ~1 m on Spot. Per robot (`robot_profiles.yaml`): footprint, self-filters, goal
  tolerances and the final-approach speed floor. Spot gets 0.30 m / 20° and 0.3 m/s (its deadband);
  ANYmal-D 0.20 m / 11° and 0.15 m/s (tighter, 0.15 m / 9°, made it dither at a goal for 80 s).

## Code layout

`isaac/legged_ros2_sim.py` parses the options, starts Isaac Sim and runs `legged_sim.simulation.LeggedSimulation`.

| `isaac/legged_sim/` | |
|---|---|
| `config.py` | command-line options; robot-dependent defaults come from the robot |
| `simulation.py` | builds everything below and runs the frame loop |
| `robots/` | `base.py` (the interface), `spot.py`, `anymal_d.py` |
| `scene.py`, `physics_materials.py` | environment, doors, friction, physics settings |
| `sensors.py`, `mesh_probe.py`, `ros_graph.py` | OS0 and D455 mounting on the body's real surface; the OmniGraph behind the topics above |
| `odometry.py`, `commands.py` | floor-aligned odometry; cmd_vel handling |
| `rough_policy.py`, `networks.py`, `height_scanner.py` | rough-terrain observation and policy, NumPy networks, height scan |
| `policy_assets.py` | exported weights (`~/.cache/isaacsim_legged_nav2/policies`), checked against torch on load |
| `actuators.py` | Spot's actuator model |
| `viewport.py` | initial GUI view and debugging snapshots (`--snapshot-dir`) |

Launch: `sim.launch.xml` (simulator), `navigation.launch.py` (picks the robot profile) →
`nav2_stack.launch.xml` (the Nav2 stack). `scripts/export_policies.py` downloads NVIDIA's checkpoints
and writes NumPy weights with torch reference outputs and SHA-256 hashes. `scripts/build_office_floor.py`
(Isaac Sim's Python) generates the first-floor layer.

The modules that do not import Isaac Sim are covered by unit tests:
`colcon test --packages-select isaacsim_legged_nav2`, or `python3 -m pytest test`.

## Performance

Real-time factor in the office (Ryzen 9 9900X3D, RTX 5090, headless, all sensors): Spot 0.72, ANYmal-D
0.8-0.9. Without cameras: Spot 1.03, ANYmal-D ~1.4 on the empty grid. CPU physics (the default) beats
`--device cuda` for a single robot. The simulator logs its real-time factor every 10 s.

## Verified

Office scene, headless, through `sim.launch.xml` and `navigation.launch.py`, for both robots:

| | Spot | ANYmal-D |
|---|---|---|
| Five-goal tour (~52 m, through doorways), ground-truth localization | 5/5, mean 0.22 m and 16° from goals | 5/5, mean 0.17 m and 7° from goals |
| Same tour with `localization:=amcl` | 5/5 (before the default changed) | 5/5, mean 0.15 m; AMCL within 6.6 cm of ground truth on average |
| Unmapped 0.6 m box across the route, both directions | 2/2, 0.38 m clear | 2/2, 0.44 m clear |
| Office stairwell over `cmd_vel` (floor to 1.80 m landing / back down) | not capable | 3/3 up, 3/3 down |
| Main staircase down to the basement's mid-landing (-1.5 m) over `cmd_vel`, with Nav2 running | not capable | 1/1: `map -> base_footprint` z followed 0.00, -0.44, -0.74, -1.49 m; `base_footprint -> base_link` stayed 0.51-0.65 m |
| Main staircase up to the new first floor (z 3.3) over `cmd_vel`, both flights | not capable | 2/3 full climbs (~52 s); 3/3 for the second flight alone. One run froze on the second flight's half-landing (2.56 m); not reproduced in 5 later runs |

Final position is measured against ground truth; the goal tolerance applies to AMCL's estimate.
Final headings land near the yaw tolerance, because Regulated Pure Pursuit stops turning as soon as
it is inside it. Before integration, the ANYmal-D policy was also evaluated in a test harness:
10/10 runs up and down an 11-step flight of the office's dimensions at 0.5 and 1.0 m/s, and 5/5 up and
5/5 down the real stairwell from randomized starts (G1 and H1 policies were compared the same way; Go2's
is trained only on steps up to 0.12 m and never climbed the 0.15 m office steps).

## Limitations

- **Nav2 does not plan across floors.** Localization holds on every floor (ground truth), but the map
  server serves the main floor's map only: on another floor the global costmap and planner use the wrong
  layout (the local costmap, built from the LiDAR, is still right). On the main floor's map the stairs
  are free space and the flight down to the basement looks like floor; keep Spot out of stairwells
  with a keepout zone.
- **The first floor is a copy of the ground floor**, including its entrance and exterior glass, which open
  onto a 3.3 m drop outside the building. The main staircase reaches it (tested); the small stair core
  should too, through its upstairs door `BP_PanicDoor6`, which `--stair-access` opens (not tested).
- **ANYmal-D's actuator network is ANYmal-C's** (NVIDIA publishes no ANYmal-D network); fine in
  simulation, a sim-to-real gap on hardware.
- **Stairwells are narrow** (1.3 m in the office). ANYmal-D jammed against a wall when walked 0.15 m
  off the centreline; stairs must be approached centred.
- **Carpet strips are 1 cm ledges** with `--stair-access` (ANYmal-D's default): the office's carpet paths
  are raised overlays on the floor tiles, and once the floors have colliders a foot can catch on their
  edge while turning in place (seen once in two tours; Nav2's progress checker recovered it).
