# SPDX-License-Identifier: Apache-2.0
"""Spot in Isaac Sim, walking under the bundled Isaac Lab locomotion policy, driven by ROS 2.

Run with Isaac Sim's Python:

    ~/isaacsim/python.sh spot_ros2_sim.py [--headless] [--scene office|grid|<file.usd>]

ROS 2 interface (domain from ROS_DOMAIN_ID, namespace from --namespace, default "spot"):
    sub  /spot/cmd_vel                         geometry_msgs/Twist (body frame: linear.x, linear.y, angular.z)
    pub  /clock                                rosgraph_msgs/Clock
    pub  /spot/odom                            nav_msgs/Odometry   odom -> base_footprint (ground truth)
    pub  /spot/tf                              odom -> base_footprint -> base_link
    pub  /spot/tf_static                       base_link -> OS0, front_stereo_{left,right}_optical_frame
    pub  /spot/OS0/point_cloud                 sensor_msgs/PointCloud2 (Ouster OS0, frame OS0)
    pub  /spot/front_stereo/{left,right}/image_raw, .../{left,right}/camera_info  (D455 stereo pair)
    pub  /spot/front_stereo/depth/image_rect_raw   (32FC1, metres, aligned with the left camera)

Frames: base_link is Spot's body (it bobs and tilts while walking). base_footprint is its projection
on the floor (x, y, yaw only), and odom lies on the floor at the spawn pose, so every height that
Nav2 uses (costmap bands, scan slices) is measured from the floor.

Safety: Isaac's Twist subscriber holds the last message forever, so a message counter detects fresh
messages and the command drops to zero after --cmd-timeout seconds of sim time without one.
Commands are clamped well inside the policy's trained range (vx -2..3, vy +-1.5, wz +-2). When the
command is zero, heading hold counters the ~40 deg turn the policy makes while settling.
"""

import argparse
import csv
import math
import os
import shlex
import sys

from isaacsim import SimulationApp

# realpath: with colcon --symlink-install this file is reached through install/, but scenes/ sits next to src/
SCENES_DIR = os.environ.get("SPOT_SIM_SCENES_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "scenes"))
SCENE_PRESETS = {
    # environment = the DD100 office scene with the DD100 and its clock graph switched off
    "office": (os.path.join(SCENES_DIR, "dd100_office_navigation.usd"), "/Robots,/World/clock"),
}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cpu",
    help="physics device; cpu is never slower for a single robot (measured on 6.1)",
)
parser.add_argument("--engine", choices=["physx", "newton"], default="physx")
parser.add_argument("--actuators", choices=["fast", "isaac"], default="fast",
                    help="fast: Spot's actuator model in one numpy pass (same model, ~20x cheaper); "
                         "isaac: Isaac Sim's per-joint ArticulationActuators")
parser.add_argument("--namespace", default="spot")
parser.add_argument("--scene", default="office", help="'office', 'grid', or a USD file to load as the environment")
parser.add_argument("--deactivate", default=None, help="comma-separated prims to switch off in the scene")
parser.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "YAW_DEG"))
parser.add_argument("--floor-z", type=float, default=0.0, help="height of the floor Spot stands on")
parser.add_argument("--no-lidar", action="store_true")
parser.add_argument("--no-camera", action="store_true")
parser.add_argument("--lidar-config", default="OS0_REV6_128ch10hz512res",
                    help="OS0 sensor variant. The asset defaults to 128x1024 (~110k points, ~1.3 MB per scan), "
                         "which arrives with multi-second gaps at Nav2's best-effort subscribers; 128x512 halves it")
parser.add_argument("--camera-skip", type=int, default=1, help="frames skipped between camera publishes")
parser.add_argument("--camera-rate", type=float, default=None,
                    help="render and publish the cameras at this rate (Hz) instead of every frame (uses --camera-skip 0)")
parser.add_argument("--camera-resolution", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
parser.add_argument("--lidar-mount", type=float, nargs=3, default=[0.0, 0.0, 0.10], metavar=("X", "Y", "DZ"),
                    help="OS0 position on the body: x, y, and height above the body's top surface")
parser.add_argument("--add-box", action="append", default=[], metavar="X,Y,SX,SY,SZ",
                    help="add a static obstacle box standing on the floor (testing)")
parser.add_argument("--cmd-timeout", type=float, default=0.5, help="sim seconds without cmd_vel before stopping")
parser.add_argument("--max-vx", type=float, default=1.5)
parser.add_argument("--min-vx", type=float, default=-1.0)
parser.add_argument("--max-vy", type=float, default=0.8)
parser.add_argument("--max-wz", type=float, default=1.2)
parser.add_argument("--min-cmd", type=float, nargs=3, default=[0.30, 0.22, 0.32], metavar=("VX", "VY", "WZ"),
                    help="deadband compensation: the policy stands still for small commands (measured from a "
                         "settled stand: vx <=0.2, vy <=0.15, wz <=0.2-0.25 are ignored), so a small non-zero "
                         "command is scaled up, keeping its direction, until one axis reaches these values")
parser.add_argument("--deadband-comp", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--hold-heading", action=argparse.BooleanOptionalAction, default=True,
                    help="when the command is zero, steer back to the heading held at stop")
parser.add_argument("--hold-gain", type=float, default=1.0, help="rad/s of yaw rate per rad of heading error")
parser.add_argument("--hold-max-wz", type=float, default=0.3)
parser.add_argument("--hold-engage-deg", type=float, default=3.0)
parser.add_argument("--hold-release-deg", type=float, default=1.0)
parser.add_argument("--view-offset", type=float, nargs=3, default=None, metavar=("FWD", "LEFT", "UP"),
                    help="initial GUI camera relative to Spot (metres ahead, to the left, above the floor); "
                         "default: the first of several views around Spot with a clear line of sight")
parser.add_argument("--snapshot-dir", default="", help="debugging: save follow-camera and close-up PNGs of Spot here")
parser.add_argument("--snapshot-every", type=float, default=5.0, help="sim seconds between snapshots")
parser.add_argument("--duration", type=float, default=0.0, help="stop after this many sim seconds (0 = run forever)")
parser.add_argument("--log", default="", help="optional CSV of per-frame commands and pose (for testing)")
# ros2 launch hands a substituted string such as sim_args:="--headless --spawn 1 2 0" over as ONE
# argv entry; split such entries the way a shell would, or every option in them is silently ignored.
_argv = [part for a in sys.argv[1:] for part in (shlex.split(a) if a.startswith("--") and " " in a else [a])]
# "--add-box -1.4,5.6,..." would be read as an option; bind such a value to its flag explicitly.
for _i in range(len(_argv) - 1):
    if _argv[_i] == "--add-box":
        _argv[_i], _argv[_i + 1] = f"--add-box={_argv[_i + 1]}", ""
_argv = [a for a in _argv if a != ""]
args, _unknown = parser.parse_known_args(_argv)
if _unknown:
    print(f"SPOTSIM: ignoring unknown arguments {_unknown}", flush=True)

extra_args = [f"--/exts/isaacsim.core.simulation_manager/default_engine={args.engine}"]
if args.engine == "newton":
    extra_args += ["--enable", "isaacsim.physics.newton", "--enable", "isaacsim.physics.newton.tensors"]
simulation_app = SimulationApp({"headless": args.headless, "extra_args": extra_args})

import carb
import numpy as np
import omni.graph.core as og
import omni.kit.app
import omni.timeline
import omni.usd

ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext in ("isaacsim.core.nodes", "isaacsim.ros2.bridge", "isaacsim.ros2.nodes"):
    ext_manager.set_extension_enabled_immediate(ext, True)

from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.robot.policy.examples import PolicyEnvConfig, RobotPolicyRunner, get_spot_spec
from isaacsim.ros2.nodes import set_isaac_name_override
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

NS = args.namespace
ROBOT_PRIM = "/World/Spot"
GRAPH = "/ROS_Spot"
ASSETS = get_assets_root_path()


def info(msg):
    print(f"SPOTSIM: {msg}", flush=True)


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quat_mul(a, b):  # WXYZ
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def yaw_quat(yaw):  # WXYZ
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def quat_yaw(q):  # WXYZ
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---------------------------------------------------------------- environment
stage = omni.usd.get_context().get_stage()
if args.scene == "grid":
    set_stage_up_axis("Z")
    set_stage_units(meters_per_unit=1.0)
    define_prim("/World/Ground", "Xform").GetReferences().AddReference(
        ASSETS + "/Isaac/Environments/Grid/default_environment.usd")
    define_prim("/World/PhysicsScene", "PhysicsScene")
else:
    scene_file, default_off = SCENE_PRESETS.get(args.scene, (args.scene, ""))
    if not os.path.isfile(scene_file):
        raise SystemExit(f"scene not found: {scene_file}")
    # Load as a sublayer of the in-memory stage, so nothing below ever edits the scene file.
    stage.GetRootLayer().subLayerPaths.append(scene_file)
    for _ in range(10):
        simulation_app.update()
    for path in filter(None, (args.deactivate if args.deactivate is not None else default_off).split(",")):
        prim = stage.GetPrimAtPath(path.strip())
        if prim.IsValid():
            prim.SetActive(False)
            info(f"deactivated {path.strip()}")
    info(f"scene: {scene_file}")

for i, spec_str in enumerate(args.add_box):
    x, y, sx, sy, sz = (float(v) for v in spec_str.split(","))
    cube = UsdGeom.Cube.Define(stage, f"/World/TestObstacle_{i}")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, args.floor_z + sz / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    info(f"test obstacle box at ({x}, {y}) size ({sx}, {sy}, {sz})")

SimulationManager.set_physics_sim_device(args.device)

# ---------------------------------------------------------------- robot
spawn_x, spawn_y, spawn_yaw = args.spawn[0], args.spawn[1], math.radians(args.spawn[2])
spec = get_spot_spec()
spot = RobotPolicyRunner(spec, prim_path=ROBOT_PRIM, position=[spawn_x, spawn_y, args.floor_z + 0.5],
                         orientation=list(yaw_quat(spawn_yaw)))
spot.spawn()
timing = PolicyEnvConfig.from_file(spec.engines[args.engine].env_config_path).timing
frame_dt = timing.render_interval * timing.physics_dt
RenderingManager.set_dt(frame_dt)
SimulationManager.set_physics_dt(spot.physics_dt)
# SimulationManager only updates physics scenes it registered itself; a scene that arrives with a
# loaded USD file keeps its own rate (60 Hz in the office scene), and the policy, trained at
# 500 Hz, collapses immediately. Set the rate on every physics scene explicitly.
# The same goes for --device: the office scene ships with GPU dynamics and a GPU broadphase, so
# "cpu" used to run GPU physics read back to the CPU every step (half the frame time with sensors).
_steps = int(round(1.0 / spot.physics_dt))
_gpu = args.device == "cuda"
for _p in stage.Traverse():
    if _p.IsA(UsdPhysics.Scene):
        _api = PhysxSchema.PhysxSceneAPI.Apply(_p)
        _api.CreateTimeStepsPerSecondAttr().Set(_steps)
        _api.CreateEnableGPUDynamicsAttr().Set(_gpu)
        _api.CreateBroadphaseTypeAttr().Set("GPU" if _gpu else "MBP")
        info(f"physics scene {_p.GetPath()}: {_steps} steps/s, {'GPU' if _gpu else 'CPU'} dynamics")

BODY = f"{ROBOT_PRIM}/body"
if not stage.GetPrimAtPath(BODY).IsValid():
    raise SystemExit(f"no body link at {BODY}; links: {[c.GetName() for c in stage.GetPrimAtPath(ROBOT_PRIM).GetChildren()]}")
set_isaac_name_override(BODY, "base_link")
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
body_box = bbox.ComputeUntransformedBound(stage.GetPrimAtPath(BODY)).ComputeAlignedRange()
bmin, bmax = body_box.GetMin(), body_box.GetMax()
info(f"body bounds (body frame): x[{bmin[0]:.3f},{bmax[0]:.3f}] y[{bmin[1]:.3f},{bmax[1]:.3f}] z[{bmin[2]:.3f},{bmax[2]:.3f}]")


def mount(path, asset, translate, yaw_deg=0.0, pitch_deg=0.0):
    prim = define_prim(path, "Xform")
    prim.GetReferences().AddReference(asset)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, pitch_deg, yaw_deg))
    return prim


tf_targets = []
LIDAR_PRIM = f"{BODY}/OS0/sensor"
if not args.no_lidar:
    mx, my, dz = args.lidar_mount
    os0 = mount(f"{BODY}/OS0", ASSETS + "/Isaac/Sensors/Ouster/OS0/OS0.usd", (mx, my, bmax[2] + dz))
    variants = os0.GetVariantSets().GetVariantSet("sensor")
    if args.lidar_config not in variants.GetVariantNames():
        raise SystemExit(f"--lidar-config {args.lidar_config!r} not in {variants.GetVariantNames()}")
    variants.SetVariantSelection(args.lidar_config)
    simulation_app.update()
    if not stage.GetPrimAtPath(LIDAR_PRIM).IsValid():
        raise SystemExit(f"OS0 sensor prim missing at {LIDAR_PRIM}")
    set_isaac_name_override(LIDAR_PRIM, "OS0")  # TF frame == point cloud frame_id, at the real sensor origin
    tf_targets.append(LIDAR_PRIM)
    if dz > 0.005:
        # Visible mast from the back to the sensor base (visual only: no collider, no mass), so the
        # raised OS0 is mounted rather than floating. The height keeps the sensor above Spot's back,
        # which the self-filters in spot_navigation.launch.xml were measured with.
        mast = UsdGeom.Cylinder.Define(stage, f"{BODY}/OS0_mast")
        mast.CreateRadiusAttr(0.035)
        mast.CreateHeightAttr(dz)
        mast.CreateAxisAttr("Z")
        mast.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.12)])
        UsdGeom.Xformable(mast).AddTranslateOp().Set(Gf.Vec3d(mx, my, bmax[2] + dz / 2))
        plate = UsdGeom.Cube.Define(stage, f"{BODY}/OS0_mount_plate")
        plate.CreateSizeAttr(1.0)
        plate.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.12)])
        pxf = UsdGeom.Xformable(plate)
        pxf.AddTranslateOp().Set(Gf.Vec3d(mx, my, bmax[2] + 0.004))
        pxf.AddScaleOp().Set(Gf.Vec3f(0.16, 0.12, 0.008))
    info(f"OS0 ({args.lidar_config}) mounted at body ({mx}, {my}, {bmax[2] + dz:.3f}) on a {dz:.2f} m mast")

CAM_LEFT = f"{BODY}/front_stereo/RSD455/Camera_OmniVision_OV9782_Left"
CAM_RIGHT = f"{BODY}/front_stereo/RSD455/Camera_OmniVision_OV9782_Right"
if not args.no_camera:
    # front face of the body, near its top edge, looking forward
    mount(f"{BODY}/front_stereo", ASSETS + "/Isaac/Sensors/RealSense/D455/rsd455.usd",
          (bmax[0] + 0.02, 0.0, bmax[2] - 0.03))
    simulation_app.update()
    # NVIDIA's D455 asset is a rigid body with mass but no collider. Nested under a robot link it is
    # not attached: it becomes its own body and falls through the floor forever. Make it visual-only.
    for p in Usd.PrimRange(stage.GetPrimAtPath(f"{BODY}/front_stereo")):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            if p.HasAPI(api):
                p.RemoveAPI(api)
    for cam in (CAM_LEFT, CAM_RIGHT):
        if not stage.GetPrimAtPath(cam).IsValid():
            kids = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(f"{BODY}/front_stereo")) if p.IsA(UsdGeom.Camera)]
            raise SystemExit(f"camera prim missing: {cam}; cameras found: {kids}")
    set_isaac_name_override(CAM_LEFT, "front_stereo_left_optical_frame")
    set_isaac_name_override(CAM_RIGHT, "front_stereo_right_optical_frame")
    tf_targets += [CAM_LEFT, CAM_RIGHT]
    info(f"D455 stereo mounted at body front ({bmax[0] + 0.02:.3f}, 0, {bmax[2] - 0.03:.3f})")

# ---------------------------------------------------------------- ROS 2 graph
K = og.Controller.Keys
nodes = [
    ("Tick", "omni.graph.action.OnPlaybackTick"),
    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
    ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
    ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
    ("MsgCounter", "omni.graph.action.Counter"),
    ("Odom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
    ("TfOdomFootprint", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ("TfFootprintBody", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
]
values = [
    ("SimTime.inputs:resetOnStop", True),
    ("Clock.inputs:topicName", "clock"),
    ("SubTwist.inputs:topicName", "cmd_vel"), ("SubTwist.inputs:nodeNamespace", NS),
    ("Odom.inputs:topicName", "odom"), ("Odom.inputs:nodeNamespace", NS),
    ("Odom.inputs:odomFrameId", "odom"), ("Odom.inputs:chassisFrameId", "base_footprint"),
    ("TfOdomFootprint.inputs:topicName", "tf"), ("TfOdomFootprint.inputs:nodeNamespace", NS),
    ("TfOdomFootprint.inputs:parentFrameId", "odom"), ("TfOdomFootprint.inputs:childFrameId", "base_footprint"),
    ("TfFootprintBody.inputs:topicName", "tf"), ("TfFootprintBody.inputs:nodeNamespace", NS),
    ("TfFootprintBody.inputs:parentFrameId", "base_footprint"), ("TfFootprintBody.inputs:childFrameId", "base_link"),
]
connects = [
    ("SimTime.outputs:simulationTime", "Clock.inputs:timeStamp"),
    ("Tick.outputs:tick", "Clock.inputs:execIn"),
    ("Tick.outputs:tick", "SubTwist.inputs:execIn"),
    ("SubTwist.outputs:execOut", "MsgCounter.inputs:execIn"),  # fires only on new messages
]
for pub in ("Clock", "SubTwist", "Odom", "TfOdomFootprint", "TfFootprintBody"):
    connects.append(("Context.outputs:context", f"{pub}.inputs:context"))
for pub in ("Odom", "TfOdomFootprint", "TfFootprintBody"):
    connects += [("Tick.outputs:tick", f"{pub}.inputs:execIn"), ("SimTime.outputs:simulationTime", f"{pub}.inputs:timeStamp")]

if tf_targets:  # static sensor frames, computed from the prims themselves
    nodes += [("ComputeTfStatic", "isaacsim.core.nodes.IsaacComputeTransformTree"),
              ("TfStatic", "isaacsim.ros2.bridge.ROS2PublishTransformTree")]
    values += [("ComputeTfStatic.inputs:parentPrim", [Sdf.Path(BODY)]),
               ("ComputeTfStatic.inputs:targetPrims", [Sdf.Path(p) for p in tf_targets]),
               ("TfStatic.inputs:topicName", "tf_static"), ("TfStatic.inputs:nodeNamespace", NS),
               ("TfStatic.inputs:staticPublisher", True)]
    connects += [("Tick.outputs:tick", "ComputeTfStatic.inputs:execIn"),
                 ("ComputeTfStatic.outputs:execOut", "TfStatic.inputs:execIn"),
                 ("ComputeTfStatic.outputs:parentFrames", "TfStatic.inputs:parentFrames"),
                 ("ComputeTfStatic.outputs:childFrames", "TfStatic.inputs:childFrames"),
                 ("ComputeTfStatic.outputs:translations", "TfStatic.inputs:translations"),
                 ("ComputeTfStatic.outputs:orientations", "TfStatic.inputs:orientations"),
                 ("SimTime.outputs:simulationTime", "TfStatic.inputs:timeStamp"),
                 ("Context.outputs:context", "TfStatic.inputs:context")]

if not (args.no_lidar and args.no_camera):  # render products need one simulation frame first
    nodes.append(("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"))
    connects.append(("Tick.outputs:tick", "RunOnce.inputs:execIn"))

if not args.no_lidar:
    nodes += [("LidarRP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
              ("LidarPub", "isaacsim.ros2.bridge.ROS2RtxLidarHelper")]
    values += [("LidarRP.inputs:cameraPrim", [Sdf.Path(LIDAR_PRIM)]),
               ("LidarPub.inputs:type", "point_cloud"), ("LidarPub.inputs:topicName", "point_cloud"),
               ("LidarPub.inputs:nodeNamespace", f"{NS}/OS0"), ("LidarPub.inputs:frameId", "OS0")]
    connects += [("RunOnce.outputs:step", "LidarRP.inputs:execIn"),
                 ("LidarRP.outputs:execOut", "LidarPub.inputs:execIn"),
                 ("LidarRP.outputs:renderProductPath", "LidarPub.inputs:renderProductPath"),
                 ("Context.outputs:context", "LidarPub.inputs:context")]

if not args.no_camera:
    w, h = args.camera_resolution
    cam_ns = f"{NS}/front_stereo"
    nodes += [("LeftRP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
              ("RightRP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
              ("LeftPub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
              ("RightPub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
              ("DepthPub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
              ("StereoInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
    values += [("LeftRP.inputs:cameraPrim", [Sdf.Path(CAM_LEFT)]), ("LeftRP.inputs:width", w), ("LeftRP.inputs:height", h),
               ("RightRP.inputs:cameraPrim", [Sdf.Path(CAM_RIGHT)]), ("RightRP.inputs:width", w), ("RightRP.inputs:height", h),
               ("LeftPub.inputs:type", "rgb"), ("LeftPub.inputs:topicName", "left/image_raw"),
               ("LeftPub.inputs:frameId", "front_stereo_left_optical_frame"),
               ("RightPub.inputs:type", "rgb"), ("RightPub.inputs:topicName", "right/image_raw"),
               ("RightPub.inputs:frameId", "front_stereo_right_optical_frame"),
               ("DepthPub.inputs:type", "depth"), ("DepthPub.inputs:topicName", "depth/image_rect_raw"),
               ("DepthPub.inputs:frameId", "front_stereo_left_optical_frame"),
               ("StereoInfo.inputs:topicName", "left/camera_info"),
               ("StereoInfo.inputs:frameId", "front_stereo_left_optical_frame"),
               ("StereoInfo.inputs:topicNameRight", "right/camera_info"),
               ("StereoInfo.inputs:frameIdRight", "front_stereo_right_optical_frame")]
    if args.camera_rate:  # the renderer then draws the cameras only at this rate, not every frame
        for cam in (CAM_LEFT, CAM_RIGHT):
            prim = stage.GetPrimAtPath(cam)
            prim.ApplyAPI("OmniSensorAPI")
            prim.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(float(args.camera_rate))
    for n in ("LeftPub", "RightPub", "DepthPub", "StereoInfo"):
        values += [(f"{n}.inputs:nodeNamespace", cam_ns), (f"{n}.inputs:resetSimulationTimeOnStop", True),
                   (f"{n}.inputs:frameSkipCount", 0 if args.camera_rate else args.camera_skip)]
        connects.append(("Context.outputs:context", f"{n}.inputs:context"))
    connects += [("RunOnce.outputs:step", "LeftRP.inputs:execIn"), ("RunOnce.outputs:step", "RightRP.inputs:execIn"),
                 ("LeftRP.outputs:execOut", "LeftPub.inputs:execIn"),
                 ("LeftRP.outputs:execOut", "DepthPub.inputs:execIn"),
                 ("LeftRP.outputs:execOut", "StereoInfo.inputs:execIn"),
                 ("RightRP.outputs:execOut", "RightPub.inputs:execIn"),
                 ("LeftRP.outputs:renderProductPath", "LeftPub.inputs:renderProductPath"),
                 ("LeftRP.outputs:renderProductPath", "DepthPub.inputs:renderProductPath"),
                 ("LeftRP.outputs:renderProductPath", "StereoInfo.inputs:renderProductPath"),
                 ("RightRP.outputs:renderProductPath", "RightPub.inputs:renderProductPath"),
                 ("RightRP.outputs:renderProductPath", "StereoInfo.inputs:renderProductPathRight")]

og.Controller.edit({"graph_path": GRAPH, "evaluator_name": "execution"},
                   {K.CREATE_NODES: nodes, K.SET_VALUES: values, K.CONNECT: connects})


def attr(name):
    return og.Controller.attribute(f"{GRAPH}/{name}")


twist_lin, twist_ang, msg_count = attr("SubTwist.outputs:linearVelocity"), attr("SubTwist.outputs:angularVelocity"), attr("MsgCounter.outputs:count")
odom_in = {k: attr(f"Odom.inputs:{k}") for k in ("position", "orientation", "linearVelocity", "angularVelocity")}
tf_fp = {k: attr(f"TfOdomFootprint.inputs:{k}") for k in ("translation", "rotation")}
tf_body = {k: attr(f"TfFootprintBody.inputs:{k}") for k in ("translation", "rotation")}


def to_og_quat(q):  # WXYZ -> OmniGraph quatd (IJKR = x, y, z, w)
    return [q[1], q[2], q[3], q[0]]


# ---------------------------------------------------------------- odometry (ground truth, floor-aligned)
class OdomState:
    def __init__(self):
        self.prev = None          # (x, y, yaw) of base_footprint in odom
        self.vel = np.zeros(3)    # filtered (vx, vy, wz) in base_footprint frame
        self.yaw = 0.0

    def update(self, dt):
        p, q = spot.articulation.get_world_poses()
        p = p.numpy().reshape(-1)
        q = tuple(float(v) for v in q.numpy().reshape(-1))
        yaw_w = quat_yaw(q)
        self.yaw = yaw_w
        # base_footprint in odom (odom = spawn pose on the floor)
        dx, dy = float(p[0]) - spawn_x, float(p[1]) - spawn_y
        c, s = math.cos(-spawn_yaw), math.sin(-spawn_yaw)
        fx, fy, fyaw = c * dx - s * dy, s * dx + c * dy, wrap(yaw_w - spawn_yaw)
        # base_link relative to base_footprint: height + roll/pitch only
        q_rel = quat_mul(yaw_quat(-yaw_w), q)
        z_rel = float(p[2]) - args.floor_z
        if self.prev is not None and dt > 0:
            px, py, pyaw = self.prev
            ddx, ddy = fx - px, fy - py
            cy, sy = math.cos(-pyaw), math.sin(-pyaw)
            raw = np.array([(cy * ddx - sy * ddy) / dt, (sy * ddx + cy * ddy) / dt, wrap(fyaw - pyaw) / dt])
            self.vel = 0.8 * self.vel + 0.2 * raw  # gait bob makes raw finite differences noisy
        self.prev = (fx, fy, fyaw)
        q_fp = yaw_quat(fyaw)
        odom_in["position"].set([fx, fy, 0.0])
        odom_in["orientation"].set(to_og_quat(q_fp))
        odom_in["linearVelocity"].set([float(self.vel[0]), float(self.vel[1]), 0.0])
        odom_in["angularVelocity"].set([0.0, 0.0, float(self.vel[2])])
        tf_fp["translation"].set([fx, fy, 0.0])
        tf_fp["rotation"].set(to_og_quat(q_fp))
        tf_body["translation"].set([0.0, 0.0, z_rel])
        tf_body["rotation"].set(to_og_quat(q_rel))
        return float(p[0]), float(p[1]), float(p[2]), yaw_w


odom = OdomState()
last_pose = (spawn_x, spawn_y, args.floor_z + 0.5, spawn_yaw)

# ---------------------------------------------------------------- command handling
class HeadingHold:
    """Holds the heading captured when the command goes to zero, with hysteresis so the policy
    isn't fed a stream of tiny commands once it is back on heading."""

    def __init__(self):
        self.target, self.active = None, False

    def __call__(self, command, yaw):
        if not args.hold_heading or np.any(np.abs(command) > 1e-3):
            self.target, self.active = None, False
            return command
        if self.target is None:
            self.target = yaw
        error = wrap(self.target - yaw)
        if abs(error) > math.radians(args.hold_engage_deg):
            self.active = True
        elif abs(error) < math.radians(args.hold_release_deg):
            self.active = False
        if not self.active:
            return command
        return np.array([0.0, 0.0, min(max(args.hold_gain * error, -args.hold_max_wz), args.hold_max_wz)], dtype=np.float32)


heading_hold = HeadingHold()


def clamped_command():
    lin, ang = twist_lin.get(), twist_ang.get()
    return np.array([min(max(float(lin[0]), args.min_vx), args.max_vx),
                     min(max(float(lin[1]), -args.max_vy), args.max_vy),
                     min(max(float(ang[2]), -args.max_wz), args.max_wz)], dtype=np.float32)


MIN_CMD = np.array(args.min_cmd, dtype=np.float32)


def deadband_compensated(command):
    """Scale a small non-zero command up to where the policy actually walks, keeping its direction.

    Without this, a planner's slow final approach and small corrections produce no motion at all:
    Spot just stands, short of the goal or drifting past it."""
    if not args.deadband_comp:
        return command
    if np.all(np.abs(command) < 0.02):  # treat as an intentional stop
        return command
    # Hysteresis (measured): only starting from a settled stand needs the boost. Once Spot is
    # stepping it follows small commands, and boosting them then just causes overshoot.
    v = odom.vel
    if math.hypot(float(v[0]), float(v[1])) > 0.05 or abs(float(v[2])) > 0.08:
        return command
    reach = float(np.max(np.abs(command) / MIN_CMD))  # 1.0 = dominant axis at its threshold
    return command / reach if reach < 1.0 else command


# ---------------------------------------------------------------- actuators
class FastActuators:
    """Spot's actuator model (Newton PD + command delay + effort clamping, as Isaac Sim builds it)
    computed for all joints in one numpy pass.

    Isaac's ArticulationActuators steps every joint as its own Newton actuator: 12 joints x 5 Warp
    launches before every 2 ms physics step, ~3 ms of Python launch overhead per step, measured as
    ~80% of the frame time. The parameters are read from those same actuator objects, so the
    model does not change; anything this class does not reproduce exactly raises, and the caller
    keeps Isaac's implementation."""

    def __init__(self, isaac_actuators):
        import warp as wp

        self.wp, self.isaac = wp, isaac_actuators
        self.art = isaac_actuators._articulation
        if len(self.art) != 1:
            raise NotImplementedError("one robot per articulation only")
        dofs, kp, kd, delay = [], [], [], []
        box = []  # symmetric effort limit (inf = none)
        dc = []  # (saturation, velocity limit, max effort) or None
        lut = []  # (positions, efforts) or None
        for act in isaac_actuators._actuators:
            ctrl = act.controller
            if type(ctrl).__name__ != "ControllerPD" or ctrl.const_effort is not None:
                raise NotImplementedError(f"controller {type(ctrl).__name__}")
            if len(act.clamping) > 1:
                raise NotImplementedError("more than one clamp per joint")
            dofs.append(int(act.indices.numpy()[0]))
            kp.append(float(ctrl.kp.numpy()[0]))
            kd.append(float(ctrl.kd.numpy()[0]))
            delay.append(int(act.delay.delay_steps.numpy()[0]) if act.delay is not None else 0)
            box.append(np.inf); dc.append(None); lut.append(None)
            for clamp in act.clamping:
                kind = type(clamp).__name__
                if kind == "ClampingMaxEffort":
                    box[-1] = float(clamp.max_effort.numpy()[0])
                elif kind == "ClampingDCMotor":
                    dc[-1] = tuple(float(a.numpy()[0]) for a in (clamp.saturation_effort, clamp.velocity_limit,
                                                                  clamp.max_motor_effort))
                elif kind == "ClampingPositionBased":
                    lut[-1] = (clamp.lookup_positions.numpy().astype(np.float64),
                               clamp.lookup_efforts.numpy().astype(np.float64))
                else:
                    raise NotImplementedError(f"clamp {kind}")
        if dofs != sorted(dofs):
            raise NotImplementedError("actuators not in DOF order")
        self.dofs = np.array(dofs)
        if dofs != list(isaac_actuators._actuated_dof_indices):
            raise NotImplementedError("actuated DOFs differ from the actuator list")
        self.device = isaac_actuators._device  # where Isaac keeps its effort buffer and DOF indices
        self.dofs_wp = isaac_actuators._actuated_dof_indices_wp
        self.kp, self.kd = np.array(kp), np.array(kd)
        self.delay, self.box, self.dc, self.lut = np.array(delay), np.array(box), dc, lut
        self.depth = max(1, int(self.delay.max()))
        self.lag = np.maximum(self.delay - 1, 0)
        self.history = []  # pushed (target_pos, target_vel, feedforward), newest last
        self.description = (f"{len(dofs)} joints, kp {sorted(set(kp))}, kd {sorted(set(kd))}, "
                            f"delay {sorted(set(delay))} steps, {sum(c is not None for c in lut)} position-limited, "
                            f"{sum(c is not None for c in dc)} DC-motor, {int(np.isfinite(self.box).sum())} box-limited")

    def install(self):
        # SimulationManager calls the pre-physics callback by name on the instance, and the policy
        # runner calls reset() on restarts, so instance attributes replace both.
        self.isaac.step_actuators = self.step
        self.isaac.reset = self.reset
        self.reset()
        return self

    def reset(self):
        self.history = []

    def step(self, step_dt, context=None):
        effort = self.compute()
        self.art.set_dof_efforts(self.wp.array(effort.astype(np.float32).reshape(1, -1), dtype=self.wp.float32,
                                               device=self.device), dof_indices=self.dofs_wp)

    def compute(self):
        art, d = self.art, self.dofs
        q = art.get_dof_positions().numpy()[0, d].astype(np.float64)
        qd = art.get_dof_velocities().numpy()[0, d].astype(np.float64)
        cmd = (art.get_dof_position_targets().numpy()[0, d].astype(np.float64),
               art.get_dof_velocity_targets().numpy()[0, d].astype(np.float64),
               self.isaac._feedforward_target_efforts.numpy()[0, d].astype(np.float64))
        if self.history:  # Newton's delay: the command pushed `delay` steps ago, or the oldest one
            n = len(self.history)
            delayed = [np.where(self.delay > 0,
                                np.array([self.history[n - 1 - min(lag, n - 1)][k][j] for j, lag in enumerate(self.lag)]),
                                cmd[k]) for k in range(3)]
        else:
            delayed = cmd
        effort = self.kp * (delayed[0] - q) + self.kd * (delayed[1] - qd) + delayed[2]
        effort = np.clip(effort, -self.box, self.box)
        for j, table in enumerate(self.lut):
            if table is not None:
                limit = np.interp(q[j], table[0], table[1])
                effort[j] = min(max(effort[j], -limit), limit)
        for j, motor in enumerate(self.dc):
            if motor is not None:
                sat, vlim, emax = motor
                corner = vlim * (1.0 + emax / sat) if sat > 0.0 else vlim
                v = min(max(qd[j], -corner), corner)
                effort[j] = min(max(effort[j], max(sat * (-1.0 - v / vlim), -emax)), min(sat * (1.0 - v / vlim), emax))
        self.history.append(cmd)
        if len(self.history) > self.depth:
            self.history.pop(0)
        return effort


def install_fast_actuators():
    if args.actuators != "fast" or getattr(spot, "_actuators", None) is None:
        return None
    try:
        fast = FastActuators(spot._actuators).install()
        info(f"fast actuators: {fast.description}")
        return fast
    except Exception as error:  # noqa: BLE001 - keep Isaac's implementation
        info(f"fast actuators not used, keeping Isaac's ArticulationActuators: {error}")
        return None


# ---------------------------------------------------------------- policy stepping
base_command = np.zeros(3, dtype=np.float32)
first_step, reset_needed, policy_failed = True, False, False
fast_actuators = None


def on_physics_step(step_size, context):
    global first_step, reset_needed, policy_failed, last_pose, fast_actuators
    if policy_failed:
        return
    try:
        if first_step:
            spot.restart_from_default_state(base_command)
            if fast_actuators is None:
                fast_actuators = install_fast_actuators() or False
            odom.prev = None
            first_step = False
        elif reset_needed:
            reset_needed, first_step = False, True
        else:
            spot.step(step_size, base_command)
        last_pose = odom.update(step_size)  # odom/TF inputs match the state the graph publishes this frame
    except Exception as error:  # noqa: BLE001 - must not raise inside the physics callback
        policy_failed = True
        carb.log_error(f"spot_ros2_sim: policy failed, robot stopped: {error}")


def view_around_robot(x, y, yaw, offset, look_up=0.45, look_ahead=0.3):
    """Camera eye/target in world coordinates, expressed relative to the robot's pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = offset
    eye = [x + c * ox - s * oy, y + s * ox + c * oy, args.floor_z + oz]
    target = [x + c * look_ahead, y + s * look_ahead, args.floor_z + look_up]
    return eye, target


VIEW_CANDIDATES = [(-3.0, 1.8, 1.8), (-3.0, -1.8, 1.8), (2.8, 1.8, 1.6), (2.8, -1.8, 1.6), (0.3, 3.2, 1.8), (0.3, -3.2, 1.8)]


def choose_view_offset():
    """First candidate camera offset whose line of sight to Spot is not blocked by the scene."""
    if args.view_offset is not None:
        return tuple(args.view_offset)
    try:
        import omni.physx
        query = omni.physx.get_physx_scene_query_interface()
        def blocked(origin, end, margin):
            d = np.subtract(end, origin)
            dist = float(np.linalg.norm(d))
            return query.raycast_closest(tuple(map(float, origin)), tuple(d / dist), dist - margin)["hit"]

        for offset in VIEW_CANDIDATES:
            eye, target = view_around_robot(spawn_x, spawn_y, spawn_yaw, offset)
            feet = [target[0], target[1], args.floor_z + 0.1]
            # sees Spot's body and feet (rays stop short of Spot itself) ...
            if blocked(eye, target, 0.9) or blocked(eye, feet, 0.9):
                continue
            # ... and stands in open floor: nothing within 1.2 m around the camera at 0.4 m and 1.0 m height,
            # so furniture right next to it (a reception counter, say) doesn't fill the foreground
            ring_clear = True
            for h in (0.4, 1.0):
                for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                    o = (eye[0], eye[1], args.floor_z + h)
                    if blocked(o, (o[0] + 1.2 * math.cos(a), o[1] + 1.2 * math.sin(a), o[2]), 0.0):
                        ring_clear = False
                        break
                if not ring_clear:
                    break
            if ring_clear:
                return offset
    except Exception as error:  # noqa: BLE001
        info(f"view selection fell back to the default: {error}")
    return VIEW_CANDIDATES[0]


snapshot = None
if args.snapshot_dir:  # debugging: render Spot from a follow camera and a close-up, at intervals
    import omni.replicator.core as rep
    from isaacsim.core.rendering_manager import ViewportManager
    from PIL import Image as PILImage
    os.makedirs(args.snapshot_dir, exist_ok=True)
    snapshot = {}
    for name in ("view", "closeup"):
        cam = UsdGeom.Camera.Define(stage, f"/World/SnapshotCam_{name}")
        cam.CreateFocalLengthAttr(14.0 if name == "view" else 18.0)
        rp = rep.create.render_product(str(cam.GetPath()), (1280, 720))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        snapshot[name] = (str(cam.GetPath()), ann)
    next_snapshot = 1.0

_cb = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()
view_offset = choose_view_offset()
info(f"view offset from Spot: {view_offset}")
if not args.headless:
    # The scene file stores its own viewport camera (outside the building); start next to Spot instead.
    try:
        from isaacsim.core.rendering_manager import ViewportManager
        eye, target = view_around_robot(spawn_x, spawn_y, spawn_yaw, view_offset)
        ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=eye, target=target)
        info(f"initial view: offset {view_offset} from Spot")
    except Exception as error:  # noqa: BLE001 - a viewport problem must not stop the simulation
        info(f"could not set the initial viewport camera: {error}")
import time as _time
rtf_wall, rtf_sim = _time.time(), 0.0

log_writer, log_file = None, None
if args.log:
    log_file = open(args.log, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["t", "msgs", "in_vx", "in_vy", "in_wz", "vx", "vy", "wz", "x", "y", "z", "yaw", "stale"])

info(f"ready: cmd_vel on /{NS}/cmd_vel, spawn ({spawn_x}, {spawn_y}, {args.spawn[2]} deg), physics {args.device}")
sim_t, last_count, last_msg_t = 0.0, 0, -1e9
while simulation_app.is_running():
    simulation_app.update()
    if not SimulationManager.is_simulating():
        reset_needed = True
        continue
    sim_t += frame_dt

    count = int(msg_count.get() or 0)
    if count != last_count:
        last_count, last_msg_t = count, sim_t
    stale = (sim_t - last_msg_t) > args.cmd_timeout
    requested = np.zeros(3, dtype=np.float32) if stale else deadband_compensated(clamped_command())
    base_command = heading_hold(requested, last_pose[3])

    if log_writer:
        lin, ang = twist_lin.get(), twist_ang.get()
        log_writer.writerow([f"{sim_t:.3f}", count, lin[0], lin[1], ang[2], *base_command.tolist(), *last_pose, int(stale)])

    if _time.time() - rtf_wall >= 10.0:
        info(f"real-time factor {(sim_t - rtf_sim) / (_time.time() - rtf_wall):.2f} (sim t {sim_t:.1f} s)")
        rtf_wall, rtf_sim = _time.time(), sim_t

    if snapshot is not None:
        px, py, _, pyaw = last_pose
        for name, (cam_path, _) in snapshot.items():  # cameras follow the robot
            offset = view_offset if name == "view" else (0.25, -1.5, 0.95)
            eye, target = view_around_robot(px, py, pyaw, offset, look_up=0.55 if name == "closeup" else 0.45,
                                            look_ahead=0.0 if name == "closeup" else 0.3)
            ViewportManager.set_camera_view(cam_path, eye=eye, target=target)
        if sim_t >= next_snapshot:
            for name, (_, ann) in snapshot.items():
                data = ann.get_data()
                if data is not None and getattr(data, "size", 0):
                    PILImage.fromarray(np.asarray(data)[:, :, :3]).save(os.path.join(args.snapshot_dir, f"t{sim_t:05.1f}_{name}.png"))
            next_snapshot += args.snapshot_every

    if args.duration and sim_t >= args.duration:
        break

if log_file:
    log_file.close()
timeline.stop()
SimulationManager.deregister_callback(_cb)
spot.close()
simulation_app.close()
