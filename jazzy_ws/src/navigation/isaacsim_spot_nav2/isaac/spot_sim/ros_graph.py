# SPDX-License-Identifier: Apache-2.0
"""The OmniGraph that connects the simulation to ROS 2 (topic names relative to the namespace).

    sub  cmd_vel                              geometry_msgs/Twist
    pub  /clock                               rosgraph_msgs/Clock
    pub  odom, tf                             odom -> base_footprint -> base_link (set from Python)
    pub  tf_static                            base_link -> sensor frames
    pub  OS0/point_cloud                      sensor_msgs/PointCloud2
    pub  front_stereo/{left,right}/image_raw, front_stereo/{left,right}/camera_info,
         front_stereo/depth/image_rect_raw
"""

from dataclasses import dataclass, field

import omni.graph.core as og
from pxr import Sdf

from .geometry import to_og_quat, yaw_quat
from .odometry import OdomSample
from .sensors import StereoCameras

GRAPH_PATH = "/ROS_Spot"


@dataclass
class _GraphSpec:
    nodes: list = field(default_factory=list)
    values: list = field(default_factory=list)
    connections: list = field(default_factory=list)

    def publisher(self, name: str, node_type: str, tick: str = "Tick.outputs:tick", **inputs) -> None:
        """A node that runs on ``tick`` with the ROS context, plus its inputs."""
        self.nodes.append((name, node_type))
        self.values += [(f"{name}.inputs:{k}", v) for k, v in inputs.items()]
        self.connections += [(tick, f"{name}.inputs:execIn"), ("Context.outputs:context", f"{name}.inputs:context")]


@dataclass(frozen=True)
class GraphConfig:
    namespace: str
    body_prim: str
    lidar_prim: str | None = None
    cameras: StereoCameras | None = None
    camera_resolution: tuple = (640, 480)


class RosGraph:
    def __init__(self, config: GraphConfig):
        spec = _GraphSpec()
        _add_core(spec, config.namespace)
        _add_static_tf(spec, config)
        if config.lidar_prim or config.cameras:  # render products need one simulation frame first
            spec.nodes.append(("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"))
            spec.connections.append(("Tick.outputs:tick", "RunOnce.inputs:execIn"))
        if config.lidar_prim:
            _add_lidar(spec, config.namespace, config.lidar_prim)
        if config.cameras:
            _add_stereo(spec, config.namespace, config.cameras, config.camera_resolution)
        keys = og.Controller.Keys
        og.Controller.edit({"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
                           {keys.CREATE_NODES: spec.nodes, keys.SET_VALUES: spec.values,
                            keys.CONNECT: spec.connections})
        self._linear = self._attr("SubTwist.outputs:linearVelocity")
        self._angular = self._attr("SubTwist.outputs:angularVelocity")
        self._count = self._attr("MsgCounter.outputs:count")
        self._odom = {k: self._attr(f"Odom.inputs:{k}")
                      for k in ("position", "orientation", "linearVelocity", "angularVelocity")}
        self._tf_footprint = {k: self._attr(f"TfOdomFootprint.inputs:{k}") for k in ("translation", "rotation")}
        self._tf_body = {k: self._attr(f"TfFootprintBody.inputs:{k}") for k in ("translation", "rotation")}

    @staticmethod
    def _attr(name: str):
        return og.Controller.attribute(f"{GRAPH_PATH}/{name}")

    def twist(self):
        """The last received cmd_vel as (linear, angular) vectors."""
        return self._linear.get(), self._angular.get()

    def message_count(self) -> int:
        """Increments once per received cmd_vel message."""
        return int(self._count.get() or 0)

    def set_odometry(self, sample: OdomSample) -> None:
        """Values the graph publishes on its next tick (odom and the two dynamic transforms)."""
        x, y, yaw = sample.footprint
        vx, vy, wz = sample.velocity
        footprint_rotation = to_og_quat(yaw_quat(yaw))
        self._odom["position"].set([x, y, 0.0])
        self._odom["orientation"].set(footprint_rotation)
        self._odom["linearVelocity"].set([vx, vy, 0.0])
        self._odom["angularVelocity"].set([0.0, 0.0, wz])
        self._tf_footprint["translation"].set([x, y, 0.0])
        self._tf_footprint["rotation"].set(footprint_rotation)
        self._tf_body["translation"].set([0.0, 0.0, sample.body_height])
        self._tf_body["rotation"].set(to_og_quat(sample.body_tilt))


def _add_core(spec: _GraphSpec, ns: str) -> None:
    spec.nodes += [("Tick", "omni.graph.action.OnPlaybackTick"),
                   ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                   ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                   ("MsgCounter", "omni.graph.action.Counter")]
    spec.values.append(("SimTime.inputs:resetOnStop", True))
    spec.publisher("Clock", "isaacsim.ros2.bridge.ROS2PublishClock", topicName="clock")
    spec.publisher("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist", topicName="cmd_vel", nodeNamespace=ns)
    spec.connections.append(("SubTwist.outputs:execOut", "MsgCounter.inputs:execIn"))  # fires only on new messages
    spec.publisher("Odom", "isaacsim.ros2.bridge.ROS2PublishOdometry", topicName="odom", nodeNamespace=ns,
                   odomFrameId="odom", chassisFrameId="base_footprint")
    spec.publisher("TfOdomFootprint", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree", topicName="tf",
                   nodeNamespace=ns, parentFrameId="odom", childFrameId="base_footprint")
    spec.publisher("TfFootprintBody", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree", topicName="tf",
                   nodeNamespace=ns, parentFrameId="base_footprint", childFrameId="base_link")
    for name in ("Clock", "Odom", "TfOdomFootprint", "TfFootprintBody"):
        spec.connections.append(("SimTime.outputs:simulationTime", f"{name}.inputs:timeStamp"))


def _add_static_tf(spec: _GraphSpec, config: GraphConfig) -> None:
    """Sensor frames relative to base_link, computed from the prims themselves."""
    targets = [p for p in (config.lidar_prim,) if p]
    if config.cameras:
        targets += [config.cameras.left, config.cameras.right]
    if not targets:
        return
    spec.nodes.append(("ComputeTfStatic", "isaacsim.core.nodes.IsaacComputeTransformTree"))
    spec.values += [("ComputeTfStatic.inputs:parentPrim", [Sdf.Path(config.body_prim)]),
                    ("ComputeTfStatic.inputs:targetPrims", [Sdf.Path(p) for p in targets])]
    spec.connections.append(("Tick.outputs:tick", "ComputeTfStatic.inputs:execIn"))
    spec.publisher("TfStatic", "isaacsim.ros2.bridge.ROS2PublishTransformTree", tick="ComputeTfStatic.outputs:execOut",
                   topicName="tf_static", nodeNamespace=config.namespace, staticPublisher=True)
    spec.connections += [(f"ComputeTfStatic.outputs:{k}", f"TfStatic.inputs:{k}")
                         for k in ("parentFrames", "childFrames", "translations", "orientations")]
    spec.connections.append(("SimTime.outputs:simulationTime", "TfStatic.inputs:timeStamp"))


def _add_render_product(spec: _GraphSpec, name: str, camera_prim: str, resolution=None) -> None:
    spec.nodes.append((name, "isaacsim.core.nodes.IsaacCreateRenderProduct"))
    spec.values.append((f"{name}.inputs:cameraPrim", [Sdf.Path(camera_prim)]))
    if resolution:
        spec.values += [(f"{name}.inputs:width", resolution[0]), (f"{name}.inputs:height", resolution[1])]
    spec.connections.append(("RunOnce.outputs:step", f"{name}.inputs:execIn"))


def _render_publisher(spec: _GraphSpec, name: str, node_type: str, render_product: str, **inputs) -> None:
    spec.publisher(name, node_type, tick=f"{render_product}.outputs:execOut", **inputs)
    spec.connections.append((f"{render_product}.outputs:renderProductPath", f"{name}.inputs:renderProductPath"))


def _add_lidar(spec: _GraphSpec, ns: str, lidar_prim: str) -> None:
    _add_render_product(spec, "LidarRP", lidar_prim)
    _render_publisher(spec, "LidarPub", "isaacsim.ros2.bridge.ROS2RtxLidarHelper", "LidarRP",
                      type="point_cloud", topicName="point_cloud", nodeNamespace=f"{ns}/OS0", frameId="OS0")


def _add_stereo(spec: _GraphSpec, ns: str, cameras: StereoCameras, resolution) -> None:
    left, right = "front_stereo_left_optical_frame", "front_stereo_right_optical_frame"
    common = {"nodeNamespace": f"{ns}/front_stereo", "resetSimulationTimeOnStop": True}
    _add_render_product(spec, "LeftRP", cameras.left, resolution)
    _add_render_product(spec, "RightRP", cameras.right, resolution)
    _render_publisher(spec, "LeftPub", "isaacsim.ros2.bridge.ROS2CameraHelper", "LeftRP",
                      type="rgb", topicName="left/image_raw", frameId=left, **common)
    _render_publisher(spec, "RightPub", "isaacsim.ros2.bridge.ROS2CameraHelper", "RightRP",
                      type="rgb", topicName="right/image_raw", frameId=right, **common)
    _render_publisher(spec, "DepthPub", "isaacsim.ros2.bridge.ROS2CameraHelper", "LeftRP",
                      type="depth", topicName="depth/image_rect_raw", frameId=left, **common)
    _render_publisher(spec, "StereoInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper", "LeftRP",
                      topicName="left/camera_info", frameId=left,
                      topicNameRight="right/camera_info", frameIdRight=right, **common)
    spec.connections.append(("RightRP.outputs:renderProductPath", "StereoInfo.inputs:renderProductPathRight"))
