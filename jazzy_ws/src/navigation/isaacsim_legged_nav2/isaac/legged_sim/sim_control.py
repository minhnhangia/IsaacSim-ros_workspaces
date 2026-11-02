# SPDX-License-Identifier: Apache-2.0
"""Reset the robot from ROS, and publish its health, without restarting the simulation.

    reset_pose  geometry_msgs/PoseStamped (in the robot's namespace): stand the robot up at
                (x, y, yaw) on the floor at z. The session, the sim clock and every sensor keep
                running, so a long recording survives a fall.
    sim_status  std_msgs/String, JSON once per sim second: sim time, body roll / pitch / height,
                upright, resets so far, policy failure, real-time factor

Uses the rclpy that ships with Isaac Sim's ROS 2 bridge, spun from the frame loop. Without it the
simulation runs as before and logs that resets are unavailable.
"""

import json
import math
import os

from . import log


class SimControl:
    STATUS_PERIOD = 1.0  # sim seconds

    def __init__(self, namespace: str):
        self.pending = None  # (x, y, z, yaw) to apply at the next physics step
        self.resets = 0
        self._next_status = 0.0
        self._node = None
        if os.environ.get("LEGGED_SIM_NO_CONTROL"):
            log("sim_control: disabled (LEGGED_SIM_NO_CONTROL)")
            return
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import String
        except ImportError as error:
            log(f"sim_control: rclpy unavailable ({error}); no reset_pose / sim_status")
            return
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        # a private context, without rclpy's signal handlers: sharing the default context with
        # the ROS 2 bridge's own Python nodes ended the frame loop right after start-up
        self._context = rclpy.Context()
        rclpy.init(context=self._context, signal_handler_options=SignalHandlerOptions.NO)
        self._String = String
        self._node = rclpy.create_node("legged_sim_control", namespace=namespace, context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._node.create_subscription(PoseStamped, "reset_pose", self._on_reset, 10)
        self._status = self._node.create_publisher(String, "sim_status", 10)
        log(f"sim_control: reset_pose and sim_status in /{namespace}")

    def _on_reset(self, msg) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pending = (p.x, p.y, p.z, yaw)
        log(f"sim_control: reset requested at ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) yaw {math.degrees(yaw):.0f}")

    def spin(self) -> None:
        if self._node is not None:
            self._executor.spin_once(timeout_sec=0.0)

    def take(self):
        """The pending reset pose, once."""
        pose, self.pending = self.pending, None
        if pose is not None:
            self.resets += 1
        return pose

    def status_due(self, sim_time: float) -> bool:
        return self._node is not None and sim_time >= self._next_status

    def publish(self, sim_time: float, orientation, body_height: float, policy_failed: bool, rtf: float) -> None:
        if not self.status_due(sim_time):
            return
        self._next_status = sim_time + self.STATUS_PERIOD
        w, x, y, z = (float(v) for v in orientation)
        body_height = float(body_height)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        upright = abs(roll) < 0.6 and abs(pitch) < 0.6 and body_height > 0.3
        msg = self._String()
        msg.data = json.dumps({"t": round(sim_time, 2), "roll": round(roll, 3), "pitch": round(pitch, 3),
                               "body_height": round(body_height, 3), "upright": upright, "resets": self.resets,
                               "policy_failed": policy_failed, "rtf": round(rtf, 3)})
        self._status.publish(msg)

    def close(self) -> None:
        if self._node is not None:
            self._executor.shutdown()
            self._node.destroy_node()
            self._context.try_shutdown()
