# SPDX-License-Identifier: Apache-2.0
"""Assembles the scene, the robot, its sensors and the ROS graph, and runs the frame loop."""

import csv
import math
import time

import carb
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.ros2.nodes import set_isaac_name_override
from isaacsim.storage.native import get_assets_root_path

from . import log
from .commands import CommandFilter
from .config import apply_robot_defaults
from .floor_tracker import FloorTracker
from .odometry import FloorOdometry
from .robots import create_robot
from .ros_graph import GraphConfig, RosGraph
from .scene import add_obstacle_boxes, configure_physics_scenes, load_environment
from .sensors import body_surface, mount_d455, mount_os0
from .viewport import SnapshotRecorder, choose_view_offset, set_initial_view

RTF_LOG_PERIOD = 10.0  # wall seconds


class LeggedSimulation:
    def __init__(self, app, args):
        self.app = app
        self.robot = create_robot(args.robot, args)
        self.args = args = apply_robot_defaults(args, self.robot)
        self.stage = omni.usd.get_context().get_stage()
        assets = get_assets_root_path()
        self.spawn = (args.spawn[0], args.spawn[1], math.radians(args.spawn[2]))

        load_environment(app, self.stage, args.scene, args.deactivate, assets, args.floor_friction,
                         args.stair_access)
        add_obstacle_boxes(self.stage, args.add_box, args.floor_z)
        SimulationManager.set_physics_sim_device(args.device)
        self.robot.spawn(*self.spawn, args.floor_z)
        RenderingManager.set_dt(self.robot.control_dt)
        SimulationManager.set_physics_dt(self.robot.physics_dt)
        configure_physics_scenes(self.stage, int(round(1.0 / self.robot.physics_dt)), args.device)

        body = self.robot.body_prim
        if not self.stage.GetPrimAtPath(body).IsValid():
            raise SystemExit(f"no body link at {body}")
        set_isaac_name_override(body, "base_link")
        surface = body_surface(self.stage, body)
        lidar = None if args.no_lidar else mount_os0(app, self.stage, body, surface, assets,
                                                     args.lidar_mount, args.lidar_config)
        cameras = None if args.no_camera else mount_d455(app, self.stage, body, surface, assets,
                                                         self.robot.camera_height, args.camera_rate)
        self.graph = RosGraph(GraphConfig(args.namespace, body, lidar, cameras, tuple(args.camera_resolution)))

        self.odometry = FloorOdometry()
        self.floor = FloorTracker(self.robot.prim_path, args.floor_z)
        self.commands = CommandFilter.from_args(args)
        self.command = np.zeros(3, dtype=np.float32)
        self.pose = (self.spawn[0], self.spawn[1], args.floor_z + self.robot.spawn_height, self.spawn[2])  # x, y, z, yaw
        self._restart_policy, self._policy_failed = True, False

    # -------------------------------------------------------------- physics callback (every physics step)
    def _on_physics_step(self, step_dt, context) -> None:
        if self._policy_failed:
            return
        try:
            if self._restart_policy:
                self.robot.reset(self.command)
                self.odometry.reset()
                self._restart_policy = False
            else:
                self.robot.step(step_dt, self.command)
            position, orientation = self.robot.world_pose()
            sample = self.odometry.update(position, orientation, self.floor.update(*position), step_dt)
            self.graph.set_odometry(sample)
            self.pose = sample.world_pose
        except Exception as error:  # noqa: BLE001 - must not raise inside the physics callback
            self._policy_failed = True
            carb.log_error(f"legged_sim: policy failed, robot stopped: {error}")

    # -------------------------------------------------------------- frame loop
    def run(self) -> None:
        args = self.args
        callback = SimulationManager.register_callback(self._on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        self.app.update()

        x, y, yaw = self.spawn
        view_offset = tuple(args.view_offset) if args.view_offset else choose_view_offset(x, y, yaw, args.floor_z)
        if not args.headless:
            set_initial_view(x, y, yaw, view_offset, args.floor_z)
        snapshots = (SnapshotRecorder(self.stage, args.snapshot_dir, args.snapshot_every, view_offset, args.floor_z)
                     if args.snapshot_dir else None)
        command_log = CommandLog(args.log) if args.log else None
        rtf = RealTimeFactor()

        log(f"ready: {self.robot.name}, cmd_vel on /{args.namespace}/cmd_vel, spawn {tuple(args.spawn)}, physics {args.device}")
        sim_time = 0.0
        try:
            while self.app.is_running():
                self.app.update()
                if not SimulationManager.is_simulating():
                    self._restart_policy = True
                    continue
                sim_time += self.robot.control_dt
                linear, angular = self.graph.twist()
                self.command = self.commands(linear, angular, self.graph.message_count(), sim_time,
                                             self.odometry.velocity, self.pose[3])
                if command_log:
                    command_log.write(sim_time, self.graph.message_count(), linear, angular, self.command, self.pose,
                                      self.commands.stale)
                rtf.tick(sim_time)
                if snapshots:
                    snapshots.update(sim_time, self.pose[0], self.pose[1], self.pose[3])
                if args.duration and sim_time >= args.duration:
                    break
        finally:
            if command_log:
                command_log.close()
            timeline.stop()
            SimulationManager.deregister_callback(callback)
            self.robot.close()


class RealTimeFactor:
    """Logs sim seconds per wall second, every RTF_LOG_PERIOD wall seconds."""

    def __init__(self):
        self._wall, self._sim = time.time(), 0.0

    def tick(self, sim_time: float) -> None:
        elapsed = time.time() - self._wall
        if elapsed >= RTF_LOG_PERIOD:
            log(f"real-time factor {(sim_time - self._sim) / elapsed:.2f} (sim t {sim_time:.1f} s)")
            self._wall, self._sim = time.time(), sim_time


class CommandLog:
    """Per-frame CSV of received and applied commands and the body pose."""

    HEADER = ["t", "msgs", "in_vx", "in_vy", "in_wz", "vx", "vy", "wz", "x", "y", "z", "yaw", "stale"]

    def __init__(self, path: str):
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def write(self, sim_time, message_count, linear, angular, command, pose, stale) -> None:
        self._writer.writerow([f"{sim_time:.3f}", message_count, linear[0], linear[1], angular[2],
                               *command.tolist(), *pose, int(stale)])

    def close(self) -> None:
        self._file.close()
