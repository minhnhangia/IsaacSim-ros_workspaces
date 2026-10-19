# SPDX-License-Identifier: Apache-2.0
"""The interface every simulated robot implements, and its per-robot defaults."""

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommandProfile:
    """cmd_vel handling defaults for one robot (see commands.py); each can be overridden on the command line."""

    min_vx: float
    max_vx: float
    max_vy: float
    max_wz: float
    min_command: tuple  # deadband compensation targets (vx, vy, wz)
    deadband_compensation: bool
    hold_heading: bool


class LeggedRobot(abc.ABC):
    """A robot walking under a locomotion policy.

    Lifecycle: ``spawn()`` before the timeline plays; then, from the post-physics callback,
    ``reset()`` on the first physics step after (re)starting and ``step()`` on every other one.
    """

    name: str
    body_prim: str  # the link published as base_link
    spawn_height: float  # body height above the floor at spawn
    control_dt: float  # policy period (one rendered frame)
    physics_dt: float
    climbs_stairs: bool
    commands: CommandProfile
    lidar_mount: tuple  # (x, y, mast height above the body surface at that point)
    camera_height: float  # centre of the front stereo camera on the body's front face (body frame z)

    @abc.abstractmethod
    def spawn(self, x: float, y: float, yaw: float, floor_z: float) -> None:
        """Adds the robot to the stage."""

    @abc.abstractmethod
    def reset(self, command: np.ndarray) -> None:
        """Puts the robot in its default standing state and restarts the policy."""

    @abc.abstractmethod
    def step(self, dt: float, command: np.ndarray) -> None:
        """Advances the policy by one physics step with the base velocity command (vx, vy, wz)."""

    @abc.abstractmethod
    def world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Body position (x, y, z) and orientation (w, x, y, z) in the world."""

    def close(self) -> None:
        """Releases callbacks and resources."""
