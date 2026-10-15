# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab's rough-terrain velocity task (``Isaac-Velocity-Rough-*``), deployed outside Isaac Lab.

Observation (concatenated, no normalization), as in Isaac Lab's ``LocomotionVelocityRoughEnvCfg``:
    base linear velocity (3, body frame), base angular velocity (3, body frame),
    projected gravity (3), velocity command (3: vx, vy, wz), joint positions - defaults (N),
    joint velocities (N), last action (N), height scan (187, see :class:`HeightScanGrid`).
Action: joint position targets = defaults + scale * action. Policy at 50 Hz over 200 Hz physics.
"""

from dataclasses import dataclass

import numpy as np

from .networks import MLPPolicy

HEIGHT_SCAN_OFFSET = 0.5  # Isaac Lab's mdp.height_scan default offset
HEIGHT_SCAN_CLIP = 1.0


class HeightScanGrid:
    """Isaac Lab ``GridPatternCfg(resolution=0.1, size=[1.6, 1.0])``, ordering "xy" (x varies fastest).

    Points are in the yaw-aligned frame of the scanned link: x forward, y left.
    """

    def __init__(self, resolution: float = 0.1, size=(1.6, 1.0)):
        xs = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
        ys = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
        self.points = np.array([(x, y) for y in ys for x in xs])

    def __len__(self) -> int:
        return len(self.points)

    def world_points(self, x: float, y: float, yaw: float) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        px, py = self.points[:, 0], self.points[:, 1]
        return np.stack([x + c * px - s * py, y + s * px + c * py], axis=1)


def height_scan_observation(link_height: float, hit_heights) -> np.ndarray:
    """``link_z - hit_z - 0.5``, clipped to [-1, 1] (Isaac Lab ``mdp.height_scan`` + the task's clip)."""
    return np.clip(link_height - np.asarray(hit_heights, dtype=np.float64) - HEIGHT_SCAN_OFFSET,
                   -HEIGHT_SCAN_CLIP, HEIGHT_SCAN_CLIP)


@dataclass(frozen=True)
class BaseState:
    """Root link state, everything in the body frame except where noted."""

    linear_velocity: np.ndarray  # (3,)
    angular_velocity: np.ndarray  # (3,)
    projected_gravity: np.ndarray  # (3,) gravity direction in the body frame
    joint_positions: np.ndarray  # (N,)
    joint_velocities: np.ndarray  # (N,)


class RoughTerrainPolicy:
    """Builds the observation, runs the actor, and keeps the last action."""

    def __init__(self, actor: MLPPolicy, default_joint_positions, action_scale: float, num_scan_points: int = 187):
        self.actor = actor
        self.default_joint_positions = np.asarray(default_joint_positions, dtype=np.float64)
        self.action_scale = action_scale
        n = len(self.default_joint_positions)
        expected = 12 + 3 * n + num_scan_points
        if actor.input_size != expected or actor.output_size != n:
            raise ValueError(f"policy expects {actor.input_size} inputs / {actor.output_size} outputs, "
                             f"robot gives {expected} / {n}")
        self.last_action = np.zeros(n)

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def observation(self, state: BaseState, command, height_scan) -> np.ndarray:
        return np.concatenate([state.linear_velocity, state.angular_velocity, state.projected_gravity,
                               np.asarray(command, dtype=np.float64),
                               state.joint_positions - self.default_joint_positions,
                               state.joint_velocities, self.last_action, height_scan])

    def act(self, state: BaseState, command, height_scan) -> np.ndarray:
        """Joint position targets for the next control period."""
        self.last_action = self.actor(self.observation(state, command, height_scan))
        return self.default_joint_positions + self.action_scale * self.last_action
