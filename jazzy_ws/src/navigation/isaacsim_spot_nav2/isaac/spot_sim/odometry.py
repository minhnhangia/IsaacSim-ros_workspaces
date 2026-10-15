# SPDX-License-Identifier: Apache-2.0
"""Ground-truth odometry for Nav2, aligned with the floor.

``odom`` is the spawn pose on the floor and ``base_footprint`` is the body projected onto the
floor (x, y, yaw only), so every height Nav2 uses is measured from the floor. ``base_link`` is
the body itself, which bobs and tilts while walking.
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import quat_mul, quat_yaw, wrap_angle, yaw_quat


@dataclass(frozen=True)
class OdomSample:
    footprint: tuple  # (x, y, yaw) of base_footprint in odom
    velocity: tuple  # (vx, vy, wz) in base_footprint, smoothed
    body_height: float  # base_link above base_footprint
    body_tilt: tuple  # base_link orientation relative to base_footprint (roll/pitch only), wxyz
    world_pose: tuple  # (x, y, z, yaw) of the body in the world


class FloorOdometry:
    SMOOTHING = 0.8  # the gait bob makes raw finite differences noisy

    def __init__(self, spawn_x: float, spawn_y: float, spawn_yaw: float, floor_z: float):
        self.spawn = (spawn_x, spawn_y, spawn_yaw)
        self.floor_z = floor_z
        self.velocity = np.zeros(3)
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def update(self, position, orientation, dt: float) -> OdomSample:
        """``position`` (x, y, z) and ``orientation`` (w, x, y, z) of the body in the world."""
        x, y, z = (float(v) for v in position)
        q = tuple(float(v) for v in orientation)
        yaw = quat_yaw(q)
        sx, sy, syaw = self.spawn
        c, s = math.cos(-syaw), math.sin(-syaw)
        fx, fy = c * (x - sx) - s * (y - sy), s * (x - sx) + c * (y - sy)
        fyaw = wrap_angle(yaw - syaw)
        if self._previous is not None and dt > 0:
            px, py, pyaw = self._previous
            cp, sp = math.cos(-pyaw), math.sin(-pyaw)
            raw = np.array([(cp * (fx - px) - sp * (fy - py)) / dt,
                            (sp * (fx - px) + cp * (fy - py)) / dt,
                            wrap_angle(fyaw - pyaw) / dt])
            self.velocity = self.SMOOTHING * self.velocity + (1 - self.SMOOTHING) * raw
        self._previous = (fx, fy, fyaw)
        return OdomSample(footprint=(fx, fy, fyaw),
                          velocity=tuple(float(v) for v in self.velocity),
                          body_height=z - self.floor_z,
                          body_tilt=quat_mul(yaw_quat(-yaw), q),
                          world_pose=(x, y, z, yaw))
