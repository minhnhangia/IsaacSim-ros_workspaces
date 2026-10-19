# SPDX-License-Identifier: Apache-2.0
"""Ground-truth odometry for Nav2, on whichever floor the robot is on.

``odom`` is the world frame, so ``map -> odom`` is identity when the map is in world coordinates
(ground-truth localization; AMCL, if used, finds the same). ``base_footprint`` is the body projected
onto the floor below it (x, y, yaw, and the floor's height), so every height Nav2 uses is measured
from the floor the robot stands on, on any storey. ``base_link`` is the body itself, which bobs and
tilts while walking.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import quat_mul, quat_yaw, wrap_angle, yaw_quat


@dataclass(frozen=True)
class OdomSample:
    footprint: tuple  # (x, y, yaw) of base_footprint in odom (= world)
    footprint_height: float  # z of base_footprint in odom: the floor under the robot
    velocity: tuple  # (vx, vy, wz) in base_footprint, smoothed
    body_height: float  # base_link above base_footprint
    body_tilt: tuple  # base_link orientation relative to base_footprint (roll/pitch only), wxyz
    world_pose: tuple  # (x, y, z, yaw) of the body in the world


class FloorOdometry:
    SMOOTHING = 0.8  # the gait bob makes raw finite differences noisy

    def __init__(self):
        self.velocity = np.zeros(3)
        self._previous = None

    def reset(self) -> None:
        self._previous = None

    def update(self, position, orientation, floor_height: float, dt: float) -> OdomSample:
        """``position`` (x, y, z) and ``orientation`` (w, x, y, z) of the body in the world; ``floor_height``
        is the world z of the floor under it."""
        x, y, z = (float(v) for v in position)
        q = tuple(float(v) for v in orientation)
        yaw = quat_yaw(q)
        if self._previous is not None and dt > 0:
            px, py, pyaw = self._previous
            c, s = np.cos(-pyaw), np.sin(-pyaw)
            raw = np.array([(c * (x - px) - s * (y - py)) / dt,
                            (s * (x - px) + c * (y - py)) / dt,
                            wrap_angle(yaw - pyaw) / dt])
            self.velocity = self.SMOOTHING * self.velocity + (1 - self.SMOOTHING) * raw
        self._previous = (x, y, yaw)
        return OdomSample(footprint=(x, y, yaw), footprint_height=float(floor_height),
                          velocity=tuple(float(v) for v in self.velocity),
                          body_height=z - float(floor_height),
                          body_tilt=quat_mul(yaw_quat(-yaw), q),
                          world_pose=(x, y, z, yaw))
