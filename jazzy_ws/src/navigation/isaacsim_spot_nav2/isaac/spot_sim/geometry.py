# SPDX-License-Identifier: Apache-2.0
"""Planar angles and quaternions. Quaternions are (w, x, y, z) tuples throughout."""

import math


def wrap_angle(angle: float) -> float:
    """Angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def yaw_quat(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_og_quat(q) -> list[float]:
    """(w, x, y, z) -> OmniGraph quatd order (x, y, z, w)."""
    return [q[1], q[2], q[3], q[0]]
