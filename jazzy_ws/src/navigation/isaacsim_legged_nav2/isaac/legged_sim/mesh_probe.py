# SPDX-License-Identifier: Apache-2.0
"""Ray queries against a triangle mesh (NumPy), used to place sensors on a robot's real surface.

A bounding box overstates where a curved body is: ANYmal-D's box top (its rear LiDAR housing)
is 15 cm above the shell under a centre mount, and its box front is ahead of the face at the
camera's height. Casting rays at the visual mesh finds the surface itself.
"""

import numpy as np


class MeshProbe:
    def __init__(self, triangles):
        """``triangles``: (T, 3, 3) array of triangle vertices, in the frame the queries use."""
        self.triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        self._v0 = self.triangles[:, 0]
        self._e1 = self.triangles[:, 1] - self._v0
        self._e2 = self.triangles[:, 2] - self._v0

    def cast(self, origin, direction):
        """Distance along the unit ``direction`` to the nearest triangle (Moller-Trumbore), or None."""
        o, d = np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
        h = np.cross(d, self._e2)
        a = np.einsum("ij,ij->i", self._e1, h)
        valid = np.abs(a) > 1e-12
        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s = o - self._v0
        u = f * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, self._e1)
        v = f * (q @ d)
        t = f * np.einsum("ij,ij->i", self._e2, q)
        hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
        return float(t[hit].min()) if hit.any() else None

    def highest_below(self, x_range, y_range, samples: int = 7, start: float = 10.0) -> float | None:
        """Highest surface point under the rectangle ``x_range`` x ``y_range`` (rays cast straight down)."""
        tops = [start - t for x in np.linspace(*x_range, samples) for y in np.linspace(*y_range, samples)
                if (t := self.cast((x, y, start), (0.0, 0.0, -1.0))) is not None]
        return max(tops) if tops else None

    def frontmost(self, y_range, z_range, samples: int = 7, start: float = 10.0) -> float | None:
        """Largest x of the surface over the rectangle ``y_range`` x ``z_range`` (rays cast along -x)."""
        fronts = [start - t for y in np.linspace(*y_range, samples) for z in np.linspace(*z_range, samples)
                  if (t := self.cast((start, y, z), (-1.0, 0.0, 0.0))) is not None]
        return max(fronts) if fronts else None
