# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from legged_sim.mesh_probe import MeshProbe


def box(lo, hi):
    """12 triangles of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [tri for a, b, c, d in faces for tri in ((v[a], v[b], v[c]), (v[a], v[c], v[d]))]


def test_cast_hits_nearest_face():
    probe = MeshProbe(box((-1, -1, 0), (1, 1, 0.5)))
    assert probe.cast((0, 0, 2), (0, 0, -1)) == pytest.approx(1.5)
    assert probe.cast((5, 0, 0.2), (-1, 0, 0)) == pytest.approx(4.0)
    assert probe.cast((5, 5, 0.2), (-1, 0, 0)) is None


def test_highest_below_takes_the_tallest_point_under_the_footprint():
    # a low shell with a taller block at the rear, like ANYmal-D's LiDAR housing
    probe = MeshProbe(box((-0.5, -0.2, 0), (0.5, 0.2, 0.08)) + box((-0.35, -0.1, 0), (-0.25, 0.1, 0.19)))
    assert probe.highest_below((-0.05, 0.05), (-0.05, 0.05)) == pytest.approx(0.08)
    assert probe.highest_below((-0.32, -0.28), (-0.05, 0.05)) == pytest.approx(0.19)
    assert probe.highest_below((3, 4), (3, 4)) is None


def test_frontmost_takes_the_most_protruding_point_over_the_footprint():
    # a recessed face with protruding sides
    probe = MeshProbe(box((-0.5, -0.2, -0.1), (0.41, 0.2, 0.1)) + box((0.41, 0.05, -0.1), (0.44, 0.2, 0.1)))
    assert probe.frontmost((-0.04, 0.04), (0.0, 0.03)) == pytest.approx(0.41)
    assert probe.frontmost((-0.06, 0.06), (0.0, 0.03)) == pytest.approx(0.44)
