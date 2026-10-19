# SPDX-License-Identifier: Apache-2.0
import math

import pytest

from legged_sim.geometry import quat_yaw, yaw_quat
from legged_sim.odometry import FloorOdometry


def test_footprint_is_the_body_on_the_floor_below_it_in_world_coordinates():
    sample = FloorOdometry().update((1.0, 3.0, -2.75), yaw_quat(math.pi / 2), floor_height=-3.3, dt=0.002)
    assert sample.footprint == pytest.approx((1.0, 3.0, math.pi / 2))
    assert sample.footprint_height == pytest.approx(-3.3)  # a basement floor
    assert sample.body_height == pytest.approx(0.55)
    assert quat_yaw(sample.body_tilt) == pytest.approx(0.0, abs=1e-9)


def test_velocity_is_in_the_body_frame_and_smoothed():
    odom = FloorOdometry()
    q = yaw_quat(math.pi / 2)  # facing +y
    odom.update((0.0, 0.0, 0.5), q, 0.0, dt=0.01)
    for i in range(1, 200):
        sample = odom.update((0.0, 0.01 * i, 0.5), q, 0.0, dt=0.01)  # 1 m/s along world +y = forward
    vx, vy, wz = sample.velocity
    assert vx == pytest.approx(1.0, abs=1e-3)
    assert vy == pytest.approx(0.0, abs=1e-6)
    assert wz == pytest.approx(0.0, abs=1e-6)
