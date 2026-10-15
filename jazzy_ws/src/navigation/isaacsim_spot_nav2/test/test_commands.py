# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from spot_sim.commands import (CommandFilter, CommandLimits, CommandWatchdog, DeadbandCompensator,
                               HeadingHold)

STANDING = (0.0, 0.0, 0.0)


def test_limits_clamp_each_axis():
    limits = CommandLimits(min_vx=-1.0, max_vx=1.5, max_vy=0.8, max_wz=1.2)
    np.testing.assert_allclose(limits.clamp(3.0, -2.0, 0.5), [1.5, -0.8, 0.5])
    np.testing.assert_allclose(limits.clamp(-3.0, 0.1, -9.0), [-1.0, 0.1, -1.2])


def test_watchdog_goes_stale_without_new_messages():
    watchdog = CommandWatchdog(timeout=0.5)
    assert watchdog.is_stale(0, 0.0)  # nothing received yet
    assert not watchdog.is_stale(1, 1.0)
    assert not watchdog.is_stale(1, 1.4)
    assert watchdog.is_stale(1, 1.6)
    assert not watchdog.is_stale(2, 1.7)


def test_deadband_scales_small_command_from_stand_keeping_direction():
    deadband = DeadbandCompensator([0.30, 0.22, 0.32])
    out = deadband(np.array([0.1, 0.05, 0.0], dtype=np.float32), STANDING)
    assert out[0] == pytest.approx(0.30)
    assert out[1] / out[0] == pytest.approx(0.5)


def test_deadband_leaves_stop_large_and_moving_commands_alone():
    deadband = DeadbandCompensator([0.30, 0.22, 0.32])
    for command, velocity in (([0.01, 0.0, 0.0], STANDING),  # intentional stop
                              ([0.6, 0.0, 0.0], STANDING),  # already above the deadband
                              ([0.1, 0.0, 0.0], (0.3, 0.0, 0.0))):  # already walking
        command = np.array(command, dtype=np.float32)
        np.testing.assert_array_equal(deadband(command, velocity), command)


def test_heading_hold_hysteresis():
    hold = HeadingHold(gain=1.0, max_wz=0.3, engage_deg=3.0, release_deg=1.0)
    zero = np.zeros(3, dtype=np.float32)
    assert not hold(zero, 0.0).any()  # captures heading 0
    assert not hold(zero, math.radians(2.0)).any()  # inside the engage band
    turn = hold(zero, math.radians(10.0))
    assert turn[2] == pytest.approx(-math.radians(10.0))
    assert hold(zero, math.radians(2.0))[2] < 0  # still active until below the release angle
    assert not hold(zero, math.radians(0.5)).any()


def test_heading_hold_off_while_commanded():
    hold = HeadingHold(gain=1.0, max_wz=0.3, engage_deg=3.0, release_deg=1.0)
    command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(hold(command, 1.0), command)


def test_filter_stops_on_stale_command():
    class Args:
        cmd_timeout, min_vx, max_vx, max_vy, max_wz = 0.5, -1.0, 1.5, 0.8, 1.2
        min_cmd, deadband_comp = [0.30, 0.22, 0.32], True
        hold_gain, hold_max_wz, hold_engage_deg, hold_release_deg, hold_heading = 1.0, 0.3, 3.0, 1.0, True

    filt = CommandFilter.from_args(Args)
    assert filt([1.0, 0, 0], [0, 0, 0], 1, 0.1, STANDING, 0.0)[0] == pytest.approx(1.0)
    assert not filt([1.0, 0, 0], [0, 0, 0], 1, 1.0, STANDING, 0.0).any()
    assert filt.stale
