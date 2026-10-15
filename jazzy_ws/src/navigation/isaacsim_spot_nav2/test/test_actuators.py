# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from spot_sim.actuators import ActuatorModel, ActuatorState


def model(n=2, delay=0, max_effort=np.inf, position=None, dc=None):
    return ActuatorModel(kp=np.full(n, 60.0), kd=np.full(n, 1.5), delay=np.full(n, delay),
                         max_effort=np.full(n, max_effort), position_limits=[position] * n,
                         dc_motor_limits=[dc] * n)


def step(state, target, q=0.0, qd=0.0, n=2):
    return state.compute(np.full(n, q), np.full(n, qd), np.full(n, target), np.zeros(n), np.zeros(n))


def test_pd_without_delay():
    effort = step(ActuatorState(model()), target=0.1, q=0.0, qd=0.2)
    np.testing.assert_allclose(effort, 60.0 * 0.1 - 1.5 * 0.2)


def test_delay_uses_command_from_delay_steps_ago():
    state = ActuatorState(model(delay=3))
    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    applied = [step(state, t)[0] / 60.0 for t in targets]
    # first step: no history (current command); then the oldest until 3 are stored, then 3 steps back
    np.testing.assert_allclose(applied, [0.1, 0.1, 0.1, 0.1, 0.2, 0.3])


def test_reset_clears_the_delay():
    state = ActuatorState(model(delay=2))
    step(state, 1.0)
    step(state, 1.0)
    state.reset()
    assert step(state, 0.0)[0] == pytest.approx(0.0)


def test_box_and_position_limits():
    assert step(ActuatorState(model(max_effort=45.0)), target=10.0)[0] == pytest.approx(45.0)
    table = (np.array([-1.0, 1.0]), np.array([10.0, 30.0]))  # limit 20 Nm at q = 0
    assert step(ActuatorState(model(position=table)), target=10.0)[0] == pytest.approx(20.0)


def test_dc_motor_limit_falls_with_speed():
    dc = (50.0, 10.0, 40.0)  # saturation effort, velocity limit, max effort
    assert step(ActuatorState(model(dc=dc)), target=10.0, qd=0.0)[0] == pytest.approx(40.0)
    assert step(ActuatorState(model(dc=dc)), target=10.0, qd=5.0)[0] == pytest.approx(25.0)
