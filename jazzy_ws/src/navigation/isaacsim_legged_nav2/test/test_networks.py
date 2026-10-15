# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from legged_sim.networks import LSTMActuatorNet, MLPPolicy, dc_motor_clip, elu


def test_mlp_matches_manual_forward_pass():
    rng = np.random.default_rng(0)
    w = {f"mlp_{i}_weight": rng.normal(size=s) for i, s in ((0, (5, 4)), (2, (3, 5)), (4, (2, 3)))}
    w.update({f"mlp_{i}_bias": rng.normal(size=s[0]) for i, s in ((0, (5,)), (2, (3,)), (4, (2,)))})
    x = rng.normal(size=4)
    expected = w["mlp_4_weight"] @ elu(w["mlp_2_weight"] @ elu(w["mlp_0_weight"] @ x + w["mlp_0_bias"])
                                       + w["mlp_2_bias"]) + w["mlp_4_bias"]
    policy = MLPPolicy(w)
    assert (policy.input_size, policy.output_size) == (4, 2)
    np.testing.assert_allclose(policy(x), expected)


def test_dc_motor_clip_limits_torque_with_speed():
    # saturation 120, limit 80, velocity limit 7.5 (ANYdrive): full 80 Nm at rest, less when driven fast
    assert dc_motor_clip(200.0, 0.0, 120.0, 80.0, 7.5) == pytest.approx(80.0)
    assert dc_motor_clip(200.0, 5.0, 120.0, 80.0, 7.5) == pytest.approx(120.0 * (1 - 5.0 / 7.5))
    assert dc_motor_clip(-200.0, 5.0, 120.0, 80.0, 7.5) == pytest.approx(-80.0)
    assert dc_motor_clip(10.0, 0.0, 120.0, 80.0, 7.5) == pytest.approx(10.0)


def _single_layer_weights(hidden=3, seed=1):
    rng = np.random.default_rng(seed)
    return {"lstm_weight_ih_l0": rng.normal(size=(4 * hidden, 2)), "lstm_weight_hh_l0": rng.normal(size=(4 * hidden, hidden)),
            "lstm_bias_ih_l0": rng.normal(size=4 * hidden), "lstm_bias_hh_l0": rng.normal(size=4 * hidden),
            "linear_weight": rng.normal(size=(1, hidden)), "linear_bias": rng.normal(size=1),
            "in_scale": np.array([[[2.0, 0.5]]]), "out_scale": np.array([3.0])}


def test_lstm_follows_pytorch_gate_equations():
    w = _single_layer_weights()
    net = LSTMActuatorNet(w, num_joints=1, saturation_effort=1e9, effort_limit=1e9, velocity_limit=1e9)
    sig = lambda v: 1 / (1 + np.exp(-v))
    h = c = np.zeros(3)
    for err, vel in ((0.1, -0.2), (0.3, 0.5)):
        x = np.array([err * 2.0, vel * 0.5])
        g = w["lstm_weight_ih_l0"] @ x + w["lstm_bias_ih_l0"] + w["lstm_weight_hh_l0"] @ h + w["lstm_bias_hh_l0"]
        i, f, gg, o = np.split(g, 4)
        c = sig(f) * c + sig(i) * np.tanh(gg)
        h = sig(o) * np.tanh(c)
        expected = 3.0 * (w["linear_weight"] @ h + w["linear_bias"])[0]
        assert net.raw_torque(np.array([err]), np.array([vel]))[0] == pytest.approx(expected)


def test_lstm_reset_clears_memory():
    net = LSTMActuatorNet(_single_layer_weights(), 2, 1e9, 1e9, 1e9)
    first = net.raw_torque(np.array([0.1, 0.2]), np.zeros(2))
    net.raw_torque(np.array([0.5, -0.5]), np.ones(2))
    net.reset()
    np.testing.assert_allclose(net.raw_torque(np.array([0.1, 0.2]), np.zeros(2)), first)
