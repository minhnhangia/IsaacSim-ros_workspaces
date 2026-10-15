# SPDX-License-Identifier: Apache-2.0
"""NumPy inference for the networks exported by ``scripts/export_policies.py``.

Isaac Sim 6.1's Python has no torch, so the RSL-RL actor and the ANYdrive actuator network are
exported once to ``.npz`` and evaluated here. Both are tiny; NumPy is fast enough at 50/200 Hz.
"""

import numpy as np


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(x))


class MLPPolicy:
    """RSL-RL actor: Linear-ELU-Linear-ELU-Linear-ELU-Linear (``mlp.0``, ``mlp.2``, ...)."""

    def __init__(self, weights):
        indices = sorted(int(k.split("_")[1]) for k in weights if k.startswith("mlp_") and k.endswith("_weight"))
        self.layers = [(np.asarray(weights[f"mlp_{i}_weight"], dtype=np.float64),
                        np.asarray(weights[f"mlp_{i}_bias"], dtype=np.float64)) for i in indices]
        if not self.layers:
            raise ValueError("no mlp_<i>_weight arrays in the policy weights")

    @property
    def input_size(self) -> int:
        return self.layers[0][0].shape[1]

    @property
    def output_size(self) -> int:
        return self.layers[-1][0].shape[0]

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=np.float64)
        for i, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if i < len(self.layers) - 1:
                x = elu(x)
        return x


def dc_motor_clip(effort, joint_velocity, saturation_effort, effort_limit, velocity_limit):
    """Isaac Lab's DC-motor torque-speed limit (``DCMotor._clip_effort``), element-wise."""
    corner = velocity_limit * (1.0 + effort_limit / saturation_effort)
    v = np.clip(joint_velocity, -corner, corner)
    upper = np.minimum(saturation_effort * (1.0 - v / velocity_limit), effort_limit)
    lower = np.maximum(saturation_effort * (-1.0 - v / velocity_limit), -effort_limit)
    return np.clip(effort, lower, upper)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTMActuatorNet:
    """Isaac Lab ``ActuatorNetLSTM``: per joint, (position error, velocity) -> torque, with memory.

    Each joint is one batch row of a multi-layer LSTM (PyTorch gate order i, f, g, o), followed by
    a linear layer and the DC-motor limit. Weights: ``lstm_weight_ih_l<k>`` etc., ``linear_*``,
    ``in_scale`` and ``out_scale`` as stored in the TorchScript network.
    """

    def __init__(self, weights, num_joints: int, saturation_effort: float, effort_limit: float, velocity_limit: float):
        self.layers = []
        k = 0
        while f"lstm_weight_ih_l{k}" in weights:
            self.layers.append((np.asarray(weights[f"lstm_weight_ih_l{k}"], dtype=np.float64),
                                np.asarray(weights[f"lstm_weight_hh_l{k}"], dtype=np.float64),
                                np.asarray(weights[f"lstm_bias_ih_l{k}"], dtype=np.float64)
                                + np.asarray(weights[f"lstm_bias_hh_l{k}"], dtype=np.float64)))
            k += 1
        if not self.layers:
            raise ValueError("no LSTM layers in the actuator network weights")
        self.in_scale = np.asarray(weights["in_scale"], dtype=np.float64).reshape(1, -1)
        self.out_scale = float(np.asarray(weights["out_scale"]).reshape(-1)[0])
        self.out_weight = np.asarray(weights["linear_weight"], dtype=np.float64)
        self.out_bias = np.asarray(weights["linear_bias"], dtype=np.float64)
        self.limits = (saturation_effort, effort_limit, velocity_limit)
        hidden = self.layers[0][1].shape[1]
        self.hidden = np.zeros((len(self.layers), num_joints, hidden))
        self.cell = np.zeros_like(self.hidden)

    def reset(self) -> None:
        self.hidden[:] = 0.0
        self.cell[:] = 0.0

    def raw_torque(self, position_error, joint_velocity) -> np.ndarray:
        """Network output before the motor limit; advances the recurrent state."""
        x = np.stack([position_error, joint_velocity], axis=1) * self.in_scale
        for k, (w_ih, w_hh, bias) in enumerate(self.layers):
            gates = x @ w_ih.T + self.hidden[k] @ w_hh.T + bias
            i, f, g, o = np.split(gates, 4, axis=1)
            self.cell[k] = _sigmoid(f) * self.cell[k] + _sigmoid(i) * np.tanh(g)
            self.hidden[k] = _sigmoid(o) * np.tanh(self.cell[k])
            x = self.hidden[k]
        return self.out_scale * (x @ self.out_weight.T + self.out_bias).reshape(-1)

    def __call__(self, joint_position, joint_velocity, target_position) -> np.ndarray:
        joint_position, joint_velocity = np.asarray(joint_position, float), np.asarray(joint_velocity, float)
        torque = self.raw_torque(np.asarray(target_position, float) - joint_position, joint_velocity)
        return dc_motor_clip(torque, joint_velocity, *self.limits)
