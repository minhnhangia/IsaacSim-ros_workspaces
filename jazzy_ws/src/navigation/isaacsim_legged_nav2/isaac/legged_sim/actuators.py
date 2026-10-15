# SPDX-License-Identifier: Apache-2.0
"""Spot's actuator model (Isaac Sim's Newton actuators) computed for all joints in one NumPy pass.

Isaac Sim's ``ArticulationActuators`` steps every joint as a separate Newton actuator: for Spot,
12 joints x ~5 Warp launches before every 2 ms physics step. The Python launch overhead (~3 ms
per step) was ~80% of the simulator's frame time. :class:`FastActuators` reads the parameters
from those same actuator objects and computes the same model (PD with a per-joint command delay,
then effort clamping); it matches Isaac's torques to float32 rounding. Anything it does not
reproduce exactly raises :class:`NotImplementedError`, and the caller keeps Isaac's version.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


def _scalar(warp_array) -> float:
    return float(warp_array.numpy()[0])


@dataclass
class ActuatorModel:
    """Per-joint parameters, in DOF order. A limit of ``None`` or ``inf`` means no limit."""

    kp: np.ndarray
    kd: np.ndarray
    delay: np.ndarray  # command delay in physics steps (0 = none)
    max_effort: np.ndarray  # symmetric box limit
    position_limits: list  # (lookup positions, lookup efforts) or None, per joint
    dc_motor_limits: list  # (saturation effort, velocity limit, max effort) or None, per joint

    @classmethod
    def from_newton_actuators(cls, actuators) -> "ActuatorModel":
        """Reads the per-joint ``newton.actuators.Actuator`` objects Isaac built (one robot)."""
        kp, kd, delay, max_effort, position, dc = [], [], [], [], [], []
        for act in actuators:
            ctrl = act.controller
            if type(ctrl).__name__ != "ControllerPD" or ctrl.const_effort is not None:
                raise NotImplementedError(f"controller {type(ctrl).__name__}")
            if len(act.clamping) > 1:
                raise NotImplementedError("more than one clamp per joint")
            kp.append(_scalar(ctrl.kp))
            kd.append(_scalar(ctrl.kd))
            delay.append(int(act.delay.delay_steps.numpy()[0]) if act.delay is not None else 0)
            max_effort.append(np.inf)
            position.append(None)
            dc.append(None)
            for clamp in act.clamping:
                kind = type(clamp).__name__
                if kind == "ClampingMaxEffort":
                    max_effort[-1] = _scalar(clamp.max_effort)
                elif kind == "ClampingPositionBased":
                    position[-1] = (clamp.lookup_positions.numpy().astype(np.float64),
                                    clamp.lookup_efforts.numpy().astype(np.float64))
                elif kind == "ClampingDCMotor":
                    dc[-1] = (_scalar(clamp.saturation_effort), _scalar(clamp.velocity_limit),
                              _scalar(clamp.max_motor_effort))
                else:
                    raise NotImplementedError(f"clamp {kind}")
        return cls(np.array(kp), np.array(kd), np.array(delay), np.array(max_effort), position, dc)

    def describe(self) -> str:
        return (f"{len(self.kp)} joints, kp {sorted(set(self.kp.tolist()))}, kd {sorted(set(self.kd.tolist()))}, "
                f"delay {sorted(set(self.delay.tolist()))} steps, "
                f"{sum(p is not None for p in self.position_limits)} position-limited, "
                f"{sum(d is not None for d in self.dc_motor_limits)} DC-motor, "
                f"{int(np.isfinite(self.max_effort).sum())} box-limited")


class ActuatorState:
    """Applies an :class:`ActuatorModel`, keeping the command history for the delay.

    Newton's delay semantics: with no history yet (or a zero delay) the current command is used;
    otherwise the command pushed ``delay`` steps ago, or the oldest one while the history is shorter.
    """

    def __init__(self, model: ActuatorModel):
        self.model = model
        self._history = deque(maxlen=max(1, int(model.delay.max(initial=0))))
        self._lag = np.maximum(model.delay - 1, 0)
        self._joints = np.arange(len(model.kp))

    def reset(self) -> None:
        self._history.clear()

    def compute(self, q, qd, target_q, target_qd, feedforward) -> np.ndarray:
        m = self.model
        command = np.stack([target_q, target_qd, feedforward]).astype(np.float64)  # (3, joints)
        if self._history:
            n = len(self._history)
            past = np.stack(self._history)  # (n, 3, joints), oldest first
            delayed = past[n - 1 - np.minimum(self._lag, n - 1), :, self._joints].T  # (3, joints)
            delayed = np.where(m.delay > 0, delayed, command)
        else:
            delayed = command
        effort = m.kp * (delayed[0] - q) + m.kd * (delayed[1] - qd) + delayed[2]
        effort = np.clip(effort, -m.max_effort, m.max_effort)
        for j, table in enumerate(m.position_limits):
            if table is not None:
                limit = np.interp(q[j], *table)
                effort[j] = np.clip(effort[j], -limit, limit)
        for j, motor in enumerate(m.dc_motor_limits):
            if motor is not None:
                saturation, velocity_limit, max_effort = motor
                corner = velocity_limit * (1.0 + max_effort / saturation) if saturation > 0.0 else velocity_limit
                v = np.clip(qd[j], -corner, corner)
                effort[j] = np.clip(effort[j], max(saturation * (-1.0 - v / velocity_limit), -max_effort),
                                    min(saturation * (1.0 - v / velocity_limit), max_effort))
        self._history.append(command)
        return effort


class FastActuators:
    """Replaces the stepping of one Isaac ``ArticulationActuators`` object (one robot)."""

    def __init__(self, isaac_actuators):
        self._isaac = isaac_actuators
        self._articulation = isaac_actuators._articulation
        if len(self._articulation) != 1:
            raise NotImplementedError("one robot per articulation only")
        dofs = [int(a.indices.numpy()[0]) for a in isaac_actuators._actuators]
        if dofs != list(isaac_actuators._actuated_dof_indices):
            raise NotImplementedError("actuators are not in actuated-DOF order")
        self._dofs = np.array(dofs)
        self.state = ActuatorState(ActuatorModel.from_newton_actuators(isaac_actuators._actuators))

    def install(self) -> "FastActuators":
        # SimulationManager calls the pre-physics callback by name on the instance, and the policy
        # runner calls reset() on restarts, so instance attributes replace both.
        self._isaac.step_actuators = self.step
        self._isaac.reset = self.state.reset
        self.state.reset()
        return self

    def compute_effort(self) -> np.ndarray:
        """Efforts for the current articulation state and targets (advances the delay history)."""
        art, d = self._articulation, self._dofs
        return self.state.compute(art.get_dof_positions().numpy()[0, d],
                                  art.get_dof_velocities().numpy()[0, d],
                                  art.get_dof_position_targets().numpy()[0, d],
                                  art.get_dof_velocity_targets().numpy()[0, d],
                                  self._isaac._feedforward_target_efforts.numpy()[0, d])

    def step(self, step_dt, context=None) -> None:  # signature required by SimulationManager
        import warp as wp

        effort = self.compute_effort()
        self._articulation.set_dof_efforts(
            wp.array(effort.astype(np.float32).reshape(1, -1), dtype=wp.float32, device=self._isaac._device),
            dof_indices=self._isaac._actuated_dof_indices_wp)
