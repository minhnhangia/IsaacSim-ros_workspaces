# SPDX-License-Identifier: Apache-2.0
"""Turns the latest cmd_vel into the command the locomotion policy receives.

The stages, in order, each compensate for something measured on this robot and policy:

1. Watchdog: Isaac's Twist subscriber keeps the last message forever, so the command drops to
   zero when no new message has arrived for a while.
2. Limits: commands stay well inside the policy's trained range (vx -2..3, vy +-1.5, wz +-2).
3. Deadband compensation: from a settled stand the policy ignores small commands (vx <= 0.2,
   vy <= 0.15, wz <= 0.2-0.25), so a planner's slow final approach would never start moving.
4. Heading hold: each time Spot comes to rest its policy turns it ~40 deg while it settles; with
   a zero command, steer back to the heading it stopped at.

Which stages are on, and their thresholds, are per robot (robots/<robot>.py).
"""

import math
from dataclasses import dataclass

import numpy as np

from .geometry import wrap_angle

ZERO = np.zeros(3, dtype=np.float32)


@dataclass(frozen=True)
class CommandLimits:
    min_vx: float
    max_vx: float
    max_vy: float
    max_wz: float

    def clamp(self, vx: float, vy: float, wz: float) -> np.ndarray:
        return np.array([min(max(vx, self.min_vx), self.max_vx),
                         min(max(vy, -self.max_vy), self.max_vy),
                         min(max(wz, -self.max_wz), self.max_wz)], dtype=np.float32)


class CommandWatchdog:
    """Detects fresh messages from a counter that increments once per received message."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self._count = 0
        self._last_message_time = -math.inf

    def is_stale(self, message_count: int, sim_time: float) -> bool:
        if message_count != self._count:
            self._count, self._last_message_time = message_count, sim_time
        return sim_time - self._last_message_time > self.timeout


class DeadbandCompensator:
    """Scales a small non-zero command up until one axis reaches ``min_command``, keeping its direction.

    Only from a stand: once Spot is stepping it follows small commands, and boosting them then
    just causes overshoot (measured).
    """

    STOP = 0.02  # below this on every axis the command is an intentional stop
    MOVING_LINEAR = 0.05  # m/s; faster than this, Spot counts as moving
    MOVING_ANGULAR = 0.08  # rad/s

    def __init__(self, min_command, enabled: bool = True):
        self.min_command = np.asarray(min_command, dtype=np.float32)
        self.enabled = enabled

    def __call__(self, command: np.ndarray, body_velocity) -> np.ndarray:
        if not self.enabled or np.all(np.abs(command) < self.STOP):
            return command
        vx, vy, wz = (float(v) for v in body_velocity)
        if math.hypot(vx, vy) > self.MOVING_LINEAR or abs(wz) > self.MOVING_ANGULAR:
            return command
        reach = float(np.max(np.abs(command) / self.min_command))  # 1.0: dominant axis at its minimum
        return command / reach if reach < 1.0 else command


class HeadingHold:
    """With a zero command, turns back to the heading captured when the command went to zero.

    Hysteresis (engage above ``engage_deg``, release below ``release_deg``) keeps the policy from
    being fed a stream of tiny commands once it is back on heading.
    """

    def __init__(self, gain: float, max_wz: float, engage_deg: float, release_deg: float, enabled: bool = True):
        self.gain, self.max_wz = gain, max_wz
        self.engage, self.release = math.radians(engage_deg), math.radians(release_deg)
        self.enabled = enabled
        self._target, self._active = None, False

    def __call__(self, command: np.ndarray, yaw: float) -> np.ndarray:
        if not self.enabled or np.any(np.abs(command) > 1e-3):
            self._target, self._active = None, False
            return command
        if self._target is None:
            self._target = yaw
        error = wrap_angle(self._target - yaw)
        if abs(error) > self.engage:
            self._active = True
        elif abs(error) < self.release:
            self._active = False
        if not self._active:
            return command
        return np.array([0.0, 0.0, min(max(self.gain * error, -self.max_wz), self.max_wz)], dtype=np.float32)


class CommandFilter:
    """cmd_vel -> policy command, through the stages described in the module docstring."""

    def __init__(self, watchdog: CommandWatchdog, limits: CommandLimits,
                 deadband: DeadbandCompensator, heading_hold: HeadingHold):
        self.watchdog, self.limits = watchdog, limits
        self.deadband, self.heading_hold = deadband, heading_hold
        self.stale = True

    @classmethod
    def from_args(cls, args) -> "CommandFilter":
        return cls(CommandWatchdog(args.cmd_timeout),
                   CommandLimits(args.min_vx, args.max_vx, args.max_vy, args.max_wz),
                   DeadbandCompensator(args.min_cmd, args.deadband_comp),
                   HeadingHold(args.hold_gain, args.hold_max_wz, args.hold_engage_deg, args.hold_release_deg,
                               args.hold_heading))

    def __call__(self, linear, angular, message_count: int, sim_time: float, body_velocity, yaw: float) -> np.ndarray:
        """``linear``/``angular`` are the Twist vectors; ``body_velocity`` is (vx, vy, wz) in the body frame."""
        self.stale = self.watchdog.is_stale(message_count, sim_time)
        if self.stale:
            requested = ZERO.copy()
        else:
            requested = self.deadband(self.limits.clamp(float(linear[0]), float(linear[1]), float(angular[2])),
                                      body_velocity)
        return self.heading_hold(requested, yaw)
