# SPDX-License-Identifier: Apache-2.0
"""Boston Dynamics Spot under NVIDIA's bundled flat-terrain Isaac Lab policy (``RobotPolicyRunner``)."""

from isaacsim.robot.policy.examples import PolicyEnvConfig, RobotPolicyRunner, get_spot_spec

from .. import log
from ..actuators import FastActuators
from ..geometry import yaw_quat
from .base import CommandProfile, LeggedRobot


class SpotRobot(LeggedRobot):
    name = "spot"
    body_prim = "/World/Spot/body"
    spawn_height = 0.5
    climbs_stairs = False  # flat-terrain policy: stalls at a 15 cm step and can flip over
    lidar_mount = (0.0, 0.0, 0.10)
    # Measured: from a settled stand the policy ignores vx <= 0.2, vy <= 0.15, wz <= 0.2-0.25, and it
    # turns ~40 deg while settling. Limits stay well inside the trained range (vx -2..3, vy +-1.5, wz +-2).
    commands = CommandProfile(min_vx=-1.0, max_vx=1.5, max_vy=0.8, max_wz=1.2, min_command=(0.30, 0.22, 0.32),
                              deadband_compensation=True, hold_heading=True)

    def __init__(self, engine: str = "physx", fast_actuators: bool = True):
        self._spec = get_spot_spec()
        self._engine, self._fast_actuators = engine, fast_actuators
        timing = PolicyEnvConfig.from_file(self._spec.engines[engine].env_config_path).timing
        self.physics_dt = timing.physics_dt
        self.control_dt = timing.render_interval * timing.physics_dt
        self._runner = None
        self._fast_actuators_tried = False

    def spawn(self, x, y, yaw, floor_z):
        self._runner = RobotPolicyRunner(self._spec, prim_path="/World/Spot", position=[x, y, floor_z + self.spawn_height],
                                         orientation=list(yaw_quat(yaw)))
        self._runner.spawn()

    def reset(self, command):
        self._runner.restart_from_default_state(command)
        self._use_fast_actuators()

    def step(self, dt, command):
        self._runner.step(dt, command)

    def world_pose(self):
        position, orientation = self._runner.articulation.get_world_poses()
        return position.numpy().reshape(-1), orientation.numpy().reshape(-1)

    def close(self):
        if self._runner is not None:
            self._runner.close()

    def _use_fast_actuators(self) -> None:
        """Once, after the runner has built Isaac's actuators (see actuators.py)."""
        if self._fast_actuators_tried or not self._fast_actuators:
            return
        self._fast_actuators_tried = True
        isaac_actuators = getattr(self._runner, "_actuators", None)
        if isaac_actuators is None:
            return
        try:
            fast = FastActuators(isaac_actuators).install()
            log(f"fast actuators: {fast.state.model.describe()}")
        except Exception as error:  # noqa: BLE001 - keep Isaac's implementation
            log(f"keeping Isaac's ArticulationActuators: {error}")
