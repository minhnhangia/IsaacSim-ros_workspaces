# SPDX-License-Identifier: Apache-2.0
"""ANYbotics ANYmal-D under Isaac Lab's pretrained rough-terrain policy (climbs and descends stairs).

Deployment of ``Isaac-Velocity-Rough-Anymal-D-v0`` (PhysX), reproducing Isaac Lab's settings
(isaaclab_assets ``ANYMAL_D_CFG``, isaaclab_tasks ``AnymalDRoughEnvCfg``): 200 Hz physics, 50 Hz
policy, joint position actions (scale 0.5) tracked by the ANYdrive LSTM actuator network with
DC-motor limits, robot friction 0.8/0.6, self-collisions on, 4/0 solver iterations.
Weights come from ``scripts/export_policies.py anymal_d``.
"""

import math
import re

import numpy as np
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from .. import log
from ..height_scanner import PhysxHeightScanner
from ..physics_materials import bind_physics_material, define_physics_material
from ..policy_assets import load_actor, load_actuator_net
from ..rough_policy import BaseState, RoughTerrainPolicy, height_scan_observation
from .base import CommandProfile, LeggedRobot

PRIM = "/World/Robot"
USD = "/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-D/anymal_d.usd"
DEFAULT_JOINT_POSITIONS = {".*HAA": 0.0, ".*F_HFE": 0.4, ".*H_HFE": -0.4, ".*F_KFE": -0.8, ".*H_KFE": 0.8}
ACTION_SCALE = 0.5
ANYDRIVE = dict(saturation_effort=120.0, effort_limit=80.0, velocity_limit=7.5)
FOOT_FRICTION = (0.8, 0.6)  # static, dynamic (training's startup material event)
SOLVER_ITERATIONS = (4, 0)  # position, velocity


def _quat_rotate_inverse(q, v) -> np.ndarray:
    """Rotates world vector ``v`` into the frame of orientation ``q`` (w, x, y, z)."""
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r.T @ np.asarray(v, dtype=np.float64)


class AnymalDRobot(LeggedRobot):
    name = "anymal_d"
    prim_path = PRIM
    body_prim = f"{PRIM}/base"
    spawn_height = 0.65
    physics_dt = 0.005
    control_dt = 0.02
    climbs_stairs = True
    # Measured on the visual mesh: the shell top at the centre is z 0.084; the stock LiDAR housing at
    # the rear reaches 0.19-0.23, so a 0.15 m mast keeps the OS0's view backward clear of it.
    lidar_mount = (0.0, 0.0, 0.15)
    camera_height = 0.02  # in the recess of the front face (z 0.00-0.04), where ANYmal's own camera sits
    # Trained on commands within +-1 m/s, +-1 rad/s. Measured from a settled stand: it follows
    # 0.05 m/s forward and 0.08 m/s sideways, but turns at only ~50-60% of 0.05-0.15 rad/s (fully at
    # 0.2). Regulated Pure Pursuit limits each turn command to the measured rate plus 0.1 rad/s, so
    # an unanswered 0.1 rad/s stalled turns in place on the office floor: compensate turning from a
    # stand. Its heading drifts 1-7 deg in 5 s after stopping, inside the yaw tolerance: no hold.
    commands = CommandProfile(min_vx=-1.0, max_vx=1.0, max_vy=0.8, max_wz=1.0, min_command=(0.05, 0.08, 0.20),
                              deadband_compensation=True, hold_heading=False)

    def __init__(self):
        self._decimation = int(round(self.control_dt / self.physics_dt))
        self._actor = load_actor(self.name)  # fail before the scene is built if the export is missing
        self._articulation = None
        self._policy = self._actuator_net = self._scanner = None
        self._targets = None
        self._steps = 0
        self._callback = None

    # -------------------------------------------------------------- setup
    def spawn(self, x, y, yaw, floor_z):
        stage = define_prim(PRIM, "Xform").GetStage()
        prim = stage.GetPrimAtPath(PRIM)
        prim.GetReferences().AddReference(get_assets_root_path() + USD)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, floor_z + self.spawn_height))
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
        bind_physics_material(prim, define_physics_material(stage, "/World/Materials/anymal_feet", *FOOT_FRICTION))
        self._configure_physics(stage, prim)
        self._articulation = Articulation(PRIM)
        names = list(self._articulation.dof_names)
        defaults = np.zeros(len(names))
        for pattern, value in DEFAULT_JOINT_POSITIONS.items():
            defaults[[i for i, n in enumerate(names) if re.fullmatch(pattern, n)]] = value
        self._policy = RoughTerrainPolicy(self._actor, defaults, ACTION_SCALE)
        self._actuator_net = load_actuator_net(self.name, len(names), **ANYDRIVE)
        self._scanner = PhysxHeightScanner(PRIM)
        log(f"ANYmal-D: {len(names)} joints {names}")

    @staticmethod
    def _configure_physics(stage, prim) -> None:
        """Torque-controlled joints (the actuator network supplies the torque) and articulation settings."""
        for joint in (p for p in Usd.PrimRange(prim) if p.IsA(UsdPhysics.RevoluteJoint)):
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            drive.CreateMaxForceAttr().Set(1.0e6)
        for root in (p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)):
            api = PhysxSchema.PhysxArticulationAPI.Apply(root)
            api.CreateSolverPositionIterationCountAttr().Set(SOLVER_ITERATIONS[0])
            api.CreateSolverVelocityIterationCountAttr().Set(SOLVER_ITERATIONS[1])
            api.CreateEnabledSelfCollisionsAttr().Set(True)
            # Torques written from Python do not wake a sleeping articulation, so never let it sleep.
            api.CreateSleepThresholdAttr().Set(0.0)
            api.CreateStabilizationThresholdAttr().Set(0.0)

    # -------------------------------------------------------------- runtime
    def reset(self, command):
        art = self._articulation
        defaults = self._policy.default_joint_positions.reshape(1, -1)
        art.set_dof_positions(defaults)
        art.set_dof_velocities(np.zeros_like(defaults))
        self._policy.reset()
        self._actuator_net.reset()
        self._targets = self._policy.default_joint_positions.copy()
        self._steps = 0
        if self._callback is None:  # the actuator network runs before every physics step
            self._callback = SimulationManager.register_callback(self._apply_torques, IsaacEvents.PRE_PHYSICS_STEP)

    def step(self, dt, command):
        self._steps += 1
        if self._steps % self._decimation:
            return
        position, orientation = self.world_pose()
        state = self._base_state(orientation)
        yaw = math.atan2(2 * (orientation[0] * orientation[3] + orientation[1] * orientation[2]),
                         1 - 2 * (orientation[2] ** 2 + orientation[3] ** 2))
        scan = height_scan_observation(position[2], self._scanner.hit_heights(*position, yaw))
        self._targets = self._policy.act(state, command, scan)

    def world_pose(self):
        position, orientation = self._articulation.get_world_poses()
        return position.numpy().reshape(-1), orientation.numpy().reshape(-1)

    def close(self):
        if self._callback is not None:
            SimulationManager.deregister_callback(self._callback)
            self._callback = None

    def _joint_state(self):
        art = self._articulation
        return art.get_dof_positions().numpy().reshape(-1), art.get_dof_velocities().numpy().reshape(-1)

    def _base_state(self, orientation) -> BaseState:
        linear, angular = self._articulation.get_velocities()
        q, qd = self._joint_state()
        return BaseState(linear_velocity=_quat_rotate_inverse(orientation, linear.numpy().reshape(-1)),
                         angular_velocity=_quat_rotate_inverse(orientation, angular.numpy().reshape(-1)),
                         projected_gravity=_quat_rotate_inverse(orientation, (0.0, 0.0, -1.0)),
                         joint_positions=q, joint_velocities=qd)

    def _apply_torques(self, dt, context) -> None:
        if self._targets is None:
            return
        q, qd = self._joint_state()
        self._articulation.set_dof_efforts(self._actuator_net(q, qd, self._targets).reshape(1, -1))
