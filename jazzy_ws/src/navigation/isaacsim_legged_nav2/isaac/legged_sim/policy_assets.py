# SPDX-License-Identifier: Apache-2.0
"""Where exported policy weights live, and how they are loaded and checked.

``scripts/export_policies.py`` (run once with a Python that has torch) downloads NVIDIA's
checkpoints and writes, per robot, ``<cache>/<robot>/`` with ``*.npz`` weights, ``*_reference.npz``
input/output pairs computed by torch, and ``manifest.json`` (sources and SHA-256 of the downloads).
Loading re-runs the reference inputs through the NumPy networks, so a bad export fails loudly.
"""

import json
import os

import numpy as np

from .networks import LSTMActuatorNet, MLPPolicy

ENV_VAR = "LEGGED_SIM_POLICY_DIR"
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "isaacsim_legged_nav2", "policies")


class PolicyAssetsMissing(RuntimeError):
    pass


def policy_dir(robot: str) -> str:
    return os.path.join(os.environ.get(ENV_VAR) or DEFAULT_DIR, robot)


def _load(robot: str, name: str) -> dict:
    path = os.path.join(policy_dir(robot), f"{name}.npz")
    if not os.path.isfile(path):
        raise PolicyAssetsMissing(
            f"{path} not found. Export the {robot} policy once (needs a Python with torch):\n"
            f"    python3 <isaacsim_legged_nav2>/scripts/export_policies.py {robot}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def manifest(robot: str) -> dict:
    with open(os.path.join(policy_dir(robot), "manifest.json")) as f:
        return json.load(f)


def load_actor(robot: str) -> MLPPolicy:
    actor = MLPPolicy(_load(robot, "actor"))
    ref = _load(robot, "actor_reference")
    got = np.stack([actor(x) for x in ref["inputs"]])
    if not np.allclose(got, ref["outputs"], atol=1e-4):
        raise ValueError(f"{robot} actor does not reproduce torch's outputs (max error {np.abs(got - ref['outputs']).max()})")
    return actor


def load_actuator_net(robot: str, num_joints: int, saturation_effort: float, effort_limit: float,
                      velocity_limit: float) -> LSTMActuatorNet:
    weights = _load(robot, "actuator_net")
    ref = _load(robot, "actuator_net_reference")
    check = LSTMActuatorNet(weights, ref["inputs"].shape[1], 1e9, 1e9, 1e9)
    got = np.stack([check.raw_torque(x[:, 0], x[:, 1]) for x in ref["inputs"]])
    if not np.allclose(got, ref["outputs"], atol=1e-4):
        raise ValueError(f"{robot} actuator network does not reproduce torch's outputs")
    return LSTMActuatorNet(weights, num_joints, saturation_effort, effort_limit, velocity_limit)
