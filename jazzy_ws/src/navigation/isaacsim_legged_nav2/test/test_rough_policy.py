# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from legged_sim.networks import MLPPolicy
from legged_sim.rough_policy import BaseState, HeightScanGrid, RoughTerrainPolicy, height_scan_observation


def test_grid_matches_isaac_lab_pattern():
    grid = HeightScanGrid()
    assert len(grid) == 187  # 17 x 11
    np.testing.assert_allclose(grid.points[0], (-0.8, -0.5))
    np.testing.assert_allclose(grid.points[1], (-0.7, -0.5))  # x varies fastest ("xy" ordering)
    np.testing.assert_allclose(grid.points[17], (-0.8, -0.4))
    np.testing.assert_allclose(grid.points[-1], (0.8, 0.5))


def test_grid_rotates_with_yaw():
    world = HeightScanGrid().world_points(1.0, 2.0, np.pi / 2)
    np.testing.assert_allclose(world[-1], (1.0 - 0.5, 2.0 + 0.8), atol=1e-12)  # forward is +y, left is -x


def test_height_scan_observation_offsets_and_clips():
    np.testing.assert_allclose(height_scan_observation(0.6, [0.0, 0.15, -2.0, 1.5]), [0.1, -0.05, 1.0, -1.0])


def _identity_policy(n_joints, n_scan=187):
    size = 12 + 3 * n_joints + n_scan
    w = np.zeros((n_joints, size))
    w[np.arange(n_joints), 12 + n_joints + np.arange(n_joints)] = 1.0  # action = joint velocities
    return MLPPolicy({"mlp_0_weight": w, "mlp_0_bias": np.zeros(n_joints)})


def test_observation_layout_and_action_mapping():
    policy = RoughTerrainPolicy(_identity_policy(2), default_joint_positions=[0.4, -0.8], action_scale=0.5)
    state = BaseState(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]), np.array([0, 0, -1.0]),
                      joint_positions=np.array([0.5, -0.8]), joint_velocities=np.array([0.2, -0.4]))
    obs = policy.observation(state, [0.3, 0.0, 0.1], np.full(187, 0.1))
    np.testing.assert_allclose(obs[:12], [1, 2, 3, 4, 5, 6, 0, 0, -1, 0.3, 0.0, 0.1])
    np.testing.assert_allclose(obs[12:14], [0.1, 0.0])  # positions relative to the defaults
    np.testing.assert_allclose(obs[14:16], [0.2, -0.4])
    np.testing.assert_allclose(obs[16:18], [0.0, 0.0])  # no previous action yet
    targets = policy.act(state, [0.3, 0.0, 0.1], np.full(187, 0.1))
    np.testing.assert_allclose(targets, [0.4 + 0.5 * 0.2, -0.8 - 0.5 * 0.4])
    np.testing.assert_allclose(policy.observation(state, [0, 0, 0], np.zeros(187))[16:18], [0.2, -0.4])


def test_policy_size_mismatch_is_rejected():
    with pytest.raises(ValueError):
        RoughTerrainPolicy(_identity_policy(3), default_joint_positions=[0.0, 0.0], action_scale=0.5)
