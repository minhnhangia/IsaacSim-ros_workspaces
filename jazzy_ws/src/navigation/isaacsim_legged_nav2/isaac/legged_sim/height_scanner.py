# SPDX-License-Identifier: Apache-2.0
"""Terrain heights under and around the robot, from PhysX raycasts against the scene's colliders.

Isaac Lab casts its rays from 20 m above against the terrain mesh only. An office has ceilings
and stair flights overhead, so rays here start just above the scanned link and ignore hits on the
robot itself. This is ground-truth geometry, the simulation stand-in for an elevation map.
"""

import numpy as np
import omni.physx

from .rough_policy import HeightScanGrid

RAY_START_ABOVE = 0.3  # m above the scanned link
MAX_DISTANCE = 5.0  # m; a ray that hits nothing reports the floor that far below its start


class PhysxHeightScanner:
    def __init__(self, robot_prim: str, grid: HeightScanGrid | None = None):
        self.grid = grid or HeightScanGrid()
        self._robot_prefix = robot_prim.rstrip("/") + "/"
        self._query = omni.physx.get_physx_scene_query_interface()

    def hit_heights(self, x: float, y: float, z: float, yaw: float) -> np.ndarray:
        """World z of the terrain below each grid point, for a link at (x, y, z) with heading yaw."""
        start = z + RAY_START_ABOVE
        heights = np.empty(len(self.grid))
        for i, (px, py) in enumerate(self.grid.world_points(x, y, yaw)):
            nearest = [MAX_DISTANCE]

            def report(hit, nearest=nearest):
                if not hit.collision.startswith(self._robot_prefix) and hit.distance < nearest[0]:
                    nearest[0] = hit.distance
                return True

            self._query.raycast_all((float(px), float(py), start), (0.0, 0.0, -1.0), MAX_DISTANCE, report)
            heights[i] = start - nearest[0]
        return heights
