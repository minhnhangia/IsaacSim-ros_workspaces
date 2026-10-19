# SPDX-License-Identifier: Apache-2.0
"""The height of the floor under the robot, from a PhysX raycast that ignores the robot itself."""

import omni.physx

MAX_DROP = 2.0  # m below the body; a robot never stands higher than this above its floor


class FloorTracker:
    def __init__(self, robot_prim: str, initial_height: float):
        self._robot_prefix = robot_prim.rstrip("/") + "/"
        self._query = omni.physx.get_physx_scene_query_interface()
        self.height = initial_height

    def update(self, x: float, y: float, z: float) -> float:
        """World z of the nearest surface straight below the body; keeps the last value over a hole."""
        nearest = [None]

        def report(hit):
            if not hit.collision.startswith(self._robot_prefix) and (nearest[0] is None or hit.distance < nearest[0]):
                nearest[0] = hit.distance
            return True

        self._query.raycast_all((float(x), float(y), float(z)), (0.0, 0.0, -1.0), MAX_DROP, report)
        if nearest[0] is not None:
            self.height = z - nearest[0]
        return self.height
