# SPDX-License-Identifier: Apache-2.0
"""Cameras that look at the robot: the GUI's initial view, and optional snapshot images for debugging."""

import math
import os

import numpy as np

from . import log

# (ahead, left, up) of the robot, in metres; tried in order
VIEW_CANDIDATES = [(-3.0, 1.8, 1.8), (-3.0, -1.8, 1.8), (2.8, 1.8, 1.6), (2.8, -1.8, 1.6), (0.3, 3.2, 1.8),
                   (0.3, -3.2, 1.8)]
CLOSEUP_OFFSET = (0.25, -1.5, 0.95)


def view_around_robot(x, y, yaw, offset, floor_z, look_up=0.45, look_ahead=0.3):
    """Camera eye and target in world coordinates, for an offset given relative to the robot."""
    c, s = math.cos(yaw), math.sin(yaw)
    ahead, left, up = offset
    eye = [x + c * ahead - s * left, y + s * ahead + c * left, floor_z + up]
    target = [x + c * look_ahead, y + s * look_ahead, floor_z + look_up]
    return eye, target


def choose_view_offset(x, y, yaw, floor_z):
    """The first candidate that sees the robot's body and feet and stands on open floor.

    Open floor: nothing within 1.2 m around the camera at 0.4 m and 1.0 m height, so furniture
    right next to it (a reception counter, say) does not fill the foreground.
    """
    try:
        import omni.physx

        query = omni.physx.get_physx_scene_query_interface()

        def blocked(origin, end, stop_short):
            d = np.subtract(end, origin)
            dist = float(np.linalg.norm(d))
            return query.raycast_closest(tuple(map(float, origin)), tuple(d / dist), dist - stop_short)["hit"]

        for offset in VIEW_CANDIDATES:
            eye, target = view_around_robot(x, y, yaw, offset, floor_z)
            feet = [target[0], target[1], floor_z + 0.1]
            if blocked(eye, target, 0.9) or blocked(eye, feet, 0.9):  # rays stop short of the robot itself
                continue
            ring = [((eye[0], eye[1], floor_z + h), a) for h in (0.4, 1.0)
                    for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
            if not any(blocked(o, (o[0] + 1.2 * math.cos(a), o[1] + 1.2 * math.sin(a), o[2]), 0.0) for o, a in ring):
                return offset
    except Exception as error:  # noqa: BLE001 - a viewing problem must not stop the simulation
        log(f"view selection fell back to the default: {error}")
    return VIEW_CANDIDATES[0]


def set_initial_view(x, y, yaw, offset, floor_z) -> None:
    """Points the GUI viewport at the robot (the office scene stores a view from outside the building)."""
    try:
        from isaacsim.core.rendering_manager import ViewportManager

        eye, target = view_around_robot(x, y, yaw, offset, floor_z)
        ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=eye, target=target)
        log(f"initial view: offset {offset} from the robot")
    except Exception as error:  # noqa: BLE001
        log(f"could not set the initial viewport camera: {error}")


class SnapshotRecorder:
    """Renders the robot from a follow camera and a close-up, saving PNGs every ``every`` sim seconds."""

    def __init__(self, stage, out_dir: str, every: float, view_offset, floor_z: float):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.every, self.floor_z = out_dir, every, floor_z
        self.next_time = 1.0
        self.views = {"view": dict(offset=view_offset, look_up=0.45, look_ahead=0.3, focal=14.0),
                      "closeup": dict(offset=CLOSEUP_OFFSET, look_up=0.55, look_ahead=0.0, focal=18.0)}
        for name, view in self.views.items():
            camera = UsdGeom.Camera.Define(stage, f"/World/SnapshotCam_{name}")
            camera.CreateFocalLengthAttr(view["focal"])
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([rep.create.render_product(str(camera.GetPath()), (1280, 720))])
            view.update(path=str(camera.GetPath()), annotator=annotator)

    def update(self, sim_time: float, x, y, yaw) -> None:
        from isaacsim.core.rendering_manager import ViewportManager
        from PIL import Image

        for view in self.views.values():
            eye, target = view_around_robot(x, y, yaw, view["offset"], self.floor_z, view["look_up"], view["look_ahead"])
            ViewportManager.set_camera_view(view["path"], eye=eye, target=target)
        if sim_time < self.next_time:
            return
        for name, view in self.views.items():
            data = view["annotator"].get_data()
            if data is not None and getattr(data, "size", 0):
                Image.fromarray(np.asarray(data)[:, :, :3]).save(os.path.join(self.out_dir, f"t{sim_time:05.1f}_{name}.png"))
        self.next_time += self.every
