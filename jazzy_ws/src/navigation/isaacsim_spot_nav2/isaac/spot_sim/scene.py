# SPDX-License-Identifier: Apache-2.0
"""The environment Spot walks in, and the physics settings it needs."""

import os

from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

from . import log

# realpath: with colcon --symlink-install this file is reached through install/, but scenes/ is in the workspace
SCENES_DIR = os.environ.get("SPOT_SIM_SCENES_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), *[".."] * 5, "scenes"))

# name -> (USD file, prims switched off by default): the DD100 office with the DD100 and its clock graph off
SCENE_PRESETS = {
    "office": (os.path.join(SCENES_DIR, "dd100_office_navigation.usd"), "/Robots,/World/clock"),
}


def load_environment(app, stage, scene: str, deactivate: str | None, assets_root: str) -> None:
    if scene == "grid":
        set_stage_up_axis("Z")
        set_stage_units(meters_per_unit=1.0)
        define_prim("/World/Ground", "Xform").GetReferences().AddReference(
            assets_root + "/Isaac/Environments/Grid/default_environment.usd")
        define_prim("/World/PhysicsScene", "PhysicsScene")
        return
    scene_file, default_off = SCENE_PRESETS.get(scene, (scene, ""))
    if not os.path.isfile(scene_file):
        raise SystemExit(f"scene not found: {scene_file}")
    # A sublayer of the in-memory stage: nothing the simulator does is saved into the scene file.
    stage.GetRootLayer().subLayerPaths.append(scene_file)
    for _ in range(10):
        app.update()
    for path in filter(None, (p.strip() for p in (deactivate if deactivate is not None else default_off).split(","))):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
            log(f"deactivated {path}")
    log(f"scene: {scene_file}")


def add_obstacle_boxes(stage, specs: list[str], floor_z: float) -> None:
    """Static boxes standing on the floor, each given as "X,Y,SX,SY,SZ"."""
    for i, spec in enumerate(specs):
        x, y, sx, sy, sz = (float(v) for v in spec.split(","))
        cube = UsdGeom.Cube.Define(stage, f"/World/TestObstacle_{i}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, floor_z + sz / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        log(f"obstacle box at ({x}, {y}) size ({sx}, {sy}, {sz})")


def configure_physics_scenes(stage, steps_per_second: int, device: str) -> None:
    """Sets the step rate and the device on every physics scene.

    Isaac's SimulationManager only updates scenes it created. One loaded with a USD file keeps its
    own settings: the office's ran at 60 Hz (the 500 Hz policy collapses) with GPU dynamics, even
    with --device cpu (GPU physics read back to the CPU every step, half the frame time).
    """
    gpu = device == "cuda"
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            api = PhysxSchema.PhysxSceneAPI.Apply(prim)
            api.CreateTimeStepsPerSecondAttr().Set(steps_per_second)
            api.CreateEnableGPUDynamicsAttr().Set(gpu)
            api.CreateBroadphaseTypeAttr().Set("GPU" if gpu else "MBP")
            log(f"physics scene {prim.GetPath()}: {steps_per_second} steps/s, {'GPU' if gpu else 'CPU'} dynamics")
