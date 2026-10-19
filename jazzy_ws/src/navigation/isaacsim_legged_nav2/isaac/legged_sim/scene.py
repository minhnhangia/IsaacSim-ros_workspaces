# SPDX-License-Identifier: Apache-2.0
"""The environment the robot walks in, and the physics settings it needs."""

import os
from dataclasses import dataclass

from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

from . import log
from .physics_materials import bind_physics_material, define_physics_material

# realpath: with colcon --symlink-install this file is reached through install/, but scenes/ is in the workspace
SCENES_DIR = os.environ.get("SPOT_SIM_SCENES_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), *[".."] * 5, "scenes"))


@dataclass(frozen=True)
class StairAccess:
    """What closes a scene's stairwells off, and how to open them."""

    doors: tuple = ()  # closed doors in front of stairs
    ground_plane: str = ""  # an infinite collision plane that also seals the flights going down
    floor_prefix: str = ""  # top-level environment prims whose meshes are the visible floors


@dataclass(frozen=True)
class ScenePreset:
    usd_file: str
    environment_prim: str  # root of the environment's own geometry (receives the floor material)
    deactivate: tuple = ()  # prims switched off by default
    stairs: StairAccess = StairAccess()


SCENE_PRESETS = {
    # The DD100 office with the DD100 and its clock graph off. Panic doors close off the small stair
    # core. The only ground-floor collider is an invisible infinite plane at z=0 that covers both
    # stair cores, so flights down to the basement are walked over like floor; the visible floor
    # tiles (SM_Floor*, ground floor and basement) have no colliders, and leave the stairwells open.
    "office": ScenePreset(os.path.join(SCENES_DIR, "dd100_office_navigation.usd"), "/World/office",
                          deactivate=("/Robots", "/World/clock"),
                          stairs=StairAccess(doors=("/World/office/BP_PanicDoor4", "/World/office/BP_PanicDoor2_368",
                                                    "/World/office/BP_PanicDoor6"),
                                             ground_plane="/World/office/GroundPlane/CollisionPlane",
                                             floor_prefix="SM_Floor")),
}


def load_environment(app, stage, scene: str, deactivate: str | None, assets_root: str,
                     floor_friction: float, stair_access: bool) -> None:
    if scene == "grid":
        set_stage_up_axis("Z")
        set_stage_units(meters_per_unit=1.0)
        define_prim("/World/Ground", "Xform").GetReferences().AddReference(
            assets_root + "/Isaac/Environments/Grid/default_environment.usd")
        define_prim("/World/PhysicsScene", "PhysicsScene")
        return
    preset = SCENE_PRESETS.get(scene, ScenePreset(scene, "/World"))
    if not os.path.isfile(preset.usd_file):
        raise SystemExit(f"scene not found: {preset.usd_file}")
    # A sublayer of the in-memory stage: nothing the simulator does is saved into the scene file.
    stage.GetRootLayer().subLayerPaths.append(preset.usd_file)
    for _ in range(10):
        app.update()
    off = list(preset.deactivate) if deactivate is None else [p.strip() for p in deactivate.split(",") if p.strip()]
    if stair_access:
        off += preset.stairs.doors
        _collide_with_floors(stage, preset.environment_prim, preset.stairs)
        if preset.stairs.ground_plane:
            off.append(preset.stairs.ground_plane)
    for path in off:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
            log(f"deactivated {path}")
    # Without a physics material PhysX uses friction 0.5, about half what the policies were trained on.
    material = define_physics_material(stage, "/World/Materials/floor", floor_friction, floor_friction)
    environment = stage.GetPrimAtPath(preset.environment_prim)
    if environment.IsValid():
        bind_physics_material(environment, material)
    log(f"scene: {preset.usd_file} (floor friction {floor_friction})")


def _collide_with_floors(stage, environment_prim: str, stairs: StairAccess) -> None:
    """Gives the visible floor meshes exact (triangle-mesh) colliders, holes and all."""
    environment = stage.GetPrimAtPath(environment_prim)
    if not stairs.floor_prefix or not environment.IsValid():
        return
    count = 0
    for top in environment.GetChildren():
        if not top.GetName().startswith(stairs.floor_prefix):
            continue
        for mesh in (p for p in Usd.PrimRange(top) if p.IsA(UsdGeom.Mesh)):
            UsdPhysics.CollisionAPI.Apply(mesh)
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
            count += 1
    log(f"floor colliders on {count} {stairs.floor_prefix}* meshes")


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
