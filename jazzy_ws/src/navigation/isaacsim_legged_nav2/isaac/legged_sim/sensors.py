# SPDX-License-Identifier: Apache-2.0
"""Sensors mounted on the robot's body: an Ouster OS0 on a mast and a front RealSense D455 stereo pair."""

from dataclasses import dataclass

from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.ros2.nodes import set_isaac_name_override
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from . import log

OS0_ASSET = "/Isaac/Sensors/Ouster/OS0/OS0.usd"
D455_ASSET = "/Isaac/Sensors/RealSense/D455/rsd455.usd"
MOUNT_COLOR = Gf.Vec3f(0.12, 0.12, 0.12)


@dataclass(frozen=True)
class BodyBounds:
    min: tuple  # (x, y, z) in the body frame
    max: tuple

    @classmethod
    def of(cls, stage, body_path: str) -> "BodyBounds":
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
        box = cache.ComputeUntransformedBound(stage.GetPrimAtPath(body_path)).ComputeAlignedRange()
        return cls(tuple(box.GetMin()), tuple(box.GetMax()))


@dataclass(frozen=True)
class StereoCameras:
    left: str
    right: str


def _reference(path: str, asset: str, translate) -> Usd.Prim:
    prim = define_prim(path, "Xform")
    prim.GetReferences().AddReference(asset)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return prim


def mount_os0(app, stage, body: str, bounds: BodyBounds, assets_root: str, mount, variant: str) -> str:
    """Mounts the OS0 ``mount[2]`` metres above the body's top surface and returns the sensor prim.

    The mast and plate under it are visual only (no collider, no mass). Nav2's self-filters
    (spot_navigation.launch.xml) were measured at the default 0.10 m height.
    """
    x, y, height = mount
    top = bounds.max[2]
    os0 = _reference(f"{body}/OS0", assets_root + OS0_ASSET, (x, y, top + height))
    variants = os0.GetVariantSets().GetVariantSet("sensor")
    if variant not in variants.GetVariantNames():
        raise SystemExit(f"--lidar-config {variant!r} not in {variants.GetVariantNames()}")
    variants.SetVariantSelection(variant)
    app.update()
    sensor = f"{body}/OS0/sensor"
    if not stage.GetPrimAtPath(sensor).IsValid():
        raise SystemExit(f"OS0 sensor prim missing at {sensor}")
    set_isaac_name_override(sensor, "OS0")  # TF frame == point cloud frame_id, at the sensor origin
    if height > 0.005:
        mast = UsdGeom.Cylinder.Define(stage, f"{body}/OS0_mast")
        mast.CreateRadiusAttr(0.035)
        mast.CreateHeightAttr(height)
        mast.CreateAxisAttr("Z")
        mast.CreateDisplayColorAttr([MOUNT_COLOR])
        UsdGeom.Xformable(mast).AddTranslateOp().Set(Gf.Vec3d(x, y, top + height / 2))
        plate = UsdGeom.Cube.Define(stage, f"{body}/OS0_mount_plate")
        plate.CreateSizeAttr(1.0)
        plate.CreateDisplayColorAttr([MOUNT_COLOR])
        plate_xform = UsdGeom.Xformable(plate)
        plate_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, top + 0.004))
        plate_xform.AddScaleOp().Set(Gf.Vec3f(0.16, 0.12, 0.008))
    log(f"OS0 ({variant}) at body ({x}, {y}, {top + height:.3f}) on a {height:.2f} m mast")
    return sensor


def mount_d455(app, stage, body: str, bounds: BodyBounds, assets_root: str, tick_rate: float) -> StereoCameras:
    """Mounts the D455 on the body's front face, near its top edge, looking forward."""
    root = f"{body}/front_stereo"
    translate = (bounds.max[0] + 0.02, 0.0, bounds.max[2] - 0.03)
    _reference(root, assets_root + D455_ASSET, translate)
    app.update()
    # NVIDIA's D455 is a rigid body with mass but no collider. Nested under a robot link it becomes
    # its own body and falls through the floor; make it visual-only.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    cameras = StereoCameras(f"{root}/RSD455/Camera_OmniVision_OV9782_Left", f"{root}/RSD455/Camera_OmniVision_OV9782_Right")
    for path, frame in ((cameras.left, "front_stereo_left_optical_frame"),
                        (cameras.right, "front_stereo_right_optical_frame")):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            found = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.IsA(UsdGeom.Camera)]
            raise SystemExit(f"camera prim missing: {path}; cameras found: {found}")
        set_isaac_name_override(path, frame)
        if tick_rate > 0:  # the renderer then draws the camera at this rate instead of every frame
            prim.ApplyAPI("OmniSensorAPI")
            prim.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(float(tick_rate))
    log(f"D455 stereo at body front ({translate[0]:.3f}, 0, {translate[2]:.3f})")
    return cameras
