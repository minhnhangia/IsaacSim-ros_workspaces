# SPDX-License-Identifier: Apache-2.0
"""Sensors mounted on the robot's body: an Ouster OS0 on a mast and a front RealSense D455 stereo pair."""

from dataclasses import dataclass

import numpy as np
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.ros2.nodes import set_isaac_name_override
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from . import log
from .mesh_probe import MeshProbe

OS0_ASSET = "/Isaac/Sensors/Ouster/OS0/OS0.usd"
D455_ASSET = "/Isaac/Sensors/RealSense/D455/rsd455.usd"
MOUNT_COLOR = Gf.Vec3f(0.12, 0.12, 0.12)


PLATE_SIZE = (0.16, 0.12, 0.008)  # mount plate under the mast (x, y, thickness)
MAST_RADIUS = 0.035


def body_surface(stage, body_path: str) -> MeshProbe:
    """The body link's visual mesh (instance proxies included), in the body frame."""
    body = stage.GetPrimAtPath(body_path)
    to_body = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    triangles = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputePurpose() not in ("default", "render"):
            continue
        points = prim.GetAttribute("points").Get()
        if not points:
            continue
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * to_body
        v = np.array([tuple(m.Transform(Gf.Vec3d(*p))) for p in points])
        index = np.asarray(prim.GetAttribute("faceVertexIndices").Get())
        k = 0
        for count in prim.GetAttribute("faceVertexCounts").Get():
            face = index[k:k + count]
            k += count
            triangles += [(v[face[0]], v[face[j]], v[face[j + 1]]) for j in range(1, count - 1)]
    if not triangles:
        raise SystemExit(f"no visual mesh under {body_path} to mount sensors on")
    return MeshProbe(np.array(triangles))


def _local_bounds(prim) -> Gf.Range3d:
    """Bounds of a referenced asset about its own origin."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    return cache.ComputeUntransformedBound(prim).ComputeAlignedRange()


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


def mount_os0(app, stage, body: str, surface: MeshProbe, assets_root: str, mount, variant: str) -> str:
    """Mounts the OS0 on a mast ``mount[2]`` metres tall, standing on the body at (``mount[0]``, ``mount[1]``).

    The mast stands on the highest point of the body under its mount plate, so it touches the
    shell without cutting into it. Mast and plate are visual only (no collider, no mass). Nav2's
    self-filters (params/robot_profiles.yaml) were measured with each robot's default mount.
    """
    x, y, height = mount
    base = surface.highest_below((x - PLATE_SIZE[0] / 2, x + PLATE_SIZE[0] / 2), (y - PLATE_SIZE[1] / 2, y + PLATE_SIZE[1] / 2))
    if base is None:
        raise SystemExit(f"--lidar-mount ({x}, {y}) is not above the body")
    os0 = _reference(f"{body}/OS0", assets_root + OS0_ASSET, (x, y, base + height))  # the OS0's origin is its base
    variants = os0.GetVariantSets().GetVariantSet("sensor")
    if variant not in variants.GetVariantNames():
        raise SystemExit(f"--lidar-config {variant!r} not in {variants.GetVariantNames()}")
    variants.SetVariantSelection(variant)
    app.update()
    sensor = f"{body}/OS0/sensor"
    if not stage.GetPrimAtPath(sensor).IsValid():
        raise SystemExit(f"OS0 sensor prim missing at {sensor}")
    set_isaac_name_override(sensor, "OS0")  # TF frame == point cloud frame_id, at the sensor origin
    if height > PLATE_SIZE[2]:
        plate = UsdGeom.Cube.Define(stage, f"{body}/OS0_mount_plate")
        plate.CreateSizeAttr(1.0)
        plate.CreateDisplayColorAttr([MOUNT_COLOR])
        plate_xform = UsdGeom.Xformable(plate)
        plate_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, base + PLATE_SIZE[2] / 2))
        plate_xform.AddScaleOp().Set(Gf.Vec3f(*PLATE_SIZE))
        mast = UsdGeom.Cylinder.Define(stage, f"{body}/OS0_mast")
        mast.CreateRadiusAttr(MAST_RADIUS)
        mast.CreateHeightAttr(height - PLATE_SIZE[2])
        mast.CreateAxisAttr("Z")
        mast.CreateDisplayColorAttr([MOUNT_COLOR])
        UsdGeom.Xformable(mast).AddTranslateOp().Set(Gf.Vec3d(x, y, base + PLATE_SIZE[2] + (height - PLATE_SIZE[2]) / 2))
    log(f"OS0 ({variant}) at body ({x}, {y}, {base + height:.3f}): {height:.2f} m mast on the body surface at z {base:.3f}")
    return sensor


def mount_d455(app, stage, body: str, surface: MeshProbe, assets_root: str, height: float,
               tick_rate: float) -> StereoCameras:
    """Mounts the D455 flush against the body's front face, centred at ``height``, looking forward."""
    root = f"{body}/front_stereo"
    camera = _reference(root, assets_root + D455_ASSET, (0.0, 0.0, height))
    app.update()
    size = _local_bounds(camera)  # the D455's origin is its front face; the housing extends behind it
    lo, hi = size.GetMin(), size.GetMax()
    face = surface.frontmost((lo[1], hi[1]), (height + lo[2], height + hi[2]))
    if face is None:
        raise SystemExit(f"no front face at camera height {height} m")
    translate = (face - lo[0], 0.0, height)  # housing's back against the frontmost point it covers
    UsdGeom.Xformable(camera).GetOrderedXformOps()[0].Set(Gf.Vec3d(*translate))
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
    log(f"D455 stereo at body ({translate[0]:.3f}, 0, {translate[2]:.3f}), flush with the front face at x {face:.3f}")
    return cameras
