# SPDX-License-Identifier: Apache-2.0
"""Ground-truth occupancy maps per floor (and stair surface points) from the composed office scene.

Run headless with Isaac Sim's Python:

    ~/isaacsim/python.sh scripts/extract_ground_truth.py --out ../../../scenes/gt_maps

The stage is composed exactly as the simulator does (scene preset + office_floor1 layer, stairwells
opened). One downward PhysX ray per map cell (raycast_all, from above the top floor) collects every
collider surface in that column: the floor support. Obstacles come from the VISUAL meshes (what
the cameras and the lidar see): every visible triangle is sampled every --sample m and each sample
inside a floor's body band (z_f + band) marks its cell. PhysX queries alone miss geometry: a ray
meets a wall only at its top and bottom faces, and box overlaps missed some walls (e.g. the
lobby's SM_Wall_3m_door21 at x -9, whose collider is enabled) and every collider-less mesh
(elevator doors); both still run and are or-ed in. For each floor height z_f a cell is

  free      a surface within +-tol of z_f, and nothing (visual or collider) in the body band
  occupied  everything else: walls, furniture, stairs, stairwells, drops, outside the building

There is no unknown: Nav2's planner here allows unknown cells, which must never include a drop.
The grid matches the office Nav2 map (dd100_office_opened_navigation.yaml) cell for cell.

Outputs in --out: L<z>.pgm/.yaml per floor (trinary, Nav2 map_server format), columns.npz (every
hit: cell index, z, stair flag) and stairs.npz (hit points on SM_Stairs* colliders, for centre
lines), summary.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "isaac"))

FLOORS = (-3.3, 0.0, 3.3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--origin", type=float, nargs=2, default=[-28.075, -34.675])
    p.add_argument("--size", type=int, nargs=2, default=[871, 2043], metavar=("COLS", "ROWS"))
    p.add_argument("--floors", type=float, nargs="+", default=list(FLOORS))
    p.add_argument("--tol", type=float, default=0.1, help="support within +-tol of the floor height")
    p.add_argument("--band", type=float, nargs=2, default=[0.14, 0.94], help="body band above the floor")
    p.add_argument("--top", type=float, default=9.0, help="rays start at this height")
    p.add_argument("--window", type=float, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1"),
                   help="only this xy window (testing)")
    p.add_argument("--sample", type=float, default=0.03, help="visual triangle sample spacing [m]")
    return p.parse_args()


def _visual_occupancy(stage, floors, band, origin, res, shape, step):
    """Floor height -> cells where a visible mesh passes through the body band above it."""
    import math

    import numpy as np
    from pxr import Usd, UsdGeom

    rows, cols = shape
    occ = {fz: np.zeros(rows * cols, bool) for fz in floors}
    lo = min(floors) + band[0]
    hi = max(floors) + band[1]
    n_tri = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or \
                img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts, counts, idx = (mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(),
                                mesh.GetFaceVertexIndicesAttr().Get())
            if not pts or not counts:
                continue
            p = np.asarray(pts, dtype=np.float64)
            p = np.c_[p, np.ones(len(p))] @ m
            p = p[:, :3]
            counts, idx = np.asarray(counts), np.asarray(idx)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            tris = [np.stack([idx[starts[counts >= k]], idx[starts[counts >= k] + k - 2],
                              idx[starts[counts >= k] + k - 1]], axis=1)
                    for k in range(3, int(counts.max()) + 1)]
            tri = p[np.concatenate(tris)]                       # (T, 3, 3) fan triangles
        else:                                                   # cubes etc.: their bounding box
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]) \
                .ComputeWorldBound(prim).ComputeAlignedRange()
            (x0, y0, z0), (x1, y1, z1) = box.GetMin(), box.GetMax()
            c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            tri = np.array([[[*c[0], z], [*c[1], z], [*c[2], z]] for z in (max(z0, lo), min(z1, hi))]
                           + [[[*c[0], z], [*c[2], z], [*c[3], z]] for z in (max(z0, lo), min(z1, hi))])
        zmin, zmax = tri[..., 2].min(axis=1), tri[..., 2].max(axis=1)
        tri = tri[(zmax >= lo) & (zmin <= hi)]
        if not len(tri):
            continue
        n_tri += len(tri)
        edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
        nsub = np.clip(np.ceil(edge / step).astype(int), 1, 400)
        for n in np.unique(nsub):
            t = tri[nsub == n]
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
            keep = (i + j) <= n
            u, v = (i[keep] / n), (j[keep] / n)                 # barycentric grid
            w = np.stack([1 - u - v, u, v], axis=1)             # (S, 3)
            s = np.einsum('sk,tkd->tsd', w, t).reshape(-1, 3)
            r = np.floor((s[:, 1] - origin[1]) / res).astype(np.int64)
            c = np.floor((s[:, 0] - origin[0]) / res).astype(np.int64)
            ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            for fz in floors:
                band_ok = ok & (s[:, 2] > fz + band[0]) & (s[:, 2] <= fz + band[1])
                occ[fz][r[band_ok] * cols + c[band_ok]] = True
    print(f"visual occupancy from {n_tri} triangles near the floors", flush=True)
    return occ


def _collide_with_visuals(stage) -> list:
    """Temporary triangle colliders on visible meshes that have none; their paths."""
    from pxr import Usd, UsdGeom, UsdPhysics

    while True:  # instance proxies are read-only: un-instance the assets first
        inst = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World")) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    added = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or \
                img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        added.append(str(prim.GetPath()))
    return added


def main() -> None:
    args = parse_args()
    from legged_sim.app import start_app

    class AppArgs:
        headless, engine = True, "physx"

    app = start_app(AppArgs)
    import carb
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.storage.native import get_assets_root_path
    from omni.physx import get_physx_scene_query_interface
    from legged_sim.scene import load_environment

    stage = omni.usd.get_context().get_stage()
    load_environment(app, stage, "office", None, get_assets_root_path(), 1.0, True)
    from isaacsim.core.experimental.utils.stage import define_prim
    if not any(p.GetTypeName() == "PhysicsScene" for p in stage.Traverse()):
        define_prim("/World/PhysicsScene", "PhysicsScene")
    added = _collide_with_visuals(stage)
    print(f"temporary colliders on {len(added)} visible meshes without one: {added}", flush=True)
    omni.timeline.get_timeline_interface().play()
    for _ in range(20):
        app.update()

    res, (ox, oy), (cols, rows) = args.resolution, args.origin, args.size
    xs = ox + (np.arange(cols) + 0.5) * res
    ys = oy + (np.arange(rows) + 0.5) * res
    c0, c1, r0, r1 = 0, cols, 0, rows
    if args.window:
        x0, x1, y0, y1 = args.window
        c0, c1 = int(np.searchsorted(xs, x0)), int(np.searchsorted(xs, x1))
        r0, r1 = int(np.searchsorted(ys, y0)), int(np.searchsorted(ys, y1))

    query = get_physx_scene_query_interface()

    def _any(hit):
        return False          # first overlap is enough
    cell, zs, stair = [], [], []
    hits = []

    def report(hit):
        hits.append((hit.position[2], hit.collision))
        return True

    t0 = time.time()
    n = 0
    for r in range(r0, r1):
        for c in range(c0, c1):
            hits.clear()
            query.raycast_all((float(xs[c]), float(ys[r]), args.top), (0.0, 0.0, -1.0), args.top + 6.0, report)
            for z, path in hits:
                cell.append(r * cols + c)
                zs.append(z)
                stair.append("SM_Stairs" in path and "Carpet" not in path and "Wall" not in path)
            n += 1
        if (r - r0) % 100 == 0:
            print(f"row {r - r0}/{r1 - r0}  {n} rays  {time.time() - t0:.0f} s", flush=True)
    cell, zs, stair = np.asarray(cell, np.int64), np.asarray(zs, np.float32), np.asarray(stair, bool)
    print(f"{n} rays, {len(zs)} hits in {time.time() - t0:.0f} s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "columns.npz"), cell=cell, z=zs, stair=stair,
                        origin=np.array([ox, oy]), resolution=res, shape=np.array([rows, cols]))
    rr, cc = np.divmod(cell[stair], cols)
    np.savez_compressed(os.path.join(args.out, "stairs.npz"),
                        xyz=np.stack([xs[cc], ys[rr], zs[stair]], axis=1).astype(np.float32))

    t1 = time.time()
    visual = _visual_occupancy(stage, args.floors, args.band, (ox, oy), res, (rows, cols), args.sample)
    print(f"visual occupancy in {time.time() - t1:.0f} s", flush=True)
    summary = {"rays": n, "hits": int(len(zs)), "window": args.window, "floors": {}}
    for fz in args.floors:
        dz = zs - fz
        support = np.zeros(rows * cols, bool)
        support[cell[np.abs(dz) <= args.tol]] = True
        body = np.zeros(rows * cols, bool)
        body[cell[(dz > args.band[0]) & (dz <= args.band[1])]] = True
        t1 = time.time()
        cand = np.where(support & ~body)[0]
        half = carb.Float3(0.5 * res, 0.5 * res, 0.5 * (args.band[1] - args.band[0]))
        zc = fz + 0.5 * (args.band[0] + args.band[1])
        rot = carb.Float4(0.0, 0.0, 0.0, 1.0)
        for k in cand:
            r, c = divmod(int(k), cols)
            if query.overlap_box(half, carb.Float3(float(xs[c]), float(ys[r]), zc), rot, _any, True):
                body[k] = True
        print(f"  {len(cand)} overlap boxes in {time.time() - t1:.0f} s", flush=True)
        body |= visual[fz]
        free = (support & ~body).reshape(rows, cols)
        name = "L0" if abs(fz) < 1e-6 else f"L{fz:g}"
        img = np.where(free, 254, 0).astype(np.uint8)[::-1]      # PGM row 0 = max y
        with open(os.path.join(args.out, f"{name}.pgm"), "wb") as f:
            f.write(f"P5\n{cols} {rows}\n255\n".encode())
            f.write(img.tobytes())
        with open(os.path.join(args.out, f"{name}.yaml"), "w") as f:
            f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {res}\norigin: [{ox}, {oy}, 0.0]\n"
                    "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
        summary["floors"][name] = {"z": fz, "free_m2": float(free.sum() * res * res),
                                   "support_m2": float(support.sum() * res * res)}
        print(name, summary["floors"][name], flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    omni.timeline.get_timeline_interface().stop()
    app.close()


if __name__ == "__main__":
    main()
