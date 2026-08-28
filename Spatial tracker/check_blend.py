"""Prove the .blend is right, by projecting it back through Blender's own camera.

The risky step in the export is the OpenCV -> Blender camera convention. Getting it wrong
produces a file that looks plausible (camera moves, points move) and is mirrored, and the
error only surfaces much later when something real is matched to it. So instead of trusting
the matrix algebra, ask Blender where its camera thinks each point lands and compare that
against the 2D track the model itself produced.

    blender --background --factory-startup --python-exit-code 1 --python check_blend.py -- \
        --blend out/SH006_f1-48__spatrack.blend --npz out/SH006_f1-48__spatrack.npz
"""
import argparse
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--npz", required=True)
    args = ap.parse_args(argv_after_dashes())

    bpy.ops.wm.open_mainfile(filepath=args.blend)
    d = np.load(args.npz, allow_pickle=True)
    t2 = d["track2d"]
    vis = np.asarray(d["visibs"]).reshape(t2.shape[0], t2.shape[1])
    if vis.dtype != bool:
        vis = vis > 0.5
    W, H = [int(v) for v in d["work_size"]]
    start, stride = int(d["start"]), int(d["stride"])
    T, N, _ = t2.shape

    sc = bpy.context.scene
    cam = sc.camera
    pts = [bpy.data.objects[f"SPT_{n:04d}"] for n in range(N)]

    errs = []
    for t in range(T):
        sc.frame_set(start + t * stride)
        dg = bpy.context.evaluated_depsgraph_get()
        for n in range(N):
            if not vis[t, n]:
                continue
            co = pts[n].evaluated_get(dg).matrix_world.translation
            u, v, z = world_to_camera_view(sc, cam, co)
            if z <= 0:
                continue
            # world_to_camera_view is normalised 0..1 from the BOTTOM-left
            px, py = u * W, (1.0 - v) * H
            errs.append(((px - t2[t, n, 0]) ** 2 + (py - t2[t, n, 1]) ** 2) ** 0.5)

    errs = np.asarray(errs)
    print(f"[check] {len(errs)} samples projected through the Blender camera")
    print(f"[check] vs the model's own 2D track: median {np.median(errs):.2f} px, "
          f"p90 {np.percentile(errs, 90):.2f} px, at {W}x{H}")
    print("[check] a few px means the .blend camera and the model agree; "
          "hundreds means the convention is wrong")


if __name__ == "__main__":
    main()
