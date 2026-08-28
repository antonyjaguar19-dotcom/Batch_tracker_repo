"""Turn a SpaTrackV2 result.npz into a .blend an artist can just open.

Run by `run_spatracker.py`; not meant to be called by hand:

    blender --background --factory-startup --python to_blender.py -- --npz x.npz --out x.blend

What ends up in the file:

  * `SpaTrack_Cam` -- animated camera, lens and film shift keyed per frame from the
    predicted intrinsics. SpaTrackV2 works in the OpenCV convention (+X right, +Y DOWN,
    +Z into the screen); Blender's camera looks down -Z with +Y up, so every camera matrix
    is post-multiplied by diag(1,-1,-1,1). Points are plain world positions and need no
    such flip -- applying one to both would look right and put the scene in a mirrored
    world, which only shows up later when something real is matched to it.
  * `SpaTrack_Points` -- one empty per tracked point, keyed per frame, hidden on the frames
    the tracker calls occluded. Empties rather than a vertex cloud because they are visible
    in object mode, snappable, and each keeps its own name so a point can be traced back to
    a column in the 2D export.
  * the plate wired in as a camera background image, so the projection can be eyeballed
    immediately rather than taken on trust.

Blender frame numbers are the PLATE's frame numbers, not 0..T.
"""
import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

# OpenCV camera -> Blender camera (flip Y and Z)
CV_TO_BL = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
SENSOR_W = 36.0


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def object_fcurves(obj):
    """The object's f-curves, on both the old and the slotted-action API.

    Blender 4.4 moved animation into layered actions: `action.fcurves` is gone and the
    curves live under the channelbag for the object's assigned slot. Falling back the other
    way keeps this file usable on an older Blender someone still has on the farm.
    """
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            bag = strip.channelbag(ad.action_slot)
            if bag is not None:
                out.extend(bag.fcurves)
    return out


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_camera(c2w, intrs, W, H, frames):
    cam_data = bpy.data.cameras.new("SpaTrack_Cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_W
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new("SpaTrack_Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = cam

    for t, f in enumerate(frames):
        m = Matrix([[float(v) for v in row] for row in c2w[t]]) @ CV_TO_BL
        cam.matrix_world = m
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)

        fx, fy = float(intrs[t][0][0]), float(intrs[t][1][1])
        cx, cy = float(intrs[t][0][2]), float(intrs[t][1][2])
        cam_data.lens = fx / W * SENSOR_W
        # film shift is in units of the LARGER image dimension, both axes
        cam_data.shift_x = (W / 2.0 - cx) / W
        cam_data.shift_y = (cy - H / 2.0) / W
        cam_data.keyframe_insert("lens", frame=f)
        cam_data.keyframe_insert("shift_x", frame=f)
        cam_data.keyframe_insert("shift_y", frame=f)
        if abs(fy - fx) > 1e-3 * fx and t == 0:
            print(f"[warn] non-square pixels (fx={fx:.2f} fy={fy:.2f}); "
                  f"Blender has one lens value, using fx")
    return cam


def build_points(coords, visibs, frames, scale_hint):
    root = bpy.data.objects.new("SpaTrack_Points", None)
    root.empty_display_size = 0.001
    bpy.context.scene.collection.objects.link(root)

    T, N, _ = coords.shape
    size = max(scale_hint * 0.004, 1e-4)
    for n in range(N):
        e = bpy.data.objects.new(f"SPT_{n:04d}", None)
        e.empty_display_type = "PLAIN_AXES"
        e.empty_display_size = size
        bpy.context.scene.collection.objects.link(e)
        e.parent = root
        for t, f in enumerate(frames):
            e.location = (float(coords[t, n, 0]), float(coords[t, n, 1]), float(coords[t, n, 2]))
            e.keyframe_insert("location", frame=f)
            vis = bool(visibs[t, n])
            e.hide_viewport = not vis
            e.hide_render = not vis
            e.keyframe_insert("hide_viewport", frame=f)
            e.keyframe_insert("hide_render", frame=f)
        # visibility must not interpolate -- a half-hidden point is meaningless
        for fc in object_fcurves(e):
            if fc.data_path in ("hide_viewport", "hide_render"):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
    return root


def wire_plate(cam, frame_files, frames):
    """Plate as a camera background image sequence, if the frames are still on disk."""
    if not len(frame_files):
        return
    first = str(frame_files[0])
    if not os.path.isfile(first):
        print(f"[plate] not found, skipping background: {first}")
        return
    img = bpy.data.images.load(first)
    img.source = "SEQUENCE"
    cam.data.show_background_images = True
    bg = cam.data.background_images.new()
    bg.image = img
    bg.alpha = 1.0
    bg.display_depth = "BACK"
    iu = bg.image_user
    iu.frame_start = int(frames[0])
    iu.frame_offset = int(os.path.splitext(os.path.basename(first))[0].lstrip("0") or 0) - 1
    iu.frame_duration = int(frames[-1] - frames[0]) + 1
    iu.use_auto_refresh = True
    print(f"[plate] background wired from {first}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv_after_dashes())

    d = np.load(args.npz, allow_pickle=True)
    coords = d["coords"]                       # (T, N, 3) world
    visibs = np.asarray(d["visibs"]).reshape(coords.shape[0], coords.shape[1])
    if visibs.dtype != bool:
        visibs = visibs > 0.5      # a score, not a flag -- see run_spatracker.VIS_THRESH
    c2w = d["c2w"]                             # (T, 4, 4)
    intrs = d["intrinsics"]                    # (T, 3, 3) in work-size pixels
    W, H = [int(v) for v in d["work_size"]]
    plate_w, plate_h = [int(v) for v in d["plate_size"]]
    start, stride = int(d["start"]), int(d["stride"])
    T, N, _ = coords.shape
    frames = [start + t * stride for t in range(T)]

    clear_scene()
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end, sc.frame_current = frames[0], frames[-1], frames[0]
    sc.render.resolution_x, sc.render.resolution_y = plate_w, plate_h
    sc.render.resolution_percentage = 100
    sc.unit_settings.system = "METRIC"

    # scene scale, for empty sizes only: the median distance of the cloud from the origin
    scale_hint = float(np.median(np.linalg.norm(coords.reshape(-1, 3), axis=1))) or 1.0

    cam = build_camera(c2w, intrs, W, H, frames)
    build_points(coords, visibs, frames, scale_hint)
    wire_plate(cam, d["frames"], frames)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    seen = int(visibs.sum())
    print(f"[blend] {T} frames {frames[0]}-{frames[-1]}, {N} points, "
          f"{seen}/{T * N} samples visible ({100.0 * seen / (T * N):.1f}%)")
    print(f"[blend] saved {args.out}")


if __name__ == "__main__":
    main()
