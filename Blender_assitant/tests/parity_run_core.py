"""Drive `addon/btr_assist/track_core.py` headless, writing bl_track.py's output format.

Exists so `test_track_core_parity.py` can run both implementations over identical input and
diff the results. Deliberately does the minimum: seed, forward, backward, write. No
replants -- the parity run disables them on both sides, because replant is the one stage
track_core has not taken over yet.

    blender.exe --background --factory-startup -noaudio --python parity_run_core.py -- \
        --seeds <seeds.json> --out <bl.json> [--leash 20]
"""

import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "addon", "btr_assist"))

import track_core  # noqa: E402  (path must be set first)


def log(m):
    print("[parity-core] %s" % m, flush=True)


def load_clip(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in
                       (".exr", ".dpx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        if not names:
            raise RuntimeError("no frames in " + path)
        return bpy.data.movieclips.load(os.path.join(path, names[0]))
    return bpy.data.movieclips.load(path)


def clip_editor(clip):
    win = bpy.context.window_manager.windows[0]
    area = None
    for a in win.screen.areas:
        if a.type == "CLIP_EDITOR":
            area = a
            break
    if area is None:
        area = max(win.screen.areas, key=lambda x: x.width * x.height)
        area.type = "CLIP_EDITOR"
    area.spaces.active.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    return win, area, region


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {}
    for i in range(0, len(argv) - 1, 2):
        args[argv[i].lstrip("-")] = argv[i + 1]

    spec = json.load(open(args["seeds"], encoding="utf-8"))
    seeds = spec["seeds"]
    w, h = int(spec["width"]), int(spec["height"])

    clip = load_clip(spec["plate"] if os.path.exists(spec["plate"]) else args["clip"])
    n_frames = int(spec.get("frames") or clip.frame_duration)
    if (clip.size[0], clip.size[1]) != (w, h):
        # Hard stop, not a warning: with a different size the normalised coords still
        # "work" and every tracker starts on the wrong feature.
        raise RuntimeError("clip is %dx%d but seeds are for %dx%d"
                           % (clip.size[0], clip.size[1], w, h))

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, n_frames
    win, area, region = clip_editor(clip)
    ctx = (win, area, region, clip, scene)

    opts = track_core.Opts(leash=float(args.get("leash", 20.0)))
    track_core.apply_settings(clip, opts)
    records = track_core.seed_tracks(clip, seeds, opts)

    err = track_core.seed_roundtrip_error(records, seeds)
    log("seed round-trip: max %.6f px  %s" % (err, "PASS" if err < 0.01 else "FAIL"))
    if err >= 0.01:
        raise RuntimeError("seed round-trip FAILED -- every number after this is void")

    guide = {s["id"]: s.get("guide") or {} for s in seeds}
    stats = {}
    for _ in track_core.track_job(ctx, records, n_frames, guide, opts, stats=stats):
        pass
    log("forward: %d calls, %d entered, %d deaths, %d clamps"
        % (stats["calls"], stats["entered"], stats["deaths"], stats["clamped"]))

    calls = track_core.track_backward_pass(ctx, records, opts)
    log("backward: %d calls (sequence mode)" % calls)

    out = {"width": w, "height": h, "frames": n_frames, "replants": 0, "tracks": []}
    for r in records:
        pts = [[int(m.frame), float(m.co[0]), float(m.co[1])]
               for m in r["t"].markers
               if not m.mute and 1 <= m.frame <= n_frames]
        pts.sort()
        out["tracks"].append({"id": r["id"], "kind": r["kind"], "pts": pts})
    with open(args["out"], "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    lens = [len(t["pts"]) for t in out["tracks"]]
    log("wrote %s: %d tracks, mean length %.1f/%d"
        % (os.path.basename(args["out"]), len(lens),
           (sum(lens) / len(lens)) if lens else 0.0, n_frames))
    log("DONE")


main()
