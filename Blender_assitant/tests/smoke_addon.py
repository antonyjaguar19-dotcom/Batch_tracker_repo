"""M1 gate: the addon registers, and a real 3DE file survives a round trip through it.

Run under Blender, which is the only interpreter that has bpy:

    blender.exe --background --factory-startup -noaudio --python tests\smoke_addon.py -- \
        --plate <frames folder> --tracks <a 3DE .txt> --out <scratch .txt>

A round trip is the right gate because both of this format's traps are silent. A dropped
half-pixel centre shifts everything by 0.5 px and still looks perfect in the viewport; a
`markers.find_frame()` without `exact=True` fills every gap with invented samples and
produces a file that parses cleanly and is wrong. Both show up here as a numeric diff.
"""

import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "addon"))


def log(m):
    print("[smoke] %s" % m, flush=True)


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
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    area.spaces.active.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    return win, area, region


def compare(a_path, b_path, three_de):
    """Same tracks, same frames, same positions? Returns (n_samples, worst_px, problems)."""
    a = {n: dict((f, (x, y)) for f, x, y in p) for n, p in three_de.read_3de(a_path)}
    b = {n: dict((f, (x, y)) for f, x, y in p) for n, p in three_de.read_3de(b_path)}
    problems = []
    if set(a) != set(b):
        problems.append("track names differ: %d in, %d out, %d common"
                        % (len(a), len(b), len(set(a) & set(b))))
    worst, n = 0.0, 0
    for name in set(a) & set(b):
        fa, fb = a[name], b[name]
        if set(fa) != set(fb):
            problems.append("%s: %d frames in, %d out" % (name, len(fa), len(fb)))
        for f in set(fa) & set(fb):
            dx = fa[f][0] - fb[f][0]
            dy = fa[f][1] - fb[f][1]
            worst = max(worst, (dx * dx + dy * dy) ** 0.5)
            n += 1
    return n, worst, problems


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = {}
    for i in range(0, len(argv) - 1, 2):
        args[argv[i].lstrip("-")] = argv[i + 1]
    plate, tracks_in = args.get("plate"), args.get("tracks")
    out_path = args.get("out", os.path.join(HERE, "..", "logs", "smoke_out.txt"))
    report_path = args.get("report", os.path.join(HERE, "..", "logs", "m1_smoke.json"))
    if not plate or not tracks_in:
        raise SystemExit("--plate and --tracks are required")

    results = {"blender": bpy.app.version_string}

    import btr_assist
    from btr_assist import three_de
    btr_assist.register()
    log("registered")
    results["register"] = {"PASS": True,
                           "classes": sum(len(m.CLASSES) for m in btr_assist.MODULES)}

    clip = load_clip(plate)
    win, area, region = clip_editor(clip)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, clip.frame_duration
    space = area.spaces.active
    results["clip"] = {"size": list(clip.size), "frames": clip.frame_duration}
    log("clip %dx%d, %d frames" % (clip.size[0], clip.size[1], clip.frame_duration))

    ov = dict(window=win, area=area, region=region, space_data=space,
              edit_movieclip=clip, scene=scene)

    with bpy.context.temp_override(**ov):
        r = bpy.ops.clip.btr_import_3de(filepath=os.path.abspath(tracks_in),
                                        clear_existing=True)
    n_tracks = len(three_de.active_tracks(clip))
    log("import -> %s, %d tracks on clip" % (list(r), n_tracks))
    results["import"] = {"result": list(r), "tracks": n_tracks, "PASS": n_tracks > 0}

    with bpy.context.temp_override(**ov):
        r = bpy.ops.clip.btr_check_3de()
    results["check"] = {"result": list(r), "PASS": "FINISHED" in r}

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with bpy.context.temp_override(**ov):
        r = bpy.ops.clip.btr_export_3de(filepath=out_path, skip_muted=True)
    log("export -> %s" % list(r))

    n, worst, problems = compare(os.path.abspath(tracks_in), out_path, three_de)
    results["roundtrip"] = {
        "samples_compared": n,
        "worst_px": round(worst, 9),
        "problems": problems,
        # 3DE ASCII is written at 12 decimal places, so a clean round trip is exact to
        # far below any tolerance that matters. Anything above 1e-6 px is a real bug --
        # most likely a lost half-pixel centre (which would read 0.5) or a flipped axis.
        "PASS": bool(n) and worst < 1e-6 and not problems,
    }
    log("round trip: %d samples, worst %.9f px, %d problems"
        % (n, worst, len(problems)))
    for p in problems[:5]:
        log("  ! %s" % p)

    btr_assist.unregister()
    log("unregistered")
    results["unregister"] = {"PASS": True}

    ok = all(v.get("PASS", True) for v in results.values() if isinstance(v, dict))
    results["ALL_PASS"] = ok
    report_path = os.path.abspath(report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log("=" * 60)
    log("M1 SMOKE: %s  (%s)" % ("PASS" if ok else "FAIL", report_path))
    log("=" * 60)


main()
