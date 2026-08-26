"""A slide built into the artist's own hand track, repaired through the real operator.

The answer is known by construction: a copy of `Track.003` with a growing offset added over
f100-f150, reaching 77 px. Every frame of the original is where the artist put it, so the
repair either brings the copy back to it or does not.

The half that matters just as much is the CONTROL. The artist's untouched hand track goes
through the same operator and must come out with nothing flagged, nothing moved and nothing
cut -- a repair that improves a broken track while quietly rewriting a correct one is worse
than no repair.

    blender.exe --background -noaudio --python tests/test_fix_slides.py -- \\
        --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "bl_ext.user_default.btr_assist"
MANUAL = os.path.join(HERE, "track2 manual tracked.txt")
TRACK = "Track.003"
LO, HI, MAXOFF = 100, 150, 70.0
PAT = 41.2

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[fix] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def atleast(name, got, want):
    ok = got >= want
    print("[fix] %-56s %s" % (name, "ok  %s" % (got,) if ok
                              else "FAIL  got %r want >= %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def read_3de(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 0
    n = int(tok[i]); i += 1
    out = []
    for _ in range(n):
        name = tok[i]; i += 2
        cnt = int(tok[i]); i += 1
        pts = []
        for _ in range(cnt):
            pts.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2])))
            i += 3
        out.append((name, pts))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""
    if not plate or not os.path.exists(plate):
        print("[fix] need --plate")
        return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    mod = importlib.import_module(EXT)
    oa = importlib.import_module(EXT + ".ops_assist")
    print("[fix] installed %s build %s" % (mod.VERSION, mod.BUILD))

    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    w, h = clip.size
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    sp = area.spaces.active
    sp.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")

    name, pts = [t for t in read_3de(MANUAL) if t[0] == TRACK][0]
    truth = {f: (x, h - y) for f, x, y in pts}          # 3DE y-UP -> image y-DOWN
    fs = sorted(truth)

    def make(track_name, slide):
        tr = clip.tracking.tracks.new(name=track_name, frame=fs[0])
        for f in fs:
            x, y = truth[f]
            if slide and LO <= f <= HI:
                k = (f - LO) / float(HI - LO)
                x += MAXOFF * k
                y += 0.45 * MAXOFF * k
            m = tr.markers.find_frame(f, exact=True)
            if m is None:
                m = tr.markers.insert_frame(f)
            m.co = oa.image_px_to_uv(x, y, w, h)
            m.mute = False
            oa.track_core.set_geom(m, PAT, PAT * 3.0, w, h) if hasattr(oa, "track_core") \
                else None
        return tr

    from bl_ext.user_default.btr_assist import track_core             # noqa: E402
    def geom(tr):
        for m in tr.markers:
            track_core.set_geom(m, PAT, PAT * 3.0, w, h)

    slid = make("SLID", True)
    ctrl = make("CONTROL", False)
    geom(slid)
    geom(ctrl)

    def err_of(tr):
        n_ok = 0
        for f in fs:
            m = tr.markers.find_frame(f, exact=True)
            if m is None or m.mute:
                continue
            x, y = oa.marker_to_image_px(m, w, h)
            if math.hypot(x - truth[f][0], y - truth[f][1]) <= 25.0:
                n_ok += 1
        return n_ok

    before_slid, before_ctrl = err_of(slid), err_of(ctrl)
    print("[fix] built a %.0f px slide over f%d-f%d" % (math.hypot(MAXOFF, .45 * MAXOFF),
                                                        LO, HI))
    print("[fix] on the feature before: SLID %d, CONTROL %d (of %d)"
          % (before_slid, before_ctrl, len(fs)))
    atleast("the slide really broke the copy", len(fs) - before_slid, 20)
    check("the control starts perfect", before_ctrl, len(fs))

    for tr in clip.tracking.tracks:
        tr.select = tr.name == "SLID"
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        res = bpy.ops.clip.btr_fix_slides(selected_only=True, min_match=0.66)
    check("the operator finished on the slid copy", res, {"FINISHED"})
    after_slid = err_of(slid)
    print("[fix] on the feature after : SLID %d of %d" % (after_slid, len(fs)))
    atleast("the slide is repaired", after_slid, before_slid + 20)
    atleast("  and nearly every frame is back", after_slid, int(len(fs) * 0.95))

    for tr in clip.tracking.tracks:
        tr.select = tr.name == "CONTROL"
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        res = bpy.ops.clip.btr_fix_slides(selected_only=True, min_match=0.66)
    check("the operator finished on the control", res, {"FINISHED"})
    after_ctrl = err_of(ctrl)
    print("[fix] on the feature after : CONTROL %d of %d" % (after_ctrl, len(fs)))
    check("a CORRECT track is left exactly as it was", after_ctrl, before_ctrl)

    print("")
    if FAILED:
        print("FIX SLIDES: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[fix] FIX SLIDES: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
