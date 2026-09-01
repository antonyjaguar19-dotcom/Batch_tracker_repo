"""Mark the first frame and the last frame, and demand every frame in between.

The artist's report: "the first frame i set is 1 and the exit frame is the last frame. but
the assistant is tracking the pattern only for 49 frames." Forty-nine was not the plate and
not the feature -- it was `MAX_ROUNDS = 24`, a constant in `ops_mark`. Blender and CoTracker
were advancing about two frames per turn and ran out of turns. The artist's marks are the
specification; a constant in this file is not allowed to overrule them.

`Track.004` of their own reference is the case: hand-tracked f1-f312, 312 points, no gaps --
a feature that really is visible for the whole shot, which is what "f1 to the last frame"
claims. Their track is the truth for every frame, so both coverage AND position are
measurable here, and coverage alone is not enough to call it working.

Measured after the cap was removed, same seed and box, both engines:

    Blender's own settings   312/312   129 s   median 4.24 px  p90 14.6  max 17.9  (208 filled)
    this addon's config      312/312    10 s   median 1.28 px  p90  6.1  max  9.0  (0 filled)

Both COVER the range. They are not the same track. Blender's KEYFRAME matching compares
against the seed frame forever, so it gives out as the feature turns and the fill takes over
-- and the fill walks with a rolling template, which drifts. The addon's PREV_FRAME
configuration tracked all 312 frames in one pass and filled nothing.

That is why this test asserts a position bound per engine and not just a frame count: a run
that covers the range by rebuilding most of it is a different, worse answer than one that
tracks it, and a frame count cannot tell them apart.

    blender.exe --background -noaudio --python tests/test_whole_shot.py -- \\
        --plate D:/Jefrin/IN/SH006.mp4
"""

import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "bl_ext.user_default.btr_assist"
MANUAL = os.path.join(HERE, "track3 manual tracked.txt")
TRACK = "Track.004"
PAT = 41.2
WRONG_PX = 25.0

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[ws] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def atmost(name, got, want):
    ok = got <= want
    print("[ws] %-56s %s" % (name, "ok  %.2f" % got
                             if ok else "FAIL  got %.2f want <= %.2f" % (got, want)))
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
        print("[ws] need --plate")
        return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    oa = importlib.import_module(EXT + ".ops_assist")
    om = importlib.import_module(EXT + ".ops_mark")
    tc = importlib.import_module(EXT + ".track_core")
    prefs = importlib.import_module(EXT + ".prefs")

    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    w, h = clip.size
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    sp = area.spaces.active
    sp.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    scene = bpy.context.scene

    _, pts = [t for t in read_3de(MANUAL) if t[0] == TRACK][0]
    truth = {f: (x, h - y) for f, x, y in pts if f >= 1}
    fs = sorted(truth)
    a, b = fs[0], fs[-1]
    check("the reference really covers the whole shot, ungapped",
          len(fs), b - a + 1)

    p = prefs.get(bpy.context)
    if p is not None and not p.assist_root:
        p.assist_root = os.path.abspath(os.path.join(HERE, ".."))

    # The round guard must scale with the range. A flat 24 is what returned 49 frames.
    check("the round guard is not a constant the marks can outgrow",
          om.max_rounds(a, b) >= b - a, True)

    for blender_tracking, bar in ((True, 20.0), (False, 10.0)):
        label = "Blender's own settings" if blender_tracking else "the addon's config"
        for t in list(clip.tracking.tracks):
            t.select = True
        with bpy.context.temp_override(window=win, area=area, region=region,
                                       space_data=sp, edit_movieclip=clip):
            try:
                bpy.ops.clip.delete_track()
            except RuntimeError:
                pass
        scene.btr_marks.clear()

        tr = clip.tracking.tracks.new(name="WHOLE", frame=a)
        clip.tracking.tracks.active = tr
        tr.select = True
        m = tr.markers[0]
        m.co = oa.image_px_to_uv(truth[a][0], truth[a][1], w, h)
        m.mute = False
        tc.set_geom(m, PAT, PAT * 3.0, w, h)
        for f, kind in ((a, "START"), (b, "END")):
            mk = scene.btr_marks.add()
            mk.track, mk.frame, mk.kind = tr.name, int(f), kind

        with bpy.context.temp_override(window=win, area=area, region=region,
                                       space_data=sp, edit_movieclip=clip):
            bpy.ops.clip.btr_track_runs(blender_tracking=blender_tracking)

        live = set(oa.live_frames(tr))
        missing = [f for f in range(a, b + 1) if f not in live]
        errs = []
        for f in sorted(live):
            if f not in truth:
                continue
            x, y = oa.marker_to_image_px(tr.markers.find_frame(f, exact=True), w, h)
            errs.append(math.hypot(x - truth[f][0], y - truth[f][1]))
        errs.sort()
        med = errs[len(errs) // 2] if errs else 999.0
        p90 = errs[min(len(errs) - 1, int(0.9 * len(errs)))] if errs else 999.0
        print("[ws] %-24s %d/%d frames, median %.2f px, p90 %.2f, max %.2f"
              % (label, len(live), b - a + 1, med, p90, errs[-1] if errs else -1))
        check("%s: every frame the artist marked came back" % label, missing, [])
        check("  none of them off the feature" if not errs else
              "  none of them off the feature",
              len([e for e in errs if e > WRONG_PX]), 0)
        # Per engine, because covering the range by REBUILDING it is a different answer from
        # tracking it, and a frame count cannot tell those apart.
        atmost("  and the median holds", med, bar)

    print("")
    if FAILED:
        print("WHOLE SHOT: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[ws] WHOLE SHOT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
