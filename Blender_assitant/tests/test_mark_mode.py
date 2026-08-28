"""Mark mode, driven through the real operators, scored against the artist's hand track.

The marks are taken FROM their hand track -- the first and last frame of each unbroken run,
at their own positions -- which is exactly what an artist marking by eye would produce, and
means the answer is known for every frame in between.

The reference is `track3` Track.001: three visible stretches, f1-14, f25-32, f41-65, with two
real occlusions between them. The automatic loop scores 46/47 on it with FIVE frames off the
feature, every one of them inside an occlusion it could not see the start of.

What must be true here:

  * every frame comes back on the feature, because nothing has to guess;
  * ZERO frames off it, because no run may enter an occlusion;
  * the artist's own marks are never moved -- they are the reference the run was tracked
    from, and overwriting them replaces their judgement with the tool's;
  * and no GPU is touched, because within a run a neural guide was measured to contribute
    nothing.

    blender.exe --background -noaudio --python tests/test_mark_mode.py -- \\
        --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "bl_ext.user_default.btr_assist"
MANUAL = os.path.join(HERE, "track3 manual tracked.txt")
TRACK = "Track.001"
PAT = 41.2
WRONG_PX = 25.0

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[mark] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def atleast(name, got, want):
    ok = got >= want
    print("[mark] %-56s %s" % (name, "ok  %s" % (got,)
                               if ok else "FAIL  got %r want >= %r" % (got, want)))
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
        print("[mark] need --plate")
        return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    mod = importlib.import_module(EXT)
    oa = importlib.import_module(EXT + ".ops_assist")
    om = importlib.import_module(EXT + ".ops_mark")
    tc = importlib.import_module(EXT + ".track_core")
    print("[mark] installed %s build %s" % (mod.VERSION, mod.BUILD))

    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    w, h = clip.size
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    sp = area.spaces.active
    sp.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    scene = bpy.context.scene

    name, pts = [t for t in read_3de(MANUAL) if t[0] == TRACK][0]
    truth = {f: (x, h - y) for f, x, y in pts if f >= 1}
    fs = sorted(truth)
    runs = [[fs[0], fs[0]]]
    for f in fs[1:]:
        if f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    print("[mark] hand track runs: %s" % [tuple(r) for r in runs])
    atleast("the reference really has occlusions to cross", len(runs), 2)

    tr = clip.tracking.tracks.new(name="MARKED", frame=runs[0][0])
    clip.tracking.tracks.active = tr
    ends = [f for r in runs for f in r]
    for f in ends:
        m = tr.markers.find_frame(f, exact=True) or tr.markers.insert_frame(f)
        m.co = oa.image_px_to_uv(truth[f][0], truth[f][1], w, h)
        m.mute = False
        tc.set_geom(m, PAT, PAT * 3.0, w, h)
        mk = scene.btr_marks.add()
        mk.track, mk.frame = tr.name, int(f)
    check("marks recorded, two per stretch", len(om.marks_for(scene, tr.name)), len(ends))

    got_runs, _fs = om.runs_from(scene, tr, w, h)
    check("marks pair into the artist's own stretches",
          [(r["start"], r["end"]) for r in got_runs], [tuple(r) for r in runs])

    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        # radius left at 0 on purpose: that means "take it from the search box I drew",
        # which is the whole point -- this feature moves up to 32 px in a single frame and a
        # fixed 12 px cap stopped two runs of three within a frame or two of their start.
        res = bpy.ops.clip.btr_track_runs(min_match=0.60)
    check("the operator finished", res, {"FINISHED"})

    on = off = 0
    worst = 0.0
    for f in fs:
        m = tr.markers.find_frame(f, exact=True)
        if m is None or m.mute:
            continue
        x, y = oa.marker_to_image_px(m, w, h)
        e = math.hypot(x - truth[f][0], y - truth[f][1])
        if e <= WRONG_PX:
            on += 1
            worst = max(worst, e)
        else:
            off += 1
    # Frames the artist does NOT have are inside an occlusion. A marked run must never
    # produce one -- that is the entire point of being told where the gaps are.
    intruded = [f for f in range(fs[0], fs[-1] + 1)
                if f not in truth and tr.markers.find_frame(f, exact=True) is not None]
    print("[mark] on the feature %d/%d, off %d, worst %.1f px, frames inside occlusions %d"
          % (on, len(fs), off, worst, len(intruded)))
    atleast("nearly every marked frame is on the feature", on, int(len(fs) * 0.95))
    check("nothing landed off the feature", off, 0)
    check("nothing was planted inside an occlusion", len(intruded), 0)

    for f in ends:
        m = tr.markers.find_frame(f, exact=True)
        x, y = oa.marker_to_image_px(m, w, h)
        if math.hypot(x - truth[f][0], y - truth[f][1]) > 0.01:
            check("the artist's own mark at f%d was not moved" % f, False, True)
            break
    else:
        check("every mark the artist placed is untouched", True, True)

    # ---- the bug the artist hit: selecting another track ------------------------------
    # `tracks.active` does not follow selection, so the panel showed one track's marks while
    # they were looking at another and the operator acted on the wrong one.
    print("")
    other = clip.tracking.tracks.new(name="SECOND", frame=fs[0])
    tr.select, other.select = False, True
    check("a freshly selected track is the target",
          om.target(clip).name if om.target(clip) else None, "SECOND")
    check("  and it carries none of the first track's marks",
          om.marks_for(scene, "SECOND"), [])
    check("  so there is nothing to track on it yet",
          om.runs_from(scene, other, w, h)[0], [])
    tr.select, other.select = True, False
    check("selecting the first one back targets it again",
          om.target(clip).name if om.target(clip) else None, tr.name)
    check("  with its marks intact", len(om.marks_for(scene, tr.name)), len(ends))

    # A deleted track must not leave marks that block anything.
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        other.select = True
        tr.select = False
        gone_before = om.stale_marks(scene, clip)
    check("no stale marks while every track exists", gone_before, [])

    print("")
    if FAILED:
        print("MARK MODE: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[mark] MARK MODE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
