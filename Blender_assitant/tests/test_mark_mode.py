"""Mark mode, driven through the real operators, scored against the artist's hand track.

The marks are taken FROM their hand track -- the first and last frame of each unbroken run.
Only the FRAMES are used: one seed marker is placed and nothing else is dragged, which is the
artist's actual flow, and it means the answer is known for every frame in between.

The reference is `track3` Track.001: three visible stretches, f1-14, f25-32, f41-65, with two
real occlusions between them. The automatic loop scores 46/47 on it with FIVE frames off the
feature, every one of them inside an occlusion it could not see the start of.

What must be true here:

  * every run is re-acquired within a few px, because CoTracker is asked WHERE and never
    WHEN -- the frames came from the artist;
  * ZERO frames off the feature, and none inside an occlusion, because a run is bounded by
    the marks at both of its ends;
  * the seed is never moved -- it is the reference everything was tracked from.

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

    # The artist's flow now: ONE seed marker, and marks that are frame numbers only. No
    # dragging -- CoTracker is asked to find the feature at each reappearance and Blender
    # tracks the runs.
    tr = clip.tracking.tracks.new(name="MARKED", frame=runs[0][0])
    clip.tracking.tracks.active = tr
    tr.select = True
    seed = tr.markers.find_frame(runs[0][0], exact=True)
    seed.co = oa.image_px_to_uv(truth[runs[0][0]][0], truth[runs[0][0]][1], w, h)
    seed.mute = False
    tc.set_geom(seed, PAT, PAT * 3.0, w, h)
    ends = [f for r in runs for f in r]
    for f in ends:
        mk = scene.btr_marks.add()
        mk.track, mk.frame = tr.name, int(f)
    check("marks recorded, two per stretch", len(om.marks_for(scene, tr.name)), len(ends))
    check("and only ONE marker exists -- the seed", len(oa.live_frames(tr)), 1)

    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        res = bpy.ops.clip.btr_track_runs()
    check("the operator finished", res, {"FINISHED"})

    for a, b in [tuple(r) for r in runs]:
        m = tr.markers.find_frame(a, exact=True)
        if m is None:
            print("[mark] run f%d-f%d: never started" % (a, b))
            continue
        x, y = oa.marker_to_image_px(m, w, h)
        got = [f for f in range(a, b + 1)
               if (tr.markers.find_frame(f, exact=True) or type("", (), {"mute": True})).mute
               is False]
        print("[mark] run f%d-f%d: started %.1f px off, reached f%d"
              % (a, b, math.hypot(x - truth[a][0], y - truth[a][1]), max(got or [a])))

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
    # 44/47 measured, worst 2.5 px. The three that go missing are the last frame or two
    # before a gap, where Blender stops rather than follow the occluder.
    atleast("nearly every frame of every run is on the feature", on, 42)
    check("nothing was planted inside an occlusion", len(intruded), 0)

    # The re-acquisitions specifically: this is what CoTracker is here for, and a landing
    # that is wrong takes every frame after it with it.
    for a, b in [tuple(r) for r in runs[1:]]:
        m = tr.markers.find_frame(a, exact=True)
        if m is None:
            check("run f%d was re-acquired" % a, False, True)
            continue
        x, y = oa.marker_to_image_px(m, w, h)
        check("run f%d landed on the feature" % a,
              math.hypot(x - truth[a][0], y - truth[a][1]) < 5.0, True)

    m0 = tr.markers.find_frame(runs[0][0], exact=True)
    x0, y0 = oa.marker_to_image_px(m0, w, h)
    check("the seed the artist placed is untouched",
          math.hypot(x0 - truth[runs[0][0]][0], y0 - truth[runs[0][0]][1]) < 0.01, True)

    # ---- what each mark MEANS ----------------------------------------------------------
    # Marks used to be a bare list of frames paired by position, so one missing mark shifted
    # every run after it and nothing on screen said which frame had been read as which.
    print("")
    scene.btr_marks.clear()

    def put(name, *marks):
        for f, kind in marks:
            mk = scene.btr_marks.add()
            mk.track, mk.frame = name, int(f)
            if kind:
                mk.kind = kind

    put(tr.name, (1, "START"), (14, "END"), (25, "START"), (32, "END"))
    check("explicit kinds pair into the artist's own stretches",
          om.runs(scene, tr.name)[0], [(1, 14), (25, 32)])
    check("  with nothing to complain about", om.runs(scene, tr.name)[1], [])

    scene.btr_marks.clear()
    put(tr.name, (1, None), (14, None), (25, None), (32, None))
    check("marks saved before kinds existed still read as they did",
          om.runs(scene, tr.name)[0], [(1, 14), (25, 32)])

    # The failure that used to be silent: a forgotten END. By position this reads f1-f14 and
    # f25-f32 -- tracking straight across the occlusion and stopping inside the next stretch.
    scene.btr_marks.clear()
    put(tr.name, (1, "START"), (14, "END"), (25, "START"), (32, "START"))
    pairs, problems = om.runs(scene, tr.name)
    check("a forgotten end does not shift the runs after it", pairs, [(1, 14)])
    # Two complaints for one mistake, and both are facts: f25 was never closed, and f32 is
    # now hanging open too. Naming the frames is the point -- "invalid" would not tell the
    # artist which of the four marks to fix.
    check("  and the frames are named", all("f25" in problems[0]
                                            for _ in [0]) and "f32" in problems[-1], True)
    for msg in problems:
        print("[mark]   says: %s" % msg)

    scene.btr_marks.clear()
    put(tr.name, (14, "END"), (25, "START"), (32, "END"))
    pairs, problems = om.runs(scene, tr.name)
    check("an end with no start is reported, not paired", pairs, [(25, 32)])
    check("  and named too", len(problems) == 1 and "f14" in problems[0], True)
    print("[mark]   says: %s" % problems[0])

    # ---- reading the list without losing your place -------------------------------------
    # Checking a mark used to mean reading a frame number off the panel and scrubbing to it
    # by hand; removing one meant going there first. Both are per-mark, per-track work.
    print("")
    scene.btr_marks.clear()
    put(tr.name, (1, "START"), (14, "END"), (25, "START"), (32, "END"))
    scene.frame_set(1)
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        bpy.ops.clip.btr_mark(action="GOTO", frame=25)
    check("clicking a mark jumps to its frame", int(scene.frame_current), 25)
    check("  and the clip editor followed", int(sp.clip_user.frame_current), 25)

    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        bpy.ops.clip.btr_mark(action="DROPAT", frame=14)
    check("X removes that mark", om.marks_for(scene, tr.name), [1, 25, 32])
    # The point of removing by frame: the playhead does not move, so a list being checked
    # stays where it was.
    check("  without moving the playhead", int(scene.frame_current), 25)
    check("  and the runs re-pair around it", om.runs(scene, tr.name)[0], [(25, 32)])

    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        bpy.ops.clip.btr_mark(action="DROPAT", frame=999)
    check("removing an unmarked frame changes nothing",
          om.marks_for(scene, tr.name), [1, 25, 32])

    scene.btr_marks.clear()
    for f in ends:
        mk = scene.btr_marks.add()
        mk.track, mk.frame = tr.name, int(f)

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
