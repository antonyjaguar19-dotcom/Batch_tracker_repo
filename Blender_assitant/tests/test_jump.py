"""The motion rule, against the artist's own hand track and the assist output beside it.

Two files for the same feature on SH006: what a human tracked, and what the assistant
produced. The assistant's has exactly the two faults the artist named -- "unwanted jumps" and
"unwanted slides" -- and they are the same signal at different sizes:

    f15  step 39.5 px  against a recent median of 9     the occluder arriving
    f20-23  steps 28-34 px                              sliding along it
    f24  step 70.4 px
    f27  step 116.5 px                                  snapping back by luck

None of it is visible to a correlation score: an occluder that resembles the feature keeps
Blender satisfied while the marker is dragged somewhere it could not physically have gone.
The hand track never steps more than 16 px.

This runs in Blender because `track_core` imports bpy, and against the INSTALLED extension,
so it tests the code the artist runs.

    blender.exe --background -noaudio --python tests\\test_jump.py
"""

import math
import os
import sys

import bpy

EXT = "bl_ext.user_default.btr_assist"
HERE = os.path.dirname(os.path.abspath(__file__))

FAILURES = []


def log(m):
    print("[jump] %s" % m, flush=True)


def fail(m):
    FAILURES.append(m)
    log("FAIL %s" % m)


def read_3de(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 1
    i += 2
    cnt = int(tok[i])
    i += 1
    out = []
    for _ in range(cnt):
        out.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2])))
        i += 3
    return sorted(out)


def main():
    bpy.ops.preferences.addon_enable(module=EXT)
    tc = sys.modules["%s.track_core" % EXT]

    man = read_3de(os.path.join(HERE, "reacquretracke_manual.txt"))
    ast = read_3de(os.path.join(HERE, "reacquretracke_assist tracked.txt"))

    # 1. The hand track must survive untouched. A rule that cuts an artist's own work is
    #    worse than no rule -- and this one trips at f5 without a minimum sample count,
    #    because the track starts at 3.7 px/frame and accelerates to 9.4.
    f, step, med = tc.first_jump(man)
    if f is not None:
        fail("cut the artist's own hand track at f%d (%.1f px vs %.1f)" % (f, step, med))
    else:
        log("%-44s ok" % "hand track is never cut")

    # 2. The assist output must be cut at f15 -- the frame the occluder arrives, and the
    #    first frame the artist's track does not have.
    f, step, med = tc.first_jump(ast)
    if f != 15:
        fail("assist output cut at f%s, expected f15" % f)
    else:
        log("%-44s ok (%.0f px vs its own %.0f)" % ("assist output cut at the occluder", step, med))

    # 3. Steps across a GAP mean nothing -- a resume is a new head, and the distance from
    #    the last frame before an occlusion is not motion. Feed a track with a hole in it
    #    and a large apparent step across the hole.
    a = [(f, 100.0 + 5.0 * f, 100.0) for f in range(1, 12)]
    b = [(f, 900.0 + 5.0 * f, 100.0) for f in range(30, 40)]
    f, _s, _m = tc.first_jump(a + b)
    if f is not None:
        fail("cut across a gap at f%d -- a resume is a new head, not a jump" % f)
    else:
        log("%-44s ok" % "a gap is not a jump")

    # 4. A plate that moves fast EVERYWHERE is not jumping. Judged against the track's own
    #    median, 40 px a frame is unremarkable if that is what it has been doing.
    fast = [(f, 40.0 * f, 0.0) for f in range(1, 30)]
    f, _s, _m = tc.first_jump(fast)
    if f is not None:
        fail("cut a steady fast track at f%d" % f)
    else:
        log("%-44s ok" % "steady 40 px/frame is not a jump")

    # 5. ...but a step out of character with that fast motion still is.
    fast2 = [(f, 40.0 * f, 0.0) for f in range(1, 15)]
    fast2 += [(15, 40.0 * 14 + 400.0, 0.0)]
    f, _s, _m = tc.first_jump(fast2)
    if f != 15:
        fail("missed a 400 px step on a 40 px/frame track (got %s)" % f)
    else:
        log("%-44s ok" % "an out-of-character step on a fast track is")

    log("JUMP: %s" % ("FAIL" if FAILURES else "PASS"))
    sys.exit(1 if FAILURES else 0)


main()
