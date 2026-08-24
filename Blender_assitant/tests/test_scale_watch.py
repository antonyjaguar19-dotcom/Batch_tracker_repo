"""Does the pattern-box watch stop a track at the right frame, for the right reason?

The watch is arithmetic over one number per frame, so every case here has an answer known
by construction -- which is the only way this project accepts a metric (`bench/README.md`:
both defects found in 2026-08 were metrics that looked plausible and were measuring the
plate instead of the tracker).

What is being pinned down:

  * a box that does not change must never flag, at any threshold
  * a real approach to camera -- smooth, a few per cent a frame -- must survive long enough
    to be useful, and then flag on CUMULATIVE size rather than on any single step
  * a single jump flags immediately, because no feature changes size 30 % in one frame
  * the ONSET is the frame the departure started, not the frame it was noticed; the repair
    cuts back to it, so an onset that is wrong throws away good frames or keeps bad ones
  * shrinking is the same failure as growing and is not special-cased away
  * `rebase` really does clear the history, or a repaired track flags again immediately and
    the loop repairs forever

No Blender, no numpy, no plate:

    runtime\python311\python.exe tests\test_scale_watch.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# By directory, not as `btr_assist.scale_watch`: the package's __init__ imports bpy. Keeping
# the watch free of bpy is what makes it testable here at all.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "addon", "btr_assist"))

from scale_watch import ScaleWatch, size_of                           # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print("%-52s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(name)


def run(sizes, base=None, first_frame=2, **kw):
    """Feed a sequence and return (flag, frame_it_flagged_on)."""
    w = ScaleWatch(base if base is not None else sizes[0], **kw)
    for i, s in enumerate(sizes):
        flag = w.feed(first_frame + i, s)
        if flag:
            return flag, first_frame + i, w
    return None, None, w


def main():
    # --- the box that does not move ----------------------------------------------
    flag, _, _ = run([21.0] * 60, base=21.0)
    check("a steady box never flags", flag is None, str(flag))
    # Grain-level wobble is not a signal.
    wob = [21.0 * (1.0 + 0.01 * ((-1) ** i)) for i in range(60)]
    flag, _, _ = run(wob, base=21.0)
    check("1% frame-to-frame wobble never flags", flag is None, str(flag))

    # --- a real approach to camera: 3% a frame -----------------------------------
    zoom = [21.0 * (1.03 ** n) for n in range(1, 41)]
    flag, frame, _ = run(zoom[:10], base=21.0)
    check("smooth 3%/frame survives 10 frames", flag is None, str(flag))
    flag, frame, _ = run(zoom, base=21.0)
    check("smooth 3%/frame flags on cumulative size", flag is not None
          and flag["reason"] == "ratio", str(flag and flag["reason"]))
    # 1.03**n first exceeds 1.6 at n = 16, and the run starts at frame 2 with n = 1.
    check("cumulative flag lands on the right frame", frame == 17,
          "flagged f%s at %.2fx" % (frame, flag["ratio"] if flag else 0))
    # The onset is the first frame outside the 6% band -- 1.03**2 = 1.0609, n = 2, frame 3.
    # Thirteen frames before the flag: that gap IS the reason the onset is tracked, since
    # everything after it was measured by a box that was already growing.
    check("onset is where the departure started", flag and flag["onset"] == 3,
          "onset f%s" % (flag or {}).get("onset"))

    # --- one jump ----------------------------------------------------------------
    jump = [21.0, 21.0, 21.0, 27.5, 27.5]
    flag, frame, _ = run(jump, base=21.0)
    check("a 31% one-frame jump flags", flag is not None and flag["reason"] == "rate",
          str(flag and flag["reason"]))
    check("jump flags on the frame it happened", frame == 5, "f%s" % frame)
    check("jump onset is that same frame", flag and flag["onset"] == 5,
          "onset f%s" % (flag or {}).get("onset"))
    check("the flag says what happened in words",
          flag and "grew" in flag["text"], flag["text"] if flag else "")

    # --- shrinking is the same failure -------------------------------------------
    shrink = [21.0 * (0.97 ** n) for n in range(1, 41)]
    flag, _, _ = run(shrink, base=21.0)
    check("shrinking flags too", flag is not None and flag["reason"] == "ratio",
          "%.2fx" % (flag["ratio"] if flag else 0))
    check("shrink flag reports a ratio below 1", flag and flag["ratio"] < 1.0,
          "%.3f" % (flag["ratio"] if flag else 0))

    # --- thresholds are thresholds -----------------------------------------------
    flag, _, _ = run(jump, base=21.0, rate=0.0)
    check("rate 0 disables the per-frame check", flag is None, str(flag))
    flag, _, _ = run(zoom, base=21.0, ratio=6.0, rate=0.5)
    check("both limits raised: no flag", flag is None, str(flag))

    # --- rebase ------------------------------------------------------------------
    w = ScaleWatch(21.0, rate=0.12, ratio=1.6)
    for n in range(1, 20):
        f = w.feed(1 + n, 21.0 * (1.03 ** n))
        if f:
            break
    check("watch flagged before rebase", f is not None)
    w.rebase(21.0 * (1.03 ** n), frame=1 + n)
    after = [w.feed(30 + i, 21.0 * (1.03 ** n)) for i in range(10)]
    check("rebase clears the history", not any(after), str([a for a in after if a]))
    # And the new baseline is the size it was rebased to, not the original.
    nxt = w.feed(50, 21.0 * (1.03 ** n) * 1.7)
    check("rebase keeps watching from the new size", nxt is not None,
          str(nxt and nxt["reason"]))

    # --- size_of -----------------------------------------------------------------
    check("size_of is the geometric mean", abs(size_of((16.0, 25.0)) - 20.0) < 1e-9,
          "%.4f" % size_of((16.0, 25.0)))
    check("a degenerate box is not a size", size_of((0.0, 25.0)) == 0.0)
    # A box that stretches on ONE axis reads as half the growth of one that doubles both,
    # which is what keeps an edge point sliding along its edge from tripping the same
    # threshold as a box swallowing its surroundings.
    one_axis = size_of((42.0, 21.0)) / size_of((21.0, 21.0))
    both = size_of((42.0, 42.0)) / size_of((21.0, 21.0))
    check("one-axis stretch reads smaller than both", one_axis < both,
          "%.3f vs %.3f" % (one_axis, both))

    print("\n%d check(s) failed" % len(FAILURES) if FAILURES else "\nall checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
