"""Seed a tracker at the artist's own points and score every frame against their track.

The question this answers is which MODEL to build on -- so it measures the model raw, seeded
exactly where the artist seeded, over exactly the frames they tracked. No pinning, no
re-acquire loop, no cuts: those are the same for either engine and would hide which of them
is actually holding the feature.

Two numbers, because they are different abilities and a tool needs both:

  * NATURAL -- frames inside the artist's first unbroken run. How close it stays while
    nothing is in the way.
  * AFTER A GAP -- frames in every later run. The artist stopped tracking because the
    feature went behind something; these are the frames where it came back, and whether the
    tracker is on it there is the re-acquisition question.

    runtime\\python311\\python.exe tests\\eval_engines_vs_manual.py ^
        --manual tests\\track3 manual tracked.txt --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ASSIST, os.path.join(ASSIST, "sidecar"), HERE]

import leash                                                          # noqa: E402
import repo                                                           # noqa: E402
from eval_cotracker_direct import read_3de                            # noqa: E402


def log(m):
    print("[vs] %s" % m, flush=True)


def runs_of(fs):
    out = [[fs[0], fs[0]]]
    for f in fs[1:]:
        if f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def score(pred, truth, first_run, wrong_px):
    """(natural, after-gap) each as (on_feature, total, p50_err)."""
    nat, aft = [], []
    for f, t in truth.items():
        p = pred.get(f)
        if p is None:
            (nat if first_run[0] <= f <= first_run[1] else aft).append(None)
            continue
        e = math.hypot(p[0] - t[0], p[1] - t[1])
        (nat if first_run[0] <= f <= first_run[1] else aft).append(e)

    def summ(v):
        got = sorted(e for e in v if e is not None and e <= wrong_px)
        return (len(got), len(v), got[len(got) // 2] if got else float("nan"))

    return summ(nat), summ(aft)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--wrong-px", type=float, default=25.0)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--chain", type=int, default=120)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap every track's span, for a like-for-like against a DINO model "
                         "that was only trained on the first N frames of the plate")
    ap.add_argument("--only", default="", help="one track name")
    a = ap.parse_args()

    os.environ.setdefault("BTR_COTRACKER_MAX_FRAMES", "400")
    repo.require_repo()
    import blio                                                       # noqa: PLC0415
    out_dir = os.path.join(ASSIST, "logs", "vs")
    os.makedirs(out_dir, exist_ok=True)
    plate = blio.Plate(os.path.abspath(a.plate), ifl_dir=out_dir)
    h = plate.h
    log("plate %dx%d, %d frames" % (plate.w, h, plate.count))

    rows = []
    for name, pts in read_3de(a.manual):
        if a.only and name != a.only:
            continue
        # 3DE is y-UP; frame 0 exists in some exports and there is no frame 0 on the plate.
        truth = {int(f): (float(x), float(h - y)) for f, x, y in pts if int(f) >= 1}
        fs = sorted(truth)
        if len(fs) < 5:
            log("%-12s skipped -- %d usable sample(s)" % (name, len(fs)))
            continue
        hi = min(fs[-1], int(plate.count))
        if a.max_frames:
            hi = min(hi, fs[0] + a.max_frames - 1)
        truth = {f: p for f, p in truth.items() if f <= hi}
        fs = sorted(truth)
        if len(fs) < 5:
            continue
        rs = runs_of(fs)

        g = leash._chain(plate, fs[0], truth[fs[0]], fs[0], hi,
                         a.max_side, a.chain, None, +1)
        nat, aft = score(g, truth, rs[0], a.wrong_px)
        rows.append((name, len(fs), len(rs), nat, aft))
        log("%-12s f%-4d-%-4d %2d run(s) | natural %3d/%-3d p50 %5.1f px | after a gap "
            "%3d/%-3d p50 %s"
            % (name, fs[0], hi, len(rs), nat[0], nat[1], nat[2],
               aft[0], aft[1], "  n/a" if aft[1] == 0 else "%5.1f px" % aft[2]))

    if not rows:
        return 1
    n_ok = sum(r[3][0] for r in rows)
    n_tot = sum(r[3][1] for r in rows)
    g_ok = sum(r[4][0] for r in rows)
    g_tot = sum(r[4][1] for r in rows)
    p50 = sorted(r[3][2] for r in rows if r[3][2] == r[3][2])
    log("")
    log("COTRACKER, raw, seeded on the artist's own points:")
    log("  natural tracking : %d of %d frames on the feature (%d%%), median p50 %.1f px"
        % (n_ok, n_tot, round(100.0 * n_ok / max(1, n_tot)),
           p50[len(p50) // 2] if p50 else float("nan")))
    if g_tot:
        log("  after a gap      : %d of %d frames on the feature (%d%%)"
            % (g_ok, g_tot, round(100.0 * g_ok / max(1, g_tot))))
    else:
        log("  after a gap      : no track in this file has a gap inside the scored range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
