"""Run the shot report over a real 3DE file and print it the way an artist would read it.

`test_coverage.py` proves the metrics against cameras whose answer is known by construction.
This is the other half: the same code on real tracks, where nobody knows the answer and the
only question is whether what comes out is worth reading. A number that is correct on
synthetic data and says nothing useful on a real shot is still a failure.

    runtime\\python311\\python.exe tests\\eval_coverage.py <file.txt> [--size 3840x2160]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [os.path.join(ASSIST, "sidecar"), HERE]

import coverage                                                       # noqa: E402
from eval_cotracker_direct import read_3de                            # noqa: E402


def render(rep, w, h):
    """The report as text. This wording is the draft of what the panel will say."""
    out = []
    if not rep.get("tracks"):
        return ["no tracks"]
    if rep.get("size_warning"):
        out.append("!! " + rep["size_warning"])
    lo, hi = rep["frames"]
    out.append("%d tracks over f%d-f%d   median span %d frames"
               % (rep["tracks"], lo, hi, rep["median_span"]))
    out.append("live at once: median %d, fewest %d (floor %d)"
               % (rep["median_live"], rep["min_live"], rep["floor"]))

    thin = rep["thin_runs"]
    if thin:
        out.append("THIN -- the solve has little to hold on to here:")
        for a, b, worst in thin[:8]:
            out.append("   f%d-f%d  as few as %d track(s)" % (a, b, worst))
        if len(thin) > 8:
            out.append("   ... and %d more stretches" % (len(thin) - 8))
    else:
        out.append("no stretch falls under the floor")

    par = rep["parallax"]
    out.append("parallax: %s -- %s" % (par["verdict"].upper(), par["reason"]))

    bad = rep["suspect"]
    if bad:
        out.append("%d track(s) disagree with the camera motion the rest agree on "
                   "(a mover, or slid off):" % len(bad))
        for s in bad[:10]:
            out.append("   %-14s failed %d of %d checks" % (s["id"], s["failed"], s["tested"]))
        if len(bad) > 10:
            out.append("   ... and %d more" % (len(bad) - 10))
    else:
        out.append("nothing disagrees with the camera motion "
                   "(which is not the same as no movers -- see the epipolar blind spot)")

    if rep["bare_cells"]:
        out.append("regions with little or no coverage across the shot:")
        for c in rep["bare_cells"]:
            out.append("   %-16s has a track on %d%% of frames"
                       % (c["name"], round(c["present_share"] * 100)))
    else:
        out.append("every region of the frame is covered for most of the shot")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--size", default="3840x2160")
    ap.add_argument("--floor", type=int, default=8)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))

    tracks = {}
    for name, pts in read_3de(a.path):
        # 3DE is y-UP; everything in `coverage` is image space, y-DOWN.
        tracks[name] = {int(f): (float(x), float(h - y)) for f, x, y in pts}
    rep = coverage.report(tracks, w, h, floor=a.floor)
    print("== %s  (%dx%d) ==" % (os.path.basename(a.path), w, h))
    for line in render(rep, w, h):
        print("   " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
