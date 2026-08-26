"""The leash's gates, on synthetic input -- no plate, no GPU, no model.

`eval_leash.py` proves the trust verdict on real footage and needs both. This proves the
LOGIC: that each gate refuses what it exists to refuse, and that removing any one of them
changes the answer. Both are needed. A gate that is never exercised by a test is the defect
class this project keeps rediscovering -- a panel test with no tracks in the scene, a unit
test that fed `first_loss` tuples and never called `hold_check`, a metric that measured the
plate instead of the tracker.

    runtime\\python311\\python.exe tests\\test_leash.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sidecar"))

import leash                                                          # noqa: E402

FAILED = []


def check(name, got, want):
    ok = got == want
    print("  %-58s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def approx(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print("  %-58s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def guide(n=40, dx=10.0, dy=0.0, lo=10):
    return {lo + i: (i * dx, i * dy) for i in range(n)}


def snapper(off_x=0.0, score=0.9):
    """A matcher whose peak sits `off_x` px from wherever it is asked to look."""
    return lambda f, x, y, r: (x + off_x, y, score)


def main():
    print("tolerance grows with the gap, sublinearly")
    approx("gap 1", leash.tolerance(1), leash.TOL_A + leash.TOL_B)
    check("gap 30 is under 4x gap 1 (sqrt, not linear)",
          leash.tolerance(30) < 4 * leash.tolerance(1), True)
    check("gap 0 is treated as gap 1, never zero",
          leash.tolerance(0) == leash.tolerance(1), True)

    print("\npredict applies DISPLACEMENT, never the guide's own coordinates")
    g = guide()
    check("moved by the guide's delta, not to the guide's position",
          leash.predict(g, 10, (500.0, 300.0), 14)[:2], (540.0, 300.0))
    check("no answer where the guide has no sample", leash.predict(g, 10, (0.0, 0.0), 999),
          None)
    check("no answer where the ANCHOR has no sample",
          leash.predict(g, 999, (0.0, 0.0), 14), None)

    print("\ntrust is local to the frames being used")
    # The real case this models: closure over the whole window is 66 px and over the five
    # frames actually being filled it is 15.6. A global verdict throws the fill away.
    clo = dict([(f, 5.0) for f in range(10, 16)] + [(f, 90.0) for f in range(16, 50)])
    check("trusted over the near frames", leash.trust_over(clo, range(10, 16))[0], True)
    check("not trusted over the far ones", leash.trust_over(clo, range(16, 50))[0], False)
    check("not trusted when the guide covers none of them",
          leash.trust_over(clo, range(500, 510))[0], False)

    print("\nfill_gap: each gate refuses what it is for")
    ok_clo = {f: 5.0 for f in range(10, 50)}
    vis = {f: True for f in range(10, 50)}

    filled, why = leash.fill_gap(snapper(0.5), g, vis, ok_clo, 10, (500.0, 300.0), 15)
    check("fills every frame between the cut and the resume",
          [x["frame"] for x in filled], [11, 12, 13, 14])
    check("never plants ON the resume frame", 15 in [x["frame"] for x in filled], False)
    check("never plants on the anchor", 10 in [x["frame"] for x in filled], False)

    filled, why = leash.fill_gap(snapper(0.5), g, vis, {f: 90.0 for f in g},
                                 10, (500.0, 300.0), 15)
    check("an untrustworthy guide fills nothing", filled, [])
    check("  and says why", "closure" in why, True)

    filled, why = leash.fill_gap(snapper(30.0), g, vis, ok_clo, 10, (500.0, 300.0), 15)
    check("a peak far from the prediction fills nothing", filled, [])
    check("  and says how far", "30.0 px from where the guide said" in why, True)

    filled, _ = leash.fill_gap(snapper(0.5, score=0.2), g, vis, ok_clo, 10, (500.0, 300.0), 15)
    check("a pattern that does not match fills nothing", filled, [])

    covered = dict(vis)
    covered[13] = False
    filled, why = leash.fill_gap(snapper(0.5), g, covered, ok_clo, 10, (500.0, 300.0), 16)
    check("stops where CoTracker calls the feature covered",
          [x["frame"] for x in filled], [11, 12])
    check("  and does NOT resume past it -- no island across an occlusion",
          [x["frame"] for x in filled], [11, 12])

    # The contiguity gate, isolated: a matcher that fails only on f13. Without it the fill
    # would jump the failure and plant f14/f15, which across a real occlusion is a marker
    # on the occluder with correct-looking neighbours either side.
    def holed(f, x, y, r):
        return (x + 0.5, y, 0.2 if f == 13 else 0.9)

    filled, _ = leash.fill_gap(holed, g, vis, ok_clo, 10, (500.0, 300.0), 16)
    check("a hole ends the fill; it does not skip and carry on",
          [x["frame"] for x in filled], [11, 12])

    filled, why = leash.fill_gap(snapper(0.5), g, vis, ok_clo, 10, (500.0, 300.0), 11)
    check("adjacent cut and resume leave nothing to do", filled, [])
    check("  and says so", "no frames between" in why, True)

    print("\nfilled from both ends, meeting in the middle")

    # Forward dies at f13, so the frames past it can only come from the resume end. Working
    # forward alone would leave them -- and on a drift gap that is every frame that matters,
    # because the cut end is on the wrong feature there while the resume end has just been
    # verified against the artist's own pattern.
    def fwd_dies(f, x, y, r):
        return (x + 0.5, y, 0.2 if f == 13 else 0.9)

    filled, _ = leash.fill_gap(fwd_dies, g, vis, ok_clo, 10, (500.0, 300.0), 18,
                               resume_px=(580.0, 300.0))
    check("the resume end covers what the cut end could not",
          [x["frame"] for x in filled], [11, 12, 14, 15, 16, 17])
    check("  and the frame BOTH ends refused stays empty",
          13 in [x["frame"] for x in filled], False)

    filled, why = leash.fill_gap(snapper(0.5), g, vis, {f: 90.0 for f in g},
                                 10, (500.0, 300.0), 15, resume_px=(540.0, 300.0))
    check("an untrustworthy guide fills nothing from either end", filled, [])
    check("  and names both ends", why.count("closure"), 2)

    filled, _ = leash.fill_gap(snapper(0.5), g, vis, ok_clo, 10, (500.0, 300.0), 15,
                               resume_px=None)
    check("no resume position -> the cut end alone, as before",
          [x["frame"] for x in filled], [11, 12, 13, 14])

    print("\nthe probe looks FURTHER than it accepts")
    seen = []

    def spy(f, x, y, r):
        seen.append(r)
        return (x, y, 0.9)

    leash.fill_gap(spy, g, vis, ok_clo, 10, (500.0, 300.0), 13)
    # seen[0] is the anchor seating -- a deliberate SHORT look at the frame we already
    # believe, not a search. The probe radii are the rest.
    probes = seen[1:]
    check("search radius exceeds the tolerance it will accept",
          all(r > leash.tolerance(i + 1) for i, r in enumerate(probes)), True)
    check("  and never drops below the floor",
          all(r >= leash.PROBE_MIN_PX for r in probes), True)
    check("  the anchor is seated with a short look, not a search",
          seen[0] < leash.PROBE_MIN_PX, True)

    print("")
    if FAILED:
        print("FAILED: %d -- %s" % (len(FAILED), "; ".join(FAILED)))
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
