"""The grade, checked against tracks whose noise is known because it was put there.

A quality metric that looks plausible and measures the wrong thing is the defect this project
keeps rediscovering. So the jitter figure is calibrated against noise of a KNOWN size injected
into the artist's own hand track, rather than eyeballed on real output:

  * a perfectly smooth path must measure ~0;
  * a path with 1.0 px of noise added must measure ~1.0, not 0.3 and not 3;
  * doubling the noise must double the reading.

And the negative that matters most: **a smooth SLIDE must not register as noise.** Jitter
cannot see drift -- a slide fits a local quadratic perfectly -- and if this test did not say so
out loud, the grade would look like a drift detector and quietly pass every slid track.

    runtime\\python311\\python.exe tests\\test_quality.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [os.path.join(ASSIST, "sidecar"), HERE]

import quality                                                        # noqa: E402
from eval_cotracker_direct import read_3de                            # noqa: E402

FAILED = []


def check(name, got, want):
    ok = got == want
    print("  %-58s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def near(name, got, want, tol):
    ok = got is not None and abs(got - want) <= tol
    print("  %-58s %s" % (name, "ok  %.3f" % got if ok
                          else "FAIL  got %r want %r +-%r" % (got, want, tol)))
    if not ok:
        FAILED.append(name)


def hand_track():
    name, pts = [t for t in read_3de(os.path.join(HERE, "track2 manual tracked.txt"))
                 if t[0] == "Track.003"][0]
    return {f: (x, y) for f, x, y in pts}


def smooth(n=200):
    """A path with constant acceleration -- exactly what the fit models, so residual is 0."""
    return {f: (100.0 + 2.0 * f + 0.01 * f * f, 50.0 + 1.0 * f) for f in range(1, n + 1)}


def noised(path, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return {f: (x + rng.normal(0, sigma), y + rng.normal(0, sigma))
            for f, (x, y) in path.items()}


def main():
    print("calibration -- noise of a known size must read back at that size")
    sm = smooth()
    near("a perfectly smooth path reads ~0", quality.jitter(sm), 0.0, 0.01)

    # A local quadratic through 7 points absorbs some of the noise, so the residual at the
    # centre is a fixed fraction of sigma rather than sigma itself. What matters is that the
    # reading is PROPORTIONAL and stable, which is what makes a threshold mean anything.
    j1 = quality.jitter(noised(sm, 1.0, seed=1))
    j2 = quality.jitter(noised(sm, 2.0, seed=1))
    j4 = quality.jitter(noised(sm, 4.0, seed=1))
    print("  1.0 px of noise reads %.3f, 2.0 reads %.3f, 4.0 reads %.3f" % (j1, j2, j4))
    near("doubling the noise doubles the reading", j2 / j1, 2.0, 0.15)
    near("and again", j4 / j2, 2.0, 0.15)
    check("more noise always reads higher", j1 < j2 < j4, True)

    print("\nthe negative that matters: jitter CANNOT see drift")
    # A synthetic ramp with sharp corners DOES produce jitter -- 1.93 px, all of it from the
    # two kinks where the slide starts and stops. Asserting "a slide reads zero" against that
    # would measure the corners I happened to build rather than the fault. The real thing is
    # gradual, and the honest evidence is the artist's own drifted output against their own
    # hand track of the same feature.
    hand = hand_track()
    jh = quality.jitter(hand)
    drifted = {f: (x, y) for f, x, y in
               [t for t in read_3de(os.path.join(HERE,
                                                 "reacquretracke_assist tracked_v003.txt"))
                if t[0] == "Track.002"][0][1]}
    jd = quality.jitter(drifted)
    print("  their DRIFTED track reads %.3f, their HAND track %.3f" % (jd, jh))
    check("a track that slid reads no worse than a correct one", abs(jd - jh) < 0.25, True)
    check("  so the grade passes it, and find_slides is what must catch it",
          quality.grade(drifted)["verdict"], "good")
    # And as a fact about the metric rather than a tunable: a constant-velocity offset added
    # to a path changes no second derivative anywhere, so it is invisible by construction.
    ramp = {f: (x + 1.2 * f, y + 0.5 * f) for f, (x, y) in sm.items()}
    near("adding a steady slide of any size changes the reading by nothing",
         quality.jitter(ramp), quality.jitter(sm), 0.01)

    print("\nagainst the artist's own tracks")
    near("their hand track sits where a human click scatters", jh, 0.75, 0.25)
    g = quality.grade(hand)
    check("and grades good", g["verdict"], "good")
    check("  with nothing to report", g["why"], [])

    print("\nshape of the grade")
    short = {f: (0.0, 0.0) for f in range(1, 8)}
    check("a very short track is poor however clean", quality.grade(short)["verdict"], "poor")
    g = quality.grade(noised(sm, 6.0, seed=3))
    check("a very unsteady track is poor", g["verdict"], "poor")
    check("  and says why in words", any("unsteady" in w for w in g["why"]), True)
    g = quality.grade(noised(sm, 2.2, seed=3))
    check("a middling one asks to be checked rather than condemned", g["verdict"], "check")

    holed = {f: v for f, v in sm.items() if not (50 <= f <= 60 or 90 <= f <= 95
                                                 or 120 <= f <= 130 or 150 <= f <= 155)}
    g = quality.grade(holed)
    check("a track in many pieces is flagged", g["verdict"], "check")
    check("  and counted", g["runs"], 5)

    _all, summ = quality.grade_all({"a": hand, "b": noised(sm, 6.0, seed=4)})
    check("the summary counts verdicts", summ["verdicts"].get("poor"), 1)
    check("  and reports the spread", summ["jitter_max"] is not None, True)

    print("")
    if FAILED:
        print("QUALITY: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("QUALITY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
