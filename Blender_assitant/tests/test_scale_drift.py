"""What does a swollen pattern box actually mean? Four answers, each fed a case whose
answer is known by construction.

The watch (`tests/test_scale_watch.py`) only decides WHEN to stop and look. This is the
looking, and it is the part that can be wrong in the expensive direction: calling a real
approach to camera "drift" resets a box that was right, and calling drift "genuine scale"
blesses a tracker that has walked onto the background.

  * the feature really got bigger      -> grown, and the SCALED patch wins clearly
  * the box grew, the feature did not  -> bad-box, and the artist's own box still finds the
                                          feature exactly where it was
  * the box grew a little              -> clean; nothing is repaired over grain
  * the feature is gone                -> unknown (not findable at any size)

The position the match reports is checked here too, but only to show the patch really did
find the right thing -- it is NOT what the repair acts on. On a real plate a fixed patch
matched 150 frames later sits p50 4.2 px from a healthy tracker's own position, so it can
answer presence and size and nothing finer (`sidecar/patmatch.classify_drift`).

    runtime\\python311\\python.exe tests\\test_scale_drift.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sidecar"))

import cv2                                                            # noqa: E402
import patmatch                                                       # noqa: E402

W, H = 640, 480
RNG = np.random.default_rng(3)


def _blob(seed, side):
    r = np.random.default_rng(seed)
    b = (r.random((31, 31)) * 255).astype(np.uint8)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    if side != 31:
        b = cv2.resize(b, (side, side), interpolation=cv2.INTER_CUBIC)
    return b


def _plate_frame(features):
    """features: (cx, cy, seed, side). Mid-grey ground with low-frequency structure."""
    base = (RNG.random((H // 8, W // 8)) * 60 + 90).astype(np.uint8)
    img = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.repeat(img[:, :, None], 3, axis=2)
    for cx, cy, seed, side in features:
        b = _blob(seed, side)
        x0, y0 = int(cx) - side // 2, int(cy) - side // 2
        img[y0:y0 + side, x0:x0 + side] = b[:, :, None]
    return img


class FakePlate:
    def __init__(self, frames):
        self._f = frames
        self.w, self.h, self.count = W, H, len(frames)

    def frame(self, i):
        return self._f[i] if 0 <= i < len(self._f) else None


FAILURES = []


def check(name, ok, detail=""):
    print("%-54s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(name)


def main():
    # The artist seeds a 31 px box on the feature at (200, 240) on frame 1. A second
    # feature 60 px away is what a swelling box swallows.
    seed_box = (200.0, 240.0, 31.0, 31.0)

    f1 = _plate_frame([(200, 240, 11, 31), (260, 240, 22, 31)])
    # frame 2: the SAME feature, 45 px across -- an approach to camera, 1.45x.
    f2 = _plate_frame([(200, 240, 11, 45), (300, 240, 22, 31)])
    # frame 3: nothing changed. The tracker's box has grown anyway and its centre has
    # slid 6 px towards the neighbour -- a box swelling onto the surroundings.
    f3 = _plate_frame([(200, 240, 11, 31), (260, 240, 22, 31)])
    # frame 4: the feature is replaced by different texture. It is not there at any size.
    f4 = _plate_frame([(200, 240, 77, 31), (260, 240, 22, 31)])
    plate = FakePlate([f1, f2, f3, f4])

    # --- the feature really got bigger -------------------------------------------
    # The tracker followed it and its box grew with it: 45/31 = 1.45x, still centred.
    rep = patmatch.drift_report(plate, 1, seed_box, 2, 200.0, 240.0,
                                (200.0, 240.0, 45.0, 45.0), radius=20.0)
    check("genuine scale is read as genuine", rep["verdict"] == "grown",
          "%s (ref %.2f, scaled %.2f)" % (rep["verdict"], rep["score_ref"] or 0,
                                          rep["score_scaled"] or 0))
    check("the scaled patch wins clearly",
          (rep["score_scaled"] or 0) - (rep["score_ref"] or 0) > 0.05,
          "%.3f vs %.3f" % (rep["score_scaled"] or 0, rep["score_ref"] or 0))
    check("the measured scale matches the box", abs(rep["scale"] - 45.0 / 31.0) < 0.02,
          "%.3f" % rep["scale"])
    err = ((rep["x_scaled"] - 200.0) ** 2 + (rep["y_scaled"] - 240.0) ** 2) ** 0.5
    check("scaled match lands on the feature", err < 1.0, "err %.3f px" % err)

    # --- the box grew, the feature did not ---------------------------------------
    # The box is 1.8x and its centre has slid 6 px. The feature is still exactly where it
    # was, so the artist's own box must find it and the fix must be worth applying.
    rep = patmatch.drift_report(plate, 1, seed_box, 3, 206.0, 240.0,
                                (206.0, 240.0, 56.0, 56.0), radius=20.0)
    check("a swollen box on an unchanged feature is bad-box",
          rep["verdict"] == "bad-box",
          "%s (ref %.2f, scaled %.2f)" % (rep["verdict"], rep["score_ref"] or 0,
                                          rep["score_scaled"] or 0))
    err = ((rep["x"] - 200.0) ** 2 + (rep["y"] - 240.0) ** 2) ** 0.5
    check("the artist's box still finds the feature", err < 0.15, "err %.4f px" % err)
    check("the offset is reported, not acted on",
          abs(rep["offset_px"] - 6.0) < 0.2, "%.3f px" % rep["offset_px"])
    check("the unscaled patch scores high", (rep["score_ref"] or 0) > 0.90,
          "%.3f" % (rep["score_ref"] or 0))

    # --- a false alarm ------------------------------------------------------------
    # 1.1x and dead on the feature: nothing here is worth cutting frames for.
    rep = patmatch.drift_report(plate, 1, seed_box, 3, 200.0, 240.0,
                                (200.0, 240.0, 34.0, 34.0), radius=12.0)
    check("a small swell on the feature is clean", rep["verdict"] == "clean",
          "%s (%.2f, %.2f px off)" % (rep["verdict"], rep["score_ref"] or 0,
                                      rep["offset_px"]))

    # --- the feature is gone ------------------------------------------------------
    rep = patmatch.drift_report(plate, 1, seed_box, 4, 200.0, 240.0,
                                (200.0, 240.0, 56.0, 56.0), radius=12.0)
    check("a feature that is not there is unknown", rep["verdict"] == "unknown",
          "%s (ref %.2f, scaled %.2f)" % (rep["verdict"], rep["score_ref"] or 0,
                                          rep["score_scaled"] or -1))
    check("unknown is not a near miss", max(rep["score_ref"] or -1,
                                         rep["score_scaled"] or -1) < 0.60,
          "best %.3f" % max(rep["score_ref"] or -1, rep["score_scaled"] or -1))

    # --- refusals -----------------------------------------------------------------
    off = patmatch.drift_report(plate, 1, (-40.0, 240.0, 31.0, 31.0), 3, 200.0, 240.0,
                                (200.0, 240.0, 40.0, 40.0))
    check("an off-plate reference is refused, not guessed",
          not off["ok"] and off["verdict"] == "no-reference", str(off.get("verdict")))
    bad = patmatch.drift_report(plate, 1, seed_box, 99, 200.0, 240.0,
                                (200.0, 240.0, 40.0, 40.0))
    check("an unreadable frame is refused", not bad["ok"], str(bad.get("verdict")))

    # --- the rules, on numbers alone ----------------------------------------------
    # classify_drift is what the verdicts above come out of; these pin the boundaries so a
    # threshold cannot be moved by accident.
    cd = patmatch.classify_drift
    # "unknown", not "lost": both scores being low is the ABSENCE of evidence, and a
    # verdict that deletes frames may not be reached that way. See patmatch.classify_drift.
    check("both scores below the gate -> unknown",
          cd(0.55, 0.58, 0.2, 1.4, min_match=0.60) == "unknown")
    check("scaled wins by more than the margin -> grown",
          cd(0.70, 0.90, 0.2, 1.4) == "grown")
    check("scaled wins by less than the margin -> not genuine",
          cd(0.90, 0.93, 0.2, 1.05) == "clean")
    check("mild swell -> clean", cd(0.95, None, 1.0, 1.1) == "clean")
    check("big swell -> bad-box", cd(0.95, None, 1.0, 1.5) == "bad-box")
    # The one that was measured and REMOVED: a large offset with a mild swell used to read
    # as drift. On real footage that fires on healthy tracks (p90 25 px), so the offset no
    # longer decides anything.
    check("a big offset alone decides nothing", cd(0.95, None, 25.0, 1.1) == "clean")
    check("shrink is judged like growth", cd(0.95, None, 0.2, 1.0 / 1.5) == "bad-box")

    # ------------------------------------------------------ holding the feature
    # `first_loss` decides whether a track is still on the artist's feature at all.
    #
    # The SH006 row is measured, not invented: a seed occluded at frame 14, scored against
    # the artist's own patch at every position the track claimed. 0.91 at f14, 0.22 at f15 --
    # while Blender's own correlation was satisfied the whole way, because PREV_FRAME only
    # ever compares to the frame before, and an occluder sliding in over a few frames never
    # looks like a failure.
    #
    # The SH013 row is why this is not a plain threshold. Patches on that plate score
    # 0.53-0.72 against the very NEXT frame while tracking perfectly well; an absolute floor
    # would cut every track on it. A loss has to be a FALL from what the track itself was
    # holding, and a score that was never high cannot fall.
    fl = patmatch.first_loss
    for name, scores, expect in (
        ("SH006 occlusion at f15",
         [(1, 1.00), (2, .995), (3, .995), (4, .996), (5, .995), (6, .994), (7, .996),
          (8, .993), (9, .968), (10, .846), (11, .833), (12, .951), (13, .985), (14, .908),
          (15, .218), (16, .210), (17, .192), (18, .168), (19, .169), (20, .161)], 15),
        ("SH013 low contrast never drifts",
         [(f, 0.55 + 0.1 * ((f % 3) - 1)) for f in range(1, 40)], None),
        ("one bad frame does not cut a good track",
         [(f, 0.10 if f == 10 else 0.95) for f in range(1, 20)], None),
        ("two bad frames in a row is the real thing",
         [(f, 0.10 if f in (10, 11) else 0.95) for f in range(1, 20)], 10),
        ("gentle decline (defocus) is still the feature",
         [(f, max(0.45, 0.95 - 0.02 * f)) for f in range(1, 30)], None),
        ("occlusion late in a long healthy track",
         [(f, 0.9) for f in range(1, 100)] + [(f, 0.15) for f in range(100, 110)], 100),
        # Ambiguity: a wire crosses the feature and a lookalike 52 px away becomes as good
        # an answer. Measured on SH006. The margin alone is a property of the PLATE and
        # reads the same for a correct track and a drifting one, so both conditions must
        # hold -- a tight margin AND a score that has given way at the claimed position.
        ("SH006 wire: drifting track, tight margin AND falling score",
         [(f, 0.95, 0.15) for f in range(1, 86)]
         + [(86, 0.951, 0.104), (87, 0.928, 0.067), (88, 0.949, 0.075), (89, 0.938, 0.065),
            (90, 0.859, 0.006), (91, 0.797, 0.006), (92, 0.725, 0.006),
            (93, 0.708, 0.032), (94, 0.621, 0.029), (95, 0.498, 0.037)], 91),
        # PERSPECTIVE. A feature approaching camera stops resembling the patch taken when
        # it was small and far away -- the score collapses exactly like drift does. What
        # separates them is the fourth field: whether a better match exists nearby. These
        # two rows are IDENTICAL in score and margin and differ only in that.
        ("approaching camera: score collapses, nothing better nearby",
         [(f, 0.95, 0.20, 0.01) for f in range(1, 40)]
         + [(f, 0.90 - 0.03 * (f - 40), 0.18, 0.02) for f in range(40, 55)], None),
        ("drift: the same collapse, but a better match sits nearby",
         [(f, 0.95, 0.20, 0.01) for f in range(1, 40)]
         + [(f, 0.90 - 0.03 * (f - 40), 0.18, 0.45) for f in range(40, 55)], 44),
        ("SH006 wire: the artist's own track, same margin, score holds",
         [(f, 0.95, 0.15) for f in range(1, 86)]
         + [(86, 0.951, 0.104), (87, 0.930, 0.067), (88, 0.954, 0.075), (89, 0.950, 0.065),
            (90, 0.902, 0.006), (91, 0.906, 0.006), (92, 0.893, 0.006),
            (93, 0.931, 0.032), (94, 0.911, 0.029), (95, 0.886, 0.037)], None),
    ):
        check("hold: %s" % name, fl(scores) == expect,
              "got %s, expected %s" % (fl(scores), expect))

    print("\n%d check(s) failed" % len(FAILURES) if FAILURES else "\nall checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
