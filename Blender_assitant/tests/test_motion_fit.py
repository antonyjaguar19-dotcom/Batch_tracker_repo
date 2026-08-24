"""Plate-motion measurement, and the search box it produces.

Two things are checked, and they are not the same thing:

  * `motion.measure` reports the right magnitude on plates whose motion is KNOWN, because it
    is synthesised here by shifting an image by a fixed number of pixels per frame. A flow
    measurement that is quietly wrong would size every box wrong and look like bad tracking.
  * the sizing rule turns that number into a box that actually reaches, and leaves a slow
    plate alone. The second half matters as much as the first: the built-in table is tuned,
    and a "fix" that inflates every box on a normal shot trades a rare failure for a
    permanent one -- a big search box is slower and starts matching lookalikes.

Why this exists: on SH013 (motocross, 59.94 fps) the near-ground moves 21-53 px between
frames and the shipped corner box reaches +-13 px, so every foreground seed died on its FIRST
step -- with the feature scoring 0.88-0.93 NCC at the correct position the whole time. It was
never searched for.

    runtime\\python311\\python.exe tests\\test_motion_fit.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ASSIST, "sidecar"))

import motion  # noqa: E402

FAILURES = []


def log(m):
    print("[motion] %s" % m, flush=True)


def fail(m):
    FAILURES.append(m)
    log("FAIL %s" % m)


class ShiftPlate:
    """A plate that moves by exactly `dx` px per frame, so the right answer is known.

    Textured noise, not a gradient: Farneback needs something to lock onto, and a smooth
    ramp is exactly the case where any flow method reports whatever it likes.
    """

    def __init__(self, w=1280, h=720, dx=0.0, dy=0.0, n=12, seed=0):
        self.w, self.h, self.dx, self.dy, self.n = w, h, dx, dy, n
        rng = np.random.default_rng(seed)
        # Pad for the FULL travel. Sized too small, numpy slicing silently returns a shorter
        # array instead of raising, and the frames simply stop moving -- a synthetic plate
        # that quietly disagrees with its own ground truth is worse than no test.
        self._pad = int(20 + max(abs(dx), abs(dy)) * (n + 2))
        big = rng.integers(0, 255, (h + 2 * self._pad, w + 2 * self._pad), dtype=np.uint8)
        # Blur slightly so the noise has structure at the scale flow works on.
        import cv2
        self._img = cv2.GaussianBlur(big, (0, 0), 2.0)

    def frame(self, i):
        ox = int(round(self._pad + self.dx * i))
        oy = int(round(self._pad + self.dy * i))
        out = self._img[oy:oy + self.h, ox:ox + self.w]
        assert out.shape == (self.h, self.w), "synthetic plate ran off its own padding"
        return out


def fit(p95, pattern_px, plate_w, have_px, headroom=1.5, cap_frac=0.25):
    """The shipped rule, kept in one place so the test and the addon cannot drift apart in
    what they mean by 'reach'."""
    want = 2.0 * (p95 * headroom + pattern_px / 2.0)
    want = min(want, plate_w * cap_frac)
    return max(have_px, want)


def main():
    # --- 1. does the measurement report the motion that is actually there? --------------
    for dx in (2.0, 12.0, 40.0):
        plate = ShiftPlate(dx=dx)
        mo = motion.measure(plate, plate.n, samples=5)
        if mo is None:
            fail("measure returned nothing for dx=%.0f" % dx)
            continue
        got = mo["global_p95"]
        # 25 % tolerance: this runs downscaled on purpose and is sizing a box, not solving a
        # camera. Anything tighter would be testing Farneback's tuning, not our use of it.
        if abs(got - dx) > max(1.5, dx * 0.25):
            fail("dx=%.0f px/frame measured as %.1f" % (dx, got))
        else:
            log("%-34s ok (measured %.1f)" % ("motion of %.0f px/frame" % dx, got))

    # --- 2. is the motion located, not just totalled? -----------------------------------
    # Half the plate still, half moving: a single global number would starve the moving half
    # or bloat the still half, which is why this is a grid at all.
    class SplitPlate(ShiftPlate):
        def frame(self, i):
            a = ShiftPlate.frame(self, 0).copy()
            b = ShiftPlate.frame(self, i)
            a[self.h // 2:, :] = b[self.h // 2:, :]
            return a

    plate = SplitPlate(dx=30.0)
    mo = motion.measure(plate, plate.n, samples=5)
    if mo is None:
        fail("split plate measured nothing")
    else:
        gx, gy = mo["grid"]
        top = max(mo["p95"][0])
        bot = max(mo["p95"][gy - 1])
        if not (bot > top * 3):
            fail("grid did not separate still from moving: top %.1f bottom %.1f" % (top, bot))
        else:
            log("%-34s ok (top %.1f, bottom %.1f)" % ("motion located to the right half",
                                                      top, bot))

    # --- 3. the box the rule produces ---------------------------------------------------
    # SH013's foreground, measured: 21-53 px/frame, p95 up to 67, on a 2562-wide plate with
    # the shipped 28 px pattern in a 55 px box.
    got = fit(45.0, 28.0, 2562, 55.0)
    reach = (got - 28.0) / 2.0
    if reach < 53.0:
        fail("fast plate: box %.0f px only reaches %.0f px, needs 53" % (got, reach))
    else:
        log("%-34s ok (%.0f px box, reaches %.0f)" % ("fast plate widened", got, reach))

    # SH004, the plate the table was tuned on: slow. The rule must leave it alone, or it
    # trades a rare failure for a permanent one.
    got = fit(4.0, 28.0, 2560, 55.0)
    if got > 56.0:
        fail("slow plate box inflated %.0f -> %.0f px" % (55.0, got))
    else:
        log("%-34s ok (stays %.0f px)" % ("slow plate left alone", got))

    # Never shrink: an artist who set a big box keeps it.
    got = fit(4.0, 28.0, 2560, 400.0)
    if got < 400.0:
        fail("an artist's 400 px box was shrunk to %.0f" % got)
    else:
        log("%-34s ok" % "artist's larger box kept")

    # The cap holds, or a very fast cell produces a box that matches lookalikes plate-wide.
    got = fit(900.0, 28.0, 2562, 55.0)
    if got > 2562 * 0.25 + 0.5:
        fail("cap did not hold: %.0f px on a 2562 px plate" % got)
    else:
        log("%-34s ok (%.0f px)" % ("runaway motion capped", got))

    log("MOTION FIT: %s" % ("FAIL" if FAILURES else "PASS"))
    return 1 if FAILURES else 0


sys.exit(main())
