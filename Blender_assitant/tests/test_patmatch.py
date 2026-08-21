"""Does the pattern check actually identify the artist's feature?

The claim being tested is the one the loop now leans on: correlating the marker's own
pattern patch against a candidate resume separates "the feature you set" from "a feature".
A metric that looks plausible and measures the plate instead of the match is exactly the
defect this project has been bitten by twice, so this feeds it cases whose answer is known
by construction:

  * the same feature, moved a known whole number of pixels -> found, sub-pixel exact, ~1.0
  * the same feature, moved a known FRACTION of a pixel   -> recovered to <0.15 px
  * a DIFFERENT feature sitting where the guide pointed   -> scores below the threshold
  * an exposure and grain change on the same feature      -> still scores high
  * a flat, featureless patch                             -> refused, not scored

Runs in seconds on CPU with no Blender, no CoTracker and no plate on disk:

    runtime\python311\python.exe tests\test_patmatch.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sidecar"))

import cv2                                                            # noqa: E402
import patmatch                                                       # noqa: E402

W, H = 640, 480
RNG = np.random.default_rng(7)


def _feature(img, cx, cy, seed):
    """A patch of fixed random texture -- unique, high-contrast, and not symmetric, so a
    correlation peak on it is unambiguous."""
    r = np.random.default_rng(seed)
    blob = (r.random((31, 31)) * 255).astype(np.uint8)
    blob = cv2.GaussianBlur(blob, (3, 3), 0)
    x0, y0 = int(cx) - 15, int(cy) - 15
    img[y0:y0 + 31, x0:x0 + 31] = blob[:, :, None]
    return img


def _plate_frame(features, shift=(0.0, 0.0), gain=1.0, offset=0.0, noise=0.0):
    """Mid-grey background with low-frequency structure, plus the given features."""
    base = (RNG.random((H // 8, W // 8)) * 60 + 90).astype(np.uint8)
    img = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.repeat(img[:, :, None], 3, axis=2)
    for cx, cy, seed in features:
        img = _feature(img, cx, cy, seed)
    if shift != (0.0, 0.0):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT)
    if gain != 1.0 or offset:
        img = np.clip(img.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    if noise:
        img = np.clip(img.astype(np.float32)
                      + RNG.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


class FakePlate:
    """`frame(i)` 0-based, BGR uint8 -- the only part of Plate patmatch touches."""

    def __init__(self, frames):
        self._f = frames
        self.w, self.h, self.count = W, H, len(frames)

    def frame(self, i):
        return self._f[i] if 0 <= i < len(self._f) else None


FAILURES = []


def check(name, ok, detail=""):
    print("%-46s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(name)


def main():
    # frame 1: the artist's seed. The feature they picked is A at (200, 240); B is a decoy
    # of the same size and contrast 180 px away -- what a wrong re-acquire lands on.
    f1 = _plate_frame([(200, 240, 11), (380, 240, 22)])
    # frame 2: everything moved 17 px right, 9 px down -- an integer move, so the true
    # answer is exactly (217, 249) and any error is the matcher's own.
    f2 = _plate_frame([(200, 240, 11), (380, 240, 22)], shift=(17.0, 9.0))
    # frame 3: the same, plus a sub-pixel component.
    f3 = _plate_frame([(200, 240, 11), (380, 240, 22)], shift=(17.4, 9.65))
    # frame 4: the same move, but two stops darker with heavy grain.
    f4 = _plate_frame([(200, 240, 11), (380, 240, 22)], shift=(17.0, 9.0),
                      gain=0.45, offset=-10, noise=6.0)
    plate = FakePlate([f1, f2, f3, f4])

    ref = patmatch.reference_patch(plate, 1, 200.0, 240.0, 31.0, 31.0)
    check("reference patch extracted", ref is not None)
    if ref is None:
        return 1
    patch, offset = ref
    check("patch is the box that was asked for", patch.shape == (31, 31), str(patch.shape))
    # An odd-width box centred on a whole pixel spans a half-pixel range that integer
    # indices cannot hold, so the patch centre sits up to half a pixel off the marker.
    # `offset` carries exactly that, and `match` subtracts it back -- which is why the
    # integer-move error below is 0.008 px and not 0.5.
    check("patch offset is sub-pixel and carried",
          max(abs(offset[0]), abs(offset[1])) <= 0.5, str(offset))

    # --- the right feature, moved a whole number of pixels ------------------------
    got = patmatch.match(plate, 2, patch, 200.0, 240.0, radius=40.0, offset=offset)
    x, y, s = got
    err = ((x - 217.0) ** 2 + (y - 249.0) ** 2) ** 0.5
    check("integer move found", err < 0.05, "err %.4f px, score %.3f" % (err, s))
    check("integer move scores near 1", s > 0.97, "%.3f" % s)

    # --- sub-pixel ---------------------------------------------------------------
    x, y, s = patmatch.match(plate, 3, patch, 200.0, 240.0, radius=40.0, offset=offset)
    err = ((x - 217.4) ** 2 + (y - 249.65) ** 2) ** 0.5
    check("sub-pixel move recovered", err < 0.15, "err %.4f px, score %.3f" % (err, s))

    # --- exposure and grain ------------------------------------------------------
    x, y, s = patmatch.match(plate, 4, patch, 200.0, 240.0, radius=40.0, offset=offset)
    err = ((x - 217.0) ** 2 + (y - 249.0) ** 2) ** 0.5
    check("survives 2 stops + grain", err < 0.5 and s > 0.80,
          "err %.4f px, score %.3f" % (err, s))

    # --- the WRONG feature -------------------------------------------------------
    # The guide points at the decoy and nothing else is in reach. This is the case the
    # whole check exists for: it must score LOW, not merely lower.
    x, y, s_wrong = patmatch.match(plate, 2, patch, 397.0, 249.0, radius=12.0,
                                   offset=offset)
    check("decoy scores below the 0.60 gate", s_wrong < 0.60, "%.3f" % s_wrong)
    check("decoy separated from the truth", s - s_wrong > 0.3,
          "true %.3f vs decoy %.3f" % (s, s_wrong))

    # --- best_candidate picks the right frame ------------------------------------
    # Frame 2 holds the feature; frame 1 is offered as a decoy candidate pointing at B.
    cands = [(1, (380.0, 240.0)), (2, (217.0, 249.0)), (3, (217.4, 249.65))]
    best = patmatch.best_candidate(plate, patch, offset, cands, radius=8.0)
    check("best candidate is a real match", best is not None)
    if best:
        f, x, y, s, tried = best
        check("best candidate rejects the decoy frame", f in (2, 3),
              "chose frame %d at %.2f" % (f, s))
        check("every candidate is reported", len(tried) == 3, str(tried))

    # --- a flat patch is refused, not scored -------------------------------------
    flat = np.zeros((H, W, 3), np.uint8) + 128
    flat_plate = FakePlate([flat, flat])
    check("flat patch refused", patmatch.reference_patch(
        flat_plate, 1, 200.0, 240.0, 31.0, 31.0) is None)

    # --- a box off the edge of the plate is refused ------------------------------
    check("off-plate box refused", patmatch.reference_patch(
        plate, 1, -40.0, 240.0, 31.0, 31.0) is None)

    print("\n%d check(s) failed" % len(FAILURES) if FAILURES else "\nall checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
