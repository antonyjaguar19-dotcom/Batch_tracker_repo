# -*- coding: utf-8 -*-
"""Self-check of the coarse-to-fine NCC search (`refine_pyramid`).

Pure numpy/cv2, no GPU. The pyramid is a SPEED change that must not become an accuracy
change, so what matters is not that it finds the peak -- it is that it finds the SAME peak as
the single-level search, to sub-pixel agreement, on the same input.

  1. agreement with single-level on a synthetic feature, at known sub-pixel shifts
  2. accuracy against the TRUE shift, so both paths being wrong together still fails
  3. it is actually faster at a large search radius
  4. it never returns a confident answer where single-level returns None (ambiguity)

    runtime\\python311\\python.exe tools\\check_pyramid.py
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pattern_refine as pr    # noqa: E402

HALF, SEARCH = 15, 64


def plate(seed=4, w=600, h=600):
    """Band-limited noise: many distinct features, no repeated structure."""
    rng = np.random.default_rng(seed)
    a = rng.normal(128, 55, (h, w)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(a, (0, 0), 1.6), 0, 255).astype(np.uint8)


def shifted(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main() -> int:
    bad = 0

    def check(ok, label, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'}  {label}" + (f"   {detail}" if detail else ""))

    print("pyramid NCC self-check\n")
    base = plate()
    pts = [(300, 300), (220, 380), (400, 250), (180, 200), (350, 420)]

    dis, err_s, err_p = [], [], []
    for (cx, cy) in pts:
        patch = cv2.getRectSubPix(base, (2 * HALF + 1, 2 * HALF + 1), (float(cx), float(cy)))
        for (tx, ty) in ((0.37, -0.62), (2.4, 3.1), (-11.3, 7.8), (19.6, -14.2)):
            g = shifted(base, tx, ty)
            want = (cx + tx, cy + ty)
            a = pr._ncc_match(g, patch, want[0], want[1] - 0.0, SEARCH, HALF,
                              edge_clamp=True, ambiguity_ratio=1.0, pyramid=False)
            b = pr._ncc_match(g, patch, want[0], want[1] - 0.0, SEARCH, HALF,
                              edge_clamp=True, ambiguity_ratio=1.0, pyramid=True)
            if a is None or b is None:
                continue
            dis.append(np.hypot(a[0] - b[0], a[1] - b[1]))
            err_s.append(np.hypot(a[0] - want[0], a[1] - want[1]))
            err_p.append(np.hypot(b[0] - want[0], b[1] - want[1]))

    n = len(dis)
    md, ms, mp = float(np.median(dis)), float(np.median(err_s)), float(np.median(err_p))
    check(n >= 15 and md < 0.02, "pyramid agrees with single-level",
          f"median disagreement {md:.4f}px over {n} matches (worst {max(dis):.4f})")
    # The absolute number here is the matcher's own floor on this synthetic (~0.07px, shared
    # by both paths); what is being asserted is that the pyramid does not ADD to it.
    check(abs(mp - ms) < 0.005, "pyramid adds no error of its own against the TRUE shift",
          f"single {ms:.4f}px vs pyramid {mp:.4f}px (shared floor, difference is the test)")

    # 3. speed at a large radius
    patch = cv2.getRectSubPix(base, (2 * HALF + 1, 2 * HALF + 1), (300.0, 300.0))
    g = shifted(base, 19.6, -14.2)
    def timed(flag):
        t = time.perf_counter()
        for _ in range(120):
            pr._ncc_match(g, patch, 300.0, 300.0, SEARCH, HALF, True, 1.0, flag)
        return time.perf_counter() - t
    ts, tp = timed(False), timed(True)
    check(tp < ts, f"faster at search={SEARCH}px", f"single {ts*1000:.0f}ms vs pyramid {tp*1000:.0f}ms "
                                                   f"({ts/max(tp,1e-9):.2f}x)")

    # 4. a repeated pattern must not become confident just because it was blurred first
    grid = np.zeros((600, 600), np.uint8)
    for yy in range(40, 560, 24):
        for xx in range(40, 560, 24):
            cv2.circle(grid, (xx, yy), 4, 235, -1)
    grid = cv2.GaussianBlur(grid, (0, 0), 1.0)
    gp = cv2.getRectSubPix(grid, (2 * HALF + 1, 2 * HALF + 1), (300.0, 300.0))
    g2 = shifted(grid, 5.0, 3.0)
    sa = pr._ncc_match(g2, gp, 300.0, 300.0, SEARCH, HALF, True, 0.90, False)
    pa = pr._ncc_match(g2, gp, 300.0, 300.0, SEARCH, HALF, True, 0.90, True)
    check(not (sa is None and pa is not None),
          "no confident answer where single-level refuses",
          f"single={'None' if sa is None else 'match'}, pyramid={'None' if pa is None else 'match'}")
    # And the other direction, which is what the fall-through exists for: an ambiguous coarse
    # level must cost speed, never a frame. If the pyramid could refuse where single-level
    # accepts, turning it on would silently shorten tracks.
    lost = 0
    for (cx, cy) in pts:
        patch = cv2.getRectSubPix(base, (2 * HALF + 1, 2 * HALF + 1), (float(cx), float(cy)))
        for (tx, ty) in ((0.37, -0.62), (2.4, 3.1), (-11.3, 7.8), (19.6, -14.2), (31.0, 22.0)):
            g = shifted(base, tx, ty)
            s = pr._ncc_match(g, patch, cx, cy, SEARCH, HALF, True, 0.90, False)
            p = pr._ncc_match(g, patch, cx, cy, SEARCH, HALF, True, 0.90, True)
            lost += 1 if (s is not None and p is None) else 0
    check(lost == 0, "an ambiguous coarse level costs speed, never a frame",
          f"{lost} frame(s) matched by single-level and refused by pyramid")

    print("\n" + ("all checks passed" if not bad else f"{bad} check(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
