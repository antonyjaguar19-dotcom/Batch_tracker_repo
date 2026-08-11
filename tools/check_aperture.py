# -*- coding: utf-8 -*-
"""Self-check of the aperture (edge-slide) refusal ratio.

Pure numpy/cv2, seconds, no GPU. Tracks a synthetic camera move over features of known
dimensionality and checks the ratio separates the 1-D ones from the rest -- and, critically,
that it keeps separating them when the plate changes, which is where every absolute bar tried
in this repo has failed.

    runtime\\python311\\python.exe tools\\check_aperture.py
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pattern_refine as pr    # noqa: E402

HALF, SEARCH, NF = 15, 24, 40


class Cfg:
    refine_patch_px = 2 * HALF + 1
    refine_search_px = SEARCH
    refine_search_max = 64
    refine_search_speed_k = 1.5
    match_ambiguity_ratio = 0.90
    refine_ncc_lost = 0.60
    refine_ncc_hold = 0.45
    refine_ncc_reref = 0.68
    refine_motion = "translation"
    refine_min_len = 8
    template_frames = 1
    refine_iterations = 1
    refine_ecc_polish = False
    min_corner_anisotropy = 0.0     # off: this test is about the refusal ratio, not seeding
    flip_y_for_3de = False


def scene(kind, contrast=70, blur=1.2, noise=2.0, seed=3):
    rng = np.random.default_rng(seed)
    im = np.full((420, 420), 128 - contrast / 2, np.float32)
    if kind == "edge":
        im[:, 210:] = 128 + contrast / 2
    elif kind == "corner":
        im[210:, 210:] = 128 + contrast / 2
    elif kind == "checker":
        im[210:, 210:] = 128 + contrast / 2
        im[:210, :210] = 128 + contrast / 2
    elif kind == "blob":
        cv2.circle(im, (210, 210), 9, 128 + contrast / 2, -1)
    im = cv2.GaussianBlur(im, (0, 0), blur)
    return np.clip(im + rng.normal(0, noise, im.shape), 0, 255).astype(np.uint8)


def refuse_ratio(kind, contrast, blur):
    """Refine a point through a synthetic pan and return its refusal ratio."""
    base = scene(kind, contrast, blur)
    frames = []
    for t in range(NF):
        M = np.float32([[1, 0, 0.8 * t], [0, 1, 0.35 * t]])
        frames.append(cv2.warpAffine(base, M, (420, 420), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REFLECT))
    pts = [(t + 1, 210.0 + 0.8 * t, 210.0 + 0.35 * t) for t in range(NF)]
    pr._APERTURE["asked"] = pr._APERTURE["refused"] = 0
    pr._refine_segment(pts, lambda i: frames[i] if 0 <= i < NF else None, Cfg(), None)
    a = pr._APERTURE["asked"]
    return (pr._APERTURE["refused"] / a) if a else float("nan")


def main() -> int:
    bad = 0

    def check(ok, label, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'}  {label}" + (f"   {detail}" if detail else ""))

    print("aperture refusal-ratio self-check\n")
    conds = [(70, 1.2), (25, 1.2), (140, 1.2), (70, 2.8), (25, 2.8)]
    table = {}
    print(f"  {'kind':9}" + "".join(f"{f'c{c} b{b}':>12}" for c, b in conds))
    for kind in ("edge", "corner", "checker", "blob"):
        row = [refuse_ratio(kind, c, b) for c, b in conds]
        table[kind] = row
        print(f"  {kind:9}" + "".join(f"{v:>12.2f}" for v in row))
    print()

    edge = np.array(table["edge"])
    rest = np.array(table["corner"] + table["checker"] + table["blob"])
    check(np.nanmin(edge) > np.nanmax(rest),
          "every edge scores above every non-edge, across all plate conditions",
          f"edge min {np.nanmin(edge):.2f} vs non-edge max {np.nanmax(rest):.2f}")
    check(np.nanmin(edge) >= 0.5, "edges refuse most of their frames",
          f"edge min {np.nanmin(edge):.2f}")
    check(np.nanmax(rest) <= 0.15, "real features almost never refuse",
          f"non-edge max {np.nanmax(rest):.2f}")
    # The property that matters for an unattended batch: one fixed cut works everywhere.
    gap_lo, gap_hi = float(np.nanmax(rest)), float(np.nanmin(edge))
    check(gap_hi - gap_lo > 0.3, "a single fixed threshold separates them on every plate",
          f"usable band {gap_lo:.2f}..{gap_hi:.2f}")

    print("\n" + ("all checks passed" if not bad else f"{bad} check(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
