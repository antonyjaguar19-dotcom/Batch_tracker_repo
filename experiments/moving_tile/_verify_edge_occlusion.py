# -*- coding: utf-8 -*-
"""Deterministic unit checks for edge-clamp NCC + gap-aware refine (no model)."""
import sys, os, types
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import cv2
from app.pattern_refine import _ncc_match, _extract, _refine_one


def cfg(**kw):
    d = dict(refine_patch_px=31, refine_search_px=24, refine_ncc_lost=0.60,
             refine_ncc_reref=0.85, refine_motion="translation", refine_min_len=4,
             mt_edge_track=True, refine_gap_aware=True)
    d.update(kw)
    return types.SimpleNamespace(**d)


def square_frame(cx, cy, val, size=21, W=200, H=200, bg=40):
    img = np.full((H, W), bg, np.uint8)
    h = size // 2
    x0, y0 = int(cx - h), int(cy - h)
    img[max(0, y0):y0 + size, max(0, x0):x0 + size] = val
    # a little texture so NCC has structure
    img[max(0, y0):y0 + size:4, max(0, x0):x0 + size] = min(255, val + 30)
    return img


# ---- Test 1: edge-clamp NCC near the frame border --------------------------------
def test_edge_clamp():
    cx, cy = 20.0, 100.0   # patch (half=15) fits (5px margin) but search box (24) runs off left
    g = square_frame(cx, cy, 220)
    patch = _extract(g, cx, cy, 15)
    off = _ncc_match(g, patch, cx, cy, 24, 15, edge_clamp=False)
    on = _ncc_match(g, patch, cx, cy, 24, 15, edge_clamp=True)
    ok = (off is None) and (on is not None) and abs(on[0] - cx) < 1.5 and abs(on[1] - cy) < 1.5
    print(f"[edge-clamp] off={off}  on={on}  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---- Test 2: gap-aware keeps the reappeared segment ------------------------------
def test_gap_aware():
    # pre-gap frames 1..8 (bright square), gap 9..14 (occluded, no points),
    # post-gap frames 15..22 (DIMMER square = appearance changed after occlusion).
    frames = {}
    track = []
    for f in range(1, 9):
        cx, cy = 60 + 3 * (f - 1), 100
        frames[f - 1] = square_frame(cx, cy, 220)
        track.append((f, float(cx) + 0.4, float(cy) - 0.3))   # coarse w/ small offset
    for f in range(15, 23):
        cx, cy = 60 + 3 * (f - 1), 100
        frames[f - 1] = square_frame(cx, cy, 150)             # dimmer -> old patch won't match
        track.append((f, float(cx) + 0.4, float(cy) - 0.3))

    def get(idx0):
        return frames.get(int(idx0))

    r_gap = _refine_one(track, get, cfg(refine_gap_aware=True))
    r_nogap = _refine_one(track, get, cfg(refine_gap_aware=False))

    def spans(r):
        if not r:
            return (0, 0, 0)
        fs = [p[0] for p in r]
        return (len(fs), min(fs), max(fs))

    n_g, lo_g, hi_g = spans(r_gap)
    n_n, lo_n, hi_n = spans(r_nogap)
    has_pre_g = any(p[0] <= 8 for p in (r_gap or []))
    has_post_g = any(p[0] >= 15 for p in (r_gap or []))
    ok = has_pre_g and has_post_g and n_g >= n_n
    print(f"[gap-aware]  gap_aware -> n={n_g} span[{lo_g},{hi_g}] pre={has_pre_g} post={has_post_g}")
    print(f"[gap-aware]  no_gap    -> n={n_n} span[{lo_n},{hi_n}]")
    print(f"[gap-aware]  -> {'PASS' if ok else 'FAIL'} (gap-aware keeps both segments)")
    return ok


if __name__ == "__main__":
    a = test_edge_clamp()
    b = test_gap_aware()
    print("ALL PASS" if (a and b) else "SOME FAIL")
    sys.exit(0 if (a and b) else 1)
