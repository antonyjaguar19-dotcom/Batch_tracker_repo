# -*- coding: utf-8 -*-
"""Self-check for the parallel pattern-refine (`refine_workers`).

Parallelising a stage is only acceptable if it cannot change the answer. Two things here
could, and neither fails loudly:

  * the per-thread stashes. certainty, aperture and peak-flatness are written deep in the
    matcher and read by its caller. Sharing them across threads attributes one track's
    numbers to another -- the tracks stay fine, the REPORT silently lies.
  * result ordering. Futures complete out of order, so assembling by completion would
    shuffle track ids between runs.

So this refines the same synthetic tracks at several worker counts and requires byte-equal
positions AND byte-equal per-track certainty/aperture against the 1-worker run.

    runtime\\python311\\python.exe tools\\check_refine_parallel.py
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pattern_refine as pr    # noqa: E402

# Sized to a real shot, not a toy. At 48 frames / 960x540 the whole run is ~1.4s and thread
# startup dominates -- the check reported 0.53x at 8 workers and looked like a regression.
NF, W, H = 160, 1920, 1080


class Cfg:
    refine_patch_px = 31
    refine_search_px = 24
    refine_search_max = 64
    refine_search_speed_k = 1.5
    match_ambiguity_ratio = 0.90
    refine_ncc_lost = 0.60
    refine_ncc_hold = 0.45
    refine_ncc_reref = 0.68
    refine_ncc_reacquire = 0.75
    refine_motion = "translation"
    refine_min_len = 8
    template_frames = 3
    refine_iterations = 2
    refine_ecc_polish = True
    refine_fb_max_px = 1.5
    refine_gap_aware = True
    reacquire_max_gap = 24
    min_corner_anisotropy = 0.08
    flip_y_for_3de = False
    host_ram_frac = 0.5
    refine_bandpass = 0.0
    refine_workers = 1
    refine_pyramid = False


class _Src:
    """Minimal BGR FrameSource stand-in: refine_tracks wraps it in _GrayFromBGR."""

    def __init__(self, frames):
        self.frames = frames
        self._arr = frames

    def get(self, start, count):
        return self.frames[start:start + count]


def build():
    rng = np.random.default_rng(7)
    base = np.clip(cv2.GaussianBlur(rng.normal(128, 55, (H, W)).astype(np.float32), (0, 0), 1.5),
                   0, 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    frames = np.stack([cv2.warpAffine(base, np.float32([[1, 0, 1.7 * t], [0, 1, 0.9 * t]]),
                                      (W, H), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REFLECT) for t in range(NF)])
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray0, maxCorners=24, qualityLevel=0.02,
                                  minDistance=40, blockSize=5).reshape(-1, 2)
    tracks = {}
    for i, (x, y) in enumerate(pts):
        if not (60 < x < W - 140 and 60 < y < H - 120):
            continue
        # Slightly wrong on purpose, so refine has real work to do.
        tracks[f"T{i:03d}"] = [(t + 1, float(x) + 1.7 * t + 0.4, float(y) + 0.9 * t - 0.3)
                               for t in range(NF)]
    return frames, tracks


def run(frames, tracks, workers):
    cfg = Cfg()
    cfg.refine_workers = workers
    t0 = time.time()
    out, _info = pr.refine_tracks(dict(tracks), "", W, H, NF, cfg, status=None,
                                  bgr_source=_Src(frames))
    dt = time.time() - t0
    cert = dict(getattr(pr.refine_tracks, "last_certainty", {}) or {})
    ap = dict(getattr(pr.refine_tracks, "last_aperture", {}) or {})
    return out, cert, ap, dt


def same(a, b):
    if list(a.keys()) != list(b.keys()):
        return False, "track ids or their ORDER differ"
    for k in a:
        pa, pb = a[k], b[k]
        if len(pa) != len(pb):
            return False, f"{k}: {len(pa)} vs {len(pb)} points"
        for u, v in zip(pa, pb):
            if int(u[0]) != int(v[0]) or float(u[1]) != float(v[1]) or float(u[2]) != float(v[2]):
                return False, f"{k} frame {u[0]}: {u[1:]} vs {v[1:]}"
    return True, ""


def main() -> int:
    frames, tracks = build()
    print(f"parallel refine self-check -- {len(tracks)} tracks x {NF} frames\n")
    bad = 0
    base_out, base_cert, base_ap, base_dt = run(frames, tracks, 1)
    print(f"  1 worker : {len(base_out)} tracks out, {base_dt:.1f}s  (reference)")

    for w in (4, 8, 16):
        out, cert, ap, dt = run(frames, tracks, w)
        ok, why = same(base_out, out)
        speed = base_dt / dt if dt > 0 else float("nan")
        print(f"  {w} workers: {len(out)} tracks out, {dt:.1f}s  ({speed:.2f}x)"
              + ("" if ok else f"   MISMATCH: {why}"))
        if not ok:
            bad += 1
            continue
        for label, r, b in (("certainty", cert, base_cert), ("aperture", ap, base_ap)):
            for k in b:
                x, y = r.get(k, None), b[k]
                if x is None or (x == x) != (y == y) or (x == x and abs(x - y) > 1e-12):
                    print(f"      BAD {label} for {k}: {x} vs {y}")
                    bad += 1
                    break

    print("\n" + ("all checks passed -- output is identical at every worker count"
                  if not bad else f"{bad} check(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
