# -*- coding: utf-8 -*-
"""Self-check of cross-pass fusion (`fuse_passes`) on constructed tracks.

Pure Python + numpy, no GPU, seconds to run. Feed `track_filter.stitch_passes` two estimates
of ONE known trajectory whose errors grow from opposite ends -- which is what FWD and BWD
actually are -- and check three things the feature depends on:

  1. fusing beats either input, and beats a plain unweighted average
  2. the weighting uses the right END of a backward track. A BWD pass is seeded at the last
     frame, so its error grows toward frame 0; labelling it FWD inverts every weight. That
     mistake cannot fail loudly in production -- it just quietly makes the answer worse --
     so it is asserted here instead.
  3. fusion refuses a partner that is a DIFFERENT feature sitting nearby

    runtime\\python311\\python.exe tools\\check_fusion.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import track_filter as tf     # noqa: E402

T = 200
DIAG = 2000.0


class Cfg:
    stitch_passes = True
    stitch_max_sep_px = 1.5
    stitch_min_overlap = 4
    stitch_max_gap = 2
    fuse_passes = True
    fuse_age_tau = 30.0
    fuse_max_sep_px = 0.75
    w_coverage, w_smoothness, w_stability = 0.5, 0.3, 0.2


def truth():
    f = np.arange(1, T + 1, dtype=np.float64)
    return f, 500.0 + 3.0 * f + 40.0 * np.sin(f / 23.0), 400.0 + 1.2 * f


def noisy(frames, gx, gy, seed_frame, sigma_per_100f, rng):
    """Error growing with distance from this pass's own query frame (a random walk)."""
    age = np.abs(frames - seed_frame)
    step = sigma_per_100f / 10.0
    ex = np.cumsum(rng.normal(0, step, len(frames)))
    ey = np.cumsum(rng.normal(0, step, len(frames)))
    if seed_frame > frames[0]:              # backward pass: error accumulates toward frame 0
        ex, ey = ex[::-1] - ex[-1], ey[::-1] - ey[-1]
    scale = np.sqrt(np.maximum(age, 0.0) / max(1.0, age.max()))
    return gx + ex * scale, gy + ey * scale


def rms(px, py, gx, gy):
    return float(np.sqrt(np.mean((px - gx) ** 2 + (py - gy) ** 2)))


def run(cfg, cand):
    out = tf.stitch_passes([dict(c) for c in cand], T, DIAG, cfg, log=None)
    return out


def as_cand(name, pas, frames, xs, ys, score):
    pts = [(int(f), float(x), float(y)) for f, x, y in zip(frames, xs, ys)]
    return {"id": name, "pass": pas, "pts": pts,
            "mean": (float(np.mean(xs)), float(np.mean(ys))), "score": score}


def series(out, gx, gy, frames):
    """RMS of the single surviving track against truth, over the frames it covers."""
    assert len(out) == 1, f"expected one fused track, got {len(out)}"
    pm = {f: (x, y) for f, x, y in out[0]["pts"]}
    idx = [i for i, f in enumerate(frames) if int(f) in pm]
    px = np.array([pm[int(frames[i])][0] for i in idx])
    py = np.array([pm[int(frames[i])][1] for i in idx])
    return rms(px, py, gx[idx], gy[idx])


def main() -> int:
    rng = np.random.default_rng(11)
    frames, gx, gy = truth()
    fx, fy = noisy(frames, gx, gy, frames[0], 0.35, rng)    # FWD: seeded frame 1
    bx, by = noisy(frames, gx, gy, frames[-1], 0.35, rng)   # BWD: seeded frame T

    e_f, e_b = rms(fx, fy, gx, gy), rms(bx, by, gx, gy)
    bad = 0

    def check(ok, label, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'}  {label}" + (f"   {detail}" if detail else ""))

    print("cross-pass fusion self-check\n")
    print(f"  inputs: FWD rms {e_f:.4f}px   BWD rms {e_b:.4f}px   (same feature, "
          f"error growing from opposite ends)\n")

    cand = [as_cand("FWD_0001", "FWD", frames, fx, fy, 0.90),
            as_cand("BWD_0001", "BWD", frames, bx, by, 0.80)]

    # 1. fusion beats both inputs
    cfg = Cfg()
    e_fused = series(run(cfg, cand), gx, gy, frames)
    check(e_fused < min(e_f, e_b), "age-weighted fusion beats either pass alone",
          f"fused {e_fused:.4f}px vs best input {min(e_f, e_b):.4f}px")

    # ... and beats a flat average (tau huge -> all weights equal)
    cfg_flat = Cfg(); cfg_flat.fuse_age_tau = 1e9
    e_flat = series(run(cfg_flat, cand), gx, gy, frames)
    check(e_fused < e_flat, "age weighting beats an unweighted mean",
          f"weighted {e_fused:.4f}px vs flat {e_flat:.4f}px")

    # 2. the backward pass must be seeded at its LAST frame
    cand_mislabelled = [as_cand("FWD_0001", "FWD", frames, fx, fy, 0.90),
                        as_cand("FWD_0002", "FWD", frames, bx, by, 0.80)]
    e_wrong = series(run(cfg, cand_mislabelled), gx, gy, frames)
    check(e_fused < e_wrong, "BWD seed-end is read from the pass name, not the first frame",
          f"correct {e_fused:.4f}px vs BWD-labelled-FWD {e_wrong:.4f}px")

    # 3. off is byte-identical to the old join (primary untouched where it has points)
    cfg_off = Cfg(); cfg_off.fuse_passes = False
    off = {c["id"]: c for c in run(cfg_off, cand)}
    same = len(off) == len(cand) and all(
        abs(a[1] - b[1]) < 1e-12 and abs(a[2] - b[2]) < 1e-12
        for c in cand for a, b in zip(off.get(c["id"], {"pts": []})["pts"], c["pts"]))
    check(same, "fuse_passes=False leaves the old join bit-for-bit",
          f"{len(off)} candidate(s) out, both unchanged={same}")

    # 4. a different feature 1.2px away is joinable-but-not-fusable, so it must not drag
    ox, oy = bx + 1.2, by + 0.0
    cand_other = [as_cand("FWD_0001", "FWD", frames, fx, fy, 0.90),
                  as_cand("BWD_0001", "BWD", frames, ox, oy, 0.80)]
    out_other = run(cfg, cand_other)
    prim = [c for c in out_other if c["id"] == "FWD_0001"]
    untouched = bool(prim) and all(
        abs(p[1] - x) < 1e-12 for p, x in zip(prim[0]["pts"], fx))
    check(untouched, "a neighbour beyond fuse_max_sep_px is never averaged in",
          f"separation 1.20px > {cfg.fuse_max_sep_px}px")

    print("\n" + ("all checks passed" if not bad else f"{bad} check(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
