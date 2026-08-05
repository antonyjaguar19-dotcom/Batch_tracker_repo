# -*- coding: utf-8 -*-
"""Self-check for the per-track policy plumbing. No GPU, no plate, no model.

Most of what per-track policy does is visible in the numbers: a wrong patch size shows up
in eval_refs. One part is not. If the seed measurements ever slip out of step with the track
columns, every track quietly gets its NEIGHBOUR's settings -- the run succeeds, the report
looks reasonable, and the numbers are subtly wrong with nothing pointing at the cause. That
is what this checks, on synthetic frames, in a couple of seconds.

    python tools/check_per_track.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.tracker_core as tc                                      # noqa: E402
from app import track_filter as _tf                                # noqa: E402
from app.track_meta import (SeedFeat, TrackMeta, TrackRegistry,     # noqa: E402
                            classify_seed, policy_for)

FAIL: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")
    if not cond:
        FAIL.append(name)


def _plate() -> np.ndarray:
    """A frame carrying all four seed classes, so the classifier has something to separate."""
    H, W = 720, 1280
    img = np.full((H, W, 3), 45, np.uint8)
    for gy in range(4):                                  # small sharp corners
        for gx in range(8):
            if (gx + gy) % 2 == 0:
                cv2.rectangle(img, (40 + gx * 40, 30 + gy * 36),
                              (40 + gx * 40 + 40, 30 + gy * 36 + 36), (215, 215, 215), -1)
    for k in range(14):                                  # large soft blobs
        cv2.circle(img, (110 + k * 80, 430), 26, (150, 140, 125), -1)
    for k in range(6):                                   # mild diagonal edges
        cv2.line(img, (0, 560 + k * 22), (W, 600 + k * 22), (200, 200, 200), 3)
    for gy in range(3):                                  # repetitive rivet grid
        for gx in range(24):
            cv2.circle(img, (30 + gx * 52, 660 + gy * 18), 4, (225, 225, 225), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img[430 - 26:430 + 26, :] = cv2.GaussianBlur(img[430 - 26:430 + 26, :], (0, 0), 3.0)
    return img


def test_classifier() -> None:
    print("seed classification")
    frames = np.stack([_plate()] * 8, axis=0)
    cfg = tc.RunnerConfig(input_dir=".", output_dir=".", per_track_policy=True,
                          max_tracks=600, seed_stagger=1, feature_quality=0.006,
                          min_feature_dist=8)
    q, seeds = tc.BatchTrackerRunner(cfg)._staggered_queries(
        frames, None, cfg.max_tracks, frames.shape[0])
    check("seeds are index-aligned with query rows",
          q is not None and len(seeds) == q.shape[1])

    kinds = sorted({k for _f, k in seeds})
    check("more than one class found", len(kinds) > 1, f"got {kinds}")

    pols = {k: policy_for(k, f, cfg) for f, k in seeds}
    boxes = sorted({p.get("refine_patch_px") for p in pols.values() if p.get("refine_patch_px")})
    check("classes get different pattern boxes", len(boxes) > 1, f"got {boxes}")
    check("every pattern box is odd", all(b % 2 == 1 for b in boxes), f"got {boxes}")
    # A saturated density means the rival-peak branch is measuring nothing -- the failure
    # mode that made this branch dead code the first time round.
    dens = [f.local_density for f, _k in seeds]
    check("rival density is a measurement, not a constant",
          len(set(np.round(dens, 3))) > 1 and max(dens) < 0.99,
          f"min={min(dens):.3f} max={max(dens):.3f}")
    print(f"        classes: {', '.join(kinds)}   boxes: {boxes}")


def test_alignment() -> None:
    """Columns that die must not shift the surviving tracks onto other seeds' measurements."""
    print("seed -> track alignment through filtering")
    T, N = 40, 10
    cfg = tc.RunnerConfig(input_dir=".", output_dir=".", per_track_policy=True,
                          enable_filtering=False, enable_spread_select=False,
                          smooth_window=1, min_track_points=8, flip_y_for_3de=False,
                          min_track_frames=0, min_track_score=0.0)
    r = tc.BatchTrackerRunner(cfg)
    K = ["corner", "blob", "edge"] * 3 + ["corner"]
    r._pass_seeds["FWD"] = [
        (SeedFeat(lmin=100, lmax=110, aniso=0.9, scale_px=5.0), k) for k in K]

    xy = np.zeros((T, N, 2), np.float32)
    vis = np.ones((T, N), bool)
    for t in range(T):
        for j in range(N):
            xy[t, j] = (100.0 + j * 50 + t * 0.7, 200.0 + j * 30 + t * 0.4)
    vis[:, 2] = False        # column dies outright
    vis[3:, 5] = False       # column too short to survive
    tracks, kept, *_ = r._merge_filter_export(
        [("FWD", xy, vis, np.ones((N,), bool))], T, 1920, 1080, 2202.0, 1.0)

    check("dead columns were dropped", "FWD_0003" not in tracks and "FWD_0006" not in tracks)
    bad = [t for t in tracks
           if (r.registry.get(t).kind if r.registry.get(t) else None) != K[int(t.split("_")[1]) - 1]]
    check("every surviving track kept its OWN seed's class", not bad, f"misaligned: {bad}")
    print(f"        {kept}/{N} columns exported, none misattributed")


def test_splits() -> None:
    """A split piece is the same feature, so it must keep the same policy."""
    print("id splits carry the policy")

    class Cfg:
        max_track_gaps, min_occlusion_run, refine_min_len = 2, 3, 4
        w_coverage, w_smoothness, w_stability = 0.5, 0.3, 0.2

    reg = TrackRegistry(enabled=True)
    reg.register("FWD_0001", TrackMeta(kind="blob", policy={"refine_patch_px": 49}))
    pts, f = [], 1
    for _run in range(4):
        pts += [(f + k, 100.0 + k, 200.0) for k in range(6)]
        f += 8
    out = _tf.defragment({"FWD_0001": pts}, Cfg(), registry=reg)
    check("track was split into runs", len(out) > 1, f"got {sorted(out)}")
    check("every piece kept its class and policy",
          all((reg.get(t) or TrackMeta()).policy.get("refine_patch_px") == 49 for t in out))
    check("no piece needed the parent fallback", reg.orphans == 0,
          f"{reg.orphans} lookups fell back")


def test_view() -> None:
    """The whole mechanism is one config substitution; it must be exact in both directions."""
    print("per-track config view")
    base = tc.RunnerConfig(input_dir=".", output_dir=".")
    reg = TrackRegistry(enabled=True)
    reg.register("A", TrackMeta(kind="blob", policy={"refine_patch_px": 49}))
    reg.register("B", TrackMeta(kind="corner"))
    v = reg.view("A", base)
    check("override wins", v.refine_patch_px == 49)
    check("everything else reads through", v.refine_ncc_lost == base.refine_ncc_lost)
    check("unknown names still hit their default", getattr(v, "nope", "d") == "d")
    check("a write cannot leak into the shot config",
          (setattr(v, "refine_iterations", 9) or base.refine_iterations) != 9)
    check("an untuned track gets the shot config ITSELF", reg.view("B", base) is base)
    check("disabled registry always returns the shot config",
          TrackRegistry(enabled=False).view("A", base) is base)


if __name__ == "__main__":
    for fn in (test_view, test_classifier, test_alignment, test_splits):
        fn()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
        raise SystemExit(1)
    print("all checks passed")
