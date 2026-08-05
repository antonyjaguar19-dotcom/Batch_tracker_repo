# -*- coding: utf-8 -*-
"""Score a bench export against exact synthetic ground truth, per feature class.

Every exported track is scored -- not the one or two an artist had time to hand-track --
so a change that helps corners and hurts blobs shows up as two rows moving in opposite
directions instead of averaging out to "no change". That averaging is the specific failure
`tools/eval_refs.py` was written to avoid, and with one reference on disk it cannot.

Ground truth per track: the scene is a known planar warp, so a point seen at q on its seed
frame s is at H[t] @ inv(H[s]) @ q on frame t, exactly. Error is the distance from that.

Metric definitions are shared with the hand-track path wherever they can be: fast / slow /
jerk come straight from `app.compare_tracks.compare`, so a number here means the same thing
as the same-named number in `refs/gt4k/baseline.json`. Two are computed here instead:

  mean_err     mean over the frames AFTER the seed. The seed frame is zero by construction
               (it defines the reference), so including it would flatter every track by 1/N.
  first_step   error on the first frame after the seed -- the divergence introduced by one
               step, with no anchor-picking heuristic needed because the seed IS the anchor.

WHAT THIS CANNOT SEE, stated because a metric that cannot fail is worse than no metric:
ground truth is anchored on each track's OWN seed, so shifting an entire export by a
constant leaves every column unchanged (verified by injecting +1px in x). That is not a gap
in the bench, it is a property of the scene: on a plane, the point 1px from the corner is
itself a valid scene point, and a track that follows it rigidly is a correct observation
that a solve can use. Seeding slightly off the intended corner is only bad because the
neighbouring spot is harder to hold -- and that shows up here as worse mean_err, in the
open. An absolute-placement column was tried (distance from the seed to cornerSubPix's
opinion) and dropped: it reported 5.6px on edge seeds and moved by 0.03px under a real 1px
shift, i.e. it measured cornerSubPix's own instability and nothing about the tracker.

    python bench/score_synth.py bench/synth/lab01 --run base
    python bench/score_synth.py bench/synth/lab01 --run test --baseline base
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.compare_tracks import Track, compare, load_tracks          # noqa: E402
from app.track_meta import SeedFeat, SeedStats, classify_seed       # noqa: E402

_SCALE_BLOCKS = (3, 5, 9, 15)     # tracker_core.BatchTrackerRunner._SCALE_BLOCKS
CLASS_ORDER = ["corner", "blob", "edge", "flat",
               "dense-corner", "dense-blob", "dense-edge"]


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------
def gt_track(H: List[np.ndarray], seed_xy: Tuple[float, float], seed_t: int,
             frames: List[int]) -> Track:
    """True position on every frame for a point seen at `seed_xy` on frame `seed_t` (0-based)."""
    inv = np.linalg.inv(H[seed_t])
    p = inv @ np.array([seed_xy[0], seed_xy[1], 1.0])
    p /= p[2]
    out: Track = {}
    for t in frames:
        r = H[t] @ p
        r /= r[2]
        out[t + 1] = (float(r[0]), float(r[1]))      # export frames are 1-based
    return out


# --------------------------------------------------------------------------------------
# Seed classification -- the repo's own classifier, fed the repo's own measurements
# --------------------------------------------------------------------------------------
def _measure(gray: np.ndarray, pts: np.ndarray) -> Tuple[List[SeedFeat], np.ndarray, np.ndarray]:
    """SeedFeat per point, replicating tracker_core._seed_measurements."""
    h, w = gray.shape[:2]
    xi = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, h - 1)

    eig = cv2.cornerEigenValsAndVecs(gray, blockSize=5, ksize=3)
    l1, l2 = np.abs(eig[:, :, 0]), np.abs(eig[:, :, 1])
    lmin_map, lmax_map = np.minimum(l1, l2), np.maximum(l1, l2)
    lmin, lmax = lmin_map[yi, xi], lmax_map[yi, xi]
    theta = np.arctan2(eig[yi, xi, 3], eig[yi, xi, 2])

    resp = np.stack([cv2.cornerMinEigenVal(gray, blockSize=b, ksize=3)[yi, xi]
                     for b in _SCALE_BLOCKS], axis=1)
    scale_px = np.asarray(_SCALE_BLOCKS, dtype=np.float32)[np.argmax(resp, axis=1)]

    thr = float(np.percentile(lmin_map, 99.0))
    if thr <= 0.0:
        thr = float(lmin_map.max()) * 0.05
    strong = (lmin_map >= max(thr, 1e-12)).astype(np.float32)
    density = cv2.boxFilter(strong, -1, (65, 65), normalize=True)[yi, xi]

    g32 = gray.astype(np.float32)
    m = cv2.boxFilter(g32, -1, (33, 33), normalize=True)
    m2 = cv2.boxFilter(g32 * g32, -1, (33, 33), normalize=True)
    contrast = np.sqrt(np.maximum(0.0, m2 - m * m))[yi, xi]

    ratio = np.divide(lmin, lmax, out=np.zeros_like(lmax), where=lmax > 1e-12)
    feats = [SeedFeat(lmin=float(lmin[i]), lmax=float(lmax[i]), aniso=float(ratio[i]),
                      theta=float(theta[i]), scale_px=float(scale_px[i]),
                      local_density=float(density[i]), local_contrast=float(contrast[i]))
             for i in range(pts.shape[0])]
    return feats, lmin, np.asarray([f.local_density for f in feats], dtype=np.float32)


def classify(shot_dir: str, seeds_by_frame: Dict[int, List[Tuple[str, float, float]]],
             min_aniso: float, seed_count: int, quality: float, min_dist: int
             ) -> Dict[str, str]:
    """Label each track by what its seed sits on.

    Percentile stats come from a fresh detection on the seed frame with the production
    seeding parameters, not from the exported seeds alone: `classify_seed` compares a seed
    against its FRAME's distribution, and the exported tracks are a filtered, spread-thinned
    subset whose distribution is not the frame's.
    """
    plate = os.path.join(shot_dir, "plate")
    files = sorted(f for f in os.listdir(plate) if f.lower().endswith((".png", ".jpg", ".exr")))
    out: Dict[str, str] = {}
    for t0, seeds in sorted(seeds_by_frame.items()):
        if t0 >= len(files):
            continue
        gray = cv2.imread(os.path.join(plate, files[t0]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        det = cv2.goodFeaturesToTrack(gray, maxCorners=seed_count, qualityLevel=quality,
                                      minDistance=min_dist, blockSize=5)
        if det is None or len(det) == 0:
            continue
        pop_feats, pop_lmin, pop_dens = _measure(gray, det.reshape(-1, 2).astype(np.float32))
        pop_scale = np.asarray([f.scale_px for f in pop_feats], dtype=np.float32)
        stats = SeedStats(
            lmin_p25=float(np.percentile(pop_lmin, 25.0)),
            lmin_p75=float(np.percentile(pop_lmin, 75.0)),
            density_p90=float(np.percentile(pop_dens, 90.0)),
            scale_med=float(np.median(pop_scale)),
        )
        pts = np.asarray([[x, y] for _, x, y in seeds], dtype=np.float32)
        feats, _, _ = _measure(gray, pts)
        for (name, _x, _y), fe in zip(seeds, feats):
            out[name] = classify_seed(fe, stats, min_aniso)
    return out


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_run(shot_dir: str, bot_path: str, min_aniso: float, seed_count: int,
              quality: float, min_dist: int) -> Dict[str, dict]:
    gt = json.load(open(os.path.join(shot_dir, "gt.json"), "r", encoding="utf-8"))
    H = [np.asarray(m, dtype=np.float64) for m in gt["H"]]
    H0 = int(gt["height"])

    raw = load_tracks(bot_path)
    # Production exports 3DE-flipped y (tracker_core: y = (H0-1) - y). Back to image space.
    tracks = {n: {f: (x, float(H0 - 1) - y) for f, (x, y) in pts.items()} for n, pts in raw.items()}

    seeds_by_frame: Dict[int, List[Tuple[str, float, float]]] = {}
    for name, pts in tracks.items():
        f0 = min(pts)
        seeds_by_frame.setdefault(f0 - 1, []).append((name, pts[f0][0], pts[f0][1]))
    kinds = classify(shot_dir, seeds_by_frame, min_aniso, seed_count, quality, min_dist)

    results: Dict[str, dict] = {}
    for name, pts in tracks.items():
        frames = sorted(pts)
        if len(frames) < 4:
            continue
        s = frames[0] - 1
        if s >= len(H):
            continue
        ref = gt_track(H, pts[frames[0]], s, [f - 1 for f in frames if f - 1 < len(H)])
        common = sorted(set(pts) & set(ref))
        if len(common) < 4:
            continue
        errs = [math.hypot(pts[f][0] - ref[f][0], pts[f][1] - ref[f][1]) for f in common]
        after = errs[1:]                        # seed frame is 0 by construction
        r = compare(pts, ref, out=io.StringIO())   # fast / slow / jerk, shared definitions
        results[name] = {
            "kind": kinds.get(name, "?"),
            "n": len(common),
            "mean_err": st.mean(after),
            "max_err": max(after),
            "first_step": after[0],
            "end_drift": after[-1],
            "fast": r.get("fast_ratio", float("nan")),
            "slow": r.get("slow_ratio", float("nan")),
            "jerk": r.get("jerk_ratio", float("nan")),
        }
    return results


def _agg(rows: List[dict], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key]) and not math.isinf(r[key])]
    return st.median(vals) if vals else float("nan")


def _delta(now: float, base: Optional[float]) -> str:
    if base is None or math.isnan(now) or math.isnan(base):
        return ""
    d = now - base
    if abs(d) < 0.005:
        return "    ="
    return f" {d:+5.2f}"


def report(res: Dict[str, dict], base: Optional[Dict[str, dict]] = None) -> None:
    by_kind: Dict[str, List[dict]] = {}
    for r in res.values():
        by_kind.setdefault(r["kind"], []).append(r)
    base_by_kind: Dict[str, List[dict]] = {}
    for r in (base or {}).values():
        base_by_kind.setdefault(r["kind"], []).append(r)

    print(f"\n{len(res)} tracks scored"
          + (f"   (baseline: {len(base)})" if base else ""))
    print("\nclass          n   mean_err        first_step       fast    slow    jerk")
    print("-" * 88)
    order = [k for k in CLASS_ORDER if k in by_kind] + \
            [k for k in sorted(by_kind) if k not in CLASS_ORDER]
    for kind in order:
        rows = by_kind[kind]
        brows = base_by_kind.get(kind)
        def cell(key, w=6):
            now = _agg(rows, key)
            b = _agg(brows, key) if brows else None
            return f"{now:{w}.2f}{_delta(now, b)}"
        print(f"{kind:<12} {len(rows):>3}   {cell('mean_err')}   {cell('first_step')}   "
              f"{_agg(rows, 'fast'):.2f}   {_agg(rows, 'slow'):.2f}   {_agg(rows, 'jerk'):.2f}")

    allr = list(res.values())
    allb = list(base.values()) if base else None
    print("-" * 76)
    def tot(key):
        now = _agg(allr, key)
        b = _agg(allb, key) if allb else None
        return f"{now:6.2f}{_delta(now, b)}"
    print(f"{'ALL':<12} {len(allr):>3}   {tot('mean_err')}   {tot('first_step')}   "
          f"{_agg(allr, 'fast'):.2f}   {_agg(allr, 'slow'):.2f}   {_agg(allr, 'jerk'):.2f}")

    # The goal is quality on EVERY track, so the tail is the number that matters, not the
    # median: a batch is judged by the worst tracks an artist has to fix by hand.
    errs = sorted(r["mean_err"] for r in allr)
    if errs:
        p90 = errs[int(0.9 * (len(errs) - 1))]
        print(f"\nworst tail   p90 mean_err {p90:.2f}px   worst track {errs[-1]:.2f}px"
              f"   tracks over 1px: {sum(1 for e in errs if e > 1.0)}/{len(errs)}")
        if allb:
            be = sorted(r["mean_err"] for r in allb)
            bp90 = be[int(0.9 * (len(be) - 1))]
            print(f"baseline     p90 mean_err {bp90:.2f}px   worst track {be[-1]:.2f}px"
                  f"   tracks over 1px: {sum(1 for e in be if e > 1.0)}/{len(be)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shot", help="bench shot folder")
    ap.add_argument("--run", default="base", help="run tag under <shot>/runs/")
    ap.add_argument("--baseline", default=None, help="run tag to diff against")
    ap.add_argument("--min-aniso", type=float, default=0.08)
    ap.add_argument("--seed-count", type=int, default=1200)
    ap.add_argument("--quality", type=float, default=0.02)
    ap.add_argument("--min-dist", type=int, default=12)
    ap.add_argument("--csv", default=None, help="write per-track rows here")
    a = ap.parse_args()

    shot = os.path.basename(os.path.normpath(a.shot))
    def path_for(tag):
        return os.path.join(a.shot, "runs", tag, f"{shot}__tapnext.txt")

    res = score_run(a.shot, path_for(a.run), a.min_aniso, a.seed_count, a.quality, a.min_dist)
    base = None
    if a.baseline:
        base = score_run(a.shot, path_for(a.baseline), a.min_aniso, a.seed_count,
                         a.quality, a.min_dist)
    print(f"run '{a.run}'" + (f" vs baseline '{a.baseline}'" if a.baseline else ""))
    report(res, base)

    if a.csv:
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            f.write("track,kind,n,mean_err,max_err,first_step,end_drift,fast,slow,jerk\n")
            for n, r in sorted(res.items(), key=lambda kv: -kv[1]["mean_err"]):
                f.write(f"{n},{r['kind']},{r['n']},{r['mean_err']:.4f},{r['max_err']:.4f},"
                        f"{r['first_step']:.4f},{r['end_drift']:.4f},{r['fast']:.4f},"
                        f"{r['slow']:.4f},{r['jerk']:.4f}\n")
        print(f"\nper-track rows -> {a.csv}")


if __name__ == "__main__":
    main()
