# -*- coding: utf-8 -*-
"""Controlled synthetic-blur proof (ISOLATED experiment).

SH006/SH007 turned out to be sharp shots (tracked features not motion-blurred), so they
can't show the blur feature's benefit. This injects KNOWN directional motion blur into some
frames of a sharp clip and uses the CLEAN-frame track as pseudo-ground-truth, to prove the
blur-matched refine recovers accuracy on the blurred frames where the sharp NCC drifts.

  reference = moving-tile + NCC on the CLEAN clip           (pseudo-GT)
  CUR       = moving-tile + NCC on the BLURRED clip         (current bot)
  BLUR      = moving-tile + blur-matched refine on BLURRED  (proposed)
Deviation of CUR vs reference and BLUR vs reference, measured on the BLURRED frames only.
"""
from __future__ import annotations
import os, sys, math, argparse, importlib.util
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def load_helpers():
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    def by_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    eng = by_path("btr_tapnext_engine", os.path.join(_REPO, "app", "tapnext_engine.py"))
    mt = by_path("btr_mt", os.path.join(_HERE, "track_moving_tile.py"))
    rp = by_path("btr_rp", os.path.join(_HERE, "refine_pipeline.py"))
    br = by_path("btr_blur", os.path.join(_HERE, "blur_refine.py"))
    return eng.TapNextEngine, mt, rp, br


def decode(path, start, count):
    cap = cv2.VideoCapture(path);
    if start: cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    fr = []
    while len(fr) < count:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    return np.asarray(fr)


def motion_blur(img, length, angle_deg):
    L = int(length);
    if L < 3: return img
    k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((L / 2, L / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (L, L)); k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k)


def dev(pred_by, ref_by, blur_frames):
    errs = []
    for tid, ref in ref_by.items():
        if tid not in pred_by: continue
        p = pred_by[tid]
        for f in blur_frames:
            if f in ref and f in p:
                errs.append(math.hypot(p[f][0] - ref[f][0], p[f][1] - ref[f][1]))
    errs = np.array(errs) if errs else np.array([0.0])
    return dict(mean=float(errs.mean()), med=float(np.median(errs)),
                p90=float(np.percentile(errs, 90)), n=len(errs))


def as_by(tracks):
    return {t["id"]: {int(f): (t["x"][k], t["y"][k]) for k, f in enumerate(t["f"])} for t in tracks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="D:/Jefrin/IN/SH007.mp4")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--blur-len", type=int, default=21)
    ap.add_argument("--every", type=int, default=3, help="blur every Nth frame")
    ap.add_argument("--max-pts", type=int, default=40)
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out", "blur"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Engine, mt, rp, br = load_helpers()
    clean = decode(args.video, args.start, args.frames)
    T, H, W = clean.shape[:3]
    print(f"[synth] clean {T}f {W}x{H}", flush=True)

    # blurred copy: directional blur on every Nth frame, angle from frame-to-frame flow proxy
    blurred = clean.copy()
    blur_frames = []
    for t in range(T):
        if t % args.every == 0 and t > 0:
            blurred[t] = motion_blur(clean[t], args.blur_len, 0.0)   # horizontal smear
            blur_frames.append(t + 1)
    print(f"[synth] blurred {len(blur_frames)} of {T} frames (len={args.blur_len}px horiz)", flush=True)

    grays_c = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in clean], 0)
    grays_b = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in blurred], 0)

    engine = Engine(tool_root=_REPO, device="cuda")
    seeds = rp.seed_corner_subpix(clean[0], max_pts=args.max_pts)
    q = np.zeros((1, seeds.shape[0], 3), np.float32); q[0, :, 1] = seeds[:, 0]; q[0, :, 2] = seeds[:, 1]

    # reference on CLEAN
    blc, blvc = engine.track_queries(clean, q)
    gdc, gdvc = mt.moving_tile_track_guided(engine, clean, seeds, blc, blvc)
    ref = rp.refine_all(rp.arrays_to_tracks(gdc, gdvc), grays_c, half=15, search=28,
                        motion_key="translation", anchor="sharpest", min_len=6)

    # tracks on BLURRED (shared moving-tile), two refines
    blb, blvb = engine.track_queries(blurred, q)
    gdb, gdvb = mt.moving_tile_track_guided(engine, blurred, seeds, blb, blvb)
    mtb = rp.arrays_to_tracks(gdb, gdvb)
    variants = {"CUR": rp.refine_all(mtb, grays_b, half=15, search=28, motion_key="translation",
                                     anchor="sharpest", min_len=6)}
    for est in ("velocity", "cepstral", "bank"):
        variants[est] = br.refine_all_blur(mtb, grays_b, blur_match=True, adaptive_thresh=True,
                                           sharp_anchor=True, blur_est=est)

    ref_by = as_by(ref)
    print("\n==== deviation vs CLEAN reference, on BLURRED frames (px) ====", flush=True)
    base = None
    for name in ("CUR", "velocity", "cepstral", "bank"):
        d = dev(as_by(variants[name]), ref_by, blur_frames)
        if base is None:
            base = d['mean']
        imp = 100.0 * (base - d['mean']) / max(1e-6, base)
        print(f"{name:<9}: mean={d['mean']:.2f} med={d['med']:.2f} p90={d['p90']:.2f} "
              f"n={d['n']}  ({imp:+.0f}% vs CUR)", flush=True)


if __name__ == "__main__":
    main()
