# -*- coding: utf-8 -*-
"""Track-replenishment (re-seeding) test on a fast low-angle shot (ISOLATED experiment).

Problem: the tracker seeds ONCE at frame 0. On a fast low-angle shot the close foreground
has huge screen parallax, so the frame-0 FG seeds sweep out of frame in a few frames and
nothing re-seeds the FG that enters afterward -> FG goes untracked for most of the clip.

Cumulative fixes, measured by FG-band coverage over time (tracks alive in the bottom third):
  V0  current: single seed at frame 0
  V1  + periodic re-seeding at keyframes across the clip
  V2  + coverage-driven: seed only where there are no live tracks (fill gaps)
  V3  + FG-biased: bias re-seed to the FG band (bottom) where lifetimes are shortest
  V4  + dense spacing: let short-lived FG tracks pack closer
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


def decode(path, count):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fr = []
    while len(fr) < count:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    return np.asarray(fr), float(fps)


def seed_masked(frame_bgr, rp, max_pts, mask=None, margin=60, min_eig=2.0):
    """goodFeatures+cornerSubPix, min-eigen gated, optionally restricted by `mask` (H,W bool)."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    m = np.zeros((H, W), np.uint8); m[margin:H - margin, margin:W - margin] = 255
    if mask is not None:
        m = cv2.bitwise_and(m, (mask.astype(np.uint8) * 255))
    pts = cv2.goodFeaturesToTrack(g, max_pts * 3, 0.02, 24, mask=m, blockSize=5)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    cv2.cornerSubPix(g, pts, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    pts = pts.reshape(-1, 2).astype(np.float32)
    good = [p for p in pts if (rp._extract(g, p[0], p[1], 7) is not None
                               and rp._contrast_score(rp._extract(g, p[0], p[1], 7)) >= min_eig)]
    pts = np.array(good, np.float32) if good else pts
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts


def track_batch(engine, mt, rp, br, frames, grays, kf, seeds, window=0, full_refine=False,
                blur_on=True):
    """Track `seeds` (on frame kf) forward over frames[kf:kf+window] (0=to end), return
    list of tracks with ABSOLUTE 1-based frames. full_refine = FULL STACK (moving-tile +
    blur-bank NCC + edge + gap-aware); default fast path = baseline track (coverage only)."""
    if seeds.shape[0] == 0:
        return []
    end = frames.shape[0] if window <= 0 else min(frames.shape[0], kf + window)
    sub = frames[kf:end]
    q = np.zeros((1, seeds.shape[0], 3), np.float32); q[0, :, 1] = seeds[:, 0]; q[0, :, 2] = seeds[:, 1]
    bl, blv = engine.track_queries(sub, q)
    if full_refine:
        gd, gdv = mt.moving_tile_track_guided(engine, sub, seeds, bl, blv)
        tracks = br.refine_all_full(rp.arrays_to_tracks(gd, gdv), grays[kf:end],
                                    half=15, search=28, blur_est="bank",
                                    blur_match=blur_on, adaptive_thresh=blur_on,
                                    sharp_anchor=True, edge_clamp=True, min_len=6)
    else:
        tracks = rp.arrays_to_tracks(bl, blv)      # coarse but fine for coverage counting
        tracks = [t for t in tracks if len(t["f"]) >= 6]
    return [{"f": t["f"] + kf, "x": t["x"], "y": t["y"]} for t in tracks]


def alive_at(tracks, t):
    """positions of tracks alive at 1-based frame t -> list (x,y)."""
    out = []
    for tr in tracks:
        idx = np.where(tr["f"] == t)[0]
        if len(idx):
            out.append((tr["x"][idx[0]], tr["y"][idx[0]]))
    return out


def dedup(tracks, sp):
    """Drop a track that duplicates a longer one: >50% frame overlap AND < sp px apart."""
    order = sorted(range(len(tracks)), key=lambda i: -len(tracks[i]["f"]))
    kept = []
    kf = [None] * len(tracks)
    for i in order:
        ti = tracks[i]; fi = {int(f): k for k, f in enumerate(ti["f"])}
        dup = False
        for j in kept:
            tj = tracks[j]; fj = {int(f): k for k, f in enumerate(tj["f"])}
            common = set(fi) & set(fj)
            if len(common) > 0.5 * min(len(fi), len(fj)):
                d = np.mean([math.hypot(ti["x"][fi[c]] - tj["x"][fj[c]],
                                        ti["y"][fi[c]] - tj["y"][fj[c]]) for c in common])
                if d < sp:
                    dup = True; break
        if not dup:
            kept.append(i)
    return [tracks[i] for i in kept]


def run_variant(engine, mt, rp, br, frames, grays, T, H, W, max_pts,
                periodic, cov_mask, fg_bias, dense, every=40, cover_r=44, window=0,
                full_refine=False, blur_on=True):
    keyframes = [0] if not periodic else list(range(0, T - 6, every))
    win = 0 if not periodic else max(window, int(every * 2.5))  # bounded window when re-seeding
    acc = []
    for kf in keyframes:
        mask = None
        if kf > 0 and (cov_mask or fg_bias):
            mask = np.ones((H, W), bool)
            if cov_mask:                       # exclude disks around tracks alive at kf
                cov = np.zeros((H, W), np.uint8)
                for (x, y) in alive_at(acc, kf + 1):
                    cv2.circle(cov, (int(x), int(y)), cover_r, 255, -1)
                mask &= (cov == 0)
            if fg_bias:                        # keep only the FG band (bottom 60%)
                band = np.zeros((H, W), bool); band[int(0.4 * H):, :] = True
                mask &= band
        seeds = seed_masked(frames[kf], rp, max_pts, mask=mask)
        acc.extend(track_batch(engine, mt, rp, br, frames, grays, kf, seeds,
                               window=win, full_refine=full_refine, blur_on=blur_on))
    sp = 14 if dense else 26
    acc = dedup(acc, sp)
    return acc


def coverage(tracks, T, H):
    total = np.zeros(T, int); fg = np.zeros(T, int)
    for tr in tracks:
        for k, f in enumerate(tr["f"]):
            if 1 <= f <= T:
                total[f - 1] += 1
                if tr["y"][k] > (2.0 / 3.0) * H:
                    fg[f - 1] += 1
    return total, fg


def plot_curves(curves, T, out, title):
    Wc, Hc, pad = 1000, 420, 50
    img = np.full((Hc, Wc, 3), 255, np.uint8)
    ymax = max(1, max(c.max() for _, c, _ in curves))
    def X(t): return pad + int(t / max(1, T - 1) * (Wc - 2 * pad))
    def Y(v): return Hc - pad - int(v / ymax * (Hc - 2 * pad))
    cv2.line(img, (pad, Hc - pad), (Wc - pad, Hc - pad), (0, 0, 0), 1)
    cv2.line(img, (pad, pad), (pad, Hc - pad), (0, 0, 0), 1)
    for lbl, c, col in curves:
        pts = [(X(t), Y(c[t])) for t in range(T)]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)
    for i, (lbl, c, col) in enumerate(curves):
        cv2.putText(img, f"{lbl} (mean {c.mean():.1f})", (pad + 12, pad + 22 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(img, title, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"ymax={ymax}", (Wc - 150, Hc - pad + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out, img)


def render_overlay(frames, fps, out, tracks, scale, H):
    T = frames.shape[0]
    ow, oh = int(frames.shape[2] * scale), int(frames.shape[1] * scale)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    by = {}
    for i, tr in enumerate(tracks):
        for k, f in enumerate(tr["f"]):
            by.setdefault(int(f), []).append((tr["x"][k], tr["y"][k]))
    for t in range(T):
        img = cv2.resize(frames[t], (ow, oh), interpolation=cv2.INTER_AREA)
        cv2.line(img, (0, int(2.0 / 3.0 * H * scale)), (ow, int(2.0 / 3.0 * H * scale)), (0, 180, 255), 1)
        for (x, y) in by.get(t + 1, []):
            fg = y > (2.0 / 3.0) * H
            cv2.drawMarker(img, (int(x * scale), int(y * scale)),
                           (0, 0, 255) if fg else (0, 220, 0), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.putText(img, "RED=FG-band track  GREEN=upper   f%d/%d  (V4 re-seeded)" % (t + 1, T),
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="D:/Jefrin/IN/SH013.mp4")
    ap.add_argument("--tag", default="SH013")
    ap.add_argument("--frames", type=int, default=303)
    ap.add_argument("--max-pts", type=int, default=28)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--full", action="store_true",
                    help="FULL STACK (moving-tile+blur-bank+edge+gap-aware) on V0 vs V4 only")
    ap.add_argument("--no-blur", action="store_true", help="full stack WITHOUT blur-bank")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out", "reseed"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Engine, mt, rp, br = load_helpers()
    frames, fps = decode(args.video, args.frames)
    T, H, W = frames.shape[:3]
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], 0)
    print(f"[{args.tag}] {T}f {W}x{H}  full_stack={args.full}", flush=True)
    engine = Engine(tool_root=_REPO, device="cuda")

    if args.full:
        variants = [
            ("V0", dict(periodic=False, cov_mask=False, fg_bias=False, dense=False), "single seed (full stack)"),
            ("V4", dict(periodic=True,  cov_mask=True,  fg_bias=True,  dense=True),  "re-seeded (full stack)"),
        ]
    else:
        variants = [
            ("V0", dict(periodic=False, cov_mask=False, fg_bias=False, dense=False), "single seed"),
            ("V1", dict(periodic=True,  cov_mask=False, fg_bias=False, dense=False), "+periodic"),
            ("V2", dict(periodic=True,  cov_mask=True,  fg_bias=False, dense=False), "+coverage gaps"),
            ("V3", dict(periodic=True,  cov_mask=True,  fg_bias=True,  dense=False), "+FG bias"),
            ("V4", dict(periodic=True,  cov_mask=True,  fg_bias=True,  dense=True),  "+dense spacing"),
        ]
    tag2 = (f"{args.tag}_full" + ("_noblur" if args.no_blur else "")) if args.full else args.tag
    results = {}
    print(f"\n{'var':<4} {'tracks':>6} {'mean/fr':>8} {'FG/fr':>7} {'FGmin':>6}  note", flush=True)
    for name, cfg, note in variants:
        tr = run_variant(engine, mt, rp, br, frames, grays, T, H, W, args.max_pts,
                         every=args.every, full_refine=args.full, blur_on=not args.no_blur, **cfg)
        total, fg = coverage(tr, T, H)
        results[name] = (tr, total, fg)
        print(f"{name:<4} {len(tr):>6} {total.mean():>8.1f} {fg.mean():>7.1f} {int(fg.min()):>6}  {note}", flush=True)

    plot_curves([("V0 single-seed", results["V0"][2], (200, 0, 0)),
                 ("V4 re-seeded", results["V4"][2], (0, 160, 0))],
                T, os.path.join(args.outdir, f"{tag2}_fg_coverage.png"),
                "FG-band tracks alive per frame" + (" (full stack)" if args.full else ""))
    scale = 1920.0 / W if W > 1920 else 1.0
    render_overlay(frames, fps, os.path.join(args.outdir, f"{tag2}_reseed_overlay.mp4"),
                   results["V4"][0], scale, H)
    print(f"[{args.tag}] wrote coverage plot + overlay", flush=True)


if __name__ == "__main__":
    main()
