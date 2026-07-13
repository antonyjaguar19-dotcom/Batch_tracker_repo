# -*- coding: utf-8 -*-
"""Motion-blur refine comparison vs the current bot (ISOLATED experiment).

Same coarse moving-tile positions for every variant; only the NCC refine changes, so the
comparison isolates the blur features. No manual ground truth for these shots, so quality is
judged by: points KEPT (blur should retain valid-but-soft frames the sharp NCC trims), jitter,
track count, and a visual overlay (current=GREEN vs blur=MAGENTA).

  CUR  current bot combo  (moving-tile + NCC sharp, contrast anchor)
  B1   + blur-matched template
  B2   + blur-adaptive thresholds
  B3   + least-blurred (Laplacian) anchor
"""
from __future__ import annotations
import os, sys, argparse, importlib.util
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
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    frames = []
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.asarray(frames), float(fps)


def to_dict(tracks, names=None):
    return {t["id"]: t for t in tracks}


def stats(tracks, grays, rp):
    if not tracks:
        return dict(N=0, pts=0, jit=float("nan"))
    pts = int(sum(len(t["f"]) for t in tracks))
    jit = float(np.nanmean([rp.m_jitter(t) for t in tracks]))
    return dict(N=len(tracks), pts=pts, jit=jit)


def render(frames, fps, out, cur, blur, scale, trail=14):
    T, H, W = frames.shape[:3]
    ow, oh = int(W * scale), int(H * scale)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    cur_by = {t["id"]: {int(f): (t["x"][k], t["y"][k]) for k, f in enumerate(t["f"])} for t in cur}
    blr_by = {t["id"]: {int(f): (t["x"][k], t["y"][k]) for k, f in enumerate(t["f"])} for t in blur}
    for t in range(T):
        img = cv2.resize(frames[t], (ow, oh), interpolation=cv2.INTER_AREA)
        f = t + 1
        for d, col in ((cur_by, (0, 220, 0)), (blr_by, (220, 0, 220))):
            for tid, m in d.items():
                if f in m:
                    x, y = m[f]
                    cv2.drawMarker(img, (int(x * scale), int(y * scale)), col,
                                   cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)
        cv2.putText(img, "GREEN=current bot   MAGENTA=blur-enhanced   f%d/%d" % (f, T),
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--max-pts", type=int, default=40)
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out", "blur"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Engine, mt, rp, br = load_helpers()
    frames, fps = decode(args.video, args.start, args.frames)
    T, H, W = frames.shape[:3]
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], 0)
    print(f"[{args.tag}] {T}f {W}x{H}", flush=True)

    engine = Engine(tool_root=_REPO, device="cuda")
    seeds = rp.seed_corner_subpix(frames[0], max_pts=args.max_pts)
    print(f"[{args.tag}] seeds {seeds.shape[0]}", flush=True)

    q = np.zeros((1, seeds.shape[0], 3), np.float32); q[0, :, 1] = seeds[:, 0]; q[0, :, 2] = seeds[:, 1]
    bl, blv = engine.track_queries(frames, q)
    gd, gdv = mt.moving_tile_track_guided(engine, frames, seeds, bl, blv)
    mt_tracks = rp.arrays_to_tracks(gd, gdv)
    print(f"[{args.tag}] moving-tile tracks {len(mt_tracks)}", flush=True)

    variants = {}
    variants["CUR"] = rp.refine_all(mt_tracks, grays, half=15, search=28,
                                    motion_key="translation", anchor="sharpest", min_len=6)
    for est in ("velocity", "cepstral", "bank"):
        variants[est] = br.refine_all_blur(mt_tracks, grays, blur_match=True,
                                           adaptive_thresh=True, sharp_anchor=True, blur_est=est)

    note = {"CUR": "current bot", "velocity": "velocity est", "cepstral": "cepstral est", "bank": "kernel bank"}
    order = ("CUR", "velocity", "cepstral", "bank")
    print(f"\n==== {args.tag}  (pts=frames kept; higher=more soft frames retained) ====", flush=True)
    print(f"{'var':<9} {'N':>3} {'pts':>6} {'jit(px)':>8}  note", flush=True)
    for k in order:
        s = stats(variants[k], grays, rp)
        print(f"{k:<9} {s['N']:>3} {s['pts']:>6} {s['jit']:>8.3f}  {note[k]}", flush=True)

    scale = 1920.0 / W if W > 1920 else 1.0
    out = os.path.join(args.outdir, f"{args.tag}_blur_compare.mp4")
    render(frames, fps, out, variants["CUR"], variants["bank"], scale)
    print(f"[{args.tag}] wrote {out}", flush=True)
    np.savez_compressed(os.path.join(args.outdir, f"{args.tag}_blur.npz"),
                        seeds=seeds, video=args.video, start=args.start)


if __name__ == "__main__":
    main()
