# -*- coding: utf-8 -*-
"""Magnified follow-cam compare (ISOLATED experiment).

Full-frame overlays hide the difference: 36 points, tiny crosses, and only ~2-3
tracks actually differ between methods -> "no big diff". This renders a ZOOMED
follow-cam per track so sub-pixel lock + fast-mover survival are actually visible.

Reads out/SH011_arrays.npz (written by track_moving_tile.py). For each selected
track it crops a small box around the point every frame and upscales it, drawing:
  RED   = baseline whole-frame (256 downscale)
  BLUE  = v1 fixed-tile
  GREEN = v2 baseline-guided tile
Panels are tiled into one grid video so you compare tracks side by side.
"""
from __future__ import annotations
import os, argparse
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))


def decode_all(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    return np.asarray(fr), float(fps)


def draw_marker(panel, cx, cy, ox, oy, zoom, color, label):
    """Map full-frame (cx,cy) into the zoomed panel and draw a cross."""
    if cx is None or np.isnan(cx):
        return
    px = int(round((cx - ox) * zoom)); py = int(round((cy - oy) * zoom))
    h, w = panel.shape[:2]
    if -20 <= px <= w + 20 and -20 <= py <= h + 20:
        cv2.drawMarker(panel, (px, py), color, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.circle(panel, (px, py), 1, color, -1, cv2.LINE_AA)


def panel_for(track_id, frames, t, d, box, zoom):
    """One zoomed panel for one track at frame t."""
    T, H, W = frames.shape[:3]
    # centre the crop on the guided track (fallback baseline, then fixed)
    c = None
    for key, vk in (("gd", "gd_vis"), ("bl", "bl_vis"), ("fx", "fx_vis")):
        arr, vis = d[key], d[vk]
        if vis[t, track_id] and not np.isnan(arr[t, track_id, 0]):
            c = arr[t, track_id]; break
    if c is None:      # nothing alive -> last known guided
        seed = d["seeds"][track_id]; c = seed
    ox = int(round(c[0] - box / 2)); oy = int(round(c[1] - box / 2))
    ox = max(0, min(ox, W - box)); oy = max(0, min(oy, H - box))
    crop = frames[t, oy:oy + box, ox:ox + box]
    panel = cv2.resize(crop, (box * zoom, box * zoom), interpolation=cv2.INTER_NEAREST)
    # markers: red baseline, blue v1, green v2
    for key, vk, col in (("bl", "bl_vis", (0, 0, 255)),
                         ("fx", "fx_vis", (255, 100, 0)),
                         ("gd", "gd_vis", (0, 255, 0))):
        if d[vk][t, track_id] and not np.isnan(d[key][t, track_id, 0]):
            draw_marker(panel, d[key][t, track_id, 0], d[key][t, track_id, 1],
                        ox, oy, zoom, col, key)
    cv2.putText(panel, f"trk {track_id}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(_HERE, "out", "SH011_arrays.npz"))
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "SH011_zoom_compare.mp4"))
    ap.add_argument("--tracks", default="", help="comma ids; else use --pick")
    ap.add_argument("--pick", default="good", choices=["good", "divergent"],
                    help="good = solid features surviving in all 3 methods; "
                         "divergent = where methods disagree most")
    ap.add_argument("--box", type=int, default=90, help="native crop size (px)")
    ap.add_argument("--zoom", type=int, default=5)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    d = {k: np.load(args.npz, allow_pickle=True)[k] for k in
         ("video", "seeds", "bl", "bl_vis", "fx", "fx_vis", "gd", "gd_vis")}
    video = str(d["video"])
    frames, fps = decode_all(video)
    T = frames.shape[0]
    N = d["seeds"].shape[0]

    if args.tracks.strip():
        ids = [int(x) for x in args.tracks.split(",")]
    elif args.pick == "good":
        # solid features: survive in ALL three methods AND move (not stuck on flat water).
        T_ = d["bl_vis"].shape[0]
        surv = np.minimum.reduce([d["bl_vis"].sum(0), d["fx_vis"].sum(0), d["gd_vis"].sum(0)])
        mot = np.zeros(N)
        for i in range(N):
            p = d["gd"][:, i][d["gd_vis"][:, i]]
            if len(p) > 1:
                mot[i] = np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
        ok = (surv > 0.7 * T_) & (mot > 15)     # long-lived + actually travels
        cand = np.where(ok)[0]
        if len(cand) < args.cols * 2:           # relax if too few
            cand = np.argsort(-surv)[:args.cols * 2]
        ids = list(cand[np.argsort(-surv[cand])][:args.cols * 2])
    else:
        # rank by how differently v2(guided) and baseline behave: length gap + mean px gap
        score = np.zeros(N)
        for i in range(N):
            both = d["gd_vis"][:, i] & d["bl_vis"][:, i]
            pxgap = 0.0
            if both.any():
                dif = d["gd"][both, i] - d["bl"][both, i]
                pxgap = np.nanmean(np.hypot(dif[:, 0], dif[:, 1]))
            lgap = abs(int(d["gd_vis"][:, i].sum()) - int(d["bl_vis"][:, i].sum()))
            score[i] = pxgap + 0.3 * lgap
        ids = list(np.argsort(-score)[:args.cols * 2])
    print(f"[zoom] tracks {ids}", flush=True)

    cols = args.cols
    rows = int(np.ceil(len(ids) / cols))
    ph = args.box * args.zoom
    gw, gh = cols * ph, rows * ph
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (gw, gh))
    for t in range(T):
        grid = np.zeros((gh, gw, 3), np.uint8)
        for j, tid in enumerate(ids):
            p = panel_for(int(tid), frames, t, d, args.box, args.zoom)
            r, c = divmod(j, cols)
            grid[r * ph:(r + 1) * ph, c * ph:(c + 1) * ph] = p
        cv2.putText(grid, "RED=baseline  BLUE=v1 fixed  GREEN=v2 guided   f%d/%d" % (t + 1, T),
                    (10, gh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(grid)
    vw.release()
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
