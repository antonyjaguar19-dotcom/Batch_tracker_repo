# -*- coding: utf-8 -*-
"""Moving-tile TAPNext test (ISOLATED experiment — does not touch the bot).

Idea: TAPNext runs at a fixed 256x256. Feeding a whole 1080p/4K frame downscaled
to 256 throws away ~7x of horizontal detail -> coarse, sub-pixel-limited tracks.

"Moving tiling": for each seed point, cut a native-resolution 256x256 crop CENTRED
on the point and let it FOLLOW the point across short temporal windows. The model
then sees full-resolution pixels around the feature (no downscale), so the track is
far more precise. Windows are chained: the last position of one window re-centres the
tile and re-seeds the query for the next (that is what makes the tile "move").

Also runs the plain whole-frame TAPNext on the SAME seeds so an overlay can compare:
  GREEN = moving-tile (native res)   RED = baseline whole-frame (256 downscale)

Reuses the shipped app/tapnext_engine.py TapNextEngine unchanged (loaded by file path
so the app/ package shadow doesn't bite). Output: out/<shot>_moving_tile.mp4
"""
from __future__ import annotations

import os
import sys
import argparse
import importlib.util
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TILE = 256  # native crop size == TAPNext input size -> zero downscale of tile content


def _load_engine_cls():
    """Load TapNextEngine from app/tapnext_engine.py by path (avoid app pkg import)."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    p = os.path.join(_REPO, "app", "tapnext_engine.py")
    spec = importlib.util.spec_from_file_location("btr_tapnext_engine", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TapNextEngine


def decode_all(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return np.asarray(frames), float(fps)


def seed_points(frame_bgr, max_pts=40, quality=0.01, min_dist=48, margin=140):
    """goodFeaturesToTrack, kept away from the border so a full tile fits around each."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    mask = np.zeros((H, W), np.uint8)
    mask[margin:H - margin, margin:W - margin] = 255
    pts = cv2.goodFeaturesToTrack(g, maxCorners=max_pts * 3, qualityLevel=quality,
                                  minDistance=min_dist, mask=mask)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    pts = pts.reshape(-1, 2).astype(np.float32)
    # even spread: greedy farthest-point down to max_pts
    keep = [0]
    while len(keep) < min(max_pts, len(pts)):
        d = np.min([np.hypot(pts[:, 0] - pts[k, 0], pts[:, 1] - pts[k, 1]) for k in keep], axis=0)
        d[keep] = -1
        keep.append(int(np.argmax(d)))
    return pts[keep]


def _tile_origin(cx, cy, W, H):
    """Top-left of a _TILE crop centred on (cx,cy), clamped inside the frame."""
    ox = int(round(cx - _TILE / 2))
    oy = int(round(cy - _TILE / 2))
    ox = max(0, min(ox, W - _TILE))
    oy = max(0, min(oy, H - _TILE))
    return ox, oy


def moving_tile_track(engine, frames, seeds, win=16, min_corr_stop=True):
    """Per-point native moving-tile tracking.

    Returns tracks (T,N,2) full-frame [x,y], vis (T,N) bool.
    """
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    N = seeds.shape[0]
    tracks = np.full((T, N, 2), np.nan, np.float32)
    vis = np.zeros((T, N), bool)

    for i in range(N):
        cx, cy = float(seeds[i, 0]), float(seeds[i, 1])
        tracks[0, i] = (cx, cy)
        vis[0, i] = True
        t0 = 0
        alive = True
        while alive and t0 < T - 1:
            t1 = min(T, t0 + win)
            ox, oy = _tile_origin(cx, cy, W, H)
            crop = frames[t0:t1, oy:oy + _TILE, ox:ox + _TILE]  # (L,256,256,3) native
            # query at the point's current position in tile-local coords, local frame 0
            q = np.zeros((1, 1, 3), np.float32)
            q[0, 0] = (0.0, cx - ox, cy - oy)
            tr, vs = engine.track_queries(crop, q)   # tr (L,1,2) [x,y] in tile space
            tr = tr[:, 0, :]
            vs = vs[:, 0]
            L = tr.shape[0]
            last_ok = -1
            for k in range(1, L):     # k=0 is the seed frame (already set / overlap)
                t = t0 + k
                if not vs[k]:
                    alive = False
                    break
                gx = ox + float(tr[k, 0])
                gy = oy + float(tr[k, 1])
                # sanity: prediction must stay inside the tile (else the tile out-ran it)
                if not (2 <= tr[k, 0] <= _TILE - 2 and 2 <= tr[k, 1] <= _TILE - 2):
                    alive = False
                    break
                tracks[t, i] = (gx, gy)
                vis[t, i] = True
                last_ok = t
                cx, cy = gx, gy
            if last_ok < 0:
                break
            # chain: next window starts at the last good frame, tile re-centred on it
            t0 = last_ok
    return tracks, vis


def moving_tile_track_guided(engine, frames, seeds, bl_tracks, bl_vis,
                             max_win=16, edge_margin=40):
    """Baseline-guided moving tile — handles FAST movers.

    The plain method keeps the tile fixed for a whole window, so a point that moves
    >~128px inside the window leaves the 256 tile and gets trimmed. Here the baseline
    whole-frame track (which follows fast motion, just coarsely) PLACES the tile: each
    window is grown only while the baseline stays inside the tile, so fast points get
    short windows re-centred on where they actually go. The native tile still decides
    the sub-pixel position (query = own last precise position).

    bl_tracks (T,N,2) / bl_vis (T,N): baseline for the SAME seeds (index-aligned).
    """
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    N = seeds.shape[0]
    half = _TILE / 2.0
    reach = half - edge_margin          # baseline may wander this far from tile centre
    tracks = np.full((T, N, 2), np.nan, np.float32)
    vis = np.zeros((T, N), bool)

    for i in range(N):
        cx, cy = float(seeds[i, 0]), float(seeds[i, 1])
        tracks[0, i] = (cx, cy)
        vis[0, i] = True
        t0 = 0
        alive = True
        while alive and t0 < T - 1:
            # tile centre = baseline (falls back to own pos when baseline is gone)
            if bl_vis[t0, i] and not np.isnan(bl_tracks[t0, i, 0]):
                ccx, ccy = float(bl_tracks[t0, i, 0]), float(bl_tracks[t0, i, 1])
            else:
                ccx, ccy = cx, cy
            ox, oy = _tile_origin(ccx, ccy, W, H)
            # grow the window only while the baseline stays inside the tile
            t1 = t0 + 1
            while (t1 < T and (t1 - t0) < max_win and bl_vis[t1, i]
                   and not np.isnan(bl_tracks[t1, i, 0])
                   and abs(bl_tracks[t1, i, 0] - (ox + half)) < reach
                   and abs(bl_tracks[t1, i, 1] - (oy + half)) < reach):
                t1 += 1
            t1 = min(T, t1 + 1)         # include one frame past (edge step still native-refined)
            crop = frames[t0:t1, oy:oy + _TILE, ox:ox + _TILE]
            q = np.zeros((1, 1, 3), np.float32)
            q[0, 0] = (0.0, cx - ox, cy - oy)
            tr, vs = engine.track_queries(crop, q)
            tr = tr[:, 0, :]; vs = vs[:, 0]
            L = tr.shape[0]
            last_ok = -1
            for k in range(1, L):
                t = t0 + k
                if not vs[k]:
                    alive = False; break
                if not (2 <= tr[k, 0] <= _TILE - 2 and 2 <= tr[k, 1] <= _TILE - 2):
                    # left the native tile before baseline said so -> stop this window,
                    # let the next window re-centre (don't kill the whole track)
                    break
                tracks[t, i] = (ox + float(tr[k, 0]), oy + float(tr[k, 1]))
                vis[t, i] = True
                last_ok = t
                cx, cy = tracks[t, i]
            if last_ok < 0:
                # no progress this window; nudge forward on baseline to avoid a stall
                if t0 + 1 < T and bl_vis[t0 + 1, i]:
                    t0 += 1; continue
                break
            t0 = last_ok
    return tracks, vis


def render(frames, fps, out_path, mt_tracks, mt_vis, bl_tracks=None, bl_vis=None, trail=18):
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    N = mt_tracks.shape[1]
    for t in range(T):
        img = frames[t].copy()
        # baseline (red) first so green draws on top
        if bl_tracks is not None:
            for i in range(bl_tracks.shape[1]):
                if not bl_vis[t, i]:
                    continue
                x, y = bl_tracks[t, i]
                if np.isnan(x):
                    continue
                cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1, cv2.LINE_AA)
        # moving-tile (green) + trail + cross + id
        for i in range(N):
            for s in range(max(0, t - trail), t):
                a = mt_tracks[s, i]; b = mt_tracks[s + 1, i]
                if mt_vis[s, i] and mt_vis[s + 1, i] and not np.isnan(a[0]) and not np.isnan(b[0]):
                    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             (0, 200, 0), 1, cv2.LINE_AA)
            if mt_vis[t, i] and not np.isnan(mt_tracks[t, i, 0]):
                x, y = mt_tracks[t, i]
                xi, yi = int(round(x)), int(round(y))
                cv2.drawMarker(img, (xi, yi), (0, 255, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
                cv2.putText(img, str(i), (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # legend
        cv2.putText(img, "GREEN = moving-tile (native)   RED = baseline whole-frame",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {t+1}/{T}", (16, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="D:/Jefrin/IN/SH011.mp4")
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "SH011_moving_tile.mp4"))
    ap.add_argument("--max-pts", type=int, default=36)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--fixed-tile", action="store_true",
                    help="use the old fixed-per-window tile (default is baseline-guided)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip baseline overlay (forces --fixed-tile too)")
    args = ap.parse_args()

    print(f"[load] {args.video}", flush=True)
    frames, fps = decode_all(args.video)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"[load] {T} frames {W}x{H} @ {fps:.1f}fps", flush=True)

    Engine = _load_engine_cls()
    print("[engine] loading TAPNext++ ...", flush=True)
    engine = Engine(tool_root=_REPO, device="cuda")

    seeds = seed_points(frames[0], max_pts=args.max_pts)
    print(f"[seed] {seeds.shape[0]} points", flush=True)

    # baseline whole-frame first — it also GUIDES the tile in guided mode
    bl_tracks = bl_vis = None
    guided = (not args.fixed_tile) and (not args.no_baseline)
    if not args.no_baseline:
        print("[baseline] whole-frame tracking ...", flush=True)
        q = np.zeros((1, seeds.shape[0], 3), np.float32)
        q[0, :, 1] = seeds[:, 0]
        q[0, :, 2] = seeds[:, 1]
        bl_tracks, bl_vis = engine.track_queries(frames, q)
        print(f"[baseline] mean track length {bl_vis.sum(0).mean():.1f}/{T} frames", flush=True)

    print("[v1] fixed-window tile ...", flush=True)
    fx_tracks, fx_vis = moving_tile_track(engine, frames, seeds, win=args.win)
    print(f"[v1] mean {fx_vis.sum(0).mean():.1f}/{T}", flush=True)
    print("[v2] baseline-guided tile ...", flush=True)
    gd_tracks, gd_vis = moving_tile_track_guided(engine, frames, seeds,
                                                 bl_tracks, bl_vis, max_win=args.win)
    print(f"[v2] mean {gd_vis.sum(0).mean():.1f}/{T}", flush=True)
    mt_tracks, mt_vis = (gd_tracks, gd_vis) if guided else (fx_tracks, fx_vis)

    stem = os.path.splitext(os.path.basename(args.video))[0]
    npz = os.path.join(_HERE, "out", f"{stem}_arrays.npz")
    np.savez_compressed(npz, video=args.video, seeds=seeds,
                        bl=bl_tracks, bl_vis=bl_vis, fx=fx_tracks, fx_vis=fx_vis,
                        gd=gd_tracks, gd_vis=gd_vis)
    print(f"[dump] {npz}", flush=True)

    if bl_vis is not None:
        mlen = mt_vis.sum(0); blen = bl_vis.sum(0)
        # per-frame max motion of the baseline = "how fast" each point moves
        spd = np.full(mt_tracks.shape[1], 0.0)
        for i in range(mt_tracks.shape[1]):
            p = bl_tracks[:, i][bl_vis[:, i]]
            if len(p) > 1:
                spd[i] = np.nanmax(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))
        print("[per-track] id  mt_len  bl_len  maxpx/frame (fast movers last):", flush=True)
        for i in np.argsort(spd):
            print(f"   {i:2d}   {int(mlen[i]):3d}    {int(blen[i]):3d}    {spd[i]:6.1f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[render] {args.out}", flush=True)
    render(frames, fps, args.out, mt_tracks, mt_vis, bl_tracks, bl_vis)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
