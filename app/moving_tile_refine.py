# -*- coding: utf-8 -*-
"""Moving-tile native re-tracking for TAPNext tracks (Apache-2.0, commercial-safe).

Why: TAPNext runs at a fixed 256x256, so on a 4K plate the whole frame is squashed
~15x on the way in -> the coarse position is several px off the real feature. NCC
pattern-refine alone cannot fix that: its search box is centred on the coarse point,
so when the coarse point is far off it locks onto the WRONG nearby patch (measured:
baseline+NCC 4.03px vs manual, often WORSE than baseline on strong features).

Moving tiling fixes the position BEFORE NCC. For each already-selected track it cuts a
native-resolution 256x256 crop that FOLLOWS the coarse path (window by window, re-centred
on the coarse guide) and re-runs TAPNext on that crop. The model then sees full-resolution
pixels around the feature, so the point lands on the real feature, not a 15x-blurred blob.

Measured on a 4K plate against 17 manual artist tracks (mean px deviation):
    baseline whole-frame ......... 4.88
    baseline + NCC (old bot) ..... 4.03
    moving-tile .................. 2.46
    moving-tile + NCC (this) ..... 1.30   <- 3x closer to the human track

Pipeline order in tracker_core: TAPNext passes -> select -> MOVING-TILE (this) ->
pattern_refine NCC -> export. This stage only moves positions; it never drops a track
(frames it cannot improve keep their coarse value), so pattern_refine still does the
trimming/gating afterwards.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2  # type: ignore

from app.video_io import FrameSource, estimate_clip_bytes

Track = List[Tuple[int, float, float]]
StatusCB = Optional[Callable[[str], None]]

_TILE = 256  # native crop == TAPNext input size -> zero downscale of the tile content


def _tile_origin(cx: float, cy: float, W: int, H: int) -> Tuple[int, int]:
    """Top-left of a _TILE crop centred on (cx,cy), clamped inside the frame."""
    ox = int(round(cx - _TILE / 2.0))
    oy = int(round(cy - _TILE / 2.0))
    ox = max(0, min(ox, W - _TILE))
    oy = max(0, min(oy, H - _TILE))
    return ox, oy


def _retrack_one(frs: np.ndarray, xs: np.ndarray, ys: np.ndarray, src: FrameSource,
                 engine, W: int, H: int, win: int, edge_margin: int) -> Dict[int, Tuple[float, float]]:
    """Re-track one coarse path with a native tile that follows it. Returns {frame:(x,y)}.

    The coarse path (xs,ys) only PLACES the tile per window; the tile's own TAPNext run
    decides the refined sub-pixel position. Frames the tile cannot reach keep coarse.
    """
    n = len(frs)
    out: Dict[int, Tuple[float, float]] = {int(frs[k]): (float(xs[k]), float(ys[k])) for k in range(n)}
    if n < 2:
        return out
    half = _TILE / 2.0
    reach = half - edge_margin
    cx, cy = float(xs[0]), float(ys[0])
    i0 = 0
    while i0 < n - 1:
        ox, oy = _tile_origin(xs[i0], ys[i0], W, H)
        # grow the window over CONTIGUOUS frames while the coarse guide stays in the tile
        i1 = i0 + 1
        while (i1 < n and (i1 - i0) < win and int(frs[i1]) - int(frs[i1 - 1]) == 1
               and abs(xs[i1] - (ox + half)) < reach and abs(ys[i1] - (oy + half)) < reach):
            i1 += 1
        L = i1 - i0
        if L < 2:
            i0 += 1
            if i0 < n:
                cx, cy = float(xs[i0]), float(ys[i0])
            continue
        win_bgr = src.get(int(frs[i0]) - 1, L)  # (L,H,W,3) native BGR, full-clip index
        if (win_bgr.shape[0] < 2 or win_bgr.shape[1] < oy + _TILE or win_bgr.shape[2] < ox + _TILE):
            i0 += 1
            if i0 < n:
                cx, cy = float(xs[i0]), float(ys[i0])
            continue
        crop = np.ascontiguousarray(win_bgr[:L, oy:oy + _TILE, ox:ox + _TILE])
        q = np.zeros((1, 1, 3), np.float32)
        q[0, 0] = (0.0, cx - ox, cy - oy)   # query at current pos, tile-local, local frame 0
        tr, vs = engine.track_queries(crop, q)   # tr (L,1,2) [x,y] in tile space, vs (L,1)
        tr = tr[:, 0, :]
        vs = vs[:, 0]
        last_ok = i0
        for k in range(1, tr.shape[0]):
            if not vs[k]:
                break
            if not (2.0 <= tr[k, 0] <= _TILE - 2.0 and 2.0 <= tr[k, 1] <= _TILE - 2.0):
                break  # left the native tile -> stop this window, next re-centres
            gx = ox + float(tr[k, 0]); gy = oy + float(tr[k, 1])
            out[int(frs[i0 + k])] = (gx, gy)
            cx, cy = gx, gy
            last_ok = i0 + k
        if last_ok <= i0:            # no progress this window -> step forward on the coarse path
            i0 += 1
            if i0 < n:
                cx, cy = float(xs[i0]), float(ys[i0])
        else:
            i0 = last_ok
    return out


def moving_tile_refine(final_tracks: Dict[str, Track], video_path: str, W0: int, H0: int,
                       total_frames: int, engine, cfg, status: StatusCB = None
                       ) -> Tuple[Dict[str, Track], str]:
    """Native moving-tile re-track of already-selected tracks (before pattern_refine).

    `final_tracks` frames are 1-based absolute; y may be 3DE-flipped (cfg.flip_y_for_3de)
    -> un-flip to image space, re-track, re-flip on output. Non-destructive: track count
    and length are preserved; only positions move.
    """
    if not final_tracks:
        return final_tracks, "no tracks"
    if W0 < _TILE or H0 < _TILE:
        return final_tracks, f"plate {W0}x{H0} < tile {_TILE}; skipped"

    win = int(getattr(cfg, "mt_window", 16) or 16)
    edge_margin = int(getattr(cfg, "mt_edge_margin", 40) or 40)
    flip = bool(getattr(cfg, "flip_y_for_3de", True)) and H0 > 0

    # native BGR frame provider: hold whole clip if it fits the RAM budget, else stream.
    frac = float(getattr(cfg, "host_ram_frac", 0.5) or 0.5)
    stream = False
    try:
        need = int(estimate_clip_bytes(video_path, 1.0))
        import psutil  # type: ignore
        budget = int(psutil.virtual_memory().available * frac)
        stream = need > budget
    except Exception:
        stream = False
    sd = str(getattr(cfg, "stream_decode", "auto") or "auto").lower()
    if sd == "always":
        stream = True
    elif sd == "never":
        stream = False
    src = FrameSource(video_path, 1.0, stream=stream)

    if status:
        status(f"Moving-tile: {len(final_tracks)} tracks, tile {_TILE}px win={win} "
               f"({'streamed' if stream else 'full-decode'} {W0}x{H0})")

    out: Dict[str, Track] = {}
    moved = 0
    for name, tr in final_tracks.items():
        pts = sorted(tr, key=lambda t: t[0])
        frs = np.array([int(p[0]) for p in pts], dtype=int)
        xs = np.array([float(p[1]) for p in pts], dtype=float)
        ys = np.array([(float(H0 - 1) - float(p[2])) if flip else float(p[2]) for p in pts], dtype=float)
        refined = _retrack_one(frs, xs, ys, src, engine, W0, H0, win, edge_margin)
        new_tr: Track = []
        any_moved = False
        for k in range(len(frs)):
            f = int(frs[k])
            rx, ry = refined.get(f, (xs[k], ys[k]))
            if abs(rx - xs[k]) > 1e-3 or abs(ry - ys[k]) > 1e-3:
                any_moved = True
            oy = (float(H0 - 1) - ry) if flip else ry
            new_tr.append((f, float(rx), float(oy)))
        out[name] = new_tr
        if any_moved:
            moved += 1

    return out, f"retracked={len(out)} moved={moved}"
