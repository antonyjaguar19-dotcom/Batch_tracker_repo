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

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2  # type: ignore

from app.video_io import FrameSource, estimate_clip_bytes

Track = List[Tuple[int, float, float]]
StatusCB = Optional[Callable[[str], None]]

_TILE = 256  # native crop == TAPNext input size -> zero downscale of the tile content


def _tile_origin(cx: float, cy: float, W: int, H: int) -> Tuple[int, int]:
    """Top-left of a _TILE crop centred on (cx,cy), clamped inside the frame.

    NOTE: the point's sub-pixel phase within the tile necessarily wraps once per pixel of
    motion, because the crop is taken on the integer grid. Rounding vs flooring only moves
    WHERE that wrap falls, not whether it happens, so neither is better -- holding the phase
    genuinely constant would mean resampling the crop, which destroys the native pixels this
    stage exists to give the model. The per-window artifact is fixed at the seam instead
    (see the overlap blending in _retrack_one).
    """
    ox = int(round(cx - _TILE / 2.0))
    oy = int(round(cy - _TILE / 2.0))
    ox = max(0, min(ox, W - _TILE))
    oy = max(0, min(oy, H - _TILE))
    return ox, oy


def _retrack_one(frs: np.ndarray, xs: np.ndarray, ys: np.ndarray, src: FrameSource,
                 engine, W: int, H: int, win: int, edge_margin: int,
                 edge_track: bool = True, overlap: int = 4) -> Dict[int, Tuple[float, float]]:
    """Re-track one coarse path with a native tile that follows it. Returns {frame:(x,y)}.

    The coarse path (xs,ys) only PLACES the tile per window; the tile's own TAPNext run
    decides the refined sub-pixel position. Frames the tile cannot reach keep coarse.

    Windows OVERLAP by `overlap` frames and their estimates are cross-faded across the
    overlap. Previously each window butt-jointed onto the last (`i0 = last_ok`) and was a
    fresh model call seeded from the previous window's drifted end, so every seam put a
    small STEP into the path -- repeating each window, which is exactly the regular beat
    seen in centre-2D. A blended seam cannot step, and a beat made of steps goes with them.
    overlap=0 restores the old butt-joint behaviour.
    """
    n = len(frs)
    out: Dict[int, Tuple[float, float]] = {int(frs[k]): (float(xs[k]), float(ys[k])) for k in range(n)}
    if n < 2:
        return out
    half = _TILE / 2.0
    reach = half - edge_margin
    # Accumulate weighted estimates so an overlapped frame is a mix of both windows rather
    # than whichever wrote last.
    acc: Dict[int, Tuple[float, float, float]] = {}   # frame -> (sum_wx, sum_wy, sum_w)

    def _add(frame: int, x: float, y: float, w: float):
        sx, sy, sw = acc.get(frame, (0.0, 0.0, 0.0))
        acc[frame] = (sx + x * w, sy + y * w, sw + w)

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
        # Seed from the SMOOTH coarse guide, not the previous window's drifted end. Handing
        # the end position forward made every seam inherit the last one's error instead of
        # correcting it, which is what let the per-window step accumulate into a visible beat.
        sx0, sy0 = float(xs[i0]), float(ys[i0])
        q[0, 0] = (0.0, sx0 - ox, sy0 - oy)   # query at the guide, tile-local, local frame 0
        tr, vs = engine.track_queries(crop, q)   # tr (L,1,2) [x,y] in tile space, vs (L,1)
        tr = tr[:, 0, :]
        vs = vs[:, 0]
        last_ok = i0
        got: List[Tuple[int, float, float, int]] = []   # (frame, x, y, k) for this window
        for k in range(1, tr.shape[0]):
            if not vs[k]:
                break
            tx, ty = float(tr[k, 0]), float(tr[k, 1])
            at_l, at_r = tx <= 2.0, tx >= _TILE - 2.0
            at_t, at_b = ty <= 2.0, ty >= _TILE - 2.0
            if at_l or at_r or at_t or at_b:
                # Hitting a tile edge on a side where the tile is CLAMPED against the frame
                # means the point is at the real frame border -> keep tracking to it. Hitting
                # an UNclamped side means the point out-ran the tile -> stop, next re-centres.
                hit_unclamped = ((at_l and ox != 0) or (at_r and ox != W - _TILE)
                                 or (at_t and oy != 0) or (at_b and oy != H - _TILE))
                if hit_unclamped or not edge_track:
                    break
            gx = ox + float(tr[k, 0]); gy = oy + float(tr[k, 1])
            got.append((int(frs[i0 + k]), gx, gy, k))
            cx, cy = gx, gy
            last_ok = i0 + k

        # Weight this window with a symmetric taper, applied only now that its real length
        # is known. Ramping in but not OUT still leaves a discontinuity where the window
        # stops -- the seam step just moves rather than disappearing. Fading both ends means
        # neighbouring windows sum smoothly across the overlap.
        if got:
            kmax = got[-1][3]
            for (frame, gx, gy, k) in got:
                if overlap > 0:
                    w = min(1.0, k / float(overlap + 1), (kmax - k + 1) / float(overlap + 1))
                else:
                    w = 1.0
                _add(frame, gx, gy, max(w, 1e-3))
        if last_ok <= i0:            # no progress this window -> step forward on the coarse path
            i0 += 1
            if i0 < n:
                cx, cy = float(xs[i0]), float(ys[i0])
        else:
            # Step back by `overlap` so the next window RE-TRACKS the tail of this one and
            # the two estimates can be blended there. Always advance at least one frame.
            nxt = last_ok - overlap if overlap > 0 else last_ok
            i0 = max(i0 + 1, min(nxt, last_ok))

    for frame, (sx, sy, sw) in acc.items():
        if sw > 0.0:
            out[frame] = (sx / sw, sy / sw)
    return out


def moving_tile_refine(final_tracks: Dict[str, Track], video_path: str, W0: int, H0: int,
                       total_frames: int, engine, cfg, status: StatusCB = None,
                       src: "FrameSource | None" = None, registry=None
                       ) -> Tuple[Dict[str, Track], str]:
    """Native moving-tile re-track of already-selected tracks (before pattern_refine).

    `final_tracks` frames are 1-based absolute; y may be 3DE-flipped (cfg.flip_y_for_3de)
    -> un-flip to image space, re-track, re-flip on output. Non-destructive: track count
    and length are preserved; only positions move.

    `src`: an optional pre-built native (scale 1.0) FrameSource to REUSE, so the caller can
    share ONE native decode with pattern_refine instead of each stage decoding the clip again
    (three concurrent full decodes of a 4K/1440p clip is what exhausts host RAM).
    """
    if not final_tracks:
        return final_tracks, "no tracks"
    if W0 < _TILE or H0 < _TILE:
        return final_tracks, f"plate {W0}x{H0} < tile {_TILE}; skipped"

    win = int(getattr(cfg, "mt_window", 16) or 16)
    overlap = max(0, min(int(getattr(cfg, "mt_overlap", 4) or 0), max(0, win - 2)))
    edge_margin = int(getattr(cfg, "mt_edge_margin", 40) or 40)
    edge_track = bool(getattr(cfg, "mt_edge_track", True))
    flip = bool(getattr(cfg, "flip_y_for_3de", True)) and H0 > 0

    if src is None:
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
        src_note = "streamed" if stream else "full-decode"
    else:
        # Named separately because `stream` only exists on the branch above: reading it here
        # raised UnboundLocalError on every shot that shared a decode, and the caller catches
        # the exception as "Moving-tile skipped" -- so the stage silently did not run at all
        # on the normal path.
        src_note = "shared decode"

    if status:
        status(f"Moving-tile: {len(final_tracks)} tracks, tile {_TILE}px win={win} "
               f"overlap={overlap} ({src_note} {W0}x{H0})")

    out: Dict[str, Track] = {}
    moved = 0
    shortened = 0
    for name, tr in final_tracks.items():
        pts = sorted(tr, key=lambda t: t[0])
        frs = np.array([int(p[0]) for p in pts], dtype=int)
        xs = np.array([float(p[1]) for p in pts], dtype=float)
        ys = np.array([(float(H0 - 1) - float(p[2])) if flip else float(p[2]) for p in pts], dtype=float)

        # Per-track window. These four were read once for the whole shot, so a point crossing
        # the frame and a point sitting still got the same 16-frame tile. The window only has
        # to end before the guide reaches the tile edge -- _retrack_one breaks out there
        # anyway -- so sizing it from THIS track's own speed keeps fast points inside the
        # tile instead of relying on that break, and leaves slow points long (fewer seams,
        # and the seam beat is what the wobble report keeps finding at mt_window).
        tcfg = registry.view(name, cfg) if registry is not None else cfg
        t_win = int(getattr(tcfg, "mt_window", win) or win)
        t_overlap = max(0, min(int(getattr(tcfg, "mt_overlap", overlap) or 0), max(0, t_win - 2)))
        t_margin = int(getattr(tcfg, "mt_edge_margin", edge_margin) or edge_margin)
        if registry is not None and len(frs) >= 3:
            step = np.hypot(np.diff(xs), np.diff(ys)) / np.maximum(1.0, np.diff(frs))
            speed = float(np.median(step))
            reach = float(_TILE // 2 - t_margin)
            if speed > 0.0 and reach > 0.0 and speed * t_win > reach:
                t_win = max(4, min(t_win, int(reach / speed)))
                t_overlap = max(0, min(t_overlap, t_win - 2))
                shortened += 1

        refined = _retrack_one(frs, xs, ys, src, engine, W0, H0, t_win, t_margin, edge_track,
                               overlap=t_overlap)
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

    info = f"retracked={len(out)} moved={moved}"
    if shortened:
        info += f" window-shortened={shortened} (fast tracks)"
    return out, info
