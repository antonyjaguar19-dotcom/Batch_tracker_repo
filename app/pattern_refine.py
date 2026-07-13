# -*- coding: utf-8 -*-
"""Full-resolution NCC + affine (ECC) pattern refinement for TAPNext tracks.

Why: TAPNext runs at a fixed 256px, so on a 4K plate it is coarse and sub-pixel
limited, and it locks to a *learned* point, not the actual contrast patch. This
stage re-tracks each already-selected track the way a 3DE pattern-box/search-box
tracker does:

  1. Anchor a reference patch (pattern box) at the track's HIGHEST-CONTRAST frame
     (min-eigenvalue of the structure tensor), so a weak first point can't sink a
     good track.
  2. Refine OUTWARD both directions from the anchor. Per frame, take a search box
     centred on TAPNext's coarse position, NCC template-match at NATIVE resolution
     (cv2.matchTemplate TM_CCOEFF_NORMED), sub-pixel-refine the peak (parabola),
     then optionally ECC-refine rotation+scale (cv2.findTransformECC).
  3. Gate on correlation: HYBRID re-reference (re-grab the patch when corr sags but
     is not lost) + TRIM on lost (cut the track where lock is truly lost).

TAPNext only decides where to *look*; the contrast patch decides the exact
sub-pixel position -> tracks stick to the pattern like 3DE and break the 256px
ceiling because refinement happens on the full-res plate.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import cv2  # type: ignore

from app.video_io import read_window_bgr_scaled, read_video_frames_bgr_scaled, estimate_clip_bytes

Track = List[Tuple[int, float, float]]
StatusCB = Optional[Callable[[str], None]]

_MOTION = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
}


def _extract(gray: np.ndarray, x: float, y: float, half: int) -> Optional[np.ndarray]:
    """Integer-centred patch of size (2*half+1); None if it would fall off frame."""
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > gray.shape[1] or yi + half + 1 > gray.shape[0]:
        return None
    return gray[yi - half:yi + half + 1, xi - half:xi + half + 1]


def _contrast_score(patch: np.ndarray) -> float:
    """Min eigenvalue of the structure tensor (goodFeaturesToTrack-style cornerness)."""
    p = patch.astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    a = float(np.mean(gx * gx)); b = float(np.mean(gx * gy)); c = float(np.mean(gy * gy))
    t = a + c
    disc = max(0.0, t * t / 4.0 - (a * c - b * b))
    return t / 2.0 - math.sqrt(disc)


def _subpix_peak(resp: np.ndarray, loc: Tuple[int, int]) -> Tuple[float, float]:
    """Quadratic sub-pixel offset of the NCC response peak at integer `loc`."""
    mx, my = loc
    h, w = resp.shape
    dx = dy = 0.0
    if 0 < mx < w - 1:
        l, c, r = float(resp[my, mx - 1]), float(resp[my, mx]), float(resp[my, mx + 1])
        d = l - 2 * c + r
        if abs(d) > 1e-9: dx = 0.5 * (l - r) / d
    if 0 < my < h - 1:
        u, c, d0 = float(resp[my - 1, mx]), float(resp[my, mx]), float(resp[my + 1, mx])
        d = u - 2 * c + d0
        if abs(d) > 1e-9: dy = 0.5 * (u - d0) / d
    return (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)))


def _ncc_match(gray: np.ndarray, patch: np.ndarray, cx: float, cy: float,
               search: int, half: int, edge_clamp: bool = False
               ) -> Optional[Tuple[float, float, float]]:
    """NCC template match of `patch` in a search box around (cx,cy). -> (x,y,cc).

    `edge_clamp`: near the frame border, instead of giving up, CLAMP the search window to
    the frame (reduce search on the clipped side) so an edge point still gets refined. Only
    bail when the patch itself can't fit (target within `half` of the border). Without it the
    original behaviour is kept (bail if the full search box falls off frame).
    """
    P = 2 * half + 1
    H, W = gray.shape[:2]
    if not edge_clamp:
        x0 = int(round(cx)) - half - search
        y0 = int(round(cy)) - half - search
        if x0 < 0 or y0 < 0 or x0 + P + 2 * search > W or y0 + P + 2 * search > H:
            return None
        win = gray[y0:y0 + P + 2 * search, x0:x0 + P + 2 * search]
    else:
        # patch must fit fully inside the frame at the target, else there is nothing to match
        if cx - half < 0 or cy - half < 0 or cx + half + 1 > W or cy + half + 1 > H:
            return None
        x0 = max(0, int(round(cx)) - half - search)
        y0 = max(0, int(round(cy)) - half - search)
        x1 = min(W, int(round(cx)) + half + search + 1)
        y1 = min(H, int(round(cy)) + half + search + 1)
        win = gray[y0:y1, x0:x1]
        if win.shape[0] < P or win.shape[1] < P:   # window smaller than the patch -> can't match
            return None
    resp = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(resp)
    dx, dy = _subpix_peak(resp, maxloc)
    px = x0 + maxloc[0] + dx + half
    py = y0 + maxloc[1] + dy + half
    return (float(px), float(py), float(maxv))


def _ecc_refine(gray: np.ndarray, patch: np.ndarray, px: float, py: float,
                half: int, motion: int) -> Optional[Tuple[float, float, float]]:
    """Refine (px,py) for rotation/scale via ECC. -> (x,y,cc) or None on failure."""
    P = 2 * half + 1
    cand = cv2.getRectSubPix(gray, (P, P), (float(px), float(py)))
    tmpl = patch.astype(np.float32)
    cand = cand.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    try:
        cc, warp = cv2.findTransformECC(tmpl, cand, warp, motion, crit, None, 5)
    except cv2.error:
        return None
    # findTransformECC finds `warp` s.t. cand(warp * x) ~= tmpl(x). The feature is the
    # template centre (half,half); warp maps it to its location in `cand` local coords,
    # which we lift back to full-image coords (cand centre = (px,py) in `gray`).
    xc = np.array([half, half, 1.0], dtype=np.float64)
    uv = warp.astype(np.float64) @ xc
    px2 = px - half + float(uv[0])
    py2 = py - half + float(uv[1])
    return (px2, py2, float(cc))


class _FrameGray:
    """Full-res grayscale frame provider. Full-decode when it fits host RAM budget,
    else per-frame window decode with a small LRU cache (bounds RAM on 4K clips)."""

    def __init__(self, path: str, total: int, host_ram_frac: float = 0.5, lru: int = 96):
        self.path = path
        self.total = int(total)
        self._all: Optional[np.ndarray] = None
        self._cache: Dict[int, np.ndarray] = {}
        self._order: List[int] = []
        self._lru = int(lru)
        need_gray = int(estimate_clip_bytes(path, 1.0)) // 3  # bgr estimate -> gray
        budget = None
        try:
            import psutil  # type: ignore
            budget = int(psutil.virtual_memory().available * float(host_ram_frac))
        except Exception:
            budget = None
        if budget is not None and need_gray < budget:
            try:
                frames, _ = read_video_frames_bgr_scaled(path, 1.0)
                self._all = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0)
            except Exception:
                self._all = None

    def get(self, idx0: int) -> Optional[np.ndarray]:
        idx0 = int(idx0)
        if idx0 < 0 or idx0 >= self.total:
            return None
        if self._all is not None:
            return self._all[idx0]
        g = self._cache.get(idx0)
        if g is not None:
            return g
        try:
            win = read_window_bgr_scaled(self.path, idx0, 1, 1.0)
        except Exception:
            return None
        if win is None or win.shape[0] < 1:
            return None
        g = cv2.cvtColor(win[0], cv2.COLOR_BGR2GRAY)
        self._cache[idx0] = g
        self._order.append(idx0)
        if len(self._order) > self._lru:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return g


def _refine_segment(pts: Track, get: Callable[[int], Optional[np.ndarray]], cfg,
                    edge_clamp: bool) -> Optional[Track]:
    """NCC/ECC refine one CONTIGUOUS visible segment (anchor at its sharpest frame,
    refine outward both ways, trim on loss of lock). Returns refined points or None."""
    if len(pts) < 2:
        return None
    half = int(cfg.refine_patch_px) // 2
    search = int(cfg.refine_search_px)
    motion = _MOTION.get(str(cfg.refine_motion).lower(), cv2.MOTION_AFFINE)
    lost = float(cfg.refine_ncc_lost)
    reref = float(cfg.refine_ncc_reref)

    # 1. anchor = highest-contrast valid patch (within THIS segment -> re-acquires per segment)
    best_i, best_c, best_patch = -1, -1.0, None
    for i, (f, x, y) in enumerate(pts):
        g = get(f - 1)
        if g is None:
            continue
        patch = _extract(g, x, y, half)
        if patch is None:
            continue
        c = _contrast_score(patch)
        if c > best_c:
            best_c, best_i, best_patch = c, i, patch
    if best_i < 0 or best_patch is None:
        return None

    refined: Dict[int, Tuple[int, float, float]] = {
        best_i: (pts[best_i][0], float(pts[best_i][1]), float(pts[best_i][2]))
    }

    # 2. refine outward both directions from the anchor
    for direction in (1, -1):
        patch = best_patch.copy()
        i = best_i + direction
        while 0 <= i < len(pts):
            f, cx, cy = pts[i]  # cx,cy = coarse position (search centre)
            g = get(f - 1)
            if g is None:
                break
            res = _ncc_match(g, patch, cx, cy, search, half, edge_clamp)
            if res is None or res[2] < lost:
                break  # trim this side at loss of lock
            x, y, cc = res
            if motion != cv2.MOTION_TRANSLATION:
                er = _ecc_refine(g, patch, x, y, half, motion)
                if er is not None and er[2] >= lost:
                    x, y, cc = er
            refined[i] = (f, float(x), float(y))
            if cc < reref:  # hybrid: re-grab the pattern before it degrades further
                np2 = _extract(g, x, y, half)
                if np2 is not None:
                    patch = np2
            i += direction

    return [refined[k] for k in sorted(refined.keys())]


def _refine_one(track: Track, get: Callable[[int], Optional[np.ndarray]],
                cfg) -> Optional[Track]:
    pts = sorted(track, key=lambda t: t[0])
    if len(pts) < 2:
        return None
    edge_clamp = bool(getattr(cfg, "mt_edge_track", True))
    gap_aware = bool(getattr(cfg, "refine_gap_aware", True))
    min_len = int(cfg.refine_min_len)

    if not gap_aware:
        out = _refine_segment(pts, get, cfg, edge_clamp)
        if out is None or len(out) < min_len:
            return None
        return out

    # Gap-aware: split into contiguous VISIBLE segments (occlusion = frame number jump > 1),
    # refine each on its OWN reference patch, and reassemble under one id. A segment that
    # refines cleanly (>= min_len) is used refined; otherwise its ORIGINAL points are kept so
    # a disappear/reappear point is never trimmed away just because its patch changed.
    segments: List[Track] = []
    cur: Track = [pts[0]]
    for p in pts[1:]:
        if int(p[0]) - int(cur[-1][0]) == 1:
            cur.append(p)
        else:
            segments.append(cur)
            cur = [p]
    segments.append(cur)

    combined: Track = []
    for seg in segments:
        ref = _refine_segment(seg, get, cfg, edge_clamp) if len(seg) >= 2 else None
        combined.extend(ref if (ref is not None and len(ref) >= min_len) else seg)
    if len(combined) < min_len:
        return None
    return sorted(combined, key=lambda t: t[0])


class _GrayFromBGR:
    """Gray provider adapter over a shared BGR FrameSource, so pattern-refine can reuse the
    native decode moving-tile already holds instead of decoding the clip a second time.
    Caches converted gray frames in a small LRU (bounds RAM when the BGR source streams)."""

    def __init__(self, src, lru: int = 128):
        self.src = src
        self._all = getattr(src, "_arr", None)   # for the 'full-decode' status line
        self._cache: Dict[int, np.ndarray] = {}
        self._order: List[int] = []
        self._lru = int(lru)

    def get(self, idx0: int) -> Optional[np.ndarray]:
        idx0 = int(idx0)
        g = self._cache.get(idx0)
        if g is not None:
            return g
        try:
            win = self.src.get(idx0, 1)
        except Exception:
            return None
        if win is None or win.shape[0] < 1:
            return None
        g = cv2.cvtColor(win[0], cv2.COLOR_BGR2GRAY)
        self._cache[idx0] = g
        self._order.append(idx0)
        if len(self._order) > self._lru:
            self._cache.pop(self._order.pop(0), None)
        return g


def refine_tracks(final_tracks: Dict[str, Track], video_path: str, W0: int, H0: int,
                  total_frames: int, cfg, status: StatusCB = None,
                  bgr_source=None) -> Tuple[Dict[str, Track], str]:
    """NCC+affine pattern-refine already-selected tracks at native resolution.

    `final_tracks` frames are 1-based absolute; y may be flipped for 3DE
    (cfg.flip_y_for_3de) -> un-flip to image space, refine, re-flip on output.
    Returns (refined_tracks, info). Tracks that lose lock immediately are dropped.

    `bgr_source`: optional shared native FrameSource to derive gray from (reuses one decode
    across moving-tile + refine instead of a second full decode -> avoids the host-RAM freeze).
    """
    if not final_tracks:
        return final_tracks, "no tracks to refine"
    flip = bool(cfg.flip_y_for_3de) and H0 > 0

    def to_img(t: Track) -> Track:
        return [(f, x, (float(H0 - 1) - y) if flip else y) for (f, x, y) in t]

    def to_out(t: Track) -> Track:
        return [(f, x, (float(H0 - 1) - y) if flip else y) for (f, x, y) in t]

    if bgr_source is not None:
        prov = _GrayFromBGR(bgr_source)
    else:
        prov = _FrameGray(video_path, total_frames, host_ram_frac=float(getattr(cfg, "host_ram_frac", 0.5)))
    if status:
        status(f"Pattern-refine: {len(final_tracks)} tracks, patch={cfg.refine_patch_px}px "
               f"search=±{cfg.refine_search_px}px motion={cfg.refine_motion} "
               f"(full-decode={'yes' if prov._all is not None else 'streamed'})")

    out: Dict[str, Track] = {}
    trimmed = dropped = 0
    for name, tr in final_tracks.items():
        ref = _refine_one(to_img(tr), prov.get, cfg)
        if ref is None:
            dropped += 1
            continue
        if len(ref) < len(tr):
            trimmed += 1
        out[name] = to_out(ref)

    info = f"refined={len(out)}/{len(final_tracks)} trimmed={trimmed} dropped={dropped}"
    return out, info
