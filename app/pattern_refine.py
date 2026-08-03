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
    """Patch of size (2*half+1) centred at the SUB-PIXEL position; None if off frame.

    This used to slice the array at int(round(x)), so the reference pattern was quantised to
    whole pixels and could not represent where the feature actually sat -- re-quantised on
    every re-reference, which is a floor on how still a track can possibly be. Sampling with
    getRectSubPix (what _ecc_refine below already does, and what 3DE and SynthEyes do) costs
    a slight bilinear softening and buys the sub-pixel truth.
    """
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > gray.shape[1] or yi + half + 1 > gray.shape[0]:
        return None
    P = 2 * half + 1
    try:
        return cv2.getRectSubPix(gray, (P, P), (float(x), float(y)))
    except cv2.error:
        return gray[yi - half:yi + half + 1, xi - half:xi + half + 1]


def _structure_eigs(patch: np.ndarray) -> Tuple[float, float]:
    """Both structure-tensor eigenvalues (lambda_min, lambda_max) for a patch."""
    p = patch.astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    a = float(np.mean(gx * gx)); b = float(np.mean(gx * gy)); c = float(np.mean(gy * gy))
    t = a + c
    disc = math.sqrt(max(0.0, t * t / 4.0 - (a * c - b * b)))
    return (t / 2.0 - disc, t / 2.0 + disc)


def _contrast_score(patch: np.ndarray) -> float:
    """Min eigenvalue of the structure tensor (goodFeaturesToTrack-style cornerness)."""
    return _structure_eigs(patch)[0]


def _anisotropy(patch: np.ndarray) -> float:
    """lambda_min / lambda_max: ~1 for a corner, ~0 for a pure edge. Contrast-invariant, so
    it separates 'this is a 1-D feature' from 'this is faint', which _contrast_score alone
    cannot. A 1-D patch gives NCC a response RIDGE, and the point slides along it."""
    lo, hi = _structure_eigs(patch)
    return (lo / hi) if hi > 1e-12 else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation between two equally-sized patches. Used to ask 'is this still
    the same feature?' -- against the anchor for drift, and across a gap for re-acquisition.

    Zero-variance patches are rejected outright: TM_CCOEFF_NORMED divides by the standard
    deviation, so a FLAT patch scores ~1.0 against anything. Without this guard a pattern
    that drifted onto a blank wall would sail through the drift guard -- the exact case it
    exists to catch.
    """
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return -1.0
    try:
        fa = a.astype(np.float32)
        fb = b.astype(np.float32)
        if float(fa.std()) < 1e-3 or float(fb.std()) < 1e-3:
            return -1.0
        return float(cv2.matchTemplate(fa, fb, cv2.TM_CCOEFF_NORMED)[0, 0])
    except Exception:
        return -1.0


def _subpix_peak(resp: np.ndarray, loc: Tuple[int, int]) -> Tuple[float, float]:
    """Sub-pixel offset of the correlation peak at integer `loc`.

    Two upgrades over the separable parabola this replaces, both aimed at PIXEL-LOCKING --
    the tendency of a crude fit to pull estimates toward whole pixels, which is what shows
    up as a track wobbling in place while the feature never moved:

      * a full 2-D quadratic over the 3x3 neighbourhood, so the cross term is used. Fitting
        x and y independently discards it entirely, which is worst on a diagonal feature.
      * where all nine samples are positive, the fit is done on the LOG of the response. A
        normalised correlation peak is much closer to Gaussian than to parabolic, and the
        log-parabola estimator is correspondingly less biased.

    Falls back to the plain quadratic when any sample is <= 0 (TM_CCOEFF_NORMED can go
    negative), and to the separable fit when the Hessian is singular.
    """
    mx, my = loc
    h, w = resp.shape
    if not (0 < mx < w - 1 and 0 < my < h - 1):
        return (0.0, 0.0)          # peak on the window border: no neighbourhood to fit
    r = resp[my - 1:my + 2, mx - 1:mx + 2].astype(np.float64)

    if float(r.min()) > 0.0:
        r = np.log(r)              # Gaussian peak -> parabolic in log space

    fx = (r[1, 2] - r[1, 0]) * 0.5
    fy = (r[2, 1] - r[0, 1]) * 0.5
    fxx = r[1, 2] - 2.0 * r[1, 1] + r[1, 0]
    fyy = r[2, 1] - 2.0 * r[1, 1] + r[0, 1]
    fxy = (r[2, 2] - r[2, 0] - r[0, 2] + r[0, 0]) * 0.25
    det = fxx * fyy - fxy * fxy

    if abs(det) > 1e-12:
        dx = -(fyy * fx - fxy * fy) / det
        dy = -(fxx * fy - fxy * fx) / det
    else:
        # Degenerate surface (a ridge, or a flat top): fall back to the separable fit.
        dx = 0.5 * (r[1, 0] - r[1, 2]) / fxx if abs(fxx) > 1e-12 else 0.0
        dy = 0.5 * (r[0, 1] - r[2, 1]) / fyy if abs(fyy) > 1e-12 else 0.0

    if not (math.isfinite(dx) and math.isfinite(dy)):
        return (0.0, 0.0)
    return (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)))


def _adaptive_search(pts: Track, cfg) -> int:
    """Search radius sized from how fast this point is actually moving.

    A fixed radius is wrong at both ends. On a fast shot the true feature can fall OUTSIDE
    the box -- NCC then returns the best match inside it, which is simply the wrong place,
    and the point slides. On a slow shot a wide box is worse than useless: it invites the
    rival-peak problem _peak_is_ambiguous exists to catch. So: keep the tight default when
    the point is barely moving, and open up only as far as the measured motion demands.
    Bounded because NCC cost grows with the SQUARE of the radius.
    """
    base = int(getattr(cfg, "refine_search_px", 24) or 24)
    smax = int(getattr(cfg, "refine_search_max", base) or base)
    k = float(getattr(cfg, "refine_search_speed_k", 0.0) or 0.0)
    if smax <= base or k <= 0.0 or len(pts) < 3:
        return base
    steps = []
    for a, b in zip(pts, pts[1:]):
        dt = max(1, int(b[0]) - int(a[0]))
        steps.append(math.hypot(float(b[1]) - float(a[1]), float(b[2]) - float(a[2])) / dt)
    if not steps:
        return base
    speed = float(np.median(steps))
    # Below ~1px/frame the point is effectively static and the coarse position is already
    # good, so keep the tight box EXACTLY. That matters beyond cost: a wider box is what
    # lets a rival peak in, which is precisely what _peak_is_ambiguous then has to reject.
    if speed < 1.0:
        return base
    return int(max(base, min(smax, round(base + k * speed))))


def _peak_flatness(resp: np.ndarray, maxloc: Tuple[int, int], maxv: float) -> float:
    """How BROAD the correlation peak is: mean of the 8 neighbours / peak. In [0,1].

    This is the quantity that separates a trackable feature from an untrackable one, and
    nothing in the pipeline measured it. A sharp corner's response falls away steeply, so the
    ratio is small; a defocused blob correlates almost as well one pixel over, so the ratio
    approaches 1 and the position is barely constrained at all -- which is precisely why such
    a point wobbles however good the sub-pixel maths is.

    It is a RATIO, so it is invariant to overall contrast: a faint but sharp feature is not
    punished for being faint, only for being soft.
    """
    my, mx = maxloc[1], maxloc[0]
    h, w = resp.shape[:2]
    if maxv <= 1e-6:
        return 1.0
    # Sample a ring several pixels out, NOT the immediate neighbours. Normalised correlation
    # is smooth, so even a crisp feature still scores ~0.9 one pixel from its peak -- judging
    # by adjacent samples squeezes every track into a narrow band and discriminates poorly.
    # Over a few pixels a sharp peak has fallen away substantially while a defocused blob has
    # barely moved, which is the whole distinction.
    r = max(2, min(5, (min(h, w) - 1) // 2))
    if r < 2 or not (r <= mx < w - r and r <= my < h - r):
        return 1.0                       # no room for the ring -> assume the worst
    ring_vals = []
    for dy in (-r, 0, r):
        for dx in (-r, 0, r):
            if dx == 0 and dy == 0:
                continue
            ring_vals.append(float(resp[my + dy, mx + dx]))
    centre = float(resp[my, mx])
    if centre <= 1e-6 or not ring_vals:
        return 1.0
    return float(max(0.0, min(1.0, float(np.mean(ring_vals)) / centre)))


def _peak_is_ambiguous(resp: np.ndarray, maxloc: Tuple[int, int], maxv: float,
                       ratio: float, half: int) -> bool:
    """True when a rival peak is nearly as strong as the winner, somewhere else in the box.

    Repetitive detail -- bolts, rivets, a window grid, tiles -- gives NCC several almost
    equally good answers, and taking minMaxLoc's single best silently snapped the point to
    the identical feature NEXT DOOR. Nothing downstream could catch it, because the
    correlation really was high. This is the standard distinctiveness (Lowe-style ratio)
    test: suppress a neighbourhood around the winner, look at the best of what is left, and
    refuse to answer when the two are too close to call.
    """
    if ratio >= 1.0 or maxv <= 0.0:
        return False
    h, w = resp.shape[:2]
    r = max(2, int(half) // 2)          # suppression radius around the winner
    x0 = max(0, maxloc[0] - r); x1 = min(w, maxloc[0] + r + 1)
    y0 = max(0, maxloc[1] - r); y1 = min(h, maxloc[1] + r + 1)
    masked = resp.copy()
    masked[y0:y1, x0:x1] = -1.0
    if masked.size == 0 or float(masked.max()) <= -1.0:
        return False                     # the box holds only the one peak -> unambiguous
    return float(masked.max()) >= ratio * float(maxv)


def _ncc_match(gray: np.ndarray, patch: np.ndarray, cx: float, cy: float,
               search: int, half: int, edge_clamp: bool = False,
               ambiguity_ratio: float = 1.0
               ) -> Optional[Tuple[float, float, float]]:
    """NCC template match of `patch` in a search box around (cx,cy). -> (x,y,cc).

    `edge_clamp`: near the frame border, instead of giving up, CLAMP the search window to
    the frame (reduce search on the clipped side) so an edge point still gets refined. Only
    bail when the patch itself can't fit (target within `half` of the border). Without it the
    original behaviour is kept (bail if the full search box falls off frame).

    `ambiguity_ratio`: reject the match when a rival peak scores >= this fraction of the
    winner (see _peak_is_ambiguous). Returning None -- "I don't know" -- is far better than a
    confident wrong answer: the caller's hysteresis simply holds the previous position.
    1.0 disables the test.
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
    if _peak_is_ambiguous(resp, maxloc, float(maxv), float(ambiguity_ratio), half):
        return None
    dx, dy = _subpix_peak(resp, maxloc)
    px = x0 + maxloc[0] + dx + half
    py = y0 + maxloc[1] + dy + half
    # How well this frame pins the point down, stashed for the caller. Kept off the return
    # tuple so every existing `x, y, cc = res` unpack keeps working.
    _LAST_FLATNESS["v"] = _peak_flatness(resp, maxloc, float(maxv))
    return (float(px), float(py), float(maxv))


# Peak flatness of the most recent _ncc_match. A module-level stash rather than a wider
# return tuple, so the ~10 existing call sites are untouched; read it immediately after a
# successful match. Single-threaded per refine call, which is how refine_tracks runs.
_LAST_FLATNESS: Dict[str, float] = {"v": 1.0}

# Median localisation certainty of the segment _refine_segment last returned, in [0,1].
# Same stash-not-signature approach as _LAST_FLATNESS: _refine_segment's (points, reason)
# contract is consumed in several places and is not worth widening for a diagnostic.
_CERTAINTY: Dict[str, float] = {"v": 0.0}


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
                    edge_clamp: bool, predict=None) -> Tuple[Optional[Track], str]:
    """NCC/ECC refine one CONTIGUOUS visible segment (anchor at its sharpest frame,
    refine outward both ways, trim on loss of lock).

    Returns (points, reason). The reason matters because the caller treats the two failure
    modes oppositely:
      "ok"        refined normally.
      "no-anchor" mechanically un-refinable (patch fell off frame, no readable frame).
                  The caller keeps the ORIGINAL points -- better raw than deleted.
      "edge"      the anchor is a 1-D feature. The caller must DROP these points: keeping
                  them raw is the worst outcome, since an unrefined point sits at TAPNext's
                  coarse 256px position (~4.9px error at 4K vs ~1.3px refined) AND still
                  slides along the edge.
    """
    if len(pts) < 2:
        return None, "no-anchor"
    half = int(cfg.refine_patch_px) // 2
    search = _adaptive_search(pts, cfg)
    motion = _MOTION.get(str(cfg.refine_motion).lower(), cv2.MOTION_AFFINE)
    lost = float(cfg.refine_ncc_lost)
    reref = float(cfg.refine_ncc_reref)
    ambig = float(getattr(cfg, "match_ambiguity_ratio", 1.0) or 1.0)
    # Hysteresis: it takes `lost` to be confident, but only `hold` to stay locked, so a
    # correlation hovering around one hard threshold can no longer drop the point for a
    # frame at a time while its pattern is plainly visible.
    hold = min(float(getattr(cfg, "refine_ncc_hold", lost) or lost), lost)
    # ECC polish. _ecc_refine is proper iterative sub-pixel estimation against the real
    # pixels, but it only ever ran when the motion model was NOT translation -- and
    # translation is the default, so on normal settings the exported position was always a
    # 3-sample curve fit. Running it in translation mode too replaces that with gradient
    # descent. This does not revisit the affine->translation decision: that was about
    # affine's extra rotation/scale freedom adding wobble; translation-only ECC has none.
    ecc_polish = bool(getattr(cfg, "refine_ecc_polish", False))

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
        return None, "no-anchor"

    # Aperture guard: even the sharpest patch in this segment may be a 1-D feature (the track
    # seeded on a corner and drifted onto an edge, or re-referenced onto one). NCC can only
    # pin it across the edge, so refining it just slides it along.
    thr = float(getattr(cfg, "min_corner_anisotropy", 0.0) or 0.0)
    if thr > 0.0 and _anisotropy(best_patch) < thr:
        return None, "edge"

    refined: Dict[int, Tuple[int, float, float]] = {
        best_i: (pts[best_i][0], float(pts[best_i][1]), float(pts[best_i][2]))
    }

    # Re-acquisition + drift-guard settings.
    max_gap = int(getattr(cfg, "reacquire_max_gap", 0) or 0)
    reacq = float(getattr(cfg, "refine_ncc_reacquire", 0.75) or 0.75)
    drift_floor = float(getattr(cfg, "refine_drift_floor", 0.0) or 0.0)
    drift_every = int(getattr(cfg, "refine_drift_check_every", 0) or 0)
    anchor = best_patch.copy()          # NEVER replaced: the drift guard's ground truth
    # Per-frame localisation certainty, reduced to one number for the caller (see _CERTAINTY).
    certs: List[float] = []

    # 2. refine outward both directions from the anchor
    for direction in (1, -1):
        patch = best_patch.copy()
        i = best_i + direction
        since_anchor_check = 0
        gap = 0                          # consecutive frames with no lock (candidate occlusion)
        last_good = refined[best_i]      # last verified position, for neighbour prediction
        while 0 <= i < len(pts):
            f, cx, cy = pts[i]  # cx,cy = coarse position (search centre)
            g = get(f - 1)
            if g is None:
                break
            # Try BOTH patterns and take the better match, rather than preferring one.
            #
            # Anchor-first was wrong: it accepted the anchor's answer whenever it merely
            # cleared `hold` (0.45), so on footage where appearance drifts -- handheld,
            # changing light -- a mediocre anchor match was used even when the current
            # pattern matched far better. A weak match has a BROAD, noisy peak, so the
            # position wobbles; and because certainty is read off that same peak, the
            # certainty figures became noise and the gate then kept arbitrary tracks.
            # Best-of-both keeps the anchor's pull against drift without ever preferring a
            # worse match to a better one.
            res = _ncc_match(g, anchor, cx, cy, search, half, edge_clamp, ambig)
            if patch is not anchor:
                alt = _ncc_match(g, patch, cx, cy, search, half, edge_clamp, ambig)
                if alt is not None and (res is None or alt[2] > res[2]):
                    # Re-run the winner last so _LAST_FLATNESS describes the peak actually
                    # used -- certainty must reflect the match we kept, not the one we lost.
                    res = _ncc_match(g, patch, cx, cy, search, half, edge_clamp, ambig)
            if res is None or res[2] < hold:
                # Lock lost. This used to end the track here, which is how points were lost
                # to occluders SAM3 never masked (a pole, a prop, a hand). Treat it as a
                # candidate occlusion: leave the frame out and keep trying to re-find the
                # ORIGINAL anchor further along.
                if max_gap <= 0 or gap >= max_gap:
                    break
                gap += 1
                # Search where the NEIGHBOURS say it should have gone, not where the coarse
                # tracker left it -- during an occlusion the coarse point has usually been
                # dragged along by whatever crossed it.
                # Re-acquisition is where a repetitive feature is most dangerous: the anchor
                # matches an identical neighbour just as well. Search a TIGHT radius around
                # the neighbour-predicted position first, and demand the ambiguity test pass
                # -- better to stay lost than come back on the bolt next door.
                px, py = cx, cy
                tight = search
                if predict is not None and last_good is not None:
                    px, py = predict(last_good[1], last_good[2], int(last_good[0]), int(f))
                    tight = max(4, int(round(search * float(
                        getattr(cfg, "reacquire_search_frac", 1.0) or 1.0))))
                ra = _ncc_match(g, anchor, px, py, tight, half, edge_clamp, ambig)
                if ra is None or ra[2] < reacq:
                    ra = _ncc_match(g, anchor, cx, cy, search, half, edge_clamp, ambig)
                if ra is not None and ra[2] >= reacq:
                    refined[i] = (f, float(ra[0]), float(ra[1]))
                    last_good = refined[i]
                    patch = anchor.copy()      # resume from the verified original pattern
                    gap = 0
                    since_anchor_check = 0
                i += direction
                continue
            gap = 0
            x, y, cc = res
            # Localisation certainty for this frame: how sharply the correlation peak falls
            # away. This is the ONLY signal that separates a defocused point from a good one
            # -- its motion statistics (long, smooth, no jumps) look excellent, which is why
            # the quality score rates it highly.
            #
            # Deliberately NOT scaled by the patch's own structure: that test would have to
            # be relative to the track's own anchor, and a uniformly soft track passes its
            # own floor, so it discriminates nothing. The threshold is instead set per shot
            # from the spread of certainties actually measured (see track_filter).
            certs.append(1.0 - float(_LAST_FLATNESS.get("v", 1.0)))
            if motion != cv2.MOTION_TRANSLATION:
                er = _ecc_refine(g, patch, x, y, half, motion)
                if er is not None and er[2] >= lost:
                    x, y, cc = er
            elif ecc_polish:
                # Only adopt it if the correlation did not get worse, so the polish can
                # never turn a good NCC answer into a poorer one.
                er = _ecc_refine(g, patch, x, y, half, cv2.MOTION_TRANSLATION)
                if er is not None and er[2] >= cc:
                    x, y, cc = er
            refined[i] = (f, float(x), float(y))
            last_good = refined[i]

            # Drift guard: re-referencing used to re-grab the patch wherever the point
            # currently sat, with nothing tying it to the original -- over a long shot the
            # pattern random-walks off the feature. Accept a new patch only while it still
            # resembles the anchor, and re-check against the anchor periodically.
            since_anchor_check += 1
            # Only re-grab from a CONFIDENT frame. A frame merely held by hysteresis is by
            # definition a poor match, so adopting its pixels as the new pattern is how a
            # track walks off its feature.
            if cc < reref and cc >= lost:  # hybrid: re-grab before the pattern degrades further
                np2 = _extract(g, x, y, half)
                if np2 is not None:
                    if drift_floor <= 0.0 or _corr(np2, anchor) >= drift_floor:
                        # Re-align the replacement to the anchor before adopting it. A patch
                        # grabbed at a position that is off by d bakes d into the reference,
                        # and every later frame inherits it -- the mechanism behind a track
                        # smoothly wandering across its own feature. Locating the anchor
                        # INSIDE the new patch and correcting by that offset makes adopting
                        # it positionally neutral.
                        rel = _ncc_match(np2, anchor, float(half), float(half),
                                         max(2, half // 3), half, True, 1.0)
                        if rel is not None:
                            ddx, ddy = rel[0] - half, rel[1] - half
                            if math.hypot(ddx, ddy) <= max(1.0, half / 3.0):
                                x, y = x + ddx, y + ddy
                                refined[i] = (f, float(x), float(y))
                                last_good = refined[i]
                                re2 = _extract(g, x, y, half)   # explicit: `ndarray or x`
                                if re2 is not None:             # is ambiguous in numpy
                                    np2 = re2
                        patch = np2
                    else:
                        patch = anchor.copy()   # snapped too far: fall back to the original
                        since_anchor_check = 0
            elif drift_floor > 0.0 and drift_every > 0 and since_anchor_check >= drift_every:
                cur = _extract(g, x, y, half)
                since_anchor_check = 0
                if cur is not None and _corr(cur, anchor) < drift_floor:
                    break                       # drifted off the feature -> stop this side
            i += direction

    _CERTAINTY["v"] = float(np.median(certs)) if certs else 0.0
    return [refined[k] for k in sorted(refined.keys())], "ok"


def measure_wobble(track: Track, max_period: int = 64) -> Tuple[float, int]:
    """(amplitude_px, dominant_period_frames) of a track's deviation from its own smooth path.

    Three rounds of "the tracks wobble" have been diagnosed by eye. The PERIOD is what names
    the cause: a beat clustering at the moving-tile window length points at that stage's
    seams, whereas a random walk has no dominant period at all. Detrending with a moving
    average leaves the wobble; the strongest FFT bin names its beat.

    Diagnostic only -- nothing is smoothed or filtered on the basis of it.
    """
    n = len(track)
    if n < 8:
        return (0.0, 0)
    pts = sorted(track, key=lambda p: p[0])
    xs = np.array([p[1] for p in pts], dtype=np.float64)
    ys = np.array([p[2] for p in pts], dtype=np.float64)

    k = max(3, min(9, (n // 4) | 1))          # odd moving-average window
    ker = np.ones(k) / float(k)
    rx = xs - np.convolve(xs, ker, mode="same")
    ry = ys - np.convolve(ys, ker, mode="same")
    edge = k // 2                              # convolve edges are unreliable -- drop them
    if n - 2 * edge < 6:
        return (0.0, 0)
    rx, ry = rx[edge:n - edge], ry[edge:n - edge]

    amp = float(np.hypot(rx.std(), ry.std()))
    m = len(rx)
    spec = np.abs(np.fft.rfft(rx)) + np.abs(np.fft.rfft(ry))
    spec[0] = 0.0                              # ignore DC
    bin_i = int(np.argmax(spec))
    period = int(round(m / bin_i)) if bin_i > 0 else 0
    if period > max_period:
        period = 0                             # too slow to be a periodic artifact
    return (amp, period)


def build_neighbour_predictor(tracks: Dict[str, Track], cfg):
    """Predict where an occluded point reappears, from the motion of its NEAREST neighbours.

    This is the one thing CoTracker genuinely does better than a per-point tracker -- using
    other points to infer a hidden one -- taken without adding a second network. Deliberately
    LOCAL: the median motion of the k nearest tracks, never a global model, because a global
    fit would drag foreground (parallax) points toward the background solution. The prediction
    only supplies a search CENTRE; the sub-pixel position still comes from native-res NCC.

    Returns predict(x, y, from_frame, to_frame) -> (x, y).
    """
    k = int(getattr(cfg, "reacquire_neighbours", 8) or 0)
    # frame -> list of (x, y, dx, dy) for every track visible on consecutive frames
    by_frame: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if k > 0:
        for tr in tracks.values():
            pts = sorted(tr, key=lambda t: t[0])
            for a, b in zip(pts, pts[1:]):
                if int(b[0]) - int(a[0]) != 1:
                    continue
                by_frame.setdefault(int(a[0]), []).append(
                    (float(a[1]), float(a[2]), float(b[1]) - float(a[1]), float(b[2]) - float(a[2])))

    def predict(x: float, y: float, f_from: int, f_to: int) -> Tuple[float, float]:
        if k <= 0 or f_to == f_from:
            return x, y
        step = 1 if f_to > f_from else -1
        cx, cy = float(x), float(y)
        for f in range(int(f_from), int(f_to), step):
            # motion of frame f -> f+1; walking backwards means subtracting it
            cand = by_frame.get(f if step > 0 else f - 1)
            if not cand:
                continue
            near = sorted(cand, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)[:k]
            if not near:
                continue
            dx = float(np.median([c[2] for c in near]))
            dy = float(np.median([c[3] for c in near]))
            cx += dx * step
            cy += dy * step
        return cx, cy

    return predict


def _fb_filter(seg: Track, get: Callable[[int], Optional[np.ndarray]], cfg,
               edge_clamp: bool) -> Track:
    """Forward-backward consistency: re-track the refined segment BACKWARDS and judge it.

    A correctly tracked point comes back to where it started; a mistrack does not. Purely
    self-referential, so parallax-safe -- it never compares a track against global motion,
    which would punish exactly the fast foreground points we want to keep.

    This is a verdict on the SEGMENT, not a per-frame delete. Deleting individual frames
    wherever the return pass disagreed punched holes through the middle of perfectly visible
    tracks -- they flickered on and off. The literature uses FB error the same way (Kalal's
    forward-backward error rejects a point, it does not perforate it), and 3DE wants a
    contiguous run, not a comb. So: a bad MEDIAN rejects the whole segment, and only a
    contiguous bad TAIL is trimmed (there, lock genuinely was lost). 0 = off.
    """
    tol = float(getattr(cfg, "refine_fb_max_px", 0.0) or 0.0)
    if tol <= 0.0 or len(seg) < 3:
        return seg
    half = int(cfg.refine_patch_px) // 2
    search = _adaptive_search(seg, cfg)
    lost = float(cfg.refine_ncc_lost)
    reref = float(cfg.refine_ncc_reref)
    ambig = float(getattr(cfg, "match_ambiguity_ratio", 1.0) or 1.0)

    g_last = get(int(seg[-1][0]) - 1)
    if g_last is None:
        return seg
    patch = _extract(g_last, float(seg[-1][1]), float(seg[-1][2]), half)
    if patch is None:
        return seg

    # Walk back, re-referencing exactly like the forward pass. Without this the single
    # end-frame patch decorrelates as it travels, so the FB error grew with distance from
    # the end and the test failed frames that were tracked perfectly well.
    errs: List[Tuple[int, float]] = []
    for f, fx, fy in reversed(seg[:-1]):
        g = get(int(f) - 1)
        if g is None:
            break
        res = _ncc_match(g, patch, fx, fy, search, half, edge_clamp, ambig)
        if res is None or res[2] < lost:
            break
        bx, by, cc = res
        errs.append((int(f), math.hypot(bx - float(fx), by - float(fy))))
        if cc < reref:
            np2 = _extract(g, bx, by, half)
            if np2 is not None:
                patch = np2
    if not errs:
        return seg

    if float(np.median([e for _f, e in errs])) > tol:
        return []              # the segment as a whole does not survive its own return trip

    # Trim only a contiguous bad TAIL (errs is newest-first, so that is its head).
    cut_from = None
    for f, e in errs:
        if e > tol:
            cut_from = f if cut_from is None else min(cut_from, f)
        else:
            break
    if cut_from is None:
        return seg
    return [p for p in seg if int(p[0]) < cut_from]


def _refine_one(track: Track, get: Callable[[int], Optional[np.ndarray]],
                cfg, predict=None) -> Optional[Track]:
    """Refine one track. Returns the refined points, or None to drop it.

    With split_unverified_segments on, this can return a LIST of segments instead (see
    _refine_one_multi) — kept as a thin wrapper so existing callers are unchanged.
    """
    res = _refine_one_multi(track, get, cfg, predict)
    if not res:
        return None
    # Callers that want one track get the segments welded back; verification already ran.
    out: Track = []
    for seg in res:
        out.extend(seg)
    return sorted(out, key=lambda t: t[0]) or None


def _refine_one_multi(track: Track, get: Callable[[int], Optional[np.ndarray]],
                      cfg, predict=None) -> List[Track]:
    """Refine one track into one or MORE verified pieces.

    A gap is only welded back into a single id when the post-gap patch still matches the
    pre-gap anchor. If it does not, the pieces come back separately: welding two different
    features under one id is the failure mode that stays invisible until the solve blows up,
    so on doubt it splits.
    """
    pts = sorted(track, key=lambda t: t[0])
    if len(pts) < 2:
        return []
    edge_clamp = bool(getattr(cfg, "mt_edge_track", True))
    gap_aware = bool(getattr(cfg, "refine_gap_aware", True))
    min_len = int(cfg.refine_min_len)

    if not gap_aware:
        out, _reason = _refine_segment(pts, get, cfg, edge_clamp, predict)
        if out is None:
            return []
        out = _fb_filter(out, get, cfg, edge_clamp)
        return [out] if len(out) >= min_len else []

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

    split_on_doubt = bool(getattr(cfg, "split_unverified_segments", True))
    reacq = float(getattr(cfg, "refine_ncc_reacquire", 0.75) or 0.75)
    half = int(cfg.refine_patch_px) // 2

    def _patch_at(pt):
        g = get(int(pt[0]) - 1)
        return _extract(g, float(pt[1]), float(pt[2]), half) if g is not None else None

    pieces: List[Track] = []
    combined: Track = []
    seg_certs: List[float] = []  # localisation certainty of each refined segment
    prev_tail_patch = None       # patch at the end of the last accepted segment
    for seg in segments:
        if len(seg) >= 2:
            ref, reason = _refine_segment(seg, get, cfg, edge_clamp, predict)
        else:
            ref, reason = None, "no-anchor"
        if reason == "edge":
            continue          # 1-D feature: drop these points; keeping them RAW would be
                              # both jittery (coarse 256px position) and still sliding.
        if reason == "ok":
            seg_certs.append(float(_CERTAINTY.get("v", 0.0)))
        use = ref if (ref is not None and len(ref) >= min_len) else seg
        if ref is not None:
            use = _fb_filter(use, get, cfg, edge_clamp)
            if len(use) < min_len:
                use = seg
        if not use:
            continue

        # Is this really the same feature we had before the gap? Compare the patch either
        # side of it; a mismatch means the tracker came back on something else.
        if combined and prev_tail_patch is not None:
            head_patch = _patch_at(use[0])
            same = head_patch is not None and _corr(head_patch, prev_tail_patch) >= reacq
            if not same and split_on_doubt:
                pieces.append(combined)
                combined = []
        combined.extend(use)
        tail = _patch_at(use[-1])          # explicit: `ndarray or x` is ambiguous in numpy
        if tail is not None:
            prev_tail_patch = tail

    if combined:
        pieces.append(combined)
    # Track-level certainty = the weakest refined segment, so one badly-localised stretch is
    # not averaged away by good ones.
    _CERTAINTY["v"] = float(min(seg_certs)) if seg_certs else 0.0
    return [sorted(p, key=lambda t: t[0]) for p in pieces if len(p) >= min_len]


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

    # Neighbour motion model, built once from the pre-refine tracks (image space, same frame
    # of reference the refine loop works in).
    predict = build_neighbour_predictor({k: to_img(v) for k, v in final_tracks.items()}, cfg)

    out: Dict[str, Track] = {}
    certainty: Dict[str, float] = {}
    trimmed = dropped = split = gapped = 0
    for name, tr in final_tracks.items():
        pieces = _refine_one_multi(to_img(tr), prov.get, cfg, predict)
        cert = float(_CERTAINTY.get("v", 0.0))
        if not pieces:
            dropped += 1
            continue
        total_pts = sum(len(p) for p in pieces)
        if total_pts < len(tr):
            trimmed += 1
        if len(pieces) > 1:
            split += 1
        for k, piece in enumerate(pieces):
            # A piece spanning more frames than it has points survived an occlusion and
            # carries a genuine gap -- 3DE reads that natively (a frame number per point).
            if piece and (int(piece[-1][0]) - int(piece[0][0]) + 1) > len(piece):
                gapped += 1
            # Only a piece that failed verification gets a new id; the first keeps the
            # original name so downstream naming is unchanged for the common case.
            tid = name if k == 0 else f"{name}_{chr(ord('b') + k - 1)}"
            out[tid] = to_out(piece)
            certainty[tid] = cert

    # Wobble report: amplitude says how bad, period says WHERE it comes from. A period
    # clustering near mt_window points at the moving-tile seams; no dominant period means a
    # random walk instead. Diagnostic only.
    if out:
        meas = [measure_wobble(t) for t in out.values()]
        amps = [a for a, _p in meas if a > 0.0]
        pers = [p for _a, p in meas if p > 1]
        if amps:
            med = float(np.median(amps))
            if pers:
                vals, counts = np.unique(np.array(pers), return_counts=True)
                modal = int(vals[int(np.argmax(counts))])
                share = int(100 * counts.max() / len(pers))
                status and status(f"Wobble: median {med:.3f}px, most common period "
                                  f"{modal}f ({share}% of tracks)")
            else:
                status and status(f"Wobble: median {med:.3f}px, no dominant period")

    aniso = float(getattr(cfg, "min_corner_anisotropy", 0.0) or 0.0)
    bits = [f"refined={len(out)}/{len(final_tracks)}", f"trimmed={trimmed}", f"dropped={dropped}"]
    if gapped:
        bits.append(f"survived-occlusion={gapped}")
    if split:
        bits.append(f"split-unverified={split}")
    if aniso > 0.0:
        bits.append(f"(edge-reject anisotropy<{aniso:.2f})")
    fb = float(getattr(cfg, "refine_fb_max_px", 0.0) or 0.0)
    if fb > 0.0:
        bits.append(f"(fwd-bwd<={fb:.1f}px)")
    if certainty:
        bits.append(f"certainty med={float(np.median(list(certainty.values()))):.2f}")
    # Hand the per-track certainty to the caller for selection. Stashed on the function so
    # the (tracks, info) return contract used by tracker_core is unchanged.
    refine_tracks.last_certainty = certainty
    return out, " ".join(bits)
