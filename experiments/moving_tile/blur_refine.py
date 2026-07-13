# -*- coding: utf-8 -*-
"""Motion-blur refine prototypes (ISOLATED experiment — bot untouched).

Stacks blur-aware features onto the NCC pattern-refine that the bot uses. Same coarse
moving-tile positions go in; only the refine changes, so a comparison isolates the blur
features.

  #1 blur-matched template: NCC fails on blur because a SHARP reference patch != the
     smeared frame. Estimate blur length+direction from the coarse-track velocity, convolve
     the reference patch with that linear motion kernel, and NCC-match the BLURRED template.
  #2 blur-adaptive thresholds: blur flattens the NCC peak, so a fixed lost/reref over-trims
     valid-but-soft frames. Relax both proportional to the estimated blur length.

Primitives (_extract/_contrast_score/_ncc_match/_ecc_refine) are reused from refine_pipeline
(which copied them from app/pattern_refine.py) so nothing imports the bot.
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
# use the BOT's edge-aware NCC + primitives so the full-stack test includes edge tracking
from app.pattern_refine import _extract, _contrast_score, _ncc_match, _MOTION  # noqa: E402
from blur_cepstral import estimate_blur_kernel  # noqa: E402

# matched-kernel bank: sharp (0) + a coarse grid of length x angle. NCC picks the best,
# so no fragile single estimate; L=0 wins on sharp frames -> no regression there.
_BANK = [(0.0, 0.0)] + [(L, math.radians(a))
                        for L in (7.0, 13.0, 19.0, 25.0)
                        for a in (0, 30, 60, 90, 120, 150)]


def line_kernel(length: float, angle_rad: float) -> np.ndarray:
    """Normalized linear motion-blur kernel of `length` px at `angle_rad` (image coords)."""
    L = max(1, int(round(length)))
    if L <= 1:
        k = np.ones((1, 1), np.float32)
        return k
    s = L if L % 2 == 1 else L + 1          # odd canvas so the line is centred
    k = np.zeros((s, s), np.float32)
    c = s // 2
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    for t in np.linspace(-L / 2.0, L / 2.0, L * 2):
        x = int(round(c + dx * t)); y = int(round(c + dy * t))
        if 0 <= x < s and 0 <= y < s:
            k[y, x] = 1.0
    ssum = k.sum()
    return k / ssum if ssum > 0 else k


def _velocity(frs, xs, ys, k, direction):
    """Local per-frame velocity (px/frame) at segment index k from the coarse path."""
    j = k - direction
    if 0 <= j < len(frs) and abs(int(frs[k]) - int(frs[j])) == 1:
        return (xs[k] - xs[j], ys[k] - ys[j])
    # fall back to the other neighbour
    j = k + direction
    if 0 <= j < len(frs) and abs(int(frs[j]) - int(frs[k])) == 1:
        return (xs[j] - xs[k], ys[j] - ys[k])
    return (0.0, 0.0)


def refine_track_blur(tr, grays, half=15, search=28, anchor="sharpest",
                      lost=0.55, reref=0.85, min_len=6,
                      blur_match=True, adaptive_thresh=True,
                      shutter=0.7, blur_min_px=2.0, sharp_anchor=True, blur_ratio=0.6,
                      blur_est="bank", cep_half=32, edge_clamp=True):
    """Blur-aware NCC refine of one track. Returns refined dict {id,f,x,y} or None.

    blur_match  -> feature #1 (blur the template to match the frame)
    adaptive_thresh -> feature #2 (relax lost/reref by blur length)
    sharp_anchor -> anchor at the LEAST-blurred frame (Laplacian variance), feature #3(anchor)
    """
    f, x, y = tr["f"], tr["x"], tr["y"]
    n = len(f)
    if n < 2:
        return None

    # patches + a sharpness score per frame (Laplacian variance = blur-aware anchor pick)
    patches, sharp, contrast = [], [], []
    for k in range(n):
        g = grays[f[k] - 1]
        p = _extract(g, x[k], y[k], half)
        patches.append(p)
        if p is None:
            sharp.append(-1.0); contrast.append(-1.0)
        else:
            sharp.append(float(cv2.Laplacian(p, cv2.CV_64F).var()))
            contrast.append(_contrast_score(p))
    valid = [k for k in range(n) if patches[k] is not None]
    if not valid:
        return None
    if anchor == "sharpest" and sharp_anchor:
        a = max(valid, key=lambda k: sharp[k])       # least-blurred frame
    elif anchor == "sharpest":
        a = max(valid, key=lambda k: contrast[k])
    else:
        a = valid[0]

    s_anchor = max(1e-6, sharp[a])                    # anchor sharpness (Laplacian var)
    refined = {a: (int(f[a]), float(x[a]), float(y[a]))}
    for direction in (1, -1):
        patch = patches[a].copy()
        s_ref = s_anchor
        k = a + direction
        while 0 <= k < n:
            g = grays[f[k] - 1]
            # IMAGE-based blur gate: only treat this frame as blurred if its patch is
            # measurably softer than the reference (screen velocity from a pan does NOT
            # mean the image is blurred -> gating on velocity blurs sharp templates).
            cur_p = _extract(g, x[k], y[k], half)
            s_cur = float(cv2.Laplacian(cur_p, cv2.CV_64F).var()) if cur_p is not None else s_ref
            blurred = (s_cur < blur_ratio * s_ref)
            lost_k, reref_k = lost, reref
            if adaptive_thresh and blurred:            # blur flattens the peak -> soften gate
                soft = min(0.35, max(0.0, (blur_ratio * s_ref - s_cur) / (blur_ratio * s_ref + 1e-6)) * 0.5)
                lost_k = lost * (1.0 - soft); reref_k = reref * (1.0 - soft * 0.6)

            if not (blur_match and blurred):
                res = _ncc_match(g, patch, x[k], y[k], search, half, edge_clamp=edge_clamp)
            elif blur_est == "bank":
                # try sharp + a bank of blur kernels, keep the best-correlating match
                res = None
                for (bl, ba) in _BANK:
                    up = patch if bl <= 0 else cv2.filter2D(patch, -1, line_kernel(bl, ba))
                    r0 = _ncc_match(g, up, x[k], y[k], search, half, edge_clamp=edge_clamp)
                    if r0 is not None and (res is None or r0[2] > res[2]):
                        res = r0
            elif blur_est == "cepstral":
                reg = _extract(g, x[k], y[k], cep_half)
                L, ang, _conf = estimate_blur_kernel(reg) if reg is not None else (0.0, 0.0, 0.0)
                up = cv2.filter2D(patch, -1, line_kernel(L, ang)) if L >= blur_min_px else patch
                res = _ncc_match(g, up, x[k], y[k], search, half, edge_clamp=edge_clamp)
            else:  # velocity
                vx, vy = _velocity(f, x, y, k, direction)
                blen = math.hypot(vx, vy) * shutter
                up = cv2.filter2D(patch, -1, line_kernel(blen, math.atan2(vy, vx))) if blen >= blur_min_px else patch
                res = _ncc_match(g, up, x[k], y[k], search, half, edge_clamp=edge_clamp)

            if res is None or res[2] < lost_k:
                break
            rx, ry, cc = res
            refined[k] = (int(f[k]), float(rx), float(ry))
            if cc < reref_k and not blurred:          # re-grab only from a SHARP frame
                np2 = _extract(g, rx, ry, half)
                if np2 is not None:
                    patch = np2
                    s_ref = max(1e-6, float(cv2.Laplacian(np2, cv2.CV_64F).var()))
            k += direction

    ks = sorted(refined.keys())
    if len(ks) < min_len:
        return None
    return {"id": tr["id"],
            "f": np.array([refined[k][0] for k in ks], int),
            "x": np.array([refined[k][1] for k in ks], float),
            "y": np.array([refined[k][2] for k in ks], float)}


def refine_all_blur(tracks, grays, **kw):
    out = []
    for tr in tracks:
        r = refine_track_blur(tr, grays, **kw)
        if r is not None:
            out.append(r)
    return out


def refine_track_full(tr, grays, min_len=6, **kw):
    """Full-stack refine of one track: GAP-AWARE (per contiguous segment, own anchor) +
    blur-bank + edge-clamp. A segment that won't refine keeps its original points, so
    disappear/reappear frames are never dropped."""
    f = tr["f"]
    n = len(f)
    if n < 2:
        return None
    segs, cur = [], [0]
    for i in range(1, n):
        if int(f[i]) - int(f[i - 1]) == 1:
            cur.append(i)
        else:
            segs.append(cur); cur = [i]
    segs.append(cur)
    of, ox, oy = [], [], []
    for seg in segs:
        if len(seg) >= 2:
            sub = {"id": tr["id"], "f": f[seg], "x": tr["x"][seg], "y": tr["y"][seg]}
            r = refine_track_blur(sub, grays, min_len=min_len, **kw)
            if r is not None and len(r["f"]) >= min_len:
                of += list(r["f"]); ox += list(r["x"]); oy += list(r["y"]); continue
        for i in seg:   # keep original (never drop the reappeared segment)
            of.append(int(f[i])); ox.append(float(tr["x"][i])); oy.append(float(tr["y"][i]))
    if len(of) < 2:
        return None
    o = np.argsort(of)
    return {"id": tr["id"], "f": np.array(of)[o], "x": np.array(ox)[o], "y": np.array(oy)[o]}


def refine_all_full(tracks, grays, **kw):
    return [r for r in (refine_track_full(t, grays, **kw) for t in tracks) if r is not None]
