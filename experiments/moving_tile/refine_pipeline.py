# -*- coding: utf-8 -*-
"""Cumulative quality pipeline on top of v2 guided moving-tile (ISOLATED experiment).

Adds 7 quality features one by one, ON TOP of each other, and reports a comparative
metric table + per-stage zoomed render so each update can be judged:

  v0  green baseline           (guided moving-tile, from the npz)
  v1  + NCC sub-pixel refine   (translation, anchor=first)
  v2  + forward-backward cull  (backward guided pass, drop/trim inconsistent)
  v3  + sub-pixel corner seeds (cornerSubPix + min-eigen gate, retrack, re-apply 1+2)
  v4  + quality score & trim   (drop weak-lock tracks, trim sagging ends)
  v5  + affine pattern lock    (ECC rotation/scale instead of translation)
  v6  + anchor at sharpest fr  (reference patch from highest-contrast frame)
  v7  + spread/quality select  (farthest-point spatial spread of the best tracks)

Metrics (no ground truth needed, all self-consistent):
  N       surviving tracks
  len     mean visible frames
  jit     mean per-frame acceleration magnitude (px)  -> LOWER = smoother/less noise
  ncc     mean NCC lock of the patch at the track pos  -> HIGHER = tighter to contrast
  fberr   mean forward/backward disagreement (px)      -> LOWER = more trustworthy

The NCC/ECC primitives are copied (not imported) from app/pattern_refine.py so the bot
is untouched, matching the training/ isolation convention.
"""
from __future__ import annotations
import os, sys, math, argparse, importlib.util
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ---- copied primitives from app/pattern_refine.py (isolation: no bot import) --------
_MOTION = {"translation": cv2.MOTION_TRANSLATION,
           "euclidean": cv2.MOTION_EUCLIDEAN, "affine": cv2.MOTION_AFFINE}


def _extract(gray, x, y, half):
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > gray.shape[1] or yi + half + 1 > gray.shape[0]:
        return None
    return gray[yi - half:yi + half + 1, xi - half:xi + half + 1]


def _contrast_score(patch):
    p = patch.astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    a = float(np.mean(gx * gx)); b = float(np.mean(gx * gy)); c = float(np.mean(gy * gy))
    t = a + c; disc = max(0.0, t * t / 4.0 - (a * c - b * b))
    return t / 2.0 - math.sqrt(disc)


def _subpix_peak(resp, loc):
    mx, my = loc; h, w = resp.shape; dx = dy = 0.0
    if 0 < mx < w - 1:
        l, c, r = float(resp[my, mx - 1]), float(resp[my, mx]), float(resp[my, mx + 1])
        d = l - 2 * c + r
        if abs(d) > 1e-9: dx = 0.5 * (l - r) / d
    if 0 < my < h - 1:
        u, c, d0 = float(resp[my - 1, mx]), float(resp[my, mx]), float(resp[my + 1, mx])
        d = u - 2 * c + d0
        if abs(d) > 1e-9: dy = 0.5 * (u - d0) / d
    return (max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy)))


def _ncc_match(gray, patch, cx, cy, search, half):
    P = 2 * half + 1
    x0 = int(round(cx)) - half - search; y0 = int(round(cy)) - half - search
    win_w = P + 2 * search; win_h = P + 2 * search
    if x0 < 0 or y0 < 0 or x0 + win_w > gray.shape[1] or y0 + win_h > gray.shape[0]:
        return None
    win = gray[y0:y0 + win_h, x0:x0 + win_w]
    resp = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(resp)
    dx, dy = _subpix_peak(resp, maxloc)
    return (float(x0 + maxloc[0] + dx + half), float(y0 + maxloc[1] + dy + half), float(maxv))


def _ecc_refine(gray, patch, px, py, half, motion):
    P = 2 * half + 1
    cand = cv2.getRectSubPix(gray, (P, P), (float(px), float(py))).astype(np.float32)
    tmpl = patch.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    try:
        cc, warp = cv2.findTransformECC(tmpl, cand, warp, motion, crit, None, 5)
    except cv2.error:
        return None
    uv = warp.astype(np.float64) @ np.array([half, half, 1.0])
    return (px - half + float(uv[0]), py - half + float(uv[1]), float(cc))


# ---- track representation: list of dicts {id, f(1-based int), x, y} -----------------
def arrays_to_tracks(A, V):
    out = []
    for i in range(A.shape[1]):
        m = V[:, i] & ~np.isnan(A[:, i, 0])
        if m.sum() < 2:
            continue
        f = (np.where(m)[0] + 1).astype(int)
        out.append({"id": i, "f": f, "x": A[m, i, 0].astype(float), "y": A[m, i, 1].astype(float)})
    return out


def refine_track(tr, grays, half, search, motion_key, anchor, lost, reref, min_len):
    f, x, y = tr["f"], tr["x"], tr["y"]
    n = len(f)
    if n < 2:
        return None
    motion = _MOTION[motion_key]
    # patches + contrast
    patches, contrast = [], []
    for k in range(n):
        g = grays[f[k] - 1]
        p = _extract(g, x[k], y[k], half)
        patches.append(p)
        contrast.append(_contrast_score(p) if p is not None else -1.0)
    valid = [k for k in range(n) if patches[k] is not None]
    if not valid:
        return None
    if anchor == "sharpest":
        a = max(valid, key=lambda k: contrast[k])
    else:  # first valid
        a = valid[0]
    patch0 = patches[a]
    refined = {a: (f[a], float(x[a]), float(y[a]))}
    for direction in (1, -1):
        patch = patch0.copy(); k = a + direction
        while 0 <= k < n:
            g = grays[f[k] - 1]
            res = _ncc_match(g, patch, x[k], y[k], search, half)
            if res is None or res[2] < lost:
                break
            rx, ry, cc = res
            if motion != cv2.MOTION_TRANSLATION:
                er = _ecc_refine(g, patch, rx, ry, half, motion)
                if er is not None and er[2] >= lost:
                    rx, ry, cc = er
            refined[k] = (f[k], float(rx), float(ry))
            if cc < reref:
                np2 = _extract(g, rx, ry, half)
                if np2 is not None:
                    patch = np2
            k += direction
    ks = sorted(refined.keys())
    if len(ks) < min_len:
        return None
    return {"id": tr["id"],
            "f": np.array([refined[k][0] for k in ks], int),
            "x": np.array([refined[k][1] for k in ks], float),
            "y": np.array([refined[k][2] for k in ks], float)}


def refine_all(tracks, grays, half=15, search=24, motion_key="translation",
               anchor="first", lost=0.55, reref=0.85, min_len=8):
    out = []
    for tr in tracks:
        r = refine_track(tr, grays, half, search, motion_key, anchor, lost, reref, min_len)
        if r is not None:
            out.append(r)
    return out


# ---- metrics -----------------------------------------------------------------------
def m_jitter(tr):
    f, x, y = tr["f"], tr["x"], tr["y"]
    acc = []
    for k in range(1, len(f) - 1):
        if f[k] - f[k - 1] == 1 and f[k + 1] - f[k] == 1:
            acc.append(math.hypot(x[k + 1] - 2 * x[k] + x[k - 1], y[k + 1] - 2 * y[k] + y[k - 1]))
    return float(np.mean(acc)) if acc else float("nan")


def m_ncc(tr, grays, half=15, search=6):
    f, x, y = tr["f"], tr["x"], tr["y"]
    # anchor sharpest for a fair lock measure
    best_c, patch = -1.0, None
    for k in range(len(f)):
        p = _extract(grays[f[k] - 1], x[k], y[k], half)
        if p is None:
            continue
        c = _contrast_score(p)
        if c > best_c:
            best_c, patch = c, p
    if patch is None:
        return float("nan")
    ccs = []
    for k in range(len(f)):
        res = _ncc_match(grays[f[k] - 1], patch, x[k], y[k], search, half)
        if res is not None:
            ccs.append(res[2])
    return float(np.mean(ccs)) if ccs else float("nan")


def metrics(tracks, grays, fberr=None, T=1):
    if not tracks:
        return dict(N=0, length=0, jit=float("nan"), ncc=float("nan"), fb=float("nan"))
    lens = [len(t["f"]) for t in tracks]
    jit = np.nanmean([m_jitter(t) for t in tracks])
    ncc = np.nanmean([m_ncc(t, grays) for t in tracks])
    fb = float("nan")
    if fberr is not None:
        vals = [fberr.get(t["id"], np.nan) for t in tracks]
        vals = [v for v in vals if not np.isnan(v)]
        fb = float(np.mean(vals)) if vals else float("nan")
    return dict(N=len(tracks), length=float(np.mean(lens)), jit=float(jit),
                ncc=float(ncc), fb=fb)


# ---- forward-backward -------------------------------------------------------------
def fb_error(engine, frames, gd, gd_vis, bl, bl_vis, guided_fn):
    """Backward guided pass from each point's last visible pos; per-id mean fwd/bwd gap."""
    T, N = gd_vis.shape
    seeds_last = np.zeros((N, 2), np.float32)
    for i in range(N):
        idx = np.where(gd_vis[:, i] & ~np.isnan(gd[:, i, 0]))[0]
        seeds_last[i] = gd[idx[-1], i] if len(idx) else (bl[0, i] if bl_vis[0, i] else (0, 0))
    fr = frames[::-1]
    bt, bv = guided_fn(engine, fr, seeds_last, bl[::-1], bl_vis[::-1])
    bt = bt[::-1]; bv = bv[::-1]                          # back to original frame order
    err = {}
    for i in range(N):
        m = gd_vis[:, i] & bv[:, i] & ~np.isnan(gd[:, i, 0]) & ~np.isnan(bt[:, i, 0])
        if m.sum() >= 4:
            d = np.hypot(gd[m, i, 0] - bt[m, i, 0], gd[m, i, 1] - bt[m, i, 1])
            err[i] = float(np.median(d))
    return err, bt, bv


def fb_cull(tracks, fberr, bt, bv, drop_med=2.5, trim_px=3.0):
    """Drop tracks whose fwd/bwd median gap is large; trim frames where they diverge."""
    out = []
    for tr in tracks:
        i = tr["id"]
        if i in fberr and fberr[i] > drop_med:
            continue
        keep = np.ones(len(tr["f"]), bool)
        for k, fr in enumerate(tr["f"]):
            t = fr - 1
            if bv[t, i] and not np.isnan(bt[t, i, 0]):
                if math.hypot(tr["x"][k] - bt[t, i, 0], tr["y"][k] - bt[t, i, 1]) > trim_px:
                    keep[k] = False
        if keep.sum() >= 8:
            out.append({"id": i, "f": tr["f"][keep], "x": tr["x"][keep], "y": tr["y"][keep]})
    return out


# ---- feature 3: better seeds ------------------------------------------------------
def seed_corner_subpix(frame_bgr, max_pts=40, quality=0.02, min_dist=48, margin=140,
                       min_eig=3.0):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    mask = np.zeros((H, W), np.uint8); mask[margin:H - margin, margin:W - margin] = 255
    pts = cv2.goodFeaturesToTrack(g, max_pts * 4, quality, min_dist, mask=mask,
                                  useHarrisDetector=False)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    cv2.cornerSubPix(g, pts, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts = pts.reshape(-1, 2).astype(np.float32)
    # min-eigen gate: reject flat/soft (water, skin, haze)
    good = []
    for p in pts:
        patch = _extract(g, p[0], p[1], 7)
        if patch is not None and _contrast_score(patch) >= min_eig:
            good.append(p)
    pts = np.array(good, np.float32) if good else pts
    # even spread
    keep = [0]
    while len(keep) < min(max_pts, len(pts)):
        d = np.min([np.hypot(pts[:, 0] - pts[k, 0], pts[:, 1] - pts[k, 1]) for k in keep], axis=0)
        d[keep] = -1
        keep.append(int(np.argmax(d)))
    return pts[keep]


# ---- feature 7: spread/quality selection ------------------------------------------
def spread_select(tracks, grays, keep_n=20):
    if len(tracks) <= keep_n:
        return tracks
    score = np.array([m_ncc(t, grays) * min(1.0, len(t["f"]) / 60.0) for t in tracks])
    pos = np.array([[t["x"][0], t["y"][0]] for t in tracks])
    order = list(np.argsort(-score))
    chosen = [order[0]]
    while len(chosen) < keep_n and len(chosen) < len(tracks):
        best, bestv = -1, -1
        for j in order:
            if j in chosen:
                continue
            dmin = min(math.hypot(*(pos[j] - pos[c])) for c in chosen)
            v = dmin * (0.5 + score[j])       # spread weighted by quality
            if v > bestv:
                bestv, best = v, j
        chosen.append(best)
    return [tracks[j] for j in chosen]


# ---- rendering: fixed image locations, zoom montage -------------------------------
def nearest_track(tracks, loc, t):
    best, bd = None, 1e9
    for tr in tracks:
        k = np.where(tr["f"] == t + 1)[0]
        if len(k) == 0:
            continue
        d = math.hypot(tr["x"][k[0]] - loc[0], tr["y"][k[0]] - loc[1])
        if d < bd and d < 60:
            bd, best = d, (tr["x"][k[0]], tr["y"][k[0]])
    return best


def render_stage(frames, locs, tracks, t, out_png, box=70, zoom=6, cols=3, title=""):
    H, W = frames.shape[1], frames.shape[2]
    ph = box * zoom; rows = int(np.ceil(len(locs) / cols))
    grid = np.zeros((rows * ph, cols * ph, 3), np.uint8)
    for j, loc in enumerate(locs):
        ox = int(round(loc[0] - box / 2)); oy = int(round(loc[1] - box / 2))
        ox = max(0, min(ox, W - box)); oy = max(0, min(oy, H - box))
        crop = frames[t, oy:oy + box, ox:ox + box]
        panel = cv2.resize(crop, (ph, ph), interpolation=cv2.INTER_NEAREST)
        p = nearest_track(tracks, loc, t)
        if p is not None:
            px = int(round((p[0] - ox) * zoom)); py = int(round((p[1] - oy) * zoom))
            cv2.drawMarker(panel, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            cv2.circle(panel, (px, py), 1, (0, 255, 0), -1)
        r, c = divmod(j, cols)
        grid[r * ph:(r + 1) * ph, c * ph:(c + 1) * ph] = panel
    cv2.putText(grid, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_png, grid)


# ---- engine + guided loader --------------------------------------------------------
def load_helpers():
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    spec = importlib.util.spec_from_file_location(
        "btr_tapnext_engine", os.path.join(_REPO, "app", "tapnext_engine.py"))
    eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)
    spec2 = importlib.util.spec_from_file_location(
        "btr_mt", os.path.join(_HERE, "track_moving_tile.py"))
    mt = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mt)
    return eng.TapNextEngine, mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(_HERE, "out", "SH011_arrays.npz"))
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out", "stages"))
    ap.add_argument("--half", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Z = np.load(args.npz, allow_pickle=True)
    video = str(Z["video"])
    Engine, mt = load_helpers()
    frames, fps = mt.decode_all(video)
    T, H, W = frames.shape[:3]
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], 0)
    print(f"[data] {os.path.basename(video)} {T}f {W}x{H}", flush=True)

    gd, gd_vis = Z["gd"], Z["gd_vis"]
    bl, bl_vis = Z["bl"], Z["bl_vis"]

    print("[engine] loading ...", flush=True)
    engine = Engine(tool_root=_REPO, device="cuda")

    rows = []   # (label, metrics)
    render_t = int(T * 0.5)

    # fixed display locations from the strongest v0 tracks
    v0 = arrays_to_tracks(gd, gd_vis)
    v0_scored = sorted(v0, key=lambda t: -m_ncc(t, grays))
    locs = []
    for tr in v0_scored:
        k = np.where(tr["f"] == render_t + 1)[0]
        if len(k):
            locs.append((tr["x"][k[0]], tr["y"][k[0]]))
        if len(locs) >= 6:
            break

    def record(label, tracks, fberr=None):
        mm = metrics(tracks, grays, fberr, T)
        rows.append((label, mm))
        png = os.path.join(args.outdir, f"{label.split()[0]}.png")
        render_stage(frames, locs, tracks, render_t, png,
                     title=f"{label}  N={mm['N']} len={mm['length']:.0f} "
                           f"jit={mm['jit']:.3f} ncc={mm['ncc']:.3f} fb={mm['fb']:.2f}")
        print(f"[{label}] N={mm['N']} len={mm['length']:.1f} jit={mm['jit']:.3f} "
              f"ncc={mm['ncc']:.3f} fb={mm['fb']:.2f}", flush=True)
        return mm

    # v0 baseline
    record("v0", v0)

    # v1 NCC translation refine (anchor=first)
    v1 = refine_all(v0, grays, half=args.half, motion_key="translation", anchor="first")
    record("v1", v1)

    # v2 forward-backward cull
    fberr, bt, bv = fb_error(engine, frames, gd, gd_vis, bl, bl_vis, mt.moving_tile_track_guided)
    v2 = fb_cull(v1, fberr, bt, bv)
    record("v2", v2, fberr)

    # v3 better seeds -> retrack baseline+guided -> re-apply refine + fb
    seeds3 = seed_corner_subpix(frames[0])
    q = np.zeros((1, seeds3.shape[0], 3), np.float32); q[0, :, 1] = seeds3[:, 0]; q[0, :, 2] = seeds3[:, 1]
    bl3, blv3 = engine.track_queries(frames, q)
    gd3, gdv3 = mt.moving_tile_track_guided(engine, frames, seeds3, bl3, blv3)
    t3 = arrays_to_tracks(gd3, gdv3)
    t3 = refine_all(t3, grays, half=args.half, motion_key="translation", anchor="first")
    fberr3, bt3, bv3 = fb_error(engine, frames, gd3, gdv3, bl3, blv3, mt.moving_tile_track_guided)
    v3 = fb_cull(t3, fberr3, bt3, bv3)
    record("v3", v3, fberr3)

    # v4 quality score + trim (drop weak lock)
    v4 = [t for t in v3 if not np.isnan(m_ncc(t, grays)) and m_ncc(t, grays) >= 0.60]
    record("v4", v4, fberr3)

    # v5 affine pattern lock
    v5 = refine_all(v4, grays, half=args.half, motion_key="affine", anchor="first")
    record("v5", v5, fberr3)

    # v6 anchor at sharpest frame
    v6 = refine_all(v4, grays, half=args.half, motion_key="affine", anchor="sharpest")
    record("v6", v6, fberr3)

    # v7 spread + quality selection
    v7 = spread_select(v6, grays, keep_n=20)
    record("v7", v7, fberr3)

    # comparative table
    print("\n==== COMPARATIVE (SH011) ====", flush=True)
    print(f"{'stage':<5} {'N':>3} {'len':>6} {'jit(px)':>8} {'ncc':>6} {'fb(px)':>7}  note", flush=True)
    notes = {"v0": "guided baseline", "v1": "+NCC subpix", "v2": "+FB cull",
             "v3": "+corner seeds", "v4": "+quality drop", "v5": "+affine",
             "v6": "+sharpest anchor", "v7": "+spread select"}
    for label, m in rows:
        print(f"{label:<5} {m['N']:>3} {m['length']:>6.1f} {m['jit']:>8.3f} "
              f"{m['ncc']:>6.3f} {m['fb']:>7.2f}  {notes[label]}", flush=True)

    # combined montage v0 vs v7 at render_t already saved as separate pngs.
    np.savez_compressed(os.path.join(args.outdir, "final_tracks.npz"),
                        video=video, locs=np.array(locs))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
