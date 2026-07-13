# -*- coding: utf-8 -*-
"""SH013 ACCURACY vs manual GT: seed the pipeline at the artist's own points, measure
per-frame px deviation (ISOLATED). Bounded 1..150 to stay light. No blur (shelved)."""
from __future__ import annotations
import os, sys, math, importlib.util
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "training", "extract"))
from _common import parse_tracks_txt  # noqa

GT = os.path.join(_REPO, "training", "Assets for training", "Tracks", "Tracks_Shot_13.txt")
VIDEO = "D:/Jefrin/IN/SH013.mp4"
H, W, T = 1440, 2562, 150


def load_helpers():
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    def by_path(n, p):
        s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    eng = by_path("btr_eng", os.path.join(_REPO, "app", "tapnext_engine.py"))
    mt = by_path("btr_mt", os.path.join(_HERE, "track_moving_tile.py"))
    rp = by_path("btr_rp", os.path.join(_HERE, "refine_pipeline.py"))
    return eng.TapNextEngine, mt, rp


def main():
    _, gt = parse_tracks_txt(GT)
    # manual tracks present at frame 1 -> seed there (image coords, 3DE Y flip)
    seeds, names, man = [], [], {}
    for nm, pts in gt.items():
        d = {int(f): (x, (H - 1) - y) for (f, x, y) in pts if f <= T}
        if 1 in d:
            seeds.append(d[1]); names.append(nm); man[nm] = d
    seeds = np.array(seeds, np.float32)
    print(f"[seed] {len(seeds)} manual points at frame 1", flush=True)

    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while len(frames) < T:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    frames = np.asarray(frames)
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], 0)

    Engine, mt, rp = load_helpers()
    engine = Engine(tool_root=_REPO, device="cuda")
    q = np.zeros((1, len(seeds), 3), np.float32); q[0, :, 1] = seeds[:, 0]; q[0, :, 2] = seeds[:, 1]

    print("[BL] baseline", flush=True)
    bl, blv = engine.track_queries(frames, q)
    print("[MT] moving-tile", flush=True)
    gd, gdv = mt.moving_tile_track_guided(engine, frames, seeds, bl, blv)
    print("[CB] +NCC refine", flush=True)
    cb = rp.refine_all(rp.arrays_to_tracks(gd, gdv), grays, half=15, search=28,
                       motion_key="translation", anchor="sharpest", min_len=6)

    def dev(pred_arr, pred_vis):
        errs, surv = [], []
        for i, nm in enumerate(names):
            m = man[nm]
            e = []
            for t in range(T):
                if pred_vis[t, i] and (t + 1) in m and not np.isnan(pred_arr[t, i, 0]):
                    e.append(math.hypot(pred_arr[t, i, 0] - m[t + 1][0], pred_arr[t, i, 1] - m[t + 1][1]))
            if e:
                errs += e
                # survival = predicted visible frames / manual frames in range
                surv.append(pred_vis[:, i].sum() / max(1, len(m)))
        errs = np.array(errs) if errs else np.array([np.nan])
        return errs, (np.mean(surv) if surv else 0)

    eb, sb = dev(bl, blv)
    eg, sg = dev(gd, gdv)
    # combo -> arrays
    cb_arr = np.full((T, len(names), 2), np.nan, np.float32); cb_vis = np.zeros((T, len(names)), bool)
    idx = {nm: i for i, nm in enumerate(names)}
    for tr in cb:
        i = tr["id"]
        for k, f in enumerate(tr["f"]):
            if 1 <= f <= T:
                cb_arr[f - 1, i] = (tr["x"][k], tr["y"][k]); cb_vis[f - 1, i] = True
    ec, sc = dev(cb_arr, cb_vis)

    print("\n==== SH013 ACCURACY vs manual (seeded at manual pts, frames 1-%d) ====" % T, flush=True)
    print(f"{'stage':<10} {'mean':>6} {'med':>6} {'p90':>7} {'max':>7} {'survival':>9}", flush=True)
    for nm, e, s in (("baseline", eb, sb), ("moving-tile", eg, sg), ("+NCC(full)", ec, sc)):
        print(f"{nm:<10} {np.nanmean(e):>6.2f} {np.nanmedian(e):>6.2f} "
              f"{np.nanpercentile(e,90):>7.2f} {np.nanmax(e):>7.2f} {s*100:>8.0f}%", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
