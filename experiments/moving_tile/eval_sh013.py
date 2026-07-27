# -*- coding: utf-8 -*-
"""Evaluate the bot's SH013 output vs the manual artist track (ISOLATED, CPU only).

Metrics for a fast motion-blur shot:
  CONTINUITY  track count + lifetime distribution (are tracks long / gapless?)
  SCATTER     spatial spread + frame-grid coverage per frame (is the frame populated?)
  ACCURACY    per frame, nearest bot track to each manual feature (does the bot put a
              track where the artist did? low = good coverage of real features)
Both tracks are 3DE Y-up ascii; drawing flips y_img = (H-1)-y.
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "training", "extract"))
from _common import parse_tracks_txt  # noqa

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--bot", default=os.path.join(_HERE, "out", "bot_sh013", "SH013__tapnext.txt"))
_ap.add_argument("--T", type=int, default=150)
_ap.add_argument("--tag", default="")
_A = _ap.parse_args()

VIDEO = "D:/Jefrin/IN/SH013.mp4"
BOT = _A.bot
GT = os.path.join(_REPO, "training", "Assets for training", "Tracks", "Tracks_Shot_13.txt")
OUT = os.path.join(_HERE, "out", "eval_sh013")
os.makedirs(OUT, exist_ok=True)
W, H = 2562, 1440


def by_frame(tracks):
    d = {}
    for name, pts in tracks.items():
        for (f, x, y) in pts:
            d.setdefault(int(f), []).append((x, y))
    return d


def alive_per_frame(tracks, T):
    a = np.zeros(T + 1, int)
    for pts in tracks.values():
        for (f, x, y) in pts:
            if 1 <= f <= T:
                a[f] += 1
    return a[1:T + 1]


def grid_cover(bf, T, gx=8, gy=6):
    """fraction of gx*gy cells occupied per frame, averaged."""
    cov = []
    for t in range(1, T + 1):
        cells = set()
        for (x, y) in bf.get(t, []):
            cells.add((int(x / W * gx), int(y / H * gy)))
        cov.append(len(cells) / float(gx * gy))
    return np.array(cov)


def spread(bf, T):
    sx, sy = [], []
    for t in range(1, T + 1):
        p = bf.get(t, [])
        if len(p) >= 2:
            a = np.array(p); sx.append(a[:, 0].std()); sy.append(a[:, 1].std())
    return (np.mean(sx) if sx else 0, np.mean(sy) if sy else 0)


def nn_accuracy(gt_bf, bot_bf, T):
    """per frame, median nearest bot-point distance to each manual point (px)."""
    ds = []
    for t in range(1, T + 1):
        g = gt_bf.get(t, []); b = bot_bf.get(t, [])
        if not g or not b:
            continue
        b = np.array(b)
        for (gxp, gyp) in g:
            ds.append(np.min(np.hypot(b[:, 0] - gxp, b[:, 1] - gyp)))
    ds = np.array(ds) if ds else np.array([np.nan])
    return ds


def main():
    _, bot = parse_tracks_txt(BOT)
    _, gt = parse_tracks_txt(GT)
    T = _A.T  # bot range
    bot_bf, gt_bf = by_frame(bot), by_frame(gt)

    print("==== SH013 vs manual GT (frames 1-%d) ====" % T, flush=True)
    # CONTINUITY
    for name, tr in (("BOT", bot), ("MANUAL", gt)):
        lens = np.array([sum(1 for (f, x, y) in v if f <= T) for v in tr.values()])
        lens = lens[lens > 0]
        gapped = sum(1 for v in tr.values()
                     if (fs := sorted(int(f) for (f, x, y) in v if f <= T)) and len(fs) >= 2
                     and (fs[-1] - fs[0] + 1) > len(fs))
        print(f"[continuity] {name:6s}: tracks={len(lens):3d} len mean={lens.mean():5.1f} "
              f"med={np.median(lens):5.1f} >=50f={int((lens>=50).sum()):3d} gapped={gapped}", flush=True)

    # SCATTER
    ba, ga = alive_per_frame(bot, T), alive_per_frame(gt, T)
    bc, gc = grid_cover(bot_bf, T), grid_cover(gt_bf, T)
    bsx, bsy = spread(bot_bf, T); gsx, gsy = spread(gt_bf, T)
    print(f"[scatter]  BOT   : alive/fr mean={ba.mean():5.1f} min={ba.min():3d}  "
          f"grid-cover={bc.mean()*100:4.1f}%  spread(px) x={bsx:.0f} y={bsy:.0f}", flush=True)
    print(f"[scatter]  MANUAL: alive/fr mean={ga.mean():5.1f} min={ga.min():3d}  "
          f"grid-cover={gc.mean()*100:4.1f}%  spread(px) x={gsx:.0f} y={gsy:.0f}", flush=True)

    # ACCURACY (nearest bot to each manual feature)
    nn = nn_accuracy(gt_bf, bot_bf, T)
    print(f"[accuracy] nearest bot->manual (px): mean={np.nanmean(nn):.1f} "
          f"med={np.nanmedian(nn):.1f} p90={np.nanpercentile(nn,90):.1f}  "
          f"(<=5px:{np.nanmean(nn<=5)*100:.0f}% <=15px:{np.nanmean(nn<=15)*100:.0f}%)", flush=True)

    # coverage curve
    Wc, Hc, pad = 1000, 380, 46
    img = np.full((Hc, Wc, 3), 255, np.uint8)
    ymax = max(ba.max(), ga.max(), 1)
    def X(t): return pad + int((t) / max(1, T - 1) * (Wc - 2 * pad))
    def Y(v): return Hc - pad - int(v / ymax * (Hc - 2 * pad))
    cv2.line(img, (pad, Hc - pad), (Wc - pad, Hc - pad), (0, 0, 0), 1)
    for arr, col, lbl in ((ga, (200, 0, 0), "MANUAL"), (ba, (0, 150, 0), "BOT")):
        for t in range(1, T):
            cv2.line(img, (X(t - 1), Y(arr[t - 1])), (X(t), Y(arr[t])), col, 2, cv2.LINE_AA)
    cv2.putText(img, "tracks alive per frame  (green BOT vs red MANUAL)", (pad, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, f"SH013{_A.tag}_coverage.png"), img)

    # overlay video 1..T
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    scale = 1920.0 / W
    ow, oh = int(W * scale), int(H * scale)
    vw = cv2.VideoWriter(os.path.join(OUT, f"SH013{_A.tag}_eval_overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    for t in range(1, T + 1):
        ok, fr = cap.read()
        if not ok:
            break
        im = cv2.resize(fr, (ow, oh), interpolation=cv2.INTER_AREA)
        for (x, y) in gt_bf.get(t, []):
            cv2.circle(im, (int(x * scale), int((H - 1 - y) * scale)), 6, (0, 0, 255), 1, cv2.LINE_AA)
        for (x, y) in bot_bf.get(t, []):
            cv2.drawMarker(im, (int(x * scale), int((H - 1 - y) * scale)), (0, 220, 0),
                           cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)
        cv2.putText(im, "RED(o)=manual  GREEN(+)=bot   f%d/%d" % (t, T),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(im)
    cap.release(); vw.release()
    print("[done] wrote coverage plot + overlay", flush=True)


if __name__ == "__main__":
    main()
