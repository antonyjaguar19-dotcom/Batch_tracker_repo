# -*- coding: utf-8 -*-
"""Independently re-track EVERY exported track, and report each one's error.

The pairing problem is what makes production footage hard to measure. A reference placed on
a feature of my choosing usually has no bot track on it -- spread selection deliberately
keeps one point per neighbourhood, so it lands 30-70px away on a different feature -- and
`eval_refs` then either scores the distance between two different features as error (it
paired a track 23.9px away under a 25px threshold and called it 23.91px of error) or reports
MISSED. Neither says anything about the tracker.

Seeding the reference at the BOT'S OWN starting position removes the problem entirely: the
reference follows the same feature by construction, so every exported track gets a number
and no pairing threshold is involved.

Independence and self-check are unchanged from tools/make_lk_reference.py: pyramidal
Lucas-Kanade (a different algorithm family from the bot's NCC+ECC, so pixel-locking bias is
not shared), forward-backward gated per step, and a full round trip whose CLOSURE bounds the
reference's own precision. A row whose closure is not far below its disagreement is not
evidence about the tracker, and is marked accordingly.
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\Jefrin\batch_tracker_v001_starter")
from app.compare_tracks import load_tracks          # noqa: E402

LK = dict(winSize=(31, 31), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.001))


def lk_run(frames, p0, lo, hi):
    """LK-track one point from frame lo to hi. -> {frame_idx: (x, y)}, stops when lost."""
    step = 1 if hi >= lo else -1
    cur = np.array([[p0]], np.float32)
    out = {lo: (float(cur[0, 0, 0]), float(cur[0, 0, 1]))}
    for i in range(lo, hi, step):
        nxt, s1, _ = cv2.calcOpticalFlowPyrLK(frames[i], frames[i + step], cur, None, **LK)
        back, _s2, _ = cv2.calcOpticalFlowPyrLK(frames[i + step], frames[i], nxt, None, **LK)
        if s1.ravel()[0] != 1:
            break
        if float(np.linalg.norm(back - cur)) > 0.3:      # per-step forward-backward gate
            break
        cur = nxt
        out[i + step] = (float(cur[0, 0, 0]), float(cur[0, 0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--bot", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--min-frames", type=int, default=20)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.mp4)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    cap.release()
    H = frames[0].shape[0]
    T = len(frames)

    bot = load_tracks(a.bot)
    rows = []
    for name, p in bot.items():
        fr = sorted(p)
        if len(fr) < a.min_frames:
            continue
        f0 = fr[0]
        seed = (p[f0][0], float(H - 1) - p[f0][1])        # un-flip to image space
        lo, hi = f0 - 1, fr[-1] - 1                        # 0-based
        if not (0 <= lo < T and 0 <= hi < T):
            continue
        fwd = lk_run(frames, seed, lo, hi)
        if len(fwd) < a.min_frames:
            continue
        end = max(fwd)
        rt = lk_run(frames, fwd[end], end, lo)             # round trip back to the seed
        closure = (math.hypot(rt[lo][0] - seed[0], rt[lo][1] - seed[1])
                   if lo in rt else float("nan"))
        errs = []
        for f in fr:
            i = f - 1
            if i in fwd:
                errs.append(math.hypot(p[f][0] - fwd[i][0],
                                       (float(H - 1) - p[f][1]) - fwd[i][1]))
        if len(errs) < a.min_frames:
            continue
        rows.append((st.mean(errs), max(errs), closure, len(errs), len(fr), name))

    rows.sort()
    print(f"--- {a.label}   {len(rows)} of {len(bot)} exported tracks independently re-tracked")
    print(f"{'track':<14}{'mean_err':>9}{'max_err':>9}{'closure':>9}{'frames':>8}   verdict")
    trust = []
    for m, mx, cl, n, tot, name in rows:
        ok = (not math.isnan(cl)) and cl < max(0.5, 0.5 * m)
        if ok:
            trust.append(m)
        print(f"{name:<14}{m:9.2f}{mx:9.2f}{cl:9.2f}{n:8d}   "
              f"{'usable' if ok else 'reference not tight enough - ignore'}")
    if trust:
        t = sorted(trust)
        print(f"\nUSABLE ROWS ({len(t)}): median {st.median(t):.2f}px  "
              f"p90 {t[int(0.9 * (len(t) - 1))]:.2f}px  worst {t[-1]:.2f}px")
        print(f"  over 1px: {sum(1 for v in t if v > 1.0)}/{len(t)}   "
              f"over 3px: {sum(1 for v in t if v > 3.0)}/{len(t)}")

        # Stratify by how good the REFERENCE is, because the line above is not the bot's
        # error -- it is the disagreement between two trackers, and it inherits whatever
        # the reference got wrong. The `usable` rule allows closure up to half the
        # disagreement, so a row with a 0.9px reference can sit in that median unremarked.
        # On the shot this was written for, tightening the reference moved the figure from
        # 2.22px to 1.00px: over half of what was being reported as tracker error was the
        # reference's own imprecision. Read the tightest band that still has rows in it.
        print("\n  by reference quality (the tightest band is the real estimate):")
        for cut in (0.5, 0.3, 0.2, 0.1):
            band = sorted(m for m, mx, cl, n, tot, name in rows
                          if not math.isnan(cl) and cl < cut)
            if len(band) >= 3:
                print(f"    closure < {cut:.1f}px : n={len(band):3d}   median {st.median(band):5.2f}px"
                      f"   worst {band[-1]:6.2f}px"
                      f"   over 1px {sum(1 for v in band if v > 1.0)}/{len(band)}")
    else:
        print("\nno row had a reference tight enough to be evidence")


if __name__ == "__main__":
    main()
