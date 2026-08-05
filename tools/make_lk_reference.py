# -*- coding: utf-8 -*-
"""Build an INDEPENDENT reference for a real plate, and bound its own precision.

Why this is worth anything: the bot's accuracy on production footage could not be measured,
because refs/gt4k's plate is missing and nobody has hand-tracked the shots that are here. An
artist's hand-track is still the gold standard. This is the next best thing available
without one, and it is honest about which parts are assumptions.

Two design choices make it usable as a reference rather than just a second opinion:

  * DIFFERENT ALGORITHM FAMILY. The bot refines with NCC template matching plus an ECC
    polish. A reference built the same way would share its systematic biases -- pixel
    locking above all -- and those biases would cancel in the comparison, making the bot
    look better than it is. This uses pyramidal Lucas-Kanade (gradient-based, iterative),
    whose failure modes are different.
  * IT MEASURES ITS OWN ERROR. Each candidate is tracked all the way forward and then all
    the way back to frame 1. On a rigid feature the round trip must return to where it
    started, so the closure error is an upper bound on that reference track's precision --
    computed without knowing the truth. Only features whose closure stays well under the
    scale being measured are kept; the rest are discarded rather than trusted.

What it is NOT: ground truth. A feature that LK and NCC both mistrack the same way is
invisible here, and closure cannot catch a drift that happens to retrace itself. Treat a
disagreement as "these two methods differ by X", and read the closure column before
believing any row.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, r"D:\Jefrin\batch_tracker_v001_starter")
from app.export_3de import write_tracks_txt          # noqa: E402

LK = dict(winSize=(31, 31), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.001))


def load_gray(mp4: str, n_max: int = 0):
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if n_max and len(frames) >= n_max:
            break
    cap.release()
    return frames


def track_chain(frames, pts, lo, hi):
    """Track pts from frame lo to hi (inclusive, either direction). -> list of (idx, Nx2, ok)."""
    step = 1 if hi >= lo else -1
    cur = pts.copy()
    alive = np.ones(len(pts), bool)
    out = [(lo, cur.copy(), alive.copy())]
    for i in range(lo, hi, step):
        nxt, st_, _err = cv2.calcOpticalFlowPyrLK(frames[i], frames[i + step], cur, None, **LK)
        back, _st2, _e2 = cv2.calcOpticalFlowPyrLK(frames[i + step], frames[i], nxt, None, **LK)
        fb = np.linalg.norm(back - cur, axis=2).ravel()
        good = (st_.ravel() == 1) & (fb < 0.3)
        alive = alive & good
        cur = nxt
        out.append((i + step, cur.copy(), alive.copy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--out", required=True, help="refs/<name> folder to create")
    ap.add_argument("--max-closure", type=float, default=0.25,
                    help="px; reject a reference track whose round trip misses by more")
    ap.add_argument("--n", type=int, default=12, help="how many references to keep")
    ap.add_argument("--quality", type=float, default=0.10)
    ap.add_argument("--min-dist", type=int, default=80)
    a = ap.parse_args()

    frames = load_gray(a.mp4)
    H, W = frames[0].shape[:2]
    T = len(frames)
    print(f"{os.path.basename(a.mp4)}: {T} frames at {W}x{H}")

    # Strong, well-separated features on frame 1 -- a reference is only worth building on
    # something unambiguous.
    p0 = cv2.goodFeaturesToTrack(frames[0], maxCorners=600, qualityLevel=a.quality,
                                 minDistance=a.min_dist, blockSize=7)
    if p0 is None:
        raise SystemExit("no features")
    p0 = p0.reshape(-1, 1, 2).astype(np.float32)
    print(f"candidates on frame 1: {len(p0)}")

    fwd = track_chain(frames, p0, 0, T - 1)
    last_idx, last_pts, last_alive = fwd[-1]
    print(f"survived to the last frame: {int(last_alive.sum())}")

    # Round trip: from the final position, all the way back.
    bwd = track_chain(frames, last_pts, T - 1, 0)
    _, back_pts, back_alive = bwd[-1]
    closure = np.linalg.norm(back_pts - p0, axis=2).ravel()

    ok = last_alive & back_alive & (closure <= a.max_closure)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(closure[idx])][:a.n]
    print(f"closure <= {a.max_closure}px: {int(ok.sum())}   keeping best {len(idx)}")
    if len(idx) == 0:
        raise SystemExit("no track closed tightly enough to serve as a reference")

    # 3DE ASCII, same convention the bot exports in (1-based frames, y flipped).
    tracks, kinds = {}, {}
    for k in idx:
        pts = []
        for (i, cur, alive) in fwd:
            if not alive[k]:
                break
            x, y = float(cur[k, 0, 0]), float(cur[k, 0, 1])
            pts.append((i + 1, x, float(H - 1) - y))
        if len(pts) >= 20:
            name = f"ref{k:03d}"
            tracks[name] = pts
            kinds[name] = "corner"
    os.makedirs(a.out, exist_ok=True)
    write_tracks_txt(os.path.join(a.out, "manual.txt"), tracks, end_frame=T)
    with open(os.path.join(a.out, "refs.json"), "w", encoding="utf-8") as f:
        json.dump(kinds, f, indent=2)
    with open(os.path.join(a.out, "reference.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": a.mp4, "frames": T, "resolution": [W, H],
            "method": "pyramidal Lucas-Kanade, forward-backward gated at 0.3px per step",
            "independence": "gradient-based; the bot refines with NCC + ECC, so systematic "
                            "biases (pixel locking) are not shared",
            "closure_px": {f"ref{int(k):03d}": round(float(closure[k]), 4) for k in idx},
            "max_closure_allowed": a.max_closure,
            "caveat": "NOT an artist hand-track. Closure bounds precision only for drift "
                      "that does not retrace itself; a fault both methods share is invisible.",
        }, f, indent=2)
    print(f"wrote {len(tracks)} reference tracks -> {a.out}")
    print(f"  closure of kept refs: min {closure[idx].min():.3f}px  "
          f"median {np.median(closure[idx]):.3f}px  max {closure[idx].max():.3f}px")


if __name__ == "__main__":
    main()
