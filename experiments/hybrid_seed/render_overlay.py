"""Burn the hybrid tracks over the plate so the result can be watched, not just read.

Points are coloured by the seed class the bot's seeder assigned, and a point that has
stopped moving is drawn hollow grey -- which makes the SynthEyes Demo 10-frame cap
visible directly in the video instead of only in the numbers.

    runtime\\python311\\python.exe experiments\\hybrid_seed\\render_overlay.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.compare_tracks import load_tracks  # noqa: E402
from run_hybrid import tapnext_seeds  # noqa: E402

# BGR
KIND_COLOR = {
    "corner":       (80, 255, 80),
    "blob":         (80, 200, 255),
    "edge":         (255, 160, 60),
    "dense-corner": (180, 255, 180),
    "dense-edge":   (255, 200, 140),
    "dense":        (200, 200, 200),
    "":             (255, 255, 255),
}
FROZEN = (110, 110, 110)
TRAIL = 12
FROZEN_EPS = 0.01   # px; SynthEyes' held-coord tail repeats a position exactly


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", default=r"D:\Jefrin\IN\SH004.mp4")
    ap.add_argument("--tracks", default=os.path.join(HERE, "out", "SH004__hybrid.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "SH004__hybrid_overlay.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    tracks = load_tracks(args.tracks)
    if not tracks:
        print(f"no tracks in {args.tracks}")
        return 1

    # Seed classes, in the same order the names were generated (HYB0000..).
    try:
        _pts, kinds, _w, _h, _t = tapnext_seeds(args.mp4, 400, 0.02, 12)
    except Exception as e:
        print(f"(could not recover seed classes: {e})")
        kinds = []
    kind_of = {f"HYB{i:04d}": k for i, k in enumerate(kinds)}

    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        print(f"could not open {args.mp4}")
        return 1
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / float(W0)
    W, H = args.width, int(round(H0 * scale))

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        print("could not open the writer")
        return 1

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1                      # exported frame numbers are 1-based
        img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        live = frozen = 0
        for name, pts in tracks.items():
            if fi not in pts:
                continue
            x, y = pts[fi]
            prev = pts.get(fi - 1)
            is_frozen = (prev is not None
                         and abs(prev[0] - x) < FROZEN_EPS
                         and abs(prev[1] - y) < FROZEN_EPS)
            px, py = int(round(x * scale)), int(round(y * scale))

            if is_frozen:
                frozen += 1
                cv2.circle(img, (px, py), 4, FROZEN, 1, cv2.LINE_AA)
                continue

            live += 1
            col = KIND_COLOR.get(kind_of.get(name, ""), KIND_COLOR[""])
            # short trail so motion is readable on a still frame
            trail = [pts[f] for f in range(max(1, fi - TRAIL), fi + 1) if f in pts]
            for a, b in zip(trail, trail[1:]):
                cv2.line(img,
                         (int(round(a[0] * scale)), int(round(a[1] * scale))),
                         (int(round(b[0] * scale)), int(round(b[1] * scale))),
                         col, 1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 4, col, -1, cv2.LINE_AA)

        bar = f"frame {fi:3d}   tracking {live:3d}   frozen {frozen:3d}   (SynthEyes Demo caps real tracking at 10 frames)"
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(img, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)

        y0 = H - 12 - 18 * len(KIND_COLOR)
        for k, c in KIND_COLOR.items():
            if k == "":
                continue
            cv2.circle(img, (16, y0), 5, c, -1, cv2.LINE_AA)
            cv2.putText(img, k, (28, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
            y0 += 18
        cv2.circle(img, (16, y0), 5, FROZEN, 1, cv2.LINE_AA)
        cv2.putText(img, "frozen (demo cap)", (28, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, FROZEN, 1, cv2.LINE_AA)

        vw.write(img)

    cap.release()
    vw.release()
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({fi} frames, {W}x{H}, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
