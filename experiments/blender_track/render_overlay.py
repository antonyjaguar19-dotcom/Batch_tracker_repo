"""Burn two track files over the plate so the difference can be watched, not only read.

The bot's guide is drawn in one colour and the Blender hybrid in another, with a line
joining the two where both are alive on the same feature. A track that is currently in a
GAP -- dead, waiting for its replant -- is drawn as a hollow ring, so the thing replant is
for is visible directly instead of only in the median-length number.

    runtime\\python311\\python.exe experiments\\blender_track\\render_overlay.py ^
        --plate experiments\\blender_track\\out\\SH004\\plate ^
        --bot   experiments\\blender_track\\out\\SH004__repl__blender.txt ^
        --guide experiments\\blender_track\\out\\SH004\\runs\\guide\\SH004__tapnext.txt
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

from blio import Plate, OUT_DIR, read_3de  # noqa: E402  (sets OPENCV_IO_ENABLE_OPENEXR)

import cv2  # noqa: E402

BLENDER = (90, 255, 120)      # BGR
GUIDE = (80, 160, 255)
GAP = (110, 110, 110)
TRAIL = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--bot", required=True, help="the Blender hybrid export")
    ap.add_argument("--guide", default="", help="the bot export it was seeded from")
    ap.add_argument("--out", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()

    plate = Plate(args.plate, ifl_dir=OUT_DIR)
    bl = read_3de(args.bot)
    gd = read_3de(args.guide) if args.guide else {}
    out_path = args.out or os.path.join(
        OUT_DIR, os.path.splitext(os.path.basename(args.bot))[0] + "_overlay.mp4")

    scale = args.width / float(plate.w)
    W, H = args.width, int(round(plate.h * scale))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        print("could not open the writer")
        return 1

    def to_px(pt):
        # Exports are y-up (flip_y_for_3de); the image is y-down.
        return int(round(pt[0] * scale)), int(round((plate.h - 1.0 - pt[1]) * scale))

    fi = 0
    for f0 in range(plate.count):
        img = plate.frame(f0)
        if img is None:
            break
        fi = f0 + 1
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        live = gaps = 0

        for name, pts in bl.items():
            if fi in pts:
                live += 1
                p = to_px(pts[fi])
                trail = [pts[f] for f in range(max(1, fi - TRAIL), fi + 1) if f in pts]
                for a, b in zip(trail, trail[1:]):
                    cv2.line(img, to_px(a), to_px(b), BLENDER, 1, cv2.LINE_AA)
                cv2.circle(img, p, 4, BLENDER, -1, cv2.LINE_AA)
                g = gd.get(name)
                if g and fi in g:
                    cv2.line(img, p, to_px(g[fi]), GUIDE, 1, cv2.LINE_AA)
                    cv2.circle(img, to_px(g[fi]), 3, GUIDE, 1, cv2.LINE_AA)
            elif pts and min(pts) < fi < max(pts):
                # Inside the track's own span but with no sample: this is the gap a replant
                # has to cross. Drawn where the guide thinks the feature is.
                gaps += 1
                g = gd.get(name)
                if g and fi in g:
                    cv2.circle(img, to_px(g[fi]), 5, GAP, 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(img, f"frame {fi:4d}/{plate.count}   blender {live:3d}   in-gap {gaps:3d}"
                    f"   {os.path.basename(args.bot)}", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (lbl, col) in enumerate((("blender (tracked)", BLENDER),
                                        ("tapnext guide", GUIDE),
                                        ("in gap, awaiting replant", GAP))):
            y = H - 60 + 18 * i
            cv2.circle(img, (16, y), 5, col, -1 if i == 0 else 1, cv2.LINE_AA)
            cv2.putText(img, lbl, (28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                        cv2.LINE_AA)
        vw.write(img)

    plate.close()
    vw.release()
    print(f"wrote {out_path}  ({fi} frames, {W}x{H}, "
          f"{os.path.getsize(out_path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
