"""Movie -> a folder of PNG frames, laid out as a bench shot.

Both engines must read the SAME pixels. The bot decodes with OpenCV and Blender with
ffmpeg; on the same mp4 those do not always agree to the sub-pixel level this experiment
measures at, and any disagreement would be scored as tracking error. Decoding once, to
lossless PNG, removes that as a variable. PNG rather than JPEG for the same reason.

The output is shaped like a bench shot (`<out>/plate/000001.png`), because that is what
`bench/run_bench.py` takes -- which is how the guide gets built through the shipping
AppState path instead of a hand-rolled RunnerConfig.

    runtime\\python311\\python.exe experiments\\blender_track\\extract_frames.py ^
        --mp4 D:\\Jefrin\\IN\\SH016.mp4 --out experiments\\blender_track\\out\\SH016
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import blio  # noqa: E402,F401  (sets OPENCV_IO_ENABLE_OPENEXR before cv2)

import cv2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--out", required=True, help="shot folder; frames land in <out>/plate")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if the frame count already matches")
    args = ap.parse_args()

    dst = os.path.join(os.path.abspath(args.out), "plate")
    cap = cv2.VideoCapture(args.mp4)
    if not cap.isOpened():
        print(f"could not open {args.mp4}")
        return 1
    want = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    have = len([f for f in os.listdir(dst)
                if f.lower().endswith(".png")]) if os.path.isdir(dst) else 0
    if have == want and not args.force:
        print(f"{os.path.basename(args.out)}: {have} frames already extracted, skipping")
        cap.release()
        return 0

    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        cv2.imwrite(os.path.join(dst, "%06d.png" % i), img)
    cap.release()

    # The container's frame count is a header value and is not always the truth; the number
    # that matters downstream is how many files actually exist.
    print(f"{os.path.basename(args.out)}: {i} frames ({w}x{h}) in {time.time() - t0:.0f}s"
          + (f"  [header said {want}]" if i != want else ""))
    return 0 if i else 1


if __name__ == "__main__":
    raise SystemExit(main())
