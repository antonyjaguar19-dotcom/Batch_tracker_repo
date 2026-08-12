"""Why is tk.Run() so slow on a 4K plate?

On SH004 (2560x1440) the hybrid appeared to run at ~1,400 tracker-frames/s. That number was
an artefact: the Demo licence killed most trackers at frame 10, and Run() on a DEAD tracker
is a no-op, so the timer was mostly measuring nothing happening. On SH016 (4096x2160) five
trackers over 127 frames did not finish in 400s.

This times one tracker over a fixed frame count at several patch/search sizes. If the time
tracks the sizes, the cost is the area match and the geometry must be specified in pixels
rather than normalised units. If every size costs the same, the cost is per-Run image
access and the geometry is innocent.

    bench_run.py --plate D:\\Jefrin\\IN\\SH016.mp4 --frames 15
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

from plate_io import Plate, to_uv, pick_feature  # noqa: E402
from sylab import connect, run_szl, OUT_DIR, DIAG  # noqa: E402

DIAG_FWD = DIAG.replace("\\", "/")


def variant(eng, u, v, size, srch, nframes, label):
    lines = [
        "//SIZZLET LabBenchRun",
        "ob = Scene.activeObj", "shot = ob.shot", "start = shot.start",
        "frame = start", "tk = new ob.trk", f'tk.nm = "B{label}"',
        "tk.kind = 0", f"tk.size = {size:.5f}", "tk.asp = 1.0",
        f"tk.srchu = {srch:.5f}", f"tk.srchv = {srch:.5f}",
        "tk.smooth = 20", "tk.autokey = 20", "tk.isSel = 2",
        f"tk.key = Point({u:.6f},{v:.6f})", "tk.isEnabled = 1", "x = tk.Run()",
        f"for (f = start + 1; f <= start + {nframes}; f++)",
        "    frame = f", "    x = tk.Run()", "end",
        "cnt = 0",
        f"for (frame = start; frame <= start + {nframes}; frame++)",
        "    if (tk.valid)", "        cnt = cnt + 1", "    end", "end",
        f'openout("{DIAG_FWD}")', 'printf("valid %d\\n", cnt)', "closeout()",
    ]
    t0 = time.time()
    out = run_szl(eng, "\n".join(lines) + "\n", watchdog=1200)
    dt = time.time() - t0
    valid = out.split()[1] if out.split() else "?"
    print(f"  size={size:<8.5f} srch={srch:<8.5f}  {dt:7.1f}s for {nframes} frames "
          f"({dt/max(nframes,1)*1000:8.1f} ms/frame)  valid={valid}", flush=True)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default=r"D:\Jefrin\IN\SH016.mp4")
    ap.add_argument("--frames", type=int, default=15)
    args = ap.parse_args()

    plate = Plate(args.plate, ifl_dir=OUT_DIR)
    print(f"plate {plate.name}: {plate.w}x{plate.h}, {plate.count} frames")
    img = plate.frame(0)
    (cx, cy) = pick_feature(img)[0]
    u, v = to_uv(cx, cy, plate.w, plate.h)
    plate.close()

    eng = connect(quiet=True)
    eng.set_writable_folder(OUT_DIR)
    if eng.hlev.NewSceneAndShot(plate.load_path) is None:
        print("FAIL: NewSceneAndShot returned None")
        return 3
    time.sleep(6)

    print(f"timing tk.Run() over {args.frames} frames, one tracker:")
    # u/v are half-width units (u spans -1..1 across the plate), so a patch of P pixels is
    # size = 2*P/W. These four cover a 16x span in patch area.
    for px, spx in ((12, 10), (24, 20), (48, 40), (96, 80)):
        variant(eng, u, v, 2.0 * px / plate.w, 2.0 * spx / plate.w, args.frames,
                f"{px}_{spx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
