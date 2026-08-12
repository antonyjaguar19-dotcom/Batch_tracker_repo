"""Did the seeds land where we asked?

A wrong plate-px -> u/v conversion would start every tracker on the wrong feature and still
produce a full, plausible-looking export -- exactly the kind of silent wrong answer this
repo's tooling exists to catch. So compare the injected seed positions against the FIRST
exported point of each track, in plate pixels.

Pairing is by NAME (tracks are HYB0000.. in seed order), not by proximity: a proximity match
would quietly hide the very error this is looking for.

    check_roundtrip.py --plate <mp4 or frames dir> --tracks out\\SH004__hybrid.txt
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

from plate_io import Plate  # noqa: E402

import numpy as np  # noqa: E402

from app.compare_tracks import load_tracks  # noqa: E402
from run_hybrid import tapnext_seeds  # noqa: E402
from sylab import OUT_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default=r"D:\Jefrin\IN\SH004.mp4")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--quality", type=float, default=0.02)
    ap.add_argument("--min-dist", type=int, default=12)
    ap.add_argument("--stagger", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=0.01, help="max allowed px offset")
    args = ap.parse_args()

    plate = Plate(args.plate, ifl_dir=OUT_DIR)
    export = args.tracks or os.path.join(OUT_DIR, f"{plate.name}__hybrid.txt")
    if not os.path.isfile(export):
        print(f"FAIL: no export at {export}")
        return 3

    seeds, _kinds = tapnext_seeds(plate, args.seeds, args.quality, args.min_dist, args.stagger)
    tracks = load_tracks(export)
    plate.close()
    print(f"{len(seeds)} seeds, {len(tracks)} exported tracks, plate {plate.w}x{plate.h}")

    deltas, missing = [], 0
    for i, (fr, sx, sy) in enumerate(seeds):
        tr = tracks.get(f"HYB{i:04d}")
        if not tr:
            missing += 1
            continue
        f0 = min(tr)
        ex, ey = tr[f0]
        deltas.append((i, f0, sx, sy, ex, ey, float(np.hypot(ex - sx, ey - sy))))

    if not deltas:
        print("FAIL: no tracks matched by name - naming or export is off")
        return 1
    d = np.array([r[6] for r in deltas])
    print(f"matched {len(deltas)} by name, {missing} missing")
    print(f"first-point offset vs injected seed: mean {d.mean():.4f}px  "
          f"median {np.median(d):.4f}px  max {d.max():.4f}px")
    for i, f0, sx, sy, ex, ey, dd in sorted(deltas, key=lambda r: -r[6])[:5]:
        print(f"  HYB{i:04d} startframe={f0}  seed=({sx:8.2f},{sy:8.2f})  "
              f"export=({ex:8.2f},{ey:8.2f})  d={dd:.3f}px")

    ok = float(d.max()) <= args.tolerance
    print(f"VERDICT: round-trip {'PASS' if ok else 'FAIL'} "
          f"(max {d.max():.4f}px, tolerance {args.tolerance}px)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
