"""Did the seeds land where we asked?

A wrong plate-px -> u/v conversion would start every tracker on the wrong feature and
still produce a full, plausible-looking export -- exactly the kind of silent wrong answer
this repo's tooling exists to catch. So compare the injected seed positions against the
FIRST exported point of each track, in plate pixels.

    runtime\\python311\\python.exe experiments\\hybrid_seed\\check_roundtrip.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from app.compare_tracks import load_tracks  # noqa: E402
from run_hybrid import tapnext_seeds  # noqa: E402

MP4 = r"D:\Jefrin\IN\SH004.mp4"
EXPORT = os.path.join(HERE, "out", "SH004__hybrid.txt")


def main() -> int:
    pts, kinds, w, h, total = tapnext_seeds(MP4, 400, 0.02, 12)
    tracks = load_tracks(EXPORT)
    print(f"{len(pts)} seeds, {len(tracks)} exported tracks, plate {w}x{h}")

    # Tracks are named HYB0000.. in seed order, so the pairing is exact -- no proximity
    # matching, which is the whole point of this check.
    deltas = []
    missing = 0
    for i, (sx, sy) in enumerate(pts):
        nm = f"HYB{i:04d}"
        tr = tracks.get(nm)
        if not tr:
            missing += 1
            continue
        f0 = min(tr)
        ex, ey = tr[f0]
        deltas.append((i, f0, sx, sy, ex, ey, float(np.hypot(ex - sx, ey - sy))))

    if not deltas:
        print("no tracks matched by name -- naming or export is off")
        return 1
    d = np.array([r[6] for r in deltas])
    print(f"matched {len(deltas)} by name, {missing} missing")
    print(f"first-point offset vs injected seed: "
          f"mean {d.mean():.3f}px  median {np.median(d):.3f}px  max {d.max():.3f}px")
    worst = sorted(deltas, key=lambda r: -r[6])[:5]
    print("worst 5:")
    for i, f0, sx, sy, ex, ey, dd in worst:
        print(f"  HYB{i:04d} startframe={f0}  seed=({sx:8.2f},{sy:8.2f})  "
              f"export=({ex:8.2f},{ey:8.2f})  d={dd:.2f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
