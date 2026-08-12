"""How often does a tracker lose its grip? Countable without any ground truth.

`bench/synth` measures LOCALISATION and needs exact truth to do it. It cannot measure a
tracker getting lost, because on one clean plane nothing does -- every configuration in
`sweep.py` kept 23/23 tracks for 100/100 frames, which is why that sweep's "the defaults
win" verdict says nothing about robustness.

Robustness needs no truth at all. A death is a hole in a track, and holes are countable
straight off the export. So a real 4K plate with real motion blur is usable as a bench the
moment it is on disk:

    deaths/track   internal gaps per track -- the headline
    run            unbroken frames between deaths; what an artist actually gets
    span           first to last frame, gaps included
    coverage       tracked frames / (tracks x shot length)

    runtime\\python311\\python.exe experiments\\blender_track\\track_stats.py ^
        experiments\\blender_track\\out\\SH006__dense__blender.txt --frames 312
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

from blio import read_3de  # noqa: E402

import numpy as np  # noqa: E402


def stats(path: str, n_frames: int = 0) -> dict:
    tracks = read_3de(path)
    if not tracks:
        return {"file": path, "tracks": 0}
    deaths, runs, spans, lens = [], [], [], []
    for pts in tracks.values():
        fs = sorted(pts)
        gaps = [(a, b) for a, b in zip(fs, fs[1:]) if b - a > 1]
        deaths.append(len(gaps))
        spans.append(fs[-1] - fs[0] + 1)
        lens.append(len(fs))
        # Unbroken runs: what the artist gets before having to re-key.
        run = 1
        for a, b in zip(fs, fs[1:]):
            if b - a == 1:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)

    T = n_frames or max(max(p) for p in tracks.values())
    return {
        "file": os.path.basename(path),
        "tracks": len(tracks),
        "frames": T,
        "deaths": int(sum(deaths)),
        "deaths_per_track": sum(deaths) / len(tracks),
        "clean": sum(1 for d in deaths if d == 0),
        "median_run": float(np.median(runs)),
        "mean_run": float(np.mean(runs)),
        "median_span": float(np.median(spans)),
        "full_length": sum(1 for s in spans if s >= T),
        "samples": int(sum(lens)),
        "coverage": sum(lens) / float(len(tracks) * T),
    }


def line(s: dict) -> str:
    if not s.get("tracks"):
        return f"{s['file']}: no tracks"
    return (f"{s['file'][:44]:<46}{s['tracks']:>5}{s['deaths_per_track']:>9.2f}"
            f"{s['clean']:>7}{s['median_run']:>9.0f}{s['median_span']:>8.0f}"
            f"{s['full_length']:>7}{100 * s['coverage']:>9.1f}")


HEADER = (f"{'export':<46}{'trk':>5}{'deaths/t':>9}{'clean':>7}{'med_run':>9}"
          f"{'span':>8}{'full':>7}{'cover%':>9}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", help="one or more 3DE export files")
    ap.add_argument("--frames", type=int, default=0,
                    help="shot length; inferred from the export when omitted, which "
                         "UNDERSTATES it if no track reaches the last frame")
    args = ap.parse_args()

    print(HEADER)
    print("-" * len(HEADER))
    for p in args.exports:
        if not os.path.isfile(p):
            print(f"{os.path.basename(p)}: missing")
            continue
        print(line(stats(p, args.frames)))
    print("\ndeaths/t = internal gaps per track; clean = tracks with none; "
          "med_run = unbroken frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
