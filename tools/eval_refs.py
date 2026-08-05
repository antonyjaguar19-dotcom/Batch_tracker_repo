# -*- coding: utf-8 -*-
"""Score a bot export against a whole set of hand-tracked references at once.

`app.compare_tracks` answers "is this one track better or worse?". That is the right
question while chasing a single fault, and the wrong one for per-track policy: the whole
point of giving a corner and a blob different settings is that the change does DIFFERENT
things to different features. Averaged over one reference that is invisible -- a rule that
helps corners and hurts blobs reads as no change at all, and gets thrown away.

So: one row per reference, labelled by what kind of feature it is.

    python tools/eval_refs.py refs/SH004 --bot out/SH004/SH004__tapnext.txt
    python tools/eval_refs.py refs/SH004 --bot new.txt --baseline refs/SH004/baseline_bot.txt

Layout of a reference folder:

    refs/SH004/
      manual.txt          hand tracks, classic 3DE ASCII (one file or several)
      refs.json           {"658": "corner", "661": "blob", ...}
      baseline_bot.txt    optional: the bot export to compare against

Pairing is by position, not by name -- the bot names its tracks BWD_0010 and the artist
names theirs 658, and neither knows about the other. Each manual track is matched to the
bot track closest to it over their shared frames. A match further away than --max-dist is
reported as MISSED rather than scored, because a confident number computed from the wrong
feature is worse than no number.

No torch, no cv2, same as compare_tracks: this has to run on any machine.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.compare_tracks import Track, compare, load_tracks  # noqa: E402

# Columns pulled from <shot>__trackreport.csv when it sits next to the bot export. They are
# the proof that the policy actually differed per track: if every row shows the same patch
# size, the classifier never fired and a pixel improvement came from somewhere else.
PROOF_COLS = ("kind", "patch_px", "motion")


def _load_refs(ref_dir: str) -> Tuple[Dict[str, Track], Dict[str, str]]:
    """Load every hand-track file in the folder, plus the class labels."""
    tracks: Dict[str, Track] = {}
    for path in sorted(glob.glob(os.path.join(ref_dir, "*.txt"))):
        if os.path.basename(path).startswith("baseline_bot"):
            continue
        for name, pts in load_tracks(path).items():
            if name in tracks:
                raise SystemExit(f"duplicate reference track {name!r} in {path}")
            tracks[name] = pts
    if not tracks:
        raise SystemExit(f"no reference tracks found in {ref_dir}")

    labels: Dict[str, str] = {}
    lab_path = os.path.join(ref_dir, "refs.json")
    if os.path.isfile(lab_path):
        with open(lab_path, "r", encoding="utf-8") as f:
            labels = {str(k): str(v) for k, v in json.load(f).items()}
    return tracks, labels


def _mean_pos(pts: Track, frames: List[int]) -> Tuple[float, float]:
    xs = [pts[f][0] for f in frames]
    ys = [pts[f][1] for f in frames]
    return st.mean(xs), st.mean(ys)


def match(ref: Track, bot_tracks: Dict[str, Track], max_dist: float,
          min_common: int = 5) -> Tuple[Optional[str], float]:
    """Find the bot track sitting on the same feature as this reference.

    Distance is the mean separation over the frames both cover, not the separation at one
    frame: two features can pass close to each other for a moment, and matching on that
    instant picks the wrong one.
    """
    best_name, best_d = None, float("inf")
    for name, pts in bot_tracks.items():
        common = sorted(set(pts) & set(ref))
        if len(common) < min_common:
            continue
        d = st.mean(math.hypot(pts[f][0] - ref[f][0], pts[f][1] - ref[f][1]) for f in common)
        if d < best_d:
            best_name, best_d = name, d
    if best_name is None or best_d > max_dist:
        return None, best_d
    return best_name, best_d


def _read_report(bot_path: str) -> Dict[str, Dict[str, str]]:
    """Pick up <shot>__trackreport.csv beside the export, if the run wrote one."""
    stem = bot_path
    for suffix in ("__tapnext.txt", "__syntheyes.txt", ".txt"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    path = stem + "__trackreport.csv"
    if not os.path.isfile(path):
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        for line in f:
            cells = line.rstrip("\n").split(",")
            if len(cells) != len(header):
                continue
            row = dict(zip(header, cells))
            rows[row.get("name", "")] = row
    return rows


def score(ref: Track, bot_tracks: Dict[str, Track], max_dist: float) -> Optional[dict]:
    """Match then compare. compare() prints; we only want its return value here."""
    name, dist = match(ref, bot_tracks, max_dist)
    if name is None:
        return None
    res = compare(bot_tracks[name], ref, out=io.StringIO())
    res["bot_track"] = name
    res["match_dist"] = dist
    return res


def _cell(res: Optional[dict], base: Optional[dict], key: str, width: int = 6,
          prec: int = 2) -> str:
    """One metric, as 'new' or 'base->new' when a baseline was given."""
    def fmt(d: Optional[dict]) -> str:
        if not d or key not in d or d[key] != d[key]:   # missing, or NaN
            return "-"
        return f"{d[key]:.{prec}f}"
    if base is None:
        return fmt(res).rjust(width)
    return f"{fmt(base)}->{fmt(res)}".rjust(width * 2 + 2)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a bot export against a folder of hand-tracked references.")
    ap.add_argument("ref_dir", help="folder holding the hand tracks and refs.json")
    ap.add_argument("--bot", required=True, help="bot export to score")
    ap.add_argument("--baseline", default=None,
                    help="earlier bot export to show as 'before->after'")
    ap.add_argument("--max-dist", type=float, default=25.0,
                    help="reject a pairing further apart than this, in px (default 25)")
    a = ap.parse_args(argv)

    refs, labels = _load_refs(a.ref_dir)
    bot = load_tracks(a.bot)
    base = load_tracks(a.baseline) if a.baseline else None
    report = _read_report(a.bot)

    ab = base is not None
    w = 14 if ab else 6
    print(f"references {len(refs)} from {a.ref_dir}   bot {len(bot)} tracks from {a.bot}")
    if ab:
        print(f"baseline   {len(base)} tracks from {a.baseline}   (shown as before->after)")
    print()
    head = (f"{'track':<8}{'class':<10}{'mean_err':>{w}} {'first_step':>{w}} "
            f"{'fast':>{w}} {'slow':>{w}} {'jerk':>{w}}")
    if report:
        head += "  " + " ".join(f"{c:>8}" for c in PROOF_COLS)
    print(head)
    print("-" * len(head))

    scored: List[dict] = []
    missed: List[str] = []
    for name in sorted(refs):
        res = score(refs[name], bot, a.max_dist)
        if res is None:
            missed.append(name)
            print(f"{name:<8}{labels.get(name, '?'):<10}  MISSED -- no bot track within "
                  f"{a.max_dist:.0f}px")
            continue
        bres = score(refs[name], base, a.max_dist) if ab else None
        row = (f"{name:<8}{labels.get(name, '?'):<10}"
               f"{_cell(res, bres, 'mean_err')} {_cell(res, bres, 'first_step_loss')} "
               f"{_cell(res, bres, 'fast_ratio')} {_cell(res, bres, 'slow_ratio')} "
               f"{_cell(res, bres, 'jerk_ratio')}")
        if report:
            r = report.get(res["bot_track"], {})
            row += "  " + " ".join(f"{r.get(c, '-'):>8}" for c in PROOF_COLS)
        print(row)
        scored.append(res)

    print()
    if scored:
        print(f"scored {len(scored)}/{len(refs)}   "
              f"mean deviation across references {st.mean(r['mean_err'] for r in scored):.2f}px")
    # A reference the bot never tracked is a result, not a gap in the test: it means the
    # seeder or a gate dropped that kind of feature entirely.
    if missed:
        print(f"MISSED {len(missed)}: {', '.join(missed)}  "
              f"(bot has no track on those features -- check the seeder and the gates)")
    if len(refs) < 5:
        print(f"note: {len(refs)} reference(s). Per-class conclusions need one per class "
              f"(corner, blob, edge, parallax, occluded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
