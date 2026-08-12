"""Tune the Blender side, one row per configuration. Two modes, and both are needed.

**Precision** (`--shot <bench shot>`) scores against exact ground truth.

**Robustness** (`--plate <dir> --guide <txt>`) scores deaths per track on a real plate,
which needs no ground truth at all -- a death is a hole in the export.

The second mode exists because the first cannot see the failure that matters most. On
`bench/synth/lab02` every one of the configurations below kept 23/23 tracks for 100/100
frames: one clean plane with uniformly good features gives nothing to get lost on. That
sweep's verdict ("the defaults win") is therefore a statement about sub-pixel accuracy
ONLY, and was wrongly read as meaning Blender-side tuning was exhausted. On real plates
Blender dies 1.0-2.3 times per track while the TAPNext guide dies 0.3 times.

    runtime\\python311\\python.exe experiments\\blender_track\\sweep.py --shot bench\\synth\\lab02
    runtime\\python311\\python.exe experiments\\blender_track\\sweep.py ^
        --plate experiments\\blender_track\\out\\SH016\\plate --name SH016 ^
        --guide experiments\\blender_track\\out\\SH016\\runs\\dense_raw\\SH016__tapnext.txt

A config that wins on one and loses badly on the other is not an improvement; read both.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PY = os.path.join(ROOT, "runtime", "python311", "python.exe")

# name -> extra flags for run_blender_hybrid. The first entry is the current default, so
# every other row reads as a delta from what the experiment already does.
CONFIGS = [
    ("default",        []),
    # The old defaults, kept as controls now that stepping + leash + PREV_FRAME shipped.
    ("old_default",    ["--no-frame-step", "--pattern-match", "KEYFRAME"]),
    ("no_leash",       ["--leash", "0"]),
    ("keyframe",       ["--pattern-match", "KEYFRAME"]),
    ("prevframe",      ["--pattern-match", "PREV_FRAME"]),
    # PREV_FRAME and a small pattern are the two configurations that survive real footage
    # best, and for the same reason: both minimise how much the thing being matched can
    # change between one comparison and the next. Worth knowing whether they compound.
    ("prev_small",     ["--pattern-match", "PREV_FRAME", "--pattern-scale", "0.7"]),
    ("prev_srch",      ["--pattern-match", "PREV_FRAME", "--search-scale", "1.5"]),
    ("scale_geom",     ["--scale-geom"]),
    ("prev_scale",     ["--pattern-match", "PREV_FRAME", "--scale-geom"]),
    ("affine",         ["--motion-model", "Affine"]),
    ("affine_prev",    ["--motion-model", "Affine", "--pattern-match", "PREV_FRAME"]),
    ("locscale",       ["--motion-model", "LocScale"]),
    ("perspective",    ["--motion-model", "Perspective"]),
    ("pat_small",      ["--pattern-scale", "0.7"]),
    ("pat_big",        ["--pattern-scale", "1.5"]),
    ("srch_big",       ["--search-scale", "1.5"]),
    ("corr_high",      ["--correlation", "0.90"]),
]

ALL_RE = re.compile(r"^ALL\s+(\d+)\s+([\d.]+)", re.M)
P90_RE = re.compile(r"worst tail\s+p90 mean_err ([\d.]+)px\s+worst track ([\d.]+)px")
BASE_RE = re.compile(r"baseline\s+p90 mean_err ([\d.]+)px\s+worst track ([\d.]+)px")


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def robust_sweep(args, picked) -> int:
    """Score deaths per track on a real plate. No ground truth involved."""
    from track_stats import stats

    out_dir = os.path.join(HERE, "out", "sweep")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for cname, flags in picked:
        out = os.path.join(out_dir, f"{args.name}__{cname}__blender.txt")
        txt = run([PY, os.path.join(HERE, "run_blender_hybrid.py"),
                   "--plate", args.plate, "--name", args.name,
                   "--reuse-tapnext", args.guide, "--tag", "sw_" + cname,
                   "--out", out] + flags)
        rt = re.search(r"round-trip : max ([\d.]+)px\s+(\w+)", txt)
        if not rt or rt.group(2) != "PASS" or not os.path.isfile(out):
            print(f"  {cname:<16} FAILED (round-trip or run)")
            continue
        s = stats(out, args.frames)
        s["name"] = cname
        rows.append(s)
        print(f"  {cname:<16} deaths/t {s['deaths_per_track']:.2f}  "
              f"clean {s['clean']:>4}  med_run {s['median_run']:.0f}  "
              f"cover {100 * s['coverage']:.1f}%")

    if not rows:
        return 1
    rows.sort(key=lambda r: (r["deaths_per_track"], -r["median_run"]))
    print(f"\n{args.name}: {len(rows)} configurations, sorted by deaths per track\n")
    print(f"{'config':<16}{'trk':>5}{'deaths/t':>10}{'clean':>7}{'med_run':>9}"
          f"{'span':>8}{'cover%':>9}")
    print("-" * 64)
    for r in rows:
        print(f"{r['name']:<16}{r['tracks']:>5}{r['deaths_per_track']:>10.2f}"
              f"{r['clean']:>7}{r['median_run']:>9.0f}{r['median_span']:>8.0f}"
              f"{100 * r['coverage']:>9.1f}")
    if os.path.isfile(args.guide):
        g = stats(args.guide, args.frames)
        print("-" * 64)
        print(f"{'[tapnext guide]':<16}{g['tracks']:>5}{g['deaths_per_track']:>10.2f}"
              f"{g['clean']:>7}{g['median_run']:>9.0f}{g['median_span']:>8.0f}"
              f"{100 * g['coverage']:>9.1f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=os.path.join("bench", "synth", "lab02"))
    ap.add_argument("--baseline", default="base", help="bot run to seed from and compare to")
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--plate", default="", help="robustness mode: a real plate's frames dir")
    ap.add_argument("--guide", default="", help="robustness mode: the bot export to seed from")
    ap.add_argument("--name", default="", help="robustness mode: shot name")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    if args.plate:
        if not (args.guide and args.name):
            print("robustness mode needs --guide and --name too")
            return 1
        picked = [c for c in CONFIGS
                  if not args.only or c[0] in {s.strip() for s in args.only.split(",")}]
        return robust_sweep(args, picked)

    shot_dir = os.path.abspath(args.shot)
    name = os.path.basename(shot_dir)
    guide = os.path.join(shot_dir, "runs", args.baseline, f"{name}__tapnext.txt")
    if not os.path.isfile(guide):
        print(f"no baseline export at {guide}")
        return 1

    picked = [c for c in CONFIGS
              if not args.only or c[0] in {s.strip() for s in args.only.split(",")}]

    rows = []
    base_row = None
    for cname, flags in picked:
        tag = "sw_" + cname
        out = os.path.join(shot_dir, "runs", tag, f"{name}__tapnext.txt")
        txt = run([PY, os.path.join(HERE, "run_blender_hybrid.py"),
                   "--plate", os.path.join(shot_dir, "plate"), "--name", name,
                   "--reuse-tapnext", guide, "--tag", tag, "--out", out] + flags)
        rt = re.search(r"round-trip : max ([\d.]+)px\s+(\w+)", txt)
        if not rt or rt.group(2) != "PASS":
            # A failed round-trip means the trackers are not on the seeded features, so
            # the accuracy number below would be measuring the wrong points entirely.
            print(f"{cname:<14} ROUND-TRIP FAIL -- skipped")
            continue
        med = re.search(r"median (\d+)", txt)

        sc = run([PY, os.path.join(ROOT, "bench", "score_synth.py"), shot_dir,
                  "--run", tag, "--baseline", args.baseline, "--no-lock"])
        m_all = ALL_RE.search(sc)
        m_p90 = P90_RE.search(sc)
        m_base = BASE_RE.search(sc)
        if not (m_all and m_p90):
            print(f"{cname:<14} score failed:\n{sc[-600:]}")
            continue
        rows.append({"name": cname, "n": int(m_all.group(1)),
                     "mean": float(m_all.group(2)), "p90": float(m_p90.group(1)),
                     "worst": float(m_p90.group(2)),
                     "median_len": int(med.group(1)) if med else 0})
        if m_base and base_row is None:
            base_row = {"name": f"[bot {args.baseline}]", "n": rows[-1]["n"],
                        "p90": float(m_base.group(1)), "worst": float(m_base.group(2))}
        print(f"  {cname:<14} mean {rows[-1]['mean']:.3f}  p90 {rows[-1]['p90']:.3f}  "
              f"worst {rows[-1]['worst']:.3f}")

    if not rows:
        return 1
    rows.sort(key=lambda r: (r["mean"], r["p90"]))
    print(f"\n{name}: {len(rows)} configurations, sorted by mean error\n")
    print(f"{'config':<16}{'n':>4}{'mean_err':>10}{'p90':>9}{'worst':>9}{'med_len':>9}")
    print("-" * 57)
    for r in rows:
        print(f"{r['name']:<16}{r['n']:>4}{r['mean']:>10.3f}{r['p90']:>9.3f}"
              f"{r['worst']:>9.3f}{r['median_len']:>9}")
    if base_row:
        print("-" * 57)
        print(f"{base_row['name']:<16}{base_row['n']:>4}{'':>10}{base_row['p90']:>9.3f}"
              f"{base_row['worst']:>9.3f}")
        print("(the bot's own mean is not printed by score_synth; p90/worst are)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
