"""Tune the Blender side against exact ground truth, one row per configuration.

Every setting here is a guess until it is measured, and the guesses are not obviously
right: `Loc` cannot follow a plate that scales, `KEYFRAME` matching resists drift but
fails when the patch stops looking like its keyframe, a bigger search box survives fast
motion but invites a jump to the wrong peak. This runs them all on a bench shot whose
ground truth is exact and prints the numbers side by side.

    runtime\\python311\\python.exe experiments\\blender_track\\sweep.py --shot bench\\synth\\lab02

`--shot` must be a bench shot (holds `plate/`, `gt.json` and a `runs/base` to seed from).
Each configuration is ~15s, so the whole default sweep is a couple of minutes and needs
no GPU.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, "runtime", "python311", "python.exe")

# name -> extra flags for run_blender_hybrid. The first entry is the current default, so
# every other row reads as a delta from what the experiment already does.
CONFIGS = [
    ("default",        []),
    ("prevframe",      ["--pattern-match", "PREV_FRAME"]),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=os.path.join("bench", "synth", "lab02"))
    ap.add_argument("--baseline", default="base", help="bot run to seed from and compare to")
    ap.add_argument("--only", default="", help="comma-separated config names")
    args = ap.parse_args()

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
