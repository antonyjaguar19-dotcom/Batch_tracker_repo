"""Parity gate: `track_core.py` must reproduce `bl_track.py` exactly.

There are now two copies of the tracking body -- the original headless one in
`experiments/blender_track/bl_track.py`, and the addon's generator version. Two copies is
two chances to disagree, and a disagreement here is half a pixel nobody sees until a solve
is wrong. This test is the price of having the second copy, and it must pass before any
number produced by the addon is believed.

Runs each implementation in its own headless Blender over the same seeds JSON, with
replants disabled on both sides (replant is the one stage track_core has not taken over),
then diffs marker-for-marker.

    python tests\test_track_core_parity.py --seeds <seeds.json>

Exit code 0 = identical.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ASSIST, ".."))
BL_TRACK = os.path.join(REPO, "experiments", "blender_track", "bl_track.py")
DEFAULT_BLENDER = (r"C:\Users\jefrin\Downloads\blender-5.2.0-windows-x64"
                   r"\blender-5.2.0-windows-x64\blender.exe")


def run_blender(exe, script, extra, quiet=True):
    cmd = [exe, "--background", "--factory-startup", "-noaudio",
           "--python-exit-code", "3", "--python", script, "--"] + extra
    print("  $ %s ... %s" % (os.path.basename(script), " ".join(extra[-4:])), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = ((p.stdout or "") + (p.stderr or "")).strip().splitlines()[-25:]
        print("\n".join("    " + t for t in tail))
        raise SystemExit("%s failed (exit %d)" % (os.path.basename(script), p.returncode))
    return p.stdout


def load(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return {t["id"]: {int(f): (x, y) for f, x, y in t["pts"]} for t in d["tracks"]}, d


def compare(a_path, b_path):
    a, da = load(a_path)
    b, db = load(b_path)
    problems = []
    if set(a) != set(b):
        problems.append("track ids differ: %d vs %d, %d common"
                        % (len(a), len(b), len(set(a) & set(b))))
    worst_px, n = 0.0, 0
    span_diffs = 0
    w, h = da["width"], da["height"]
    for tid in sorted(set(a) & set(b)):
        fa, fb = a[tid], b[tid]
        if set(fa) != set(fb):
            span_diffs += 1
            if len(problems) < 8:
                problems.append("%s: frames %d vs %d (%d common)"
                                % (tid, len(fa), len(fb), len(set(fa) & set(fb))))
        for f in set(fa) & set(fb):
            dx = (fa[f][0] - fb[f][0]) * w
            dy = (fa[f][1] - fb[f][1]) * h
            worst_px = max(worst_px, (dx * dx + dy * dy) ** 0.5)
            n += 1
    return {"samples": n, "worst_px": worst_px, "span_mismatched_tracks": span_diffs,
            "problems": problems,
            "a_tracks": len(a), "b_tracks": len(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--leash", default="20")
    ap.add_argument("--clip", default="", help="plate folder; defaults to the seeds JSON")
    ap.add_argument("--skip-ref", action="store_true",
                    help="reuse the previous bl_track.py output instead of re-running it")
    ap.add_argument("--blender",
                    default=os.environ.get("BTR_BLENDER_EXE", DEFAULT_BLENDER))
    ap.add_argument("--outdir", default=os.path.join(ASSIST, "logs", "parity"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ref = os.path.join(args.outdir, "ref_bl_track.json")
    new = os.path.join(args.outdir, "new_track_core.json")

    print("parity: bl_track.py vs track_core.py")
    print("  seeds  : %s" % args.seeds)
    print("  leash  : %s px" % args.leash)

    # Both sides: replants off, backward on, leash matched. bl_track's own defaults are
    # PREV_FRAME + leash 20, and track_core.Opts matches them, so nothing is passed that
    # would make this a comparison of two different configurations.
    # bl_track.py requires --clip even when the seeds JSON names the plate; it only falls
    # back to --clip when that path does not exist on this box.
    with open(args.seeds, encoding="utf-8") as fh:
        plate = json.load(fh).get("plate", "")
    clip = args.clip or plate
    if not clip:
        raise SystemExit("seeds JSON has no 'plate' -- pass --clip")
    # The seeds JSON stores the plate path as it was written, which is relative to the REPO
    # root, not to wherever this test is run from. Resolving it against cwd silently yields
    # a path that does not exist and Blender fails on the load rather than on the argument.
    if not os.path.isabs(clip):
        clip = os.path.join(REPO, clip)
    clip = os.path.abspath(clip)
    if not os.path.exists(clip):
        raise SystemExit("plate not found: %s" % clip)

    if not (args.skip_ref and os.path.isfile(ref)):
        run_blender(args.blender, BL_TRACK,
                    ["--clip", clip, "--seeds", os.path.abspath(args.seeds), "--out", ref,
                     "--no-replant", "--leash", args.leash])
    run_blender(args.blender, os.path.join(HERE, "parity_run_core.py"),
                ["--clip", clip, "--seeds", os.path.abspath(args.seeds), "--out", new,
                 "--leash", args.leash])

    r = compare(ref, new)
    print()
    print("  tracks           : %d vs %d" % (r["a_tracks"], r["b_tracks"]))
    print("  samples compared : %d" % r["samples"])
    print("  worst difference : %.9f px" % r["worst_px"])
    print("  span mismatches  : %d tracks" % r["span_mismatched_tracks"])
    for p in r["problems"]:
        print("  ! %s" % p)

    # Both sides run the same C tracker on the same pixels through the same operator, so a
    # real match is EXACT. Any non-zero difference is a divergence in the Python around it,
    # not floating-point noise, and is worth chasing rather than tolerating.
    ok = (r["samples"] > 0 and r["worst_px"] == 0.0
          and r["span_mismatched_tracks"] == 0 and not r["problems"])
    with open(os.path.join(args.outdir, "parity.json"), "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)
    print()
    print("=" * 60)
    print("PARITY: %s" % ("PASS -- the two copies agree exactly" if ok else "FAIL"))
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
