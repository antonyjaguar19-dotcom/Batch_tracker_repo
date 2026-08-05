# -*- coding: utf-8 -*-
"""Run the production TAPNext path over a synthetic bench shot.

This deliberately calls `app.py:_track_shots_tapnext` rather than building a RunnerConfig
itself. That mapping is ~80 fields wide and changes whenever a setting is added; a bench
that re-implements it measures a configuration nobody ships. Driving the app's own function
means the bench cannot drift from production by construction.

    python bench/run_bench.py --shot bench/synth/lab01 --tag base
    python bench/run_bench.py --shot bench/synth/lab01 --tag test --set per_track_policy=1

`--set` writes onto AppState before the run, so any UI setting is reachable by its field
name (see AppState in app/app.py). Values parse as int / float / bool / str.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
os.chdir(_HERE)  # app.py resolves several paths relative to cwd


def _load_backend():
    """Load app/app.py under its own module name (an `app` PACKAGE also exists here)."""
    spec = importlib.util.spec_from_file_location(
        "btr_backend", os.path.join(_HERE, "app", "app.py"))
    be = importlib.util.module_from_spec(spec)
    sys.modules["btr_backend"] = be
    spec.loader.exec_module(be)
    return be


def _coerce(v: str):
    low = v.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True, help="bench shot folder (holds plate/ and gt.json)")
    ap.add_argument("--tag", required=True, help="run name; output goes to <shot>/runs/<tag>")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override an AppState field (repeatable)")
    # The three seeding numbers live on UI sliders, not AppState, so they are passed in.
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--seed-count", type=int, default=1200)
    ap.add_argument("--seed-min-dist", type=int, default=12)
    a = ap.parse_args()

    plate = os.path.join(a.shot, "plate")
    if not os.path.isdir(plate):
        raise SystemExit(f"no plate/ under {a.shot} -- run bench/make_synth.py first")
    out_root = os.path.join(a.shot, "runs", a.tag)
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)          # a stale export would be scored as this run's
    os.makedirs(out_root, exist_ok=True)

    be = _load_backend()
    be._ensure_tracker_loaded()
    if be.BatchTrackerRunner is None:
        raise SystemExit(f"TAPNext unavailable: {be.TRACKER_IMPORT_ERROR}")

    state = be.AppState()
    state.track_backend = "tapnext"
    applied = {}
    for kv in a.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects FIELD=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip()
        if not hasattr(state, k):
            raise SystemExit(f"AppState has no field {k!r}")
        val = _coerce(v)
        setattr(state, k, val)
        applied[k] = val

    shot = os.path.basename(os.path.normpath(a.shot))
    state.shots_data[shot] = be.ShotData(
        name=shot, use=True, render_path=os.path.abspath(plate), scale="100%")

    be.STOP_EVENT.clear()
    print(f"=== bench run '{a.tag}' | shot={shot} | overrides={applied or 'none (defaults)'}")
    t0 = time.time()
    any_ran, stopped = be._track_shots_tapnext(
        in_root=os.path.abspath(a.shot), out_root=os.path.abspath(out_root),
        shot_tasks_map={}, state=state, grid=a.grid,
        seed_count=a.seed_count, seed_min_dist=a.seed_min_dist)
    be.close_shot_log()
    secs = time.time() - t0

    txt = os.path.join(out_root, f"{shot}__tapnext.txt")
    print(f"=== ran={any_ran} stopped={stopped} in {secs:.1f}s")
    if not os.path.isfile(txt):
        raise SystemExit(f"no export produced at {txt}")
    print(f"=== export: {txt}")
    with open(os.path.join(out_root, "run.txt"), "w", encoding="utf-8") as f:
        f.write(f"tag={a.tag}\nseconds={secs:.1f}\noverrides={applied}\n"
                f"grid={a.grid} seed_count={a.seed_count} seed_min_dist={a.seed_min_dist}\n")


if __name__ == "__main__":
    main()
