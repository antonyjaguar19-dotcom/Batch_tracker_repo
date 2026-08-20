"""Find batch_tracker and make it importable, or say exactly what is missing.

The sidecar reuses the bot rather than reimplementing it: TAPNext, seed classification and
the feature seeder all come from `app/` and `experiments/blender_track/`. That is a
deliberate dependency -- a vendored copy of `app.tracker_core` would be a second copy of an
80-field config that changes whenever a setting is added, which is exactly the failure the
parity gate exists to police for the tracking loop.

So when the repo is not reachable, the failure has to name the modules rather than surface
as `ModuleNotFoundError: app` three frames deep.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))

NEEDED = [
    ("app.tracker_core", "TAPNext runner and RunnerConfig"),
    ("app.track_meta", "seed classification (corner/blob/edge/dense)"),
    ("app.compare_tracks", "3DE reader"),
    ("app.export_3de", "3DE writer"),
]


def paths():
    """What bootstrap recorded. Env vars win, then paths.json, then the obvious guess."""
    cfg = {}
    p = os.path.join(ASSIST, "config", "paths.json")
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        pass
    repo = os.environ.get("BTR_REPO") or cfg.get("repo_root") or os.path.dirname(ASSIST)
    return {
        "assist_root": ASSIST,
        "repo_root": os.path.abspath(repo),
        "tapnext_ckpt": os.environ.get("BTR_TAPNEXT_CKPT") or cfg.get("tapnext_ckpt", ""),
        "tapnext_code": cfg.get("tapnext_code", ""),
    }


def require_repo():
    """Inject the repo on sys.path and import-check every module the sidecar needs.

    Returns the paths dict. Raises RuntimeError naming what failed and why it matters.
    """
    p = paths()
    repo = p["repo_root"]
    bl = os.path.join(repo, "experiments", "blender_track")
    if not os.path.isfile(os.path.join(repo, "app", "tracker_core.py")):
        raise RuntimeError(
            "batch_tracker not found at %s. Auto-seed imports the bot's own TAPNext "
            "runner and seed classifier; set BTR_REPO or re-run bootstrap. "
            "(3DE import/export in the addon does not need this.)" % repo)
    for d in (repo, bl, os.path.join(repo, "bench")):
        if d not in sys.path:
            sys.path.insert(0, d)

    # The checkpoint is found by the bot through this env var; the sidecar may have its own
    # copy under Blender_assitant/weights, so it is set here rather than assumed.
    if p["tapnext_ckpt"] and os.path.isfile(p["tapnext_ckpt"]):
        os.environ.setdefault("BTR_TAPNEXT_CKPT", p["tapnext_ckpt"])

    missing = []
    for mod, why in NEEDED:
        try:
            __import__(mod)
        except Exception as exc:                     # noqa: BLE001 -- report, don't mask
            missing.append("%s (%s) -- %s: %s" % (mod, why, type(exc).__name__, exc))
    if missing:
        raise RuntimeError("the repo is present but these imports failed:\n  "
                           + "\n  ".join(missing))
    return p
