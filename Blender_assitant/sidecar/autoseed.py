"""Smart auto-seed: where should the trackers go, and roughly how does the plate move?

Reuses `experiments/blender_track/run_blender_hybrid.py` rather than reimplementing it --
`tapnext_export`, `seeds_from_export` and `classify_seeds` are the measured pipeline, and a
second copy would be a second thing to keep in step.

The measured recipe, and why each number is what it is:

  track_spacing_px = 15   Spacing, not quality, is what caps the count. The log that
                          settled it reads `1278 past quality bar -> 28 after spacing` at
                          the default 60; at 15 the same shot yields 122.
  moving_tile      = off  Blender REPLACES both refine stages. Bench, exact ground truth:
  pattern_refine   = off  bot with refines off 2.71px, bot full refine chain 0.06px,
                          Blender on that same raw guide 0.05px -- and the refines were the
                          expensive half of the run (5.5 + 1.9 min of 9.7).

Output is the seeds contract the Blender side already speaks, unchanged, plus a `rejected`
list so the UI can draw what was thrown away.
"""

import os
import time


def build_seeds(plate_path, out_dir, target=150, spacing_px=15, max_tracks=1200,
                quality=0.02, min_dist=12, reject_movers=False, on_status=None):
    """Run TAPNext over the plate and turn its tracks into classified seeds.

    Returns the seeds dict. `plate_path` is a folder of frames or a movie file; the sidecar
    reads pixels off disk itself, so nothing large crosses the wire.
    """
    from repo import require_repo                                     # noqa: PLC0415
    require_repo()

    import blio                                                       # noqa: PLC0415
    import run_blender_hybrid as rbh                                  # noqa: PLC0415

    say = on_status or (lambda m: None)
    t0 = time.time()

    plate = blio.Plate(plate_path)
    say("plate %s  %dx%d  %d frames" % (plate.name, plate.width, plate.height,
                                        plate.frames))

    os.makedirs(out_dir, exist_ok=True)
    say("TAPNext guide pass (this is the slow part) ...")
    _apply_recipe(spacing_px)
    txt = rbh.tapnext_export(plate, out_dir, max_tracks=max_tracks, tag="assist")
    say("guide export: %s" % os.path.basename(txt))

    seeds = rbh.seeds_from_export(txt, plate.width, plate.height, plate.frames,
                                  keep=max(1, int(target)))
    say("%d seeds from the guide" % len(seeds))

    seeds = rbh.classify_seeds(plate, seeds, quality, min_dist)
    kinds = {}
    for s in seeds:
        kinds[s.get("kind") or "?"] = kinds.get(s.get("kind") or "?", 0) + 1
    say("kinds: %s" % ", ".join("%s %d" % kv for kv in sorted(kinds.items())))

    rejected = []
    if reject_movers:
        # Deliberately OFF by default and unmeasured -- see FINDINGS.md. A mover detector
        # that flags parallax foreground would be the third metric in this project to look
        # plausible while measuring the plate instead of the tracker.
        seeds, rejected = _reject_movers(seeds, plate)
        say("mover rejection: %d kept, %d rejected" % (len(seeds), len(rejected)))

    say("auto-seed done in %.1fs" % (time.time() - t0))
    return {
        "plate": plate.path,
        "width": plate.width,
        "height": plate.height,
        "frames": plate.frames,
        "seeds": seeds,
        "rejected": rejected,
        "guide_export": txt,
    }


def _apply_recipe(spacing_px):
    """Push the measured settings into the bot's AppState defaults for this run.

    `RunnerConfig` is built from `AppState` by an ~80-field mapping that changes whenever a
    setting is added, so the recipe is applied by name on the state rather than by
    rebuilding the config here -- the same reason `bench/run_bench.py` calls the app's own
    `_track_shots_tapnext` instead of assembling a config itself.
    """
    from app.tracker_core import RunnerConfig                          # noqa: PLC0415
    for field, value in (("track_spacing_px", float(spacing_px)),
                         ("enable_moving_tile", False),
                         ("enable_pattern_refine", False)):
        if hasattr(RunnerConfig, field) or field in getattr(RunnerConfig, "__annotations__", {}):
            setattr(RunnerConfig, field, value)


def _reject_movers(seeds, plate):
    """Placeholder for the geometry-only mover gate. Returns (kept, rejected).

    The method is decided -- local neighbour motion, not a global homography and not a
    segmentation model. Measured on SH013 with synthetic gaps of known truth: local
    neighbour motion 8.6px at 10-frame gaps, global homography 144.3px, DINOv2 36.2px,
    RoMa 163.4px. A global fit cannot hold foreground dirt and a distant mountain at once,
    which is why it flags parallax as motion.

    It is not implemented yet because it has no reference set to be scored against, and
    shipping an unmeasured detector is how the two 2026-08 metric defects happened. See
    FINDINGS.md, milestone M4.
    """
    return seeds, []
