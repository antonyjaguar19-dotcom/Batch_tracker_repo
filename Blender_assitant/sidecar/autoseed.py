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

    os.makedirs(out_dir, exist_ok=True)
    # `Plate(path, ifl_dir)`: for a folder of frames the constructor also writes an .ifl
    # index, which only SynthEyes consumes -- Blender never sees it. It still needs
    # somewhere to put it, so the job's own output dir is passed rather than the repo's.
    # Attributes are `.w` / `.h` / `.count`, not width/height/frames.
    plate = blio.Plate(plate_path, ifl_dir=out_dir)
    say("plate %s  %dx%d  %d frames" % (plate.name, plate.w, plate.h, plate.count))

    say("TAPNext guide pass (this is the slow part) ...")
    txt = _guide_export(plate, out_dir, max_tracks, spacing_px, on_status=say)
    say("guide export: %s" % os.path.basename(txt))

    seeds = rbh.seeds_from_export(txt, plate.w, plate.h, plate.count,
                                  keep=max(1, int(target)))
    say("%d seeds from the guide" % len(seeds))

    # classify_seeds LABELS `seeds` in place and returns how many it labelled -- it does
    # not return the list. Rebinding its result silently replaced the seeds with an int.
    n_kind = rbh.classify_seeds(plate, seeds, quality, min_dist)
    say("classified %d of %d seeds" % (n_kind, len(seeds)))
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
        # `plate.path`, never `plate.load_path`: the latter is the .ifl index SynthEyes
        # wants, and Blender opens it as a 256x256 image.
        "plate": plate.path,
        "width": plate.w,
        "height": plate.h,
        "frames": plate.count,
        "seeds": seeds,
        "rejected": rejected,
        "guide_export": txt,
    }


def _guide_export(plate, out_dir, max_tracks, spacing_px, on_status=None):
    """Run the bot's TAPNext backend with the measured recipe, return its 3DE export.

    This mirrors `run_blender_hybrid.tapnext_export` (`:39-63`) rather than calling it,
    because the recipe has to reach `RunnerConfig` as real constructor arguments. The
    first version set class attributes instead and two of the three names were wrong:
    the UI knob is `track_spacing_px` on **AppState**, and `app.py:2424` maps it to
    `spread_min_dist_px` on RunnerConfig -- so the spacing that is the whole point of the
    recipe was silently not applied. Named arguments fail loudly on a typo; monkeypatched
    defaults do not.

    `spread_min_dist_px` is quoted against `spread_ref_width` (1920) and scaled to the
    plate, so 15 here means 15 at 1920 and 20 at 2560 -- the same convention the hybrid's
    `--set track_spacing_px=15` used.
    """
    from app.tracker_core import RunnerConfig, BatchTrackerRunner      # noqa: PLC0415

    common = dict(
        output_dir=out_dir,
        max_tracks=int(max_tracks),
        output_tag="assist",
        spread_min_dist_px=int(round(spacing_px)),
        # Blender REPLACES both refine stages: bot with refines off 2.71px, bot with them
        # 0.06px, Blender on that same raw guide 0.05px -- and they were the expensive half
        # of a run (5.5 + 1.9 min of 9.7). This guide is deliberately raw.
        enable_moving_tile=False,
        enable_pattern_refine=False,
    )
    if not plate.is_movie:
        cfg = RunnerConfig(input_dir=os.path.dirname(plate.path),
                           sequence_path=plate.path, sequence_name=plate.name, **common)
    else:
        cfg = RunnerConfig(input_dir=os.path.dirname(plate.path),
                           selected_files=[os.path.basename(plate.path)], **common)

    say = on_status or (lambda m: None)
    BatchTrackerRunner(cfg, on_status=lambda m: say(str(m))).run()
    out = os.path.join(out_dir, "%s__assist__tapnext.txt" % plate.name)
    if not os.path.isfile(out):
        raise RuntimeError("the bot produced no export at %s" % out)
    return out


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
