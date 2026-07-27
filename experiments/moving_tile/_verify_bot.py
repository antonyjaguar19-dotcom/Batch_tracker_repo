import sys, os
sys.path.insert(0, os.path.abspath("."))
from app.tracker_core import RunnerConfig, BatchTrackerRunner
cfg = RunnerConfig(
    input_dir="D:/Jefrin/IN",
    output_dir="experiments/moving_tile/out/bot_verify",
    selected_files=["SH011.mp4"],
    selected_scales={"SH011.mp4": 1.0},
    seeding_mode="features", bidirectional=True,
    enable_mask_gating=False, mask_root_dir="",
    enable_spread_select=True, spread_min_dist_px=60, max_output_tracks=20,
    enable_moving_tile=True, enable_pattern_refine=True,
    flip_y_for_3de=True,
)
os.makedirs(cfg.output_dir, exist_ok=True)
r = BatchTrackerRunner(cfg, on_status=lambda m: print("ST:", m, flush=True))
r.run()
print("RUN DONE")
