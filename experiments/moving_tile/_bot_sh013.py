import sys, os
sys.path.insert(0, os.path.abspath("."))
from app.tracker_core import RunnerConfig, BatchTrackerRunner
cfg = RunnerConfig(
    input_dir="D:/Jefrin/IN",
    output_dir="experiments/moving_tile/out/bot_sh013",
    selected_files=["SH013.mp4"],
    selected_scales={"SH013.mp4": 1.0},
    frame_start=1, frame_end=150,          # bounded range so it completes without freezing
    seeding_mode="features", bidirectional=True,
    max_tracks=200,                        # cap seeds/window (avoid memory blowup)
    enable_mask_gating=False, mask_root_dir="",
    enable_spread_select=True, spread_min_dist_px=45, max_output_tracks=0,
    enable_moving_tile=True, enable_pattern_refine=True,
    enable_reseed=True, reseed_every=30,
    mt_edge_track=True, refine_gap_aware=True,
    stream_decode="never",                # bound host RAM (no full-clip decode in refine)
    flip_y_for_3de=True,
)
os.makedirs(cfg.output_dir, exist_ok=True)
r = BatchTrackerRunner(cfg, on_status=lambda m: print("ST:", m, flush=True))
r.run()
print("RUN DONE")
