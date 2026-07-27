import sys, os, threading, time
sys.path.insert(0, os.path.abspath("."))
try:
    import psutil
    def mon():
        while True:
            m=psutil.virtual_memory()
            print("MEM: used=%.1fG avail=%.1fG (%.0f%%)"%(m.used/1e9,m.available/1e9,m.percent),flush=True)
            time.sleep(20)
    threading.Thread(target=mon,daemon=True).start()
except Exception as e: print("no psutil",e)
from app.tracker_core import RunnerConfig, BatchTrackerRunner
cfg = RunnerConfig(
    input_dir="D:/Jefrin/IN", output_dir="experiments/moving_tile/out/bot_sh013_stress",
    selected_files=["SH013.mp4"], selected_scales={"SH013.mp4": 1.0},
    frame_start=0, frame_end=0,               # FULL range 303
    seeding_mode="features", bidirectional=True, max_tracks=1200,
    enable_mask_gating=False, mask_root_dir="",
    enable_spread_select=True, spread_min_dist_px=45,
    enable_moving_tile=True, enable_pattern_refine=True,
    enable_reseed=True, reseed_every=30,
    mt_edge_track=True, refine_gap_aware=True,
    stream_decode="auto", host_ram_frac=0.5, flip_y_for_3de=True,
)
os.makedirs(cfg.output_dir, exist_ok=True)
r = BatchTrackerRunner(cfg, on_status=lambda m: print("ST:", m, flush=True))
r.run(); print("RUN DONE")
