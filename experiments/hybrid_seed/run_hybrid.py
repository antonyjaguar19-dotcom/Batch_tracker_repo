"""HYBRID: TAPNext's seeder chooses the points, SynthEyes tracks them.

The bot's TAPNext path and its SynthEyes path each pick their own features. This asks
what happens when one engine's seed selection is handed to the other engine's tracker.

Nothing in `app/` is modified. The bot's real seeder is imported and called
(`BatchTrackerRunner._staggered_queries` -> goodFeaturesToTrack + edge/anisotropy
rejection + seed classification), and the resulting points are injected into SynthEyes
as area-match trackers.

Mechanism, all three parts measured on build 2026.2.4679 (see FINDINGS.md):
  * `tk = new ob.trk` / `tk.key = Point(u,v)`   -- create a tracker at a chosen point
  * `tk.Run()` once per frame                   -- THIS is the per-frame track step;
                                                   Scene.RunTrackersFwd() does nothing here
  * the engine's existing Sizzle export         -- writes classic 3DE ASCII

Seeds are all placed on the first tracked frame. Mid-shot creation does not take on this
build (a tracker created at frame 80 never becomes valid), so seed staggering is forced
off rather than silently producing dead trackers.

    runtime\\python311\\python.exe experiments\\hybrid_seed\\run_hybrid.py --mp4 D:\\Jefrin\\IN\\SH004.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.tracker_core import RunnerConfig, BatchTrackerRunner  # noqa: E402
from app.syntheyes_engine import SynthEyesEngine  # noqa: E402

OUT_DIR = os.path.join(HERE, "out")


# ----------------------------------------------------------------------------- seeding

def tapnext_seeds(mp4: str, n_seeds: int, quality: float, min_dist: int):
    """Run the bot's own seeder on frame 0. Returns (pts Nx2 in plate px, kinds, W, H, T)."""
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {mp4}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read frame 0")
    h, w = frame0.shape[:2]

    cfg = RunnerConfig(
        input_dir=os.path.dirname(mp4),
        output_dir=OUT_DIR,
        auto_tune=False,            # a fixed seeder makes the comparison legible
        per_track_policy=True,      # so seeds come back classified corner/blob/edge/dense
        seed_stagger=1,             # all seeds on frame 0 -- see module docstring
        max_tracks=n_seeds,
        feature_quality=quality,
        min_feature_dist=min_dist,
    )
    runner = BatchTrackerRunner(cfg)
    frames = frame0[None]           # stagger=1 only ever indexes offset 0
    q, seeds = runner._staggered_queries(frames, None, n_seeds, total)
    if q is None:
        raise RuntimeError("seeder returned no points")
    pts = q[0][:, 1:3].astype(np.float32)
    kinds = [k for (_f, k) in seeds] if seeds else [""] * len(pts)
    return pts, kinds, w, h, total


# ------------------------------------------------------------------------ sizzle build

def to_uv(px: float, py: float, w: int, h: int):
    """Inverse of the engine's export mapping (app/syntheyes_engine.py:2062)."""
    return (2.0 * px / w) - 1.0, 1.0 - (2.0 * py / h)


# Tracker geometry per seed class. A corner can be localised in both axes so it gets a
# small tight patch; a blob needs more support; an edge point slides along the edge, so it
# gets a wider search across it. This is the per-track policy idea (app/track_meta.py)
# expressed in SynthEyes' own tracker fields -- something neither engine does today.
KIND_GEOM = {
    "corner": (0.014, 0.012, 0.014),
    "blob":   (0.026, 0.018, 0.020),
    "edge":   (0.020, 0.022, 0.024),
    "dense":  (0.018, 0.015, 0.018),
    "":       (0.020, 0.015, 0.020),
}


def build_script(uv, kinds, diag_path: str, per_kind_geom: bool) -> str:
    lines = [
        "//SIZZLET BTRHybrid",
        "ob = Scene.activeObj",
        "shot = ob.shot",
        "start = shot.start",
        "stop = shot.stop",
        "frame = start",
    ]
    for i, (u, v) in enumerate(uv):
        kind = kinds[i] if i < len(kinds) else ""
        size, su, sv = KIND_GEOM.get(kind, KIND_GEOM[""]) if per_kind_geom else KIND_GEOM[""]
        lines += [
            "frame = start",
            "tk = new ob.trk",
            f'tk.nm = "HYB{i:04d}"',
            "tk.kind = 0",
            f"tk.size = {size:.4f}",
            "tk.asp = 1.0",
            f"tk.srchu = {su:.4f}",
            f"tk.srchv = {sv:.4f}",
            "tk.smooth = 20",
            "tk.autokey = 20",
            "tk.isSel = 2",
            f"tk.key = Point({u:.6f},{v:.6f})",
            "tk.isEnabled = 1",
            "tk.Run()",
        ]
    # Frame-major so each frame's image is touched once for all trackers.
    # Scene.SetFrame is NOT optional despite tracking appearing to work without it: the
    # Sizzle `frame` global alone moves the evaluation time but not the app's image, and
    # about ten frames past the app's current frame every tracker freezes on a stale one.
    lines += [
        "for (f = start + 1; f <= stop; f++)",
        "    Scene.SetFrame(f)",
        "    frame = f",
        "    for (tk in ob.trk)",
        "        tk.Run()",
        "    end",
        "end",
        f'openout("{diag_path}")',
        'printf("ntrk %d\\n", #ob.trk)',
        "for (tk in ob.trk)",
        "    cnt = 0",
        "    for (frame = start; frame <= stop; frame++)",
        "        if (tk.valid)",
        "            cnt = cnt + 1",
        "        end",
        "    end",
        '    printf("%s %d\\n", tk.nm, cnt)',
        "end",
        "closeout()",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mp4", default=r"D:\Jefrin\IN\SH004.mp4")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--quality", type=float, default=0.02)
    ap.add_argument("--min-dist", type=int, default=12)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--flat-geom", action="store_true",
                    help="one tracker size for every seed (control for the per-kind geometry)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    shot = os.path.splitext(os.path.basename(args.mp4))[0]
    out_txt = os.path.join(out_dir, f"{shot}__{args.tag}.txt")

    t_seed = time.time()
    pts, kinds, w, h, total = tapnext_seeds(args.mp4, args.seeds, args.quality, args.min_dist)
    counts = {}
    for k in kinds:
        counts[k or "unclassified"] = counts.get(k or "unclassified", 0) + 1
    print(f"plate {w}x{h}, {total} frames", flush=True)
    print(f"seeder returned {len(pts)} points in {time.time()-t_seed:.1f}s: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)

    uv = [to_uv(float(x), float(y), w, h) for (x, y) in pts]
    diag = os.path.join(out_dir, f"{args.tag}.diag").replace("\\", "/")
    script = build_script(uv, kinds, diag, per_kind_geom=not args.flat_geom)
    szl_path = os.path.join(out_dir, f"{args.tag}.szl")
    with open(szl_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"sizzle script: {szl_path} ({len(script.splitlines())} lines)", flush=True)

    settings = {
        "syntheyes_exe": os.environ.get(
            "BTR_SYNTHEYES_EXE",
            r"C:\Program Files\BorisFX\SynthEyes 2026\SynthEyes64.exe"),
        "port": int(os.environ.get("BTR_SE_PORT", 2222)),
        "pin": os.environ.get("BTR_SE_PIN", "listen"),
        "startup_wait": 3,
    }
    eng = SynthEyesEngine(settings, on_log=lambda m: print(f"SE: {m}", flush=True))
    if not eng.setup_sypy():
        return 2
    if not eng.connect_or_launch():
        return 3
    eng.set_writable_folder(out_dir)

    print("-> loading plate into SynthEyes", flush=True)
    if eng.hlev.NewSceneAndShot(os.path.normpath(args.mp4)) is None:
        print("FAIL: NewSceneAndShot returned None")
        return 4
    time.sleep(6)

    # Build the RAM cache before tracking, exactly as process_shot does for short shots.
    # Without it every tracker tracked cleanly for ~10 frames and then froze on one stale
    # image -- an export full of valid-looking, motionless points.
    for sh in eng.hlev.Shots():
        eng.hlev.Validate(sh)
    print("   validated shot RAM cache", flush=True)
    time.sleep(2)

    try:
        os.remove(diag)
    except OSError:
        pass

    print(f"-> injecting {len(uv)} trackers and tracking {total} frames", flush=True)
    t0 = time.time()
    eng._resync_socket()
    eng._run_sizzle(script, watchdog_secs=3600)
    dt = time.time() - t0
    print(f"   inject+track: {dt:.1f}s "
          f"({len(uv)*total/max(dt,1e-6):,.0f} tracker-frames/s)", flush=True)

    if os.path.isfile(diag):
        rows = [ln.split() for ln in open(diag).read().strip().splitlines() if ln.strip()]
        spans = [int(r[1]) for r in rows if len(r) == 2 and r[0] != "ntrk"]
        if spans:
            full = sum(1 for s in spans if s >= total - 1)
            print(f"   spans: {len(spans)} trackers, {full} held all {total} frames, "
                  f"median {int(np.median(spans))}, min {min(spans)}", flush=True)
    else:
        print("   WARNING: no diag written - the Sizzle script did not reach the end")

    print("-> exporting 3DE ASCII", flush=True)
    n = eng._sizzle_export_3de(out_txt)
    print(f"RESULT: {n} tracks -> {out_txt}", flush=True)
    return 0 if n and n > 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
