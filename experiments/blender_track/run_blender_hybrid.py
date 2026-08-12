"""HYBRID: TAPNext seeds and replants, Blender tracks.

The bot's TAPNext path is globally robust -- it knows where a feature went even across an
occlusion -- but its per-frame localisation is what `moving_tile_refine` and
`pattern_refine` exist to fix. Blender's movie clip tracker is the opposite: a classic
NCC/affine planar tracker with sub-pixel refinement and no idea what happens once it loses
correlation. This asks what the two are worth combined:

  * TAPNext chooses WHICH features to track, and where they start        (seed)
  * TAPNext says where a dead feature reappears, and it is resumed       (replant)
  * Blender does every per-frame measurement in between                  (track)

Nothing in `app/` is modified. The bot is imported and called, never edited.

    run_blender_hybrid.py --plate D:\\Jefrin\\IN\\SH004.mp4
    run_blender_hybrid.py --plate ... --reuse-tapnext out\\SH004__tapnext.txt
    run_blender_hybrid.py --plate ... --flat-geom --tag flat      (geometry control)
    run_blender_hybrid.py --plate ... --no-replant --tag norepl   (replant control)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blio import (Plate, OUT_DIR, px_to_uv, uv_to_px, read_3de, write_3de,  # noqa: E402
                  dump_json, load_json, run_blender)


# --------------------------------------------------------------------- seed sources

def tapnext_export(plate: Plate, out_dir: str, max_tracks: int, tag: str = "guide") -> str:
    """Run the bot's own TAPNext backend on this plate and return its 3DE export.

    This is the bot exactly as it ships -- auto-tune, per-track policy, both refines. The
    export is used two ways: as the seed/replant source below, and as the control the
    hybrid is scored against, so the comparison is against a real bot run rather than a
    stripped-down one.
    """
    from app.tracker_core import RunnerConfig, BatchTrackerRunner

    if not plate.is_movie:
        cfg = RunnerConfig(input_dir=os.path.dirname(plate.path), output_dir=out_dir,
                           sequence_path=plate.path, sequence_name=plate.name,
                           max_tracks=max_tracks, output_tag=tag)
    else:
        cfg = RunnerConfig(input_dir=os.path.dirname(plate.path), output_dir=out_dir,
                           selected_files=[os.path.basename(plate.path)],
                           max_tracks=max_tracks, output_tag=tag)
    runner = BatchTrackerRunner(cfg, on_status=lambda m: print("   " + str(m)))
    runner.run()
    out = os.path.join(out_dir, f"{plate.name}__{tag}__tapnext.txt")
    if not os.path.isfile(out):
        raise RuntimeError(f"the bot produced no export at {out}")
    return out


def seeds_from_export(txt: str, w: int, h: int, n_frames: int, keep: int) -> list[dict]:
    """Every TAPNext track becomes one Blender seed plus its own replant guide.

    The seed is the track's first point; the guide is the whole track. Keeping the two
    tied to one identity is the point -- a replant has to resume the SAME feature, and the
    only thing that knows which feature a dead tracker was on is the track it came from.
    """
    tracks = read_3de(txt)
    order = sorted(tracks.items(), key=lambda kv: -len(kv[1]))
    if keep > 0:
        order = order[:keep]
    seeds = []
    for name, pts in order:
        if len(pts) < 2:
            continue
        f0 = min(pts)
        x, y = pts[f0]
        u, v = px_to_uv(x, y, w, h)
        if not (0.0 < u < 1.0 and 0.0 < v < 1.0):
            continue
        guide = {}
        for f, (gx, gy) in pts.items():
            if 1 <= f <= n_frames:
                gu, gv = px_to_uv(gx, gy, w, h)
                if 0.0 < gu < 1.0 and 0.0 < gv < 1.0:
                    guide[str(int(f))] = [gu, gv]
        seeds.append({"id": name, "frame": int(f0), "u": u, "v": v,
                      "kind": "", "guide": guide})
    return seeds


def classify_seeds(plate: Plate, seeds: list[dict], quality: float, min_dist: int,
                   seed_count: int = 1200, min_aniso: float = 0.08) -> int:
    """Label every seed corner/blob/edge/dense with the repo's own classifier.

    The export does not carry the seed class and the per-track report has no `kind` column,
    so it is re-derived here. `_measure` and `SeedStats` come from `bench/score_synth.py`
    so that the class a seed is GIVEN here is the same class it is later SCORED under --
    two classifiers would make a per-class result unreadable.

    `classify_seed` judges a seed against its frame's distribution, so the percentile stats
    come from a fresh detection on that frame, not from the seeds themselves.
    """
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    import cv2
    from score_synth import _measure
    from app.track_meta import SeedStats, classify_seed

    by_frame: dict[int, list[dict]] = {}
    for s in seeds:
        by_frame.setdefault(int(s["frame"]), []).append(s)

    n = 0
    for f1, group in sorted(by_frame.items()):
        img = plate.frame(f1 - 1)                      # export frames are 1-based
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det = cv2.goodFeaturesToTrack(gray, maxCorners=seed_count, qualityLevel=quality,
                                      minDistance=min_dist, blockSize=5)
        if det is None or len(det) == 0:
            continue
        pop_feats, pop_lmin, pop_dens = _measure(gray, det.reshape(-1, 2).astype(np.float32))
        stats = SeedStats(
            lmin_p25=float(np.percentile(pop_lmin, 25.0)),
            lmin_p75=float(np.percentile(pop_lmin, 75.0)),
            density_p90=float(np.percentile(pop_dens, 90.0)),
            scale_med=float(np.median([f.scale_px for f in pop_feats])),
        )
        # Seeds are stored in the y-flipped export space; the measurement is top-down.
        pts = np.asarray([[s["u"] * plate.w - 0.5,
                           (plate.h - 1.0) - (s["v"] * plate.h - 0.5)] for s in group],
                         dtype=np.float32)
        feats, _, _ = _measure(gray, pts)
        for s, fe in zip(group, feats):
            s["kind"] = classify_seed(fe, stats, min_aniso)
            n += 1
    return n


def seeds_from_seeder(plate: Plate, n_seeds: int, quality: float, min_dist: int,
                      stagger: int) -> list[dict]:
    """The bot's raw seeder only: goodFeaturesToTrack + edge/anisotropy rejection + seed
    classification, no GPU and no guide. This is the fast smoke-test path -- it answers
    "does Blender track these points" without waiting for a full TAPNext run, but with no
    guide there is nothing to replant from."""
    import numpy as np
    from app.tracker_core import RunnerConfig, BatchTrackerRunner

    n_off = max(1, int(stagger))
    cfg = RunnerConfig(input_dir=os.path.dirname(plate.path), output_dir=OUT_DIR,
                       auto_tune=False, per_track_policy=True, seed_stagger=n_off,
                       max_tracks=n_seeds, feature_quality=quality,
                       min_feature_dist=min_dist)
    runner = BatchTrackerRunner(cfg)
    offsets = runner._stagger_offsets(int(plate.count)) if n_off > 1 else [0]
    need = max(offsets) + 1
    frames = np.stack([plate.frame(i) for i in range(need)]) if need > 1 \
        else plate.frame(0)[None]
    q, feats = runner._staggered_queries(frames, None, n_seeds, int(plate.count))
    if q is None:
        raise RuntimeError("seeder returned no points")
    kinds = [k for (_f, k) in feats] if feats else [""] * q.shape[1]

    seeds = []
    for i, (fr, x, y) in enumerate(q[0].astype(float)):
        # The seeder works top-down in pixels; the export space this experiment scores in
        # is y-flipped, and so is Blender's. Flip once, here, so there is exactly one place
        # the convention lives.
        y3de = (plate.h - 1.0) - float(y)
        u, v = px_to_uv(float(x), y3de, plate.w, plate.h)
        if not (0.0 < u < 1.0 and 0.0 < v < 1.0):
            continue
        seeds.append({"id": f"BLN_{i:04d}", "frame": int(fr) + 1, "u": u, "v": v,
                      "kind": kinds[i] if i < len(kinds) else "", "guide": {}})
    return seeds


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default=r"D:\Jefrin\IN\SH004.mp4",
                    help="movie file, or a folder of frames")
    ap.add_argument("--source", choices=["tapnext", "seeder"], default="tapnext")
    ap.add_argument("--reuse-tapnext", default="",
                    help="an existing bot export to seed from, instead of running the bot")
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--name", default="", help="shot name for the output files")
    ap.add_argument("--out", default="", help="write the 3DE export here instead of out/")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--keep", type=int, default=0, help="cap seeds taken from the guide")
    ap.add_argument("--quality", type=float, default=0.02)
    ap.add_argument("--min-dist", type=int, default=12)
    ap.add_argument("--stagger", type=int, default=1)
    ap.add_argument("--flat-geom", action="store_true")
    ap.add_argument("--no-replant", action="store_true")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--replant-gap", type=int, default=3)
    ap.add_argument("--replant-rounds", type=int, default=6)
    ap.add_argument("--correlation", type=float, default=0.75)
    ap.add_argument("--blender", default="")
    ap.add_argument("--timeout", type=float, default=7200.0)
    args = ap.parse_args()

    t_all = time.time()
    plate = Plate(args.plate, ifl_dir=OUT_DIR)
    # A frames folder is often called `plate` or `v001`, which is not the shot name. The
    # bench shots are laid out that way, and the scorer keys on the shot name.
    name = args.name or plate.name
    print(f"plate {name}: {plate.w}x{plate.h}, {plate.count} frames")

    # ---------------------------------------------------------------- seeds + guide
    guide_txt = ""
    if args.source == "tapnext":
        guide_txt = args.reuse_tapnext
        if guide_txt:
            print(f"seeding from existing export: {guide_txt}")
        else:
            print("running the bot's TAPNext to produce seeds + replant guide ...")
            t0 = time.time()
            guide_txt = tapnext_export(plate, OUT_DIR, args.seeds)
            print(f"  bot TAPNext: {time.time() - t0:.1f}s -> {os.path.basename(guide_txt)}")
        seeds = seeds_from_export(guide_txt, plate.w, plate.h, plate.count, args.keep)
        n_kind = classify_seeds(plate, seeds, args.quality, args.min_dist)
        kinds: dict[str, int] = {}
        for s in seeds:
            kinds[s["kind"] or "?"] = kinds.get(s["kind"] or "?", 0) + 1
        print(f"  {len(seeds)} seeds, {n_kind} classified: "
              + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    else:
        seeds = seeds_from_seeder(plate, args.seeds, args.quality, args.min_dist,
                                  args.stagger)
        print(f"  {len(seeds)} seeds from the raw seeder (no guide -> no replant)")
        args.no_replant = True

    if not seeds:
        print("no seeds -- nothing to track")
        return 1

    seed_json = os.path.join(OUT_DIR, f"{name}__{args.tag}__seeds.json")
    # `plate.path`, never `plate.load_path`: the latter is the IFL that SynthEyes wants and
    # Blender cannot read -- it opens the .ifl as a 256x256 image and every seed lands in a
    # different plate. The size guard in bl_track.py catches that, and did.
    dump_json(seed_json, {"plate": plate.path,
                          "width": plate.w, "height": plate.h, "frames": plate.count,
                          "seeds": seeds})

    # ---------------------------------------------------------------- track in blender
    out_json = os.path.join(OUT_DIR, f"{name}__{args.tag}__bl.json")
    bl_args = ["--clip", plate.path,
               "--seeds", seed_json, "--out", out_json,
               "--replant-gap", args.replant_gap, "--replant-rounds", args.replant_rounds,
               "--correlation", args.correlation]
    if args.flat_geom:
        bl_args.append("--flat-geom")
    if args.no_replant:
        bl_args.append("--no-replant")
    if args.no_backward:
        bl_args.append("--no-backward")

    print("tracking in Blender ...")
    t0 = time.time()
    rc, _out = run_blender(os.path.join(HERE, "bl_track.py"), bl_args,
                           blender=args.blender, timeout=args.timeout)
    if rc != 0 or not os.path.isfile(out_json):
        print("blender failed -- see the output above")
        return 2
    track_secs = time.time() - t0

    # ---------------------------------------------------------------- back to 3DE
    res = load_json(out_json)
    tracks = {}
    for t in res["tracks"]:
        pts = {}
        for f, u, v in t["pts"]:
            x, y = uv_to_px(u, v, plate.w, plate.h)
            pts[int(f)] = (x, y)
        if len(pts) >= 2:
            tracks[t["id"]] = pts
    # Round-trip: the point Blender reports on the seed frame, converted back, must be the
    # pixel the seeder asked for. If this is not ~0 then every tracker started on the wrong
    # feature and no number below means anything -- so it is checked on every run rather
    # than being a separate script somebody remembers to run.
    worst = 0.0
    for s in seeds:
        pts = tracks.get(s["id"])
        if not pts or int(s["frame"]) not in pts:
            continue
        sx, sy = uv_to_px(s["u"], s["v"], plate.w, plate.h)
        gx, gy = pts[int(s["frame"])]
        worst = max(worst, abs(gx - sx), abs(gy - sy))
    print(f"  seed round-trip : max {worst:.4f}px  " + ("PASS" if worst < 0.01 else "FAIL"))

    out_txt = args.out or os.path.join(OUT_DIR, f"{name}__{args.tag}__blender.txt")
    n = write_3de(out_txt, tracks)

    lens = [len(p) for p in tracks.values()]
    lens.sort()
    total = sum(lens)
    print(f"\nexported {n} tracks -> {out_txt}")
    print(f"  frames tracked : {total}  ({total / max(1, track_secs):.0f} tracker-frames/s)")
    if lens:
        print(f"  track length   : median {lens[len(lens) // 2]}, "
              f"max {lens[-1]}, of {plate.count} frames")
        print(f"  full length    : {sum(1 for l in lens if l >= plate.count)} tracks")
    print(f"  replants       : {res.get('replants', 0)}")
    if guide_txt:
        print(f"  guide/control  : {guide_txt}")
    print(f"total {time.time() - t_all:.1f}s")
    plate.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
