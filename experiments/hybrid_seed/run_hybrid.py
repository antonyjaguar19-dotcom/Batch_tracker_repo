"""HYBRID: TAPNext's seeder chooses the points, SynthEyes tracks them.

The bot's TAPNext path and its SynthEyes path each pick their own features. This asks what
happens when one engine's seed selection is handed to the other engine's tracker.

Nothing in `app/` is modified. The bot's real seeder is imported and called
(`BatchTrackerRunner._staggered_queries` -> goodFeaturesToTrack + edge/anisotropy rejection
+ seed classification), and the resulting points are injected into SynthEyes as area-match
trackers.

Mechanism, all measured on build 2026.2.4679 (see FINDINGS.md):
  * `tk = new ob.trk` / `tk.key = Point(u,v)`   -- create a tracker at a chosen point
  * `tk.Run()` once per frame                   -- THIS is the per-frame track step;
                                                   Scene.RunTrackersFwd() does nothing here
  * the engine's existing Sizzle export         -- writes classic 3DE ASCII

    run_hybrid.py --plate D:\\Jefrin\\IN\\SH004.mp4 --seeds 400
    run_hybrid.py --plate <frames dir> --seeds 400 --reacquire
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

from plate_io import Plate, to_uv  # noqa: E402  (sets OPENCV_IO_ENABLE_OPENEXR before cv2)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.tracker_core import RunnerConfig, BatchTrackerRunner  # noqa: E402
from app.compare_tracks import load_tracks  # noqa: E402
from app.pattern_refine import _extract, _ncc_match  # noqa: E402
from sylab import connect, OUT_DIR  # noqa: E402

# Tracker geometry per seed class: how much of the image the tracker remembers (size) and
# how far it looks for it next frame (srchu/srchv). A corner is pinned in both axes so it
# gets a small tight patch; a point on a long edge can only be located ACROSS the edge and
# slides along it, so it gets a wider search. This is app/track_meta.py's per-track policy
# expressed in SynthEyes' own fields -- something neither engine does today. Untested; run
# once with --flat-geom to get the control.
# (patch_px, search_px). In PIXELS, converted per plate at build time -- see
# build_seed_script. Keeping these as raw normalised numbers made every tracker twice as
# wide on a 4K plate as on a 2K one, and the area-match cost scales with the square of that.
KIND_GEOM_PX = {
    "corner":       (24, 20),
    "blob":         (40, 28),
    "edge":         (32, 34),
    "dense-corner": (26, 22),
    "dense-edge":   (32, 34),
    "dense":        (28, 24),
    "":             (32, 24),
}
FLAT_GEOM_PX = KIND_GEOM_PX[""]


# ----------------------------------------------------------------------------- seeding

def tapnext_seeds(plate: Plate, n_seeds: int, quality: float, min_dist: int, stagger: int):
    """Run the bot's own seeder. Returns (pts Nx3 as (frame0, x, y), kinds)."""
    n_off = max(1, int(stagger))
    cfg = RunnerConfig(
        input_dir=os.path.dirname(plate.path), output_dir=OUT_DIR,
        auto_tune=False,           # a fixed seeder keeps runs comparable
        per_track_policy=True,     # so seeds come back classified corner/blob/edge/dense
        seed_stagger=n_off,
        max_tracks=n_seeds, feature_quality=quality, min_feature_dist=min_dist,
    )
    runner = BatchTrackerRunner(cfg)

    # _staggered_queries indexes frames[off] for each stagger offset, so it needs exactly the
    # frames it will seed on. With stagger=1 that is frame 0 alone.
    offsets = runner._stagger_offsets(int(plate.count)) if n_off > 1 else [0]
    need = max(offsets) + 1
    frames = np.stack([_req(plate, i) for i in range(need)]) if need > 1 else _req(plate, 0)[None]

    q, seeds = runner._staggered_queries(frames, None, n_seeds, int(plate.count))
    if q is None:
        raise RuntimeError("seeder returned no points")
    kinds = [k for (_f, k) in seeds] if seeds else [""] * q.shape[1]
    return q[0].astype(float), kinds


def _req(plate: Plate, i: int):
    img = plate.frame(i)
    if img is None:
        raise RuntimeError(f"could not read frame {i} of {plate.path}")
    return img


# ------------------------------------------------------------------------ sizzle build

def build_seed_script(seeds, kinds, w, h, plate_w: int, per_kind_geom: bool) -> str:
    """Create the trackers and key them on the first frame. No tracking yet."""
    lines = ["//SIZZLET BTRHybridSeed",
             "ob = Scene.activeObj", "shot = ob.shot",
             "start = shot.start", "stop = shot.stop", "frame = start"]
    for i, (fr, x, y) in enumerate(seeds):
        u, v = to_uv(float(x), float(y), w, h)
        kind = kinds[i] if i < len(kinds) else ""
        px, spx = KIND_GEOM_PX.get(kind, FLAT_GEOM_PX) if per_kind_geom else FLAT_GEOM_PX
        # u/v span -1..1 across the plate, so one u unit is plate_w/2 pixels: a patch of
        # `px` pixels is 2*px/plate_w. Defining the geometry in PIXELS matters -- as fixed
        # normalised numbers the same tracker silently doubles in size on a 4K plate, and
        # the area match cost grows with the square of it.
        size = 2.0 * px / plate_w
        srch = 2.0 * spx / plate_w
        lines += [
            "frame = start", "tk = new ob.trk", f'tk.nm = "HYB{i:04d}"',
            "tk.kind = 0", f"tk.size = {size:.6f}", "tk.asp = 1.0",
            f"tk.srchu = {srch:.6f}", f"tk.srchv = {srch:.6f}",
            "tk.smooth = 20", "tk.autokey = 20", "tk.isSel = 2",
            f"tk.key = Point({u:.6f},{v:.6f})", "tk.isEnabled = 1", "x = tk.Run()",
        ]
    return "\n".join(lines) + "\n"


def build_track_chunk(f0: int, f1: int) -> str:
    """Track every tracker over frames f0..f1 inclusive.

    Frame-major, so each frame's image is touched once for all trackers. Scene.SetFrame is
    deliberately NOT called -- the Sizzle `frame` global alone advances the image, verified
    by sampling AvgImgColor with and without it.

    The shot is done in chunks rather than one call because a single RunScriptFile that
    covers a whole 4K plate gave no progress and could hang SynthEyes outright with nothing
    to report. A chunk boundary is somewhere to check liveness and print progress.
    """
    return "\n".join([
        "//SIZZLET BTRHybridTrack",
        "ob = Scene.activeObj",
        f"for (f = {f0}; f <= {f1}; f++)", "    frame = f",
        "    for (tk in ob.trk)", "        x = tk.Run()", "    end", "end",
    ]) + "\n"


def build_shotinfo_script(diag_path: str) -> str:
    """Ask SynthEyes what it thinks it has loaded.

    NewSceneAndShot can return a shot object while the plate did not actually attach -- the
    window title then reads "Camera01 -" with no path, seeding runs against no image, and
    the next Sizzle call hangs. Checking the reported width/height/length against the plate
    turns that into an error instead of a mystery.
    """
    return "\n".join([
        "//SIZZLET BTRShotInfo",
        "ob = Scene.activeObj", "shot = ob.shot",
        f'openout("{diag_path}")',
        'printf("width %d\\n", shot.width)',
        'printf("height %d\\n", shot.height)',
        'printf("start %d\\n", shot.start)',
        'printf("stop %d\\n", shot.stop)',
        "closeout()",
    ]) + "\n"


def build_span_script(diag_path: str) -> str:
    return "\n".join(["//SIZZLET BTRHybridSpans",
                      "ob = Scene.activeObj", "shot = ob.shot",
                      "start = shot.start", "stop = shot.stop"]
                     + _span_report(diag_path)) + "\n"


def build_resume_script(replants, diag_path) -> str:
    """Re-key already-existing trackers mid-shot and carry them on to the end of the shot.

    `replants` is [(col_index, name, frame0, u, v), ...]. Trackers are addressed by index --
    ob.trk is in creation order, which is seed order -- and the name is checked anyway, since
    a silent index slip would hand a track its neighbour's re-acquisition.
    """
    ordered = sorted(replants, key=lambda r: r[2])
    lines = ["//SIZZLET BTRHybridResume",
             "ob = Scene.activeObj", "shot = ob.shot",
             "start = shot.start", "stop = shot.stop", "bad = 0", "n = 0"]
    # Keying walks frames in ascending order so the plate is read forwards, not seeked about.
    for k, (idx, name, f0, u, v) in enumerate(ordered):
        lines += [
            f"r{k} = ob.trk[{idx + 1}]",
            f'if (r{k}.nm != "{name}")', "    bad = bad + 1", "else",
            f"    frame = {f0}",
            f"    r{k}.key = Point({u:.6f},{v:.6f})",
            f"    x = r{k}.Run()", "    n = n + 1", "end",
        ]
    if ordered:
        first = ordered[0][2]
        lines += [f"for (f = {first} + 1; f <= stop; f++)", "    frame = f"]
        for k, (idx, name, f0, u, v) in enumerate(ordered):
            lines += [f"    if (f > {f0})", f"        x = r{k}.Run()", "    end"]
        lines += ["end"]
    lines += ['openout("' + diag_path + '")',
              'printf("replanted %d\\n", n)', 'printf("name_mismatch %d\\n", bad)',
              "closeout()"]
    return "\n".join(lines) + "\n"


def _span_report(diag_path):
    return ['openout("' + diag_path + '")',
            'printf("ntrk %d\\n", #ob.trk)',
            "for (tk in ob.trk)", "    cnt = 0",
            "    for (frame = start; frame <= stop; frame++)",
            "        if (tk.valid)", "            cnt = cnt + 1", "        end", "    end",
            '    printf("%s %d\\n", tk.nm, cnt)', "end", "closeout()"]


# -------------------------------------------------------------------------- reacquire

def find_replants(plate: Plate, export_txt: str, seeds, max_gap: int,
                  ncc_thresh: float, patch_px: int, search_px: int):
    """Where does each dead track's feature come back?

    SynthEyes stops but does not say where the feature went, so the search happens here,
    with the same primitives the bot's own re-acquisition uses (app/pattern_refine.py:
    _extract for the sub-pixel patch, _ncc_match for the search). One sequential pass over
    the plate: a track's patch is grabbed as we pass its last good frame, and every frame
    after that is a candidate until the gap budget runs out.
    """
    tracks = load_tracks(export_txt)
    half = max(4, int(patch_px) // 2)

    # last good 0-based frame per seed column, for tracks that stop before the end
    pending = {}
    for i in range(len(seeds)):
        tr = tracks.get(f"HYB{i:04d}")
        if not tr:
            continue
        last1 = max(tr)                       # export frames are 1-based
        last0 = last1 - 1
        if last0 >= plate.count - 2:
            continue                          # ran to the end, nothing to recover
        pending[i] = {"last0": last0, "xy": tr[last1], "patch": None}
    if not pending:
        return [], 0

    replants = []
    start_at = min(p["last0"] for p in pending.values())
    live = dict(pending)
    for f in range(start_at, plate.count):
        if not live:
            break
        img = plate.frame(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        done = []
        for i, st in live.items():
            if f == st["last0"]:
                st["patch"] = _extract(gray, st["xy"][0], st["xy"][1], half)
                if st["patch"] is None:
                    done.append(i)            # too close to the frame edge to remember
                continue
            if f <= st["last0"] or st["patch"] is None:
                continue
            if f - st["last0"] > max_gap:
                done.append(i)                # past the budget: a different feature now
                continue
            got = _ncc_match(gray, st["patch"], st["xy"][0], st["xy"][1],
                             search=int(search_px), half=half, edge_clamp=True)
            if got is not None and got[2] >= ncc_thresh:
                replants.append((i, f"HYB{i:04d}", f, *to_uv(got[0], got[1], plate.w, plate.h)))
                done.append(i)
        for i in done:
            live.pop(i, None)
    return replants, len(pending)


# ------------------------------------------------------------------------------- main

def _read_spans(diag: str, total: int):
    if not os.path.isfile(diag):
        return None
    spans = []
    for ln in open(diag, encoding="utf-8", errors="ignore").read().splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[0] != "ntrk":
            try:
                spans.append(int(parts[1]))
            except ValueError:
                pass
    if not spans:
        return None
    spans_np = np.array(spans)
    full = int((spans_np >= total - 1).sum())
    med = int(np.median(spans_np))
    print(f"   spans: {len(spans)} trackers, {full} held all {total} frames, "
          f"median {med}, min {int(spans_np.min())}")
    if med <= 12 and total > 30:
        print("   *** WARNING: a median span of ~10 frames on a long shot is the SynthEyes "
              "Demo signature. Check the licence before believing any number here. ***")
    return spans


class HangError(RuntimeError):
    pass


def run_bounded(eng, script: str, timeout_s: float, label: str):
    """Run a Sizzle script but never block forever on it.

    SynthEyes stopped responding partway through a 4K plate here (its window went to the
    Windows 'Ghost' not-responding class, then the socket dropped with WinError 10054).
    RunScriptFile is a blocking socket call, so a hung SynthEyes hangs the experiment with
    nothing printed and nothing to diagnose. Running it on a worker thread bounds the wait:
    the thread is abandoned on timeout, which is fine because the caller gives up on this
    SynthEyes entirely.
    """
    import threading
    box = {}

    def _go():
        try:
            eng._run_sizzle(script, watchdog_secs=int(max(30, timeout_s)))
            box["ok"] = True
        except Exception as e:            # noqa: BLE001 - reported, not swallowed
            box["err"] = e

    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise HangError(f"SynthEyes stopped responding during {label} "
                        f"(no reply for {timeout_s:.0f}s)")
    if "err" in box:
        raise RuntimeError(f"{label}: {box['err']}")


def _spans_from_export(txt: str, total: int):
    """Spans read off the exported 3DE file. Counts POINTS, so a track with a hole in it
    reports the frames it actually holds -- which is the honest number once re-acquisition
    has stitched a gap back together."""
    tracks = load_tracks(txt)
    if not tracks:
        return
    spans = np.array([len(v) for v in tracks.values()])
    covers = np.array([max(v) - min(v) + 1 for v in tracks.values()])
    gapped = int((covers > spans).sum())
    print(f"   after re-acquisition: {len(spans)} tracks, "
          f"median points {int(np.median(spans))}, "
          f"median span {int(np.median(covers))} of {total}, "
          f"{gapped} track(s) carry a gap")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plate", default=r"D:\Jefrin\IN\SH004.mp4",
                    help="movie file, or a folder of frames")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--quality", type=float, default=0.02)
    ap.add_argument("--min-dist", type=int, default=12)
    ap.add_argument("--stagger", type=int, default=1,
                    help="seed entry points. 1 = all seeds on frame 0. Only raise this if "
                         "the midshot probe passes -- mid-shot creation fails on some builds")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--flat-geom", action="store_true",
                    help="one tracker size for every seed (the control for per-kind geometry)")
    ap.add_argument("--reacquire", action="store_true",
                    help="replant a dead track's own seed when its feature comes back")
    ap.add_argument("--reacquire-rounds", type=int, default=2)
    ap.add_argument("--reacquire-gap", type=int, default=48)
    ap.add_argument("--reacquire-ncc", type=float, default=0.75)
    ap.add_argument("--reacquire-patch", type=int, default=31)
    ap.add_argument("--reacquire-search", type=int, default=48)
    ap.add_argument("--no-validate", dest="validate", action="store_false",
                    help="never build the SynthEyes RAM cache")
    ap.add_argument("--validate-max-gb", type=float, default=2.0,
                    help="skip the RAM cache when the shot would exceed this")
    ap.add_argument("--frames-per-call", type=int, default=20,
                    help="frames tracked per Sizzle call; smaller = finer progress and a "
                         "tighter bound on a SynthEyes hang")
    ap.add_argument("--chunk-timeout", type=float, default=300.0,
                    help="give up on SynthEyes if one call takes longer than this")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    plate = Plate(args.plate, ifl_dir=out_dir)
    out_txt = os.path.join(out_dir, f"{plate.name}__{args.tag}.txt")
    diag = os.path.join(out_dir, f"{args.tag}.diag").replace("\\", "/")

    print(f"plate {plate.name}: {plate.w}x{plate.h}, {plate.count} frames", flush=True)

    t0 = time.time()
    seeds, kinds = tapnext_seeds(plate, args.seeds, args.quality, args.min_dist, args.stagger)
    counts = {}
    for k in kinds:
        counts[k or "unclassified"] = counts.get(k or "unclassified", 0) + 1
    print(f"seeder returned {len(seeds)} points in {time.time()-t0:.1f}s: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    print(f"tracker geometry: {'per seed class' if not args.flat_geom else 'flat (control)'}",
          flush=True)

    seed_script = build_seed_script(seeds, kinds, plate.w, plate.h, plate.w,
                                    per_kind_geom=not args.flat_geom)
    with open(os.path.join(out_dir, f"{args.tag}.szl"), "w", encoding="utf-8") as f:
        f.write(seed_script)

    try:
        eng = connect(quiet=False)
    except SystemExit as e:
        print(f"FAIL: {e}")
        return 3
    eng.set_writable_folder(out_dir)

    print("-> loading plate into SynthEyes", flush=True)
    if eng.hlev.NewSceneAndShot(plate.load_path) is None:
        print("FAIL: NewSceneAndShot returned None")
        return 4
    time.sleep(6)
    # Validate builds a whole-shot RAM cache. process_shot does it for any shot under 300
    # frames, but that rule was written for 2K plates: a 127-frame 4K plate is ~3.4GB of
    # 8-bit RGB and more once SynthEyes has its own copy, which crashed the socket outright
    # here (WinError 10054). Gate it on the actual size, and skip it when it is too big --
    # it was measured to make no difference to correctness, only to speed.
    est_gb = plate.w * plate.h * 3.0 * plate.count / (1024.0 ** 3)
    if args.validate and est_gb <= args.validate_max_gb:
        for sh in eng.hlev.Shots():
            eng.hlev.Validate(sh)
        print(f"   validated RAM cache (~{est_gb:.1f} GB)", flush=True)
        time.sleep(2)
    else:
        print(f"   skipping the RAM cache (~{est_gb:.1f} GB, limit "
              f"{args.validate_max_gb:.1f} GB)", flush=True)

    eng._resync_socket()           # Validate desyncs it; RunScriptFile then silently no-ops

    info_diag = os.path.join(out_dir, f"{args.tag}_shot.diag")
    try:
        os.remove(info_diag)
    except OSError:
        pass
    try:
        run_bounded(eng, build_shotinfo_script(info_diag.replace("\\", "/")),
                    args.chunk_timeout, "shot info")
    except (HangError, RuntimeError) as e:
        print(f"FAIL: {e}")
        return 6
    info = dict(_pairs(open(info_diag, encoding="utf-8", errors="ignore").read())
                if os.path.isfile(info_diag) else [])
    if info.get("width") != plate.w or info.get("height") != plate.h:
        print(f"FAIL: SynthEyes loaded {info.get('width')}x{info.get('height')} but the plate "
              f"is {plate.w}x{plate.h}. The plate did not attach -- NewSceneAndShot can "
              f"return a shot with no image. Close SynthEyes and re-run.")
        return 6
    n_frames_se = int(info.get("stop", 0)) - int(info.get("start", 0)) + 1
    print(f"   SynthEyes has {info.get('width')}x{info.get('height')}, "
          f"frames {info.get('start')}-{info.get('stop')} ({n_frames_se})", flush=True)

    print(f"-> injecting {len(seeds)} trackers", flush=True)
    t0 = time.time()
    try:
        run_bounded(eng, seed_script, args.chunk_timeout, "seeding")
    except (HangError, RuntimeError) as e:
        print(f"FAIL: {e}")
        return 6
    print(f"   seeded in {time.time()-t0:.1f}s", flush=True)

    print(f"-> tracking {plate.count} frames in chunks of {args.frames_per_call}", flush=True)
    t0 = time.time()
    step = max(1, int(args.frames_per_call))
    done = 0
    for f0 in range(1, plate.count, step):
        f1 = min(plate.count - 1, f0 + step - 1)
        tc = time.time()
        try:
            run_bounded(eng, build_track_chunk(f0, f1), args.chunk_timeout,
                        f"frames {f0}-{f1}")
        except HangError as e:
            print(f"FAIL: {e}")
            print("   SynthEyes hung partway through. On a 4K plate this happened on the "
                  "Demo build; if it happens on a licensed one, lower --frames-per-call or "
                  "--seeds and report the frame range above.")
            return 6
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return 6
        done = f1
        rate = (len(seeds) * (f1 - f0 + 1)) / max(time.time() - tc, 1e-6)
        print(f"   frames {f0:5d}-{f1:<5d}  {time.time()-tc:6.1f}s  "
              f"{rate:8,.0f} tracker-frames/s", flush=True)
        if not eng.is_alive():
            print("FAIL: SynthEyes died during tracking.")
            return 6
    dt = time.time() - t0
    print(f"   tracked to frame {done} in {dt:.1f}s "
          f"({len(seeds)*max(done,1)/max(dt,1e-6):,.0f} tracker-frames/s overall)", flush=True)

    try:
        run_bounded(eng, build_span_script(diag), args.chunk_timeout, "span report")
        _read_spans(diag.replace("/", os.sep), plate.count)
    except (HangError, RuntimeError) as e:
        print(f"   (span report skipped: {e})")

    n = eng._sizzle_export_3de(out_txt)
    if not n or n <= 0:
        print("FAIL: export produced no tracks")
        return 5

    if args.reacquire:
        for rnd in range(1, max(1, args.reacquire_rounds) + 1):
            print(f"-> re-acquisition round {rnd}", flush=True)
            t_s = time.time()
            replants, n_dead = find_replants(
                plate, out_txt, seeds, args.reacquire_gap, args.reacquire_ncc,
                args.reacquire_patch, args.reacquire_search)
            print(f"   {n_dead} track(s) ended early, {len(replants)} feature(s) found again "
                  f"({time.time()-t_s:.1f}s searching)", flush=True)
            if not replants:
                break
            rdiag = os.path.join(out_dir, f"{args.tag}_resume{rnd}.diag").replace("\\", "/")
            rscript = build_resume_script(replants, rdiag)
            with open(os.path.join(out_dir, f"{args.tag}_resume{rnd}.szl"), "w",
                      encoding="utf-8") as f:
                f.write(rscript)
            print(f"   resume script: {len(rscript.splitlines())} lines", flush=True)
            t_s = time.time()
            eng._resync_socket()
            try:
                run_bounded(eng, rscript, args.chunk_timeout * 4, "re-acquisition resume")
            except (HangError, RuntimeError) as e:
                print(f"   re-acquisition abandoned: {e}")
                break
            print(f"   resume tracked in {time.time()-t_s:.1f}s", flush=True)
            rd = rdiag.replace("/", os.sep)
            if os.path.isfile(rd):
                info = dict(_pairs(open(rd, encoding="utf-8", errors="ignore").read()))
                print(f"   replanted {info.get('replanted', 0)}", flush=True)
                if info.get("name_mismatch"):
                    print(f"   *** {info['name_mismatch']} tracker index/name mismatches - "
                          f"ob.trk is not in creation order; results are unsafe ***")
            n = eng._sizzle_export_3de(out_txt)
        # The resume script does not rewrite the span diag, so read the spans back off the
        # export itself -- otherwise this would report the pre-reacquisition numbers.
        _spans_from_export(out_txt, plate.count)

    plate.close()
    print(f"RESULT: {n} tracks -> {out_txt}", flush=True)
    return 0


def _pairs(text: str):
    for ln in (text or "").splitlines():
        parts = ln.split()
        if len(parts) == 2:
            try:
                yield parts[0], int(parts[1])
            except ValueError:
                continue


if __name__ == "__main__":
    raise SystemExit(main())
