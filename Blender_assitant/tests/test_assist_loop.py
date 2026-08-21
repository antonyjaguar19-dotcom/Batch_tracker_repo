"""End-to-end gate for the artist loop: seed -> Blender -> CoTracker -> Blender.

Runs headless with real seeds on a real plate, driving the same functions the modal
operator drives, so a failure here is a failure there.

    runtime\\python311\\python.exe tests\\test_assist_loop.py --plate <mp4 or folder> \\
        --points 0.45,0.42 0.60,0.55 --rounds 2

Points are given in normalised image coordinates (x right, y DOWN, matching what you read
off a still) so a test seed can be written down without opening Blender.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_BLENDER = (r"C:\Users\jefrin\Downloads\blender-5.2.0-windows-x64"
                   r"\blender-5.2.0-windows-x64\blender.exe")

DRIVER = r'''
import json, os, sys, time
import bpy

ASSIST = r"{assist}"
sys.path.insert(0, os.path.join(ASSIST, "addon"))
sys.path.insert(0, os.path.join(ASSIST, "sidecar"))

from btr_assist import three_de, track_core
from btr_assist.ops_assist import (marker_to_image_px, image_px_to_uv, live_frames,
                                   dead_tracks, marker_pattern_box, marker_search_px)

spec = json.load(open(r"{spec}", encoding="utf-8"))
plate, pts, rounds, gap, tail = (spec["plate"], spec["points"], spec["rounds"],
                                 spec["gap"], spec["tail"])
verify, min_match = spec["verify_pattern"], spec["min_match"]


def log(m):
    print("[loop] %s" % m, flush=True)


def load_clip(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in
                       (".exr", ".dpx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        return bpy.data.movieclips.load(os.path.join(path, names[0]))
    return bpy.data.movieclips.load(path)


clip = load_clip(plate)
w, h = clip.size
n_frames = clip.frame_duration
log("clip %dx%d %d frames source=%s" % (w, h, n_frames, clip.source))

win = bpy.context.window_manager.windows[0]
area = max(win.screen.areas, key=lambda a: a.width * a.height)
area.type = "CLIP_EDITOR"
area.spaces.active.clip = clip
region = next(r for r in area.regions if r.type == "WINDOW")
scene = bpy.context.scene
scene.frame_start, scene.frame_end = 1, n_frames
ctx = (win, area, region, clip, scene)

opts = track_core.Opts(leash=0.0)
track_core.apply_settings(clip, opts)

tracks = three_de.active_tracks(clip)
records, seeds_px, patterns, search_px = [], {{}}, {{}}, {{}}
for i, (nx, ny) in enumerate(pts):
    x, y_down = nx * w, ny * h
    u, v = image_px_to_uv(x, y_down, w, h)
    tr = tracks.new(name="USER_%02d" % i, frame=1)
    tr.motion_model = "Loc"; tr.use_brute = True; tr.use_normalization = True
    tr.correlation_min = 0.75; tr.frames_limit = 0; tr.pattern_match = "PREV_FRAME"
    m = tr.markers[0]; m.co = (u, v)
    track_core.set_geom(m, 21.0 * max(1.0, w / 1920.0), 41.0 * max(1.0, w / 1920.0), w, h)
    records.append({{"t": tr, "id": tr.name, "kind": "", "alive": True,
                    "w": w, "h": h, "seed_frame": 1}})
    seeds_px[tr.name] = (1, x, y_down)
    cx, cy, pw, ph = marker_pattern_box(m, w, h)
    patterns[tr.name] = {{"frame": 1, "cx": cx, "cy": cy, "w": pw, "h": ph}}
    search_px[tr.name] = marker_search_px(m, w, h)
    log("seed %s at image px (%.1f, %.1f), pattern %.0fx%.0f at (%.1f, %.1f)"
        % (tr.name, x, y_down, pw, ph, cx, cy))

# round-trip the coordinate conversion before believing anything downstream
worst = 0.0
for r in records:
    mx, my = marker_to_image_px(r["t"].markers[0], w, h)
    sx, sy = seeds_px[r["id"]][1], seeds_px[r["id"]][2]
    worst = max(worst, ((mx - sx) ** 2 + (my - sy) ** 2) ** 0.5)
# Bar is 0.001 px, not zero: Blender stores marker.co as float32, so a round trip
# through it carries ~7e-5 px on a 3840-wide plate -- storage precision, not an error.
# 0.001 px still catches everything that matters: a y-flip is thousands of px, a dropped
# half-pixel centre is 0.5, a resolution mismatch is a fixed fraction of the width.
log("coord round-trip (uv <-> image px, y flip): max %.6f px  %s"
    % (worst, "PASS" if worst < 1e-3 else "FAIL"))
assert worst < 1e-3, "coordinate conversion is not reversible"

from btr_assist import client   # the addon's own client, not a sidecar module
root = ASSIST
py = os.path.join(ASSIST, "runtime", "python311", "python.exe")
client.ensure(root, py, 0, timeout=120)

report = {{"rounds": [], "coord_roundtrip_px": worst}}
resumed, continue_from, gave_up = {{}}, {{}}, set()
for rnd in range(rounds + 1):
    t0 = time.time()
    st = {{}}
    for _ in track_core.track_job(ctx, records, n_frames, {{}}, opts, stats=st):
        pass
    if rnd == 0:
        track_core.track_backward_pass(ctx, records, opts)
    spans = sorted(len(live_frames(r["t"])) for r in records)
    log("round %d: tracked in %.1fs, deaths %d, spans %s"
        % (rnd, time.time() - t0, st.get("deaths", 0), spans))
    report["rounds"].append({{"round": rnd, "deaths": st.get("deaths", 0),
                             "spans": spans, "seconds": round(time.time() - t0, 1)}})
    if rnd >= rounds:
        break

    dead = dead_tracks([r["t"] for r in records], n_frames, tail)
    if not dead:
        log("nothing died -- stopping")
        break
    reqs = []
    for tr, f0, f1 in dead:
        if tr.name in gave_up:
            continue
        s = seeds_px[tr.name]
        cont = continue_from.get(tr.name)
        if cont is not None and cont[0] > f1:
            lf, lx, ly, g = int(cont[0]), float(cont[1]), float(cont[2]), 1
        else:
            m = tr.markers.find_frame(f1, exact=True)
            lf, g = f1, gap
            lx, ly = marker_to_image_px(m, w, h)
        reqs.append({{"id": tr.name, "query_frame": s[0], "query_x": s[1], "query_y": s[2],
                     "last_good_frame": lf, "last_good_x": lx, "last_good_y": ly,
                     "gap": g, "pattern": patterns.get(tr.name),
                     "search_px": search_px.get(tr.name, 0.0)}})
        log("  dead: %s spans %d..%d of %d%s"
            % (tr.name, f0, f1, n_frames,
               "  (search continues from f%d)" % lf if lf != f1 else ""))
    if not reqs:
        log("every dead track has been given up on -- stopping")
        break

    ci = {{"path": os.path.abspath(plate), "width": w, "height": h, "frames": n_frames}}
    r = client.start_reacquire(root, ci, reqs,
                               {{"frame_hi": n_frames, "verify_pattern": verify,
                                "min_match": min_match}})
    while True:
        js = client.poll(root, r["id"])
        if js["state"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    if js["state"] != "done":
        log("re-acquire FAILED: %s" % (js.get("error") or {{}}).get("message"))
        report["error"] = (js.get("error") or {{}}).get("message")
        break
    res = js["result"]
    log("  CoTracker: %d resume(s), %d miss(es)"
        % (len(res["resumes"]), len(res["misses"])))
    for miss in res["misses"]:
        log("    miss %s: %s" % (miss["id"], miss["reason"]))
        if miss.get("retry") and miss.get("tail_frame"):
            continue_from[miss["id"]] = (miss["tail_frame"], miss["tail_x"], miss["tail_y"])
        else:
            gave_up.add(miss["id"])
    by = {{r_["id"]: r_ for r_ in records}}
    for rr in res["resumes"]:
        rec = by.get(rr["id"])
        u, v = image_px_to_uv(rr["x"], rr["y"], w, h)
        mk = rec["t"].markers.insert_frame(int(rr["frame"]), co=(u, v))
        mk.mute = False
        old = rec["t"].markers.find_frame(int(rr["last_good_frame"]), exact=True)
        if old is not None:
            sx = (old.search_max[0] - old.search_min[0])
            sy = (old.search_max[1] - old.search_min[1])
            mk.pattern_corners = old.pattern_corners
            mk.search_min = (-sx, -sy); mk.search_max = (sx, sy)
        rec["alive"] = True; rec["seed_frame"] = int(rr["frame"])
        resumed.setdefault(rr["id"], []).append(int(rr["frame"]))
        continue_from.pop(rr["id"], None)
        sc = rr.get("match_score")
        log("    %s searched from f%d -> back at f%d at (%.1f, %.1f), first over the "
            "line f%s, %d frame(s) swept, %d occluded, match %s"
            % (rr["id"], rr["last_good_frame"], rr["frame"], rr["x"], rr["y"],
               rr.get("first_match_frame"), rr.get("scanned") or 0,
               rr["occluded_frames"], "n/a" if sc is None else "%.2f" % sc))
    report["rounds"][-1]["resumes"] = res["resumes"]
    report["rounds"][-1]["misses"] = res["misses"]

report["resumed"] = resumed
report["final_spans"] = sorted(len(live_frames(r["t"])) for r in records)
json.dump(report, open(r"{report}", "w"), indent=2)
log("final spans %s of %d" % (report["final_spans"], n_frames))
log("DONE")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--points", nargs="+", default=["0.45,0.42"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--tail", type=int, default=2)
    ap.add_argument("--min-match", type=float, default=0.60,
                    help="correlation a resume must reach against the seed pattern")
    ap.add_argument("--no-verify", action="store_true",
                    help="plant CoTracker's first visible frame unchecked (the old "
                         "behaviour, kept so the two can be compared on one plate)")
    ap.add_argument("--blender",
                    default=os.environ.get("BTR_BLENDER_EXE", DEFAULT_BLENDER))
    args = ap.parse_args()

    pts = [[float(v) for v in p.split(",")] for p in args.points]
    outdir = os.path.join(ASSIST, "logs", "assist_loop")
    os.makedirs(outdir, exist_ok=True)
    spec_path = os.path.join(outdir, "spec.json")
    report_path = os.path.join(outdir, "report.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump({"plate": os.path.abspath(args.plate), "points": pts,
                   "rounds": args.rounds, "gap": args.gap, "tail": args.tail,
                   "verify_pattern": not args.no_verify,
                   "min_match": args.min_match}, fh)

    driver = os.path.join(outdir, "driver.py")
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(DRIVER.format(assist=ASSIST, spec=spec_path, report=report_path))

    cmd = [args.blender, "--background", "--factory-startup", "-noaudio",
           "--python-exit-code", "3", "--python", driver]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        if line.startswith("[loop]") or "Error" in line or "Traceback" in line:
            print(line)
    if p.returncode != 0:
        print("FAILED (exit %d)" % p.returncode)
        print("\n".join(out.strip().splitlines()[-20:]))
        return 1
    print("\nreport: %s" % report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
