"""Drive the whole assist loop from a hand track's seed, then score every frame against it.

`eval_reacquire.py` scores ONE resume. A real shot does not have one occlusion -- the
reference this was written for has three runs, [[1,14],[25,32],[40,64]], meaning the feature
is covered, returns, is covered again, and returns again. There is no reason a track cannot
have five. A loop that crosses the first occlusion and abandons the track before the second
is not solving the artist's problem.

Three numbers, answering different questions:

  * RECOVERED -- frames the hand track has that we also have, within `max_px`. How much of
    the artist's work the loop actually did.
  * WRONG -- frames we produced that are further out than that, or that sit where the hand
    track has no sample at all. These are worse than missing frames: the artist has to find
    and delete them, and a confident wrong number solves.
  * MISSED -- frames the hand track has that we do not. Honest absence; a gap is legal in
    3DE and solves across.

    blender.exe --background -noaudio --python tests/eval_track_vs_manual.py -- \\
        --manual tests/reacquretracke_manual.txt --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import argparse
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ASSIST, "addon"))

from btr_assist import client, three_de, track_core                      # noqa: E402
from btr_assist.ops_assist import (image_px_to_uv, marker_to_image_px,   # noqa: E402
                                   live_frames, marker_pattern_box, marker_search_px)


def log(m):
    print("[ev] %s" % m, flush=True)


def read_3de(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 0
    n = int(tok[i])
    i += 1
    out = []
    for _ in range(n):
        name = tok[i]
        i += 1
        i += 1
        cnt = int(tok[i])
        i += 1
        pts = []
        for _ in range(cnt):
            pts.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2])))
            i += 3
        out.append((name, pts))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--min-resume-len", type=int, default=0,
                    help="abandon a track whose last resumed segment was shorter than this. "
                         "0 disables. The addon ships 12, and the reference's middle run is "
                         "EIGHT frames long")
    ap.add_argument("--min-match", type=float, default=0.6622)
    ap.add_argument("--pattern", type=float, default=41.2)
    ap.add_argument("--search", type=float, default=113.1)
    ap.add_argument("--max-px", type=float, default=5.0,
                    help="kept for the per-frame verbose listing only")
    ap.add_argument("--wrong-px", type=float, default=25.0,
                    help="beyond this the marker is on something else, not merely imprecise. "
                         "A wrong-feature landing measures in the hundreds; a human clicking "
                         "4K frames disagrees by a few")
    ap.add_argument("--track", default="",
                    help="which track in the manual file (default: the first)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    all_tracks = read_3de(a.manual)
    if a.track:
        hit = [t for t in all_tracks if t[0] == a.track]
        if not hit:
            log("no track named %r -- file has %s"
                % (a.track, ", ".join(t[0] for t in all_tracks)))
            sys.exit(2)
        name, pts = hit[0]
    else:
        name, pts = all_tracks[0]
    clip = bpy.data.movieclips.load(os.path.abspath(a.plate))
    w, h = clip.size
    truth = {f: (x, h - y) for f, x, y in pts}          # 3DE y-UP -> image y-DOWN
    tf = sorted(truth)
    runs = []
    for f in tf:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    log("hand track %s: %d samples, runs %s" % (name, len(pts), runs))
    log("%d occlusion(s) to cross" % max(0, len(runs) - 1))

    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda ar: ar.width * ar.height)
    area.type = "CLIP_EDITOR"
    area.spaces.active.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, clip.frame_duration
    ctx = (win, area, region, clip, scene)

    opts = track_core.Opts(leash=0.0, motion_model="LocScale", scale_clamp=1.6,
                           edge_stop=True)
    track_core.apply_settings(clip, opts)
    tracks = three_de.active_tracks(clip)
    seed_f = tf[0]
    sx, sy = truth[seed_f]
    tr = tracks.new(name="EV", frame=seed_f)
    tr.motion_model = "LocScale"
    tr.use_brute = True
    tr.use_normalization = True
    tr.correlation_min = 0.75
    tr.frames_limit = 0
    tr.pattern_match = "PREV_FRAME"
    tr.markers[0].co = image_px_to_uv(sx, sy, w, h)
    track_core.set_geom(tr.markers[0], a.pattern, a.search, w, h)
    rec = {"t": tr, "id": "EV", "kind": "", "alive": True, "w": w, "h": h,
           "seed_frame": seed_f, "seed_pat": (a.pattern, a.pattern)}
    pattern = {"frame": seed_f, "cx": sx, "cy": sy, "w": a.pattern, "h": a.pattern}

    py = os.path.join(ASSIST, "runtime", "python311", "python.exe")
    if not os.path.isfile(py):
        py = r"D:\Jefrin\batch_tracker_v001_starter\runtime\python311\python.exe"
    client.ensure(ASSIST, py, 0, timeout=180)
    ci = {"path": os.path.abspath(a.plate), "width": w, "height": h,
          "frames": clip.frame_duration}

    def wait(jid):
        while True:
            st = client.poll(ASSIST, jid)
            if st["state"] not in ("queued", "running"):
                return st
            time.sleep(0.4)

    gave_up = False
    cut_by_check = False
    prev_resume = seed_f
    for rnd in range(a.rounds + 1):
        for _ in track_core.track_job(ctx, [rec], clip.frame_duration, {}, opts):
            pass
        if not live_frames(tr):
            break

        # Motion check first -- plate-free, and it catches the occluder that keeps
        # correlation happy while dragging the marker.
        jpath = sorted([int(m.frame), m.co[0] * w, (1.0 - m.co[1]) * h]
                       for m in tr.markers if not m.mute)
        jf, jstep, jmed = track_core.first_jump(jpath)
        if jf:
            for g in [m.frame for m in tr.markers if m.frame >= int(jf)]:
                if len(tr.markers) > 1:
                    tr.markers.delete_frame(g)
            cut_by_check = True
            log("round %d: jump cut at f%d (%.0f px vs its own %.0f)" % (rnd, jf, jstep, jmed))

        path = [[int(m.frame), m.co[0] * w, (1.0 - m.co[1]) * h]
                for m in tr.markers if not m.mute]
        path.sort()
        st = wait(client.start_hold(ASSIST, ci,
                                    [{"id": "EV", "pattern": pattern, "path": path}])["id"])
        lost = None
        if st["state"] == "done":
            t0 = ((st["result"] or {}).get("tracks") or [{}])[0]
            lost = t0.get("lost_at")
        if lost:
            for f in [m.frame for m in tr.markers if m.frame >= int(lost)]:
                if len(tr.markers) > 1:
                    tr.markers.delete_frame(f)
            cut_by_check = True
            log("round %d: cut at f%d" % (rnd, lost))
        fr = live_frames(tr)
        if not fr or fr[-1] >= clip.frame_duration - 2:
            break

        lg = fr[-1]
        m = tr.markers.find_frame(lg, exact=True)
        lx, ly = marker_to_image_px(m, w, h)
        bcx, bcy, bpw, bph = marker_pattern_box(m, w, h)
        if a.min_resume_len and rnd and (lg - prev_resume) < a.min_resume_len:
            log("round %d: GIVING UP -- last segment %d frames, under min_resume_len %d"
                % (rnd, lg - prev_resume, a.min_resume_len))
            gave_up = True
            break
        req = [{"id": "EV", "query_frame": seed_f, "query_x": sx, "query_y": sy,
                "last_good_frame": lg, "last_good_x": lx, "last_good_y": ly,
                "gap": a.gap, "pattern": pattern,
                # Omitted after a hold/jump cut -- see ops_assist: the recent patch is the
                # drift, so the artist's own is what should be searched for.
                **({} if cut_by_check else
                   {"last_box": {"frame": lg, "cx": bcx, "cy": bcy, "w": bpw, "h": bph}}),
                "search_px": marker_search_px(m, w, h)}]
        st = wait(client.start_reacquire(ASSIST, ci, req,
                                         {"frame_hi": clip.frame_duration,
                                          "verify_pattern": True,
                                          "min_match": a.min_match})["id"])
        if st["state"] != "done":
            log("round %d: re-acquire failed: %s" % (rnd, st.get("error")))
            break
        res = st["result"]
        if not res.get("resumes"):
            for miss in res.get("misses") or []:
                log("round %d: no resume -- %s" % (rnd, (miss.get("reason") or "")[:90]))
            break
        rr = res["resumes"][0]
        prev_resume = int(rr["frame"])
        cut_by_check = False
        mk = tr.markers.insert_frame(int(rr["frame"]),
                                     co=image_px_to_uv(float(rr["x"]), float(rr["y"]), w, h))
        mk.mute = False
        mk.pattern_corners = m.pattern_corners
        mk.search_min, mk.search_max = m.search_min, m.search_max
        rec["alive"] = True
        rec["seed_frame"] = int(rr["frame"])
        log("round %d: resumed at f%d (match %s)"
            % (rnd, rr["frame"], rr.get("match_score")))
    else:
        log("ran out of rounds (%d)" % a.rounds)

    ours = {}
    for f in live_frames(tr):
        mm = tr.markers.find_frame(f, exact=True)
        ours[f] = marker_to_image_px(mm, w, h)

    if a.verbose:
        log("")
        log("%-6s %-18s %-18s %s" % ("frame", "ours (img px)", "hand track", "verdict"))
        for f in sorted(set(list(ours) + list(truth))):
            o = ours.get(f)
            t = truth.get(f)
            if o and t:
                e = ((o[0]-t[0])**2 + (o[1]-t[1])**2) ** 0.5
                v = "ok %.1f px" % e if e <= a.max_px else "WRONG %.0f px" % e
            elif o:
                v = "WRONG -- artist has no sample here (occluded)"
            else:
                v = "missed (gap)"
            log("%-6d %-18s %-18s %s"
                % (f, "%.0f,%.0f" % o if o else "-", "%.0f,%.0f" % t if t else "-", v))

    # Two different questions, and conflating them makes the gate meaningless. Landing on
    # the WRONG FEATURE is the failure this reference exists to catch -- it is hundreds of
    # px, or a frame the artist's track does not have at all because the feature was hidden.
    # Disagreeing with a human click by a few px on a 3840-wide plate is PRECISION, and the
    # reference has its own noise at that scale. They are reported apart and only the first
    # fails the gate.
    # Only frames the reference actually COVERS can be judged. A hand track that stops at
    # f250 says nothing about f251 -- counting those as wrong measures where the artist
    # stopped clicking, not where the tracker went. Inside the range, a missing sample IS
    # meaningful: it is a frame the artist deliberately left out because the feature was
    # hidden, and a marker there is on something else.
    lo_t, hi_t = min(truth), max(truth)
    beyond = 0
    errs = []
    pairs = []
    off_feature = []
    for f, (x, y) in sorted(ours.items()):
        if f < lo_t or f > hi_t:
            beyond += 1
            continue
        if f not in truth:
            off_feature.append((f, None))
            continue
        tx, ty = truth[f]
        err = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        if err > a.wrong_px:
            off_feature.append((f, err))
        else:
            errs.append(err)
            pairs.append((x - tx, y - ty))
    missed = sum(1 for f in truth if f not in ours)

    def q(v, pc):
        v = sorted(v)
        return v[min(len(v) - 1, int(round(pc * (len(v) - 1))))] if v else float("nan")

    log("")
    log("hand track frames          : %d" % len(truth))
    log("ON THE FEATURE             : %d  (%d%%)"
        % (len(errs), round(100.0 * len(errs) / max(1, len(truth)))))
    if errs:
        log("  precision vs hand track  : p50 %.1f px  p90 %.1f px  max %.1f px"
            % (q(errs, .5), q(errs, .9), max(errs)))
        log("  within 2 px / 5 px       : %d%% / %d%%"
            % (round(100.0 * sum(1 for e in errs if e <= 2) / len(errs)),
               round(100.0 * sum(1 for e in errs if e <= 5) / len(errs))))
    # Separate the constant part from the wobble. An artist clicking a feature and a
    # correlator locking to it settle on slightly different points ON THE SAME FEATURE, and
    # a fixed offset is harmless to a solve -- it is the same point every frame. What is not
    # harmless is scatter. Reporting one number hides which of the two this is.
    if pairs:
        mdx = sum(d[0] for d in pairs) / len(pairs)
        mdy = sum(d[1] for d in pairs) / len(pairs)
        import math
        var = sum((d[0] - mdx) ** 2 + (d[1] - mdy) ** 2 for d in pairs) / len(pairs)
        log("  constant offset          : %+.1f, %+.1f px (same point on the same feature)"
            % (mdx, mdy))
        log("  scatter around it        : %.1f px rms" % math.sqrt(var))
    log("OFF THE FEATURE (must delete): %d" % len(off_feature))
    for f, e in off_feature[:8]:
        log("    f%-4d %s" % (f, "hand track has no sample -- occluded" if e is None
                              else "%.0f px out" % e))
    log("MISSED (left as a gap)     : %d" % missed)
    if beyond:
        log("beyond the reference       : %d frame(s) past f%d, not judged"
            % (beyond, hi_t))
    if gave_up:
        log("NOTE: the loop gave up early on min_resume_len")
    log("TRACK vs HAND TRACK: %s" % ("FAIL" if off_feature else "PASS"))
    sys.exit(1 if off_feature else 0)


main()
