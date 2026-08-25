"""The sidecar: a localhost HTTP server that owns the GPU, so Blender never has to.

Why a separate process at all:
  * torch + CUDA is ~2.5 GB and Blender 5.2's Python is 3.13, not the 3.11 the bot runs on
  * a CUDA OOM kills this process, not the artist's session
  * every licence boundary lands on a process boundary -- the addon is GPL and imports bpy,
    the models are Apache/BSD and never share an interpreter with it

Stdlib only. `http.server` is enough for seven endpoints, and in self-contained mode every
avoided dependency is a smaller bootstrap.

Security: bound to 127.0.0.1, port 0 (the OS picks), and every request must carry the token
written into `logs/sidecar.json`. Without that, any local process -- including a web page --
could drive a service that reads client footage off a studio share.
"""

import json
import os
import secrets
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))

#: Pixels of slack the identity check is allowed around the position localisation found.
#: It is a second opinion on one spot, not a search -- widen it and a nearby lookalike can
#: answer for the artist's feature.
VERIFY_RADIUS = 6.0

#: How far below `min_match` still counts as a NEAR MISS worth showing the artist rather than
#: discarding. Measured two ways: on the SH004 known-answer cases the pattern gate refuses
#: 7 % of resumes that are CORRECT, and on SH013 the gate refused every foreground resume at
#: 0.45-0.54 while CoTracker produced a path each time. A near miss is never accepted on the
#: artist's behalf -- it is proposed, always confirmed, and labelled with the score it got.
UNVERIFIED_MARGIN = 0.15
VERSION = "0.1.0"

TOKEN = ""
JOBS = {}
JOBS_LOCK = threading.Lock()
STARTED = time.time()


class Job:
    """One unit of work. Only one runs at a time -- there is one A4000."""

    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.state = "queued"
        self.message = ""
        self.stage = ""
        self.result = None
        self.error = None
        self.cancel = threading.Event()
        self.started = time.time()
        self.log = []

    def say(self, msg):
        self.stage = str(msg)
        self.log.append("%.1fs  %s" % (time.time() - self.started, msg))
        del self.log[:-200]
        print("[job %s] %s" % (self.id, msg), flush=True)

    def public(self):
        return {"id": self.id, "kind": self.kind, "state": self.state,
                "stage": self.stage, "message": self.message,
                "seconds": round(time.time() - self.started, 1),
                "result": self.result, "error": self.error,
                "log": self.log[-12:]}


def busy_job():
    with JOBS_LOCK:
        for j in JOBS.values():
            if j.state in ("queued", "running"):
                return j
    return None


def run_job(job, fn):
    def body():
        job.state = "running"
        try:
            job.result = fn(job)
            job.state = "cancelled" if job.cancel.is_set() else "done"
        except Exception as exc:                     # noqa: BLE001
            job.state = "error"
            # The traceback goes to the log; the addon gets one sentence. A traceback in a
            # Blender popup is not information an artist can act on.
            job.error = {"code": type(exc).__name__, "message": str(exc)[:400],
                         "log": os.path.join(ASSIST, "logs", "sidecar.log")}
            job.say("ERROR %s: %s" % (type(exc).__name__, exc))
            traceback.print_exc()
    t = threading.Thread(target=body, daemon=True)
    t.start()
    return job


# ------------------------------------------------------------------ job bodies

def job_seed(payload):
    def fn(job):
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import autoseed                                              # noqa: PLC0415
        clip = payload.get("clip") or {}
        params = payload.get("params") or {}
        plate = clip.get("path", "")
        if not plate:
            raise ValueError("no clip path")
        if not os.path.exists(plate):
            # A relative filepath from Blender (`//plate/...`) that the addon failed to
            # resolve lands here. Failing on the argument beats failing on a decode.
            raise FileNotFoundError("plate not found: %s" % plate)
        out_dir = os.path.join(ASSIST, "logs", "seed", job.id)
        data = autoseed.build_seeds(
            plate, out_dir,
            target=int(params.get("target", 150)),
            spacing_px=float(params.get("spacing_px", 15)),
            max_tracks=int(params.get("max_tracks", 1200)),
            reject_movers=bool(params.get("reject_movers", False)),
            on_status=job.say)
        w, h = int(data["width"]), int(data["height"])
        cw, ch = int(clip.get("width", w)), int(clip.get("height", h))
        if (cw, ch) != (w, h):
            # Hard stop. With a different size the normalised coordinates still "work" and
            # every tracker starts on the wrong feature, which looks like bad tracking
            # rather than like a bug.
            raise RuntimeError("Blender reports %dx%d but the plate reads %dx%d -- "
                               "different plate, or a proxy is on" % (cw, ch, w, h))
        return data
    return fn


def job_motion(payload):
    """How far does this plate move between frames, per region?

    Exists because the addon's geometry table has no motion term and cannot get one on its
    own: Blender ships numpy but not cv2, and this is optical flow. Cheap and coarse by
    design -- it sizes a search box, so tens of pixels is the resolution that matters.
    """
    def fn(job):
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from repo import require_repo                                # noqa: PLC0415
        require_repo()
        import blio                                                  # noqa: PLC0415
        import motion                                                # noqa: PLC0415

        clip = payload.get("clip") or {}
        params = payload.get("params") or {}
        path = clip.get("path", "")
        if not os.path.exists(path):
            raise FileNotFoundError("plate not found: %s" % path)
        out_dir = os.path.join(ASSIST, "logs", "motion", job.id)
        os.makedirs(out_dir, exist_ok=True)
        plate = blio.Plate(path, ifl_dir=out_dir)
        cw, ch = int(clip.get("width", plate.w)), int(clip.get("height", plate.h))
        if (cw, ch) != (plate.w, plate.h):
            raise RuntimeError("Blender reports %dx%d but the plate reads %dx%d -- "
                               "different plate, or a proxy is on" % (cw, ch, plate.w, plate.h))
        job.stage = "measuring plate motion"
        mo = motion.measure(plate, int(clip.get("frames", 0) or plate.count),
                            samples=int(params.get("samples", motion.DEFAULT_SAMPLES)),
                            on_status=job.say)
        if mo is None:
            raise RuntimeError("could not read enough frames to measure motion")
        job.say("motion: p95 %.1f px/frame over the plate, worst cell %.1f"
                % (mo["global_p95"], max(max(r) for r in mo["p95"])))

        # Contrast of each seed's own patch, on its own frame. Measured on SH013 over a
        # 30-seed grid: patches with std >= 8 ran a median 113 frames, under 8 a median of
        # ~25, and every seed that died before frame 25 had a median std of 4.4. It is not
        # a gate -- an artist may have good reason to track a soft feature -- but it is the
        # difference between "the tracker is broken" and "there is nothing there to hold",
        # and it costs one patch read per seed to say so BEFORE the take.
        import patmatch                                              # noqa: PLC0415
        contrast = {}
        for sd in (payload.get("seeds") or []):
            try:
                ref = patmatch.reference_patch(plate, int(sd["frame"]), float(sd["cx"]),
                                               float(sd["cy"]), float(sd["w"]),
                                               float(sd["h"]))
            except (KeyError, TypeError, ValueError):
                continue
            if ref is not None:
                contrast[sd["id"]] = float(ref[0].std())
        if contrast:
            weak = sorted((v, k) for k, v in contrast.items() if v < 8.0)
            job.say("contrast: %d seed(s) measured, %d below 8 (soft)"
                    % (len(contrast), len(weak)))
        mo["contrast"] = contrast
        return mo
    return fn


def job_hold(payload):
    """Did each track stay on the artist's feature, and from which frame did it not?

    One correlation per frame per track against the seed patch -- no model, no GPU. The
    plate is decoded once per frame and shared across every track that wants it, the same
    shape as the reappearance sweep.
    """
    def fn(job):
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from repo import require_repo                                # noqa: PLC0415
        require_repo()
        import blio                                                  # noqa: PLC0415
        import patmatch                                              # noqa: PLC0415

        clip = payload.get("clip") or {}
        reqs = payload.get("requests") or []
        params = payload.get("params") or {}
        path = clip.get("path", "")
        if not os.path.exists(path):
            raise FileNotFoundError("plate not found: %s" % path)
        out_dir = os.path.join(ASSIST, "logs", "hold", job.id)
        os.makedirs(out_dir, exist_ok=True)
        plate = blio.Plate(path, ifl_dir=out_dir)
        floor = float(params.get("floor", 0.5))
        drop = float(params.get("drop", 0.5))

        job.stage = "checking %d track(s) held their feature" % len(reqs)
        results = []
        for r in reqs:
            pat = r.get("pattern") or {}
            ref = patmatch.reference_patch(plate, int(pat.get("frame", 1)),
                                           float(pat["cx"]), float(pat["cy"]),
                                           float(pat["w"]), float(pat["h"])) if pat else None
            if ref is None:
                results.append({"id": r["id"], "lost_at": None,
                                "reason": "no pattern to check against"})
                continue
            patch, off = ref
            std = float(patch.std())
            path_pts = [(int(p[0]), float(p[1]), float(p[2])) for p in (r.get("path") or [])]
            if not path_pts:
                results.append({"id": r["id"], "lost_at": None, "reason": "no positions"})
                continue
            scores = patmatch.hold_check(plate, patch, off, path_pts)
            lost = patmatch.first_loss(scores, floor=floor, drop=drop)
            got = [r[1] for r in scores if r[1] is not None]
            results.append({
                "id": r["id"],
                "lost_at": lost,
                "pattern_std": round(std, 2),
                "score_first": round(got[0], 3) if got else None,
                "score_last": round(got[-1], 3) if got else None,
                "score_median": round(sorted(got)[len(got) // 2], 3) if got else None,
                "scores": [[r[0], (None if r[1] is None else round(r[1], 3)),
                            (None if len(r) < 3 or r[2] is None else round(r[2], 3))]
                           for r in scores],
            })
            if lost:
                at = next((r for r in scores if r[0] == lost), None)
                job.say("%s: left your feature at f%d (was %.2f, became %.2f, margin %s)"
                        % (r["id"], lost, got[0] if got else 0.0,
                           (at[1] if at and at[1] is not None else 0.0),
                           ("%.3f" % at[2]) if at and len(at) > 2 and at[2] is not None
                           else "n/a"))
        n = sum(1 for r in results if r.get("lost_at"))
        job.say("%d of %d track(s) drifted off the feature" % (n, len(results)))
        return {"tracks": results}
    return fn


def job_patcheck(payload):
    """Why did these pattern boxes change size -- the feature, or the tracker losing it?

    No model, no GPU: this is two correlations per track against the artist's own patch, so
    it runs in well under a second on CPU and can sit inside the tracking loop rather than
    at the end of it. It shares an endpoint shape with `reacquire` only so the addon's
    polling code is the same one.
    """
    def fn(job):
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from repo import require_repo                                # noqa: PLC0415
        require_repo()
        import blio                                                  # noqa: PLC0415
        import patmatch                                              # noqa: PLC0415

        clip = payload.get("clip") or {}
        reqs = payload.get("requests") or []
        params = payload.get("params") or {}
        if not reqs:
            raise ValueError("no tracks to check")
        path = clip.get("path", "")
        if not os.path.exists(path):
            raise FileNotFoundError("plate not found: %s" % path)

        out_dir = os.path.join(ASSIST, "logs", "patcheck", job.id)
        os.makedirs(out_dir, exist_ok=True)
        plate = blio.Plate(path, ifl_dir=out_dir)
        cw, ch = int(clip.get("width", plate.w)), int(clip.get("height", plate.h))
        if (cw, ch) != (plate.w, plate.h):
            raise RuntimeError("Blender reports %dx%d but the plate reads %dx%d"
                               % (cw, ch, plate.w, plate.h))

        min_match = float(params.get("min_match", 0.60))
        out = []
        for r in reqs:
            pat = r.get("pattern") or {}
            cur = r.get("current") or {}
            if not pat or not cur:
                out.append({"id": r["id"], "ok": False, "verdict": "no-reference",
                            "reason": "no pattern box was sent"})
                continue
            rep = patmatch.drift_report(
                plate, int(pat.get("frame", 1)),
                (float(pat["cx"]), float(pat["cy"]), float(pat["w"]), float(pat["h"])),
                int(r["frame"]), float(cur["cx"]), float(cur["cy"]),
                (float(cur["cx"]), float(cur["cy"]), float(cur["w"]), float(cur["h"])),
                # The box has already moved as far as it is going to; this only has to
                # cover where the correlation peak sits relative to it.
                radius=float(r.get("radius", 32.0)),
                min_match=min_match)
            rep["id"] = r["id"]
            rep["frame"] = int(r["frame"])
            out.append(rep)
            job.say("%s f%d: %s (ref %s, scaled %s, %.2fx, %.2f px off)"
                    % (r["id"], int(r["frame"]), rep.get("verdict"),
                       "n/a" if rep.get("score_ref") is None else "%.2f" % rep["score_ref"],
                       "n/a" if rep.get("score_scaled") is None
                       else "%.2f" % rep["score_scaled"],
                       rep.get("scale", 1.0), rep.get("offset_px", 0.0)))
        return {"checks": out, "min_match": min_match,
                "width": plate.w, "height": plate.h}
    return fn


def job_reacquire(payload):
    """Where should each dead track be resumed? One CoTracker pass covers all of them.

    Queries are the ARTIST'S original seed positions, so CoTracker is following the same
    feature the artist chose rather than something a detector picked.
    """
    def fn(job):
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from repo import require_repo                                # noqa: PLC0415
        require_repo()
        import blio                                                  # noqa: PLC0415
        import cotrack                                               # noqa: PLC0415
        import patmatch                                              # noqa: PLC0415

        clip = payload.get("clip") or {}
        reqs = payload.get("requests") or []
        params = payload.get("params") or {}
        if not reqs:
            raise ValueError("no tracks to re-acquire")
        path = clip.get("path", "")
        if not os.path.exists(path):
            raise FileNotFoundError("plate not found: %s" % path)
        if not cotrack.available():
            raise RuntimeError("CoTracker is not installed -- run "
                               "bootstrap.bat --with-cotracker")

        out_dir = os.path.join(ASSIST, "logs", "reacq", job.id)
        os.makedirs(out_dir, exist_ok=True)
        plate = blio.Plate(path, ifl_dir=out_dir)
        cw, ch = int(clip.get("width", plate.w)), int(clip.get("height", plate.h))
        if (cw, ch) != (plate.w, plate.h):
            raise RuntimeError("Blender reports %dx%d but the plate reads %dx%d"
                               % (cw, ch, plate.w, plate.h))

        # A WINDOW around the deaths, not the whole clip.
        #
        # Feeding CoTracker all 312 frames to answer "where did this one point go" saturated
        # a 16 GB A4000 (16080 / 16376 MiB, 99 % util) and ran for over nine minutes without
        # finishing. Offline CoTracker attends across the whole clip and adds a support
        # grid, so cost climbs steeply with length -- and none of it is needed. The question
        # only concerns the frames just after the failure.
        #
        # The query is placed at each track's LAST GOOD frame using Blender's own position
        # there, not at the artist's original seed. Blender matched that point by
        # correlation on every frame up to it, so it is the same feature, measured better --
        # and it keeps the window short instead of forcing it back to frame 1.
        lead = int(params.get("lead", 2))
        # 150, not 120: the window is now swept frame by frame for the reappearance, so
        # its length is the longest occlusion that can be crossed in one pass. 150 + lead
        # stays inside the 160-frame budget that keeps CoTracker off the VRAM ceiling.
        search_len = int(params.get("search_len", 150))
        budget = int(params.get("max_frames", 160))
        frame_hi = min(int(plate.count), int(params.get("frame_hi") or plate.count))

        # Tracks do not all die together. One window spanning every death is as long as the
        # shot -- deaths at frame 1 and frame 91 give a 211-frame window on a 312-frame clip,
        # which is the thing that saturated the GPU. So group the requests by WHERE they
        # died and run one short pass per group; each pass is ~120 frames whatever the clip
        # length, and the total work scales with the number of distinct failure points
        # rather than with the shot.
        order = sorted(reqs, key=lambda r: int(r["last_good_frame"]))
        groups, cur = [], []
        for r in order:
            trial = cur + [r]
            lo = max(1, int(trial[0]["last_good_frame"]) - lead)
            hi = min(frame_hi, int(trial[-1]["last_good_frame"]) + search_len)
            if cur and (hi - lo + 1) > budget:
                groups.append(cur)
                cur = [r]
            else:
                cur = trial
        if cur:
            groups.append(cur)
        job.say("%d track(s) died at %d distinct point(s) -> %d CoTracker pass(es)"
                % (len(reqs), len(set(int(r["last_good_frame"]) for r in reqs)),
                   len(groups)))

        # The query is placed at each track's LAST GOOD frame using Blender's own position
        # there, not at the artist's original seed. Blender matched that point by
        # correlation on every frame up to it, so it is the same feature measured better --
        # and it keeps the window short instead of forcing it back to frame 1.
        per_req = {}
        for gi, grp in enumerate(groups):
            lo = max(1, int(grp[0]["last_good_frame"]) - lead)
            hi = min(frame_hi, int(grp[-1]["last_good_frame"]) + search_len)
            if hi - lo + 1 < 2:
                continue
            job.say("pass %d/%d: %d track(s), frames %d-%d"
                    % (gi + 1, len(groups), len(grp), lo, hi))
            queries = [(int(r["last_good_frame"]), float(r["last_good_x"]),
                        float(r["last_good_y"])) for r in grp]
            sub = cotrack.track_points(plate, queries, lo, hi,
                                       max_side=int(params.get("max_side", 768)),
                                       # Forward only: the question is where it comes BACK.
                                       # Backward doubles the cost and answers a question
                                       # Blender has already settled.
                                       backward=False,
                                       on_status=job.say)
            for j, r in enumerate(grp):
                per_req[r["id"]] = (sub["tracks"][j], sub["vis"][j])
        # ---- where does it come back? --------------------------------------------
        # CoTracker says where the point WOULD be on every frame; it does not say which
        # feature that is, and its visibility head does not say when the real one is back.
        # So the guide supplies a predicted position per frame and the artist's own pattern
        # patch -- the pixels Blender draws in the Track panel preview, taken from the
        # keyframe -- decides, by correlation at full plate resolution, which frame the
        # feature actually reappears on.
        #
        # The whole window is swept. An earlier version only ever looked at the first six
        # frames CoTracker called visible, which through a real occlusion are the frames
        # where the feature is still covered: every one failed, the track was abandoned, and
        # the artist was left to do it by hand while the feature was plainly back a dozen
        # frames later. Visibility is a report now, not a gate.
        min_match = float(params.get("min_match", 0.60))
        settle = int(params.get("settle", 4))
        verify = bool(params.get("verify_pattern", True))

        resumes, misses = [], []
        jobs, meta, unverified, pattern_std = [], {}, [], {}
        verify_ref = {}          # id -> (seed patch, offset, box scale, localised?)
        for r in reqs:
            got = per_req.get(r["id"])
            if got is None:
                misses.append({"id": r["id"], "retry": False,
                               "reason": "no frames left after the failure to look at"})
                continue
            tm, vm = got
            lg = int(r["last_good_frame"])
            last_px = (float(r["last_good_x"]), float(r["last_good_y"]))
            path = cotrack.resume_path(tm, lg, last_px, gap=int(r.get("gap", 3)),
                                       max_search=int(params.get("max_search", 400)),
                                       frame_hi=frame_hi)
            if not path:
                misses.append({"id": r["id"], "retry": False,
                               "reason": "the guide has no position after frame %d" % lg})
                continue
            meta[r["id"]] = (tm, vm, lg, last_px, path)

            pat = r.get("pattern") or {}
            ref = patmatch.reference_patch(
                plate, int(pat.get("frame", lg)), float(pat["cx"]), float(pat["cy"]),
                float(pat["w"]), float(pat["h"])) if (verify and pat) else None
            if ref is None:
                # Nothing to check against: fall back to the guide's own first visible
                # frame, and say so on the resume rather than passing it off as verified.
                unverified.append((r["id"], "no pattern to check against (box off-plate "
                                            "or featureless)" if (verify and pat) else ""))
                continue
            patch, offset = ref
            # Contrast of the artist's own box, carried through to the resume. Not a gate:
            # measured on SH004 it does not separate a good match from a bad one (a seed on
            # flat sky, std 0.42, and a usable one at std 0.43 both matched), so refusing on
            # it would throw away real tracks. It is a WARNING, because a box this flat
            # cannot be ruled on either way and the artist should know before they confirm.
            # Real artist-placed features on this plate measure std 40-64.
            pattern_std[r["id"]] = float(patch.std())
            # Search radius: the marker's own search box if the addon sent one, since that
            # is the artist's statement of how far this feature may move. The guide has
            # already carried the point most of the way, so this only has to cover the
            # guide's own error.
            radius = float(r.get("search_px") or max(24.0, patch.shape[1]))
            radius = max(8.0, min(float(params.get("max_match_radius", 96.0)), radius / 2.0))

            # Two patches, two jobs. LOCALISING wants the feature as the track last saw it;
            # VERIFYING wants the feature the artist chose, and those stop being the same
            # picture within a few dozen frames. Measured on 158 known-answer cases (SH004,
            # 59 tracks that survived the shot, simulated deaths at f40/80/120): localising
            # with the seed patch lands p50 3.87 px / 30 % within 2 px, with the last-good
            # patch p50 0.46 px / 85 %. Correcting only the seed patch's SIZE gets 3.36 px
            # -- so staleness, not scale, is what was costing the landing.
            #
            # The seed patch keeps the job it was introduced for. Localising with the
            # last-good patch would happily re-acquire a feature the track had already
            # drifted onto; the identity check below is what refuses that, and it has to
            # read the artist's own box or it is checking nothing.
            lb = r.get("last_box") or {}
            loc = None
            if lb:
                loc = patmatch.reference_patch(plate, int(lb.get("frame", lg)),
                                               float(lb["cx"]), float(lb["cy"]),
                                               float(lb["w"]), float(lb["h"]))
            # Scale of the track's box against the artist's. The identity check is run at
            # THIS size: the same 158 cases, scored at a position known to be correct, put
            # the seed patch at seed size below 0.60 on 11 % of CORRECT resumes and at the
            # track's size on 7 %. Same patch, same position -- the size is the difference
            # between refusing a good resume and keeping it.
            try:
                vscale = float(lb["w"]) / float(pat["w"]) if lb else 1.0
            except (KeyError, TypeError, ZeroDivisionError):
                vscale = 1.0
            verify_ref[r["id"]] = (patch, offset, vscale, loc is not None)
            jpatch, joffset = (loc if loc is not None else (patch, offset))
            jobs.append({"id": r["id"], "patch": jpatch, "offset": joffset,
                         "radius": radius, "path": path})

        found = patmatch.find_reappearance(plate, jobs, min_match=min_match,
                                           settle=settle, on_status=job.say) if jobs else {}

        for r in reqs:
            if r["id"] not in meta:
                continue
            tm, vm, lg, last_px, path = meta[r["id"]]
            hit = found.get(r["id"])

            if hit is None:
                # Unverified path: the guide's first visible frame, unchecked.
                cands = cotrack.resume_candidates(tm, vm, lg, last_px,
                                                  gap=int(r.get("gap", 3)),
                                                  max_search=int(params.get("max_search", 400)),
                                                  limit=1)
                if not cands:
                    misses.append({"id": r["id"], "retry": False,
                                   "reason": "CoTracker never calls it visible again "
                                             "after frame %d" % lg})
                    continue
                f, (x, y) = cands[0]
                score, first_frame, scanned = None, None, 0
                note = dict(unverified).get(r["id"], "")
            else:
                scanned = int(hit["scanned"])
                if hit["frame"] is None:
                    # Nothing over the line anywhere in the window. Whether that is worth
                    # another look depends on WHY: a window that simply ran out is a
                    # different answer from a feature that is not in the box at all, and
                    # only the first is worth spending another CoTracker pass on.
                    best = hit["best_seen"]
                    ran_out = path[-1][0] >= frame_hi

                    # NEAR MISS. The pattern gate said no, but it is the only thing that
                    # said no: CoTracker still calls the feature visible here, and the patch
                    # came close. Killing the track on that is how a hard plate ends up with
                    # nothing at all -- measured on SH013, every foreground resume was
                    # refused at 0.45-0.54 and every track stayed dead. So offer it, mark it
                    # UNVERIFIED, and let the artist look. It is never auto-accepted.
                    near = (best is not None
                            and best >= (min_match - UNVERIFIED_MARGIN)
                            and hit.get("best_frame")
                            and hit.get("best_x") is not None)
                    seen = bool(near and vm.get(int(hit["best_frame"]), False))
                    if near and not seen:
                        # Worth saying out loud. "Nothing reached 0.60" reads as a tracker
                        # that gave up; "0.52, and CoTracker says the feature is not there"
                        # is a different statement, and the artist should get the second one.
                        job.say("%s: best %.2f at f%s, but CoTracker calls the feature NOT "
                                "visible there -- not offered"
                                % (r["id"], best, hit.get("best_frame")))
                    if seen:
                        bf = int(hit["best_frame"])
                        occl = sum(1 for ff in range(lg, bf) if not vm.get(ff, True))
                        resumes.append({
                            "id": r["id"], "frame": bf,
                            "x": float(hit["best_x"]), "y": float(hit["best_y"]),
                            "last_good_frame": lg, "gap_frames": int(bf - lg),
                            "occluded_frames": int(occl),
                            "match_score": round(float(best), 3),
                            "locate_score": round(float(best), 3),
                            "first_match_frame": None,
                            "scanned": int(hit["scanned"]),
                            "verified": False,
                            "candidates": hit.get("candidates") or [],
                            "match_note": ("your pattern only reaches %.2f here, under the "
                                           "%.2f you set -- CoTracker says the feature is "
                                           "visible, so this is its best guess for you to "
                                           "judge, not a verified match"
                                           % (best, min_match))})
                        continue
                    misses.append({
                        "id": r["id"],
                        "score": None if best is None else round(float(best), 3),
                        "best_frame": hit["best_frame"],
                        "searched": [int(path[0][0]), int(path[-1][0])],
                        # Where the guide thinks the point is at the end of the swept
                        # window. Handing this back lets the next round re-query CoTracker
                        # from there and sweep the NEXT window, so an occlusion longer than
                        # one window is crossed in stages instead of ending the track. The
                        # prediction is unverified by construction -- it is a starting point
                        # for another search, never a resume, and the pattern check still
                        # decides whatever that search finds.
                        "retry": not ran_out,
                        "tail_frame": int(path[-1][0]),
                        "tail_x": float(path[-1][1][0]),
                        "tail_y": float(path[-1][1][1]),
                        "reason": "swept frames %d-%d; nothing reached %.2f against the "
                                  "feature as the track last saw it (best %s at frame "
                                  "%s). %s"
                                  % (path[0][0], path[-1][0], min_match,
                                     "n/a" if best is None else "%.2f" % best,
                                     hit["best_frame"],
                                     "The shot ends there." if ran_out else
                                     "It may come back later than this window reaches.")})
                    continue
                f, x, y = int(hit["frame"]), float(hit["x"]), float(hit["y"])
                score, first_frame = float(hit["score"]), hit["first_frame"]
                locate_score = score
                # Identity. Localisation has answered "the feature is HERE"; this answers
                # "and it is the one you picked". Only a few pixels of slack -- the position
                # is already a full-res correlation peak, so this is a second opinion on the
                # same spot, not another search.
                vp, voff, vscale, localised = verify_ref.get(r["id"], (None, None, 1.0, False))
                if localised and vp is not None:
                    vpatch = (patmatch._resized(vp, vscale)
                              if abs(vscale - 1.0) > 0.02 else vp)
                    got = None
                    if vpatch is not None:
                        got = patmatch.match(plate, f, vpatch, x, y,
                                             radius=VERIFY_RADIUS, offset=voff)
                    score = None if got is None else float(got[2])
                    if got is not None and score is not None and score >= min_match:
                        # Take the POSITION too, not only the score. Localisation used the
                        # feature as the track last saw it, and that patch was cut mid-track
                        # -- its peak sits at a different sub-position within the feature
                        # than the artist's own seed. Blender then carries that offset for
                        # the whole resumed run.
                        #
                        # Measured on the artist's v002 output against their hand track:
                        #     run f1-15   offset 2.34 px   (their click vs the correlator)
                        #     run f29-33  offset 7.22 px
                        #     run f41-64  offset 5.06 px
                        # each run internally tight (0.2-0.4 px scatter) but each resume
                        # landing at its own bias. Refining against the seed patch pulls
                        # f29 from 7.14 px to 3.69 and f41 from 4.75 to 4.08, back toward
                        # the run-1 baseline.
                        #
                        # Only within VERIFY_RADIUS, and only when the seed patch is
                        # confident: this adjusts a sub-position, it does not go looking.
                        x, y = float(got[0]), float(got[1])
                    if score is None:
                        # Cannot be judged -- the box does not fit at this size here. Say so
                        # rather than passing an unchecked resume off as verified.
                        score = locate_score
                        unverified.append((r["id"], "your pattern box does not fit at the "
                                                    "resumed position; localisation only"))
                    elif score < min_match:
                        misses.append({
                            "id": r["id"], "score": round(score, 3), "best_frame": f,
                            "searched": [int(path[0][0]), int(path[-1][0])],
                            # Nothing to retry: the feature WAS found, and it is not the
                            # one the artist picked. Another sweep finds the same thing.
                            "retry": False,
                            "reason": "found something at frame %d (localised %.2f) but "
                                      "your own pattern only scores %.2f there, under "
                                      "%.2f -- the track had most likely drifted off your "
                                      "feature before it died"
                                      % (f, locate_score, score, min_match)})
                        continue
                std = pattern_std.get(r["id"])
                note = ("your pattern box has almost no contrast (std %.1f) -- the match "
                        "score cannot mean much here" % std) if (std is not None
                                                                 and std < 5.0) else ""

            occluded = sum(1 for ff in range(lg, f) if not vm.get(ff, True))
            resumes.append({"id": r["id"], "frame": int(f), "x": x, "y": y,
                            "last_good_frame": lg,
                            "gap_frames": int(f - lg),
                            "occluded_frames": int(occluded),
                            "match_score": None if score is None else round(score, 3),
                            # What localisation scored, with whatever patch it used. Kept
                            # separate because `match_score` is the artist's own pattern and
                            # is the one worth reading before confirming.
                            "locate_score": (None if hit is None or hit.get("score") is None
                                             else round(float(hit["score"]), 3)),
                            "first_match_frame": first_frame,
                            "scanned": scanned,
                            "verified": True,
                            # Alternatives, so a wrong landing costs a keypress instead of
                            # the track. The best is what `frame`/`x`/`y` already hold.
                            "candidates": hit.get("candidates") or [],
                            "match_note": note,
                            "pattern_std": None if pattern_std.get(r["id"]) is None
                                           else round(pattern_std[r["id"]], 2),
                            "guide_dx": tm[f][0] - tm[lg][0],
                            "guide_dy": tm[f][1] - tm[lg][1]})

        scored = [x["match_score"] for x in resumes if x["match_score"] is not None]
        job.say("%d resume(s), %d without one%s"
                % (len(resumes), len(misses),
                   ("; pattern match %.2f..%.2f" % (min(scored), max(scored)))
                   if scored else ""))
        cotrack.free()
        return {"resumes": resumes, "misses": misses, "min_match": min_match,
                "width": plate.w, "height": plate.h, "passes": len(groups)}
    return fn


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):                  # quiet; jobs do their own logging
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if self.headers.get("X-BTR-Token", "") == TOKEN:
            return True
        self._send(403, {"error": {"code": "BAD_TOKEN",
                                   "message": "wrong or missing token"}})
        return False

    def _payload(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, health())
        if not self._auth():
            return None
        if self.path.startswith("/jobs/"):
            jid = self.path.split("/")[2]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._send(404, {"error": {"code": "NO_JOB",
                                                  "message": "unknown job %s" % jid}})
            return self._send(200, job.public())
        return self._send(404, {"error": {"code": "NO_ROUTE", "message": self.path}})

    def do_POST(self):
        if not self._auth():
            return None
        try:
            payload = self._payload()
        except ValueError as exc:
            return self._send(400, {"error": {"code": "BAD_JSON", "message": str(exc)}})

        if self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.2),
                                             os._exit(0)), daemon=True).start()
            return None

        if self.path in ("/jobs/seed", "/jobs/reacquire", "/jobs/patcheck", "/jobs/motion",
                         "/jobs/hold"):
            b = busy_job()
            if b is not None:
                return self._send(409, {"error": {
                    "code": "BUSY",
                    "message": "a %s job is already running (%s)" % (b.kind, b.stage)}})
            kind = self.path.rsplit("/", 1)[1]
            job = Job(kind)
            with JOBS_LOCK:
                JOBS[job.id] = job
            run_job(job, {"seed": job_seed, "reacquire": job_reacquire,
                          "patcheck": job_patcheck, "motion": job_motion,
                          "hold": job_hold}[kind](payload))
            return self._send(200, job.public())

        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            jid = self.path.split("/")[2]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if job is None:
                return self._send(404, {"error": {"code": "NO_JOB", "message": jid}})
            job.cancel.set()
            job.say("cancel requested")
            return self._send(200, {"ok": True})

        return self._send(404, {"error": {"code": "NO_ROUTE", "message": self.path}})


def health():
    """Everything the addon needs to decide whether to offer a button, in one call."""
    out = {"ok": True, "version": VERSION, "uptime_s": round(time.time() - STARTED, 1),
           "pid": os.getpid()}
    try:
        import sys
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        import repo                                                  # noqa: PLC0415
        p = repo.paths()
        out["repo_root"] = p["repo_root"]
        out["tapnext_ckpt"] = ("present" if os.path.isfile(p["tapnext_ckpt"] or "")
                               else "MISSING")
        try:
            repo.require_repo()
            out["repo"] = "ok"
        except RuntimeError as exc:
            out["repo"] = str(exc)[:300]
            out["ok"] = False
    except Exception as exc:                         # noqa: BLE001
        out["repo"] = "%s: %s" % (type(exc).__name__, exc)
        out["ok"] = False
    try:
        import torch                                                 # noqa: PLC0415
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            free, total = torch.cuda.mem_get_info()
            out["device"] = torch.cuda.get_device_name(0)
            out["vram_free_mb"] = int(free / 1e6)
            out["vram_total_mb"] = int(total / 1e6)
    except Exception as exc:                         # noqa: BLE001
        out["torch"] = "%s: %s" % (type(exc).__name__, exc)
        out["cuda"] = False
        out["ok"] = False
    b = busy_job()
    out["busy"] = b.public() if b else None
    return out


def serve(port=0, portfile=None, token=None, parent_pid=0):
    global TOKEN
    TOKEN = token or secrets.token_hex(16)
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    actual = srv.server_address[1]
    info = {"port": actual, "token": TOKEN, "pid": os.getpid()}
    pf = portfile or os.path.join(ASSIST, "logs", "sidecar.json")
    os.makedirs(os.path.dirname(pf), exist_ok=True)
    with open(pf, "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    print("[sidecar] listening on 127.0.0.1:%d  (portfile %s)" % (actual, pf), flush=True)

    if parent_pid:
        # If Blender goes away, so does this. Otherwise a crashed session leaves a process
        # holding 2.5 GB of VRAM that nothing will ever ask to stop.
        def watch():
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x00100000, False, int(parent_pid))
            if h:
                k.WaitForSingleObject(h, 0xFFFFFFFF)
                print("[sidecar] parent %s exited -- shutting down" % parent_pid,
                      flush=True)
                os._exit(0)
        threading.Thread(target=watch, daemon=True).start()

    srv.serve_forever()
