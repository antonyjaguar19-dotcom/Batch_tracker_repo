"""The artist's loop: you place the seed, Blender tracks it, CoTracker gets it back.

    select your markers
      -> Blender tracks them, dies where it loses correlation
      -> CoTracker, queried at YOUR points, says where each one is once it is visible again
      -> YOUR pattern box is correlated against every candidate at full plate resolution;
         anything that is not the same feature is refused and the track stays dead
      -> the surviving resume lands on the correlation peak, MUTED, and Blender tracks on
      -> repeat for N rounds
      -> you look at each resume against the plate and press Keep or Drop

Blender does every per-frame measurement. CoTracker is only consulted at a death, and only
to answer "where did this go" -- it never contributes a tracked position to the export.

Why the resume arrives muted: measured across three shots, a re-acquired point lands on the
RIGHT feature 26-47 % of the time, and a wrong one tracks perfectly well afterwards.
Surviving proves a seed was trackable, not that it was the thing you asked for. So nothing
un-mutes itself.

The pattern check is what attacks that 26-47 % directly rather than only surviving it. The
patch is captured from the marker's keyframe BEFORE anything is tracked -- the pixels
Blender draws in the Track panel's preview box, which is the artist's actual statement of
what the feature is -- and the sidecar refuses any candidate that scores below `min_match`
against it (`sidecar/patmatch.py`). A refusal leaves the track dead, which is the honest
outcome: no resume beats a resume on the wrong feature. The review pass stays regardless,
because a fixed patch cannot rule on a feature whose appearance genuinely changed, and
because the score has not yet been measured against hand tracks on this footage.

Why the resume frame itself is never exported: it is the guide's estimate of where the
feature went, not a measurement of it. The frame after it is the first one Blender actually
matched. `Keep` drops the estimate and keeps the measurements.
"""

import os

import bpy
from bpy.props import BoolProperty, IntProperty

from . import client, prefs, three_de, track_core
from .ops_seed import clip_info

TICK_SECONDS = 0.05


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def marker_to_image_px(marker, w, h):
    """Blender marker -> plate pixels, y-DOWN.

    Blender's clip space and 3DE are both y-up; image files are y-down, and CoTracker reads
    image files. Getting this backwards mirrors every resume about the horizon, which looks
    like a catastrophically bad matcher rather than like an axis bug.
    """
    x, y_up = three_de.uv_to_px(marker.co[0], marker.co[1], w, h)
    return x, (h - 1.0) - y_up


def image_px_to_uv(x, y_down, w, h):
    y_up = (h - 1.0) - y_down
    return three_de.px_to_uv(x, y_up, w, h)


def marker_pattern_box(marker, w, h):
    """The artist's pattern box in plate pixels, y-DOWN: (cx, cy, pw, ph).

    `pattern_corners` are four OFFSETS from the marker in normalised clip space, in the
    order bottom-left, bottom-right, top-right, top-left. Their bounding box is the region
    Blender correlates and the region it draws in the Track panel's preview -- the feature,
    as the artist set it. The box centre is not the marker: dragging a corner moves one side
    only, so the centre has to be read from the corners rather than assumed to be `co`.
    """
    xs = [c[0] for c in marker.pattern_corners]
    ys = [c[1] for c in marker.pattern_corners]
    cu = marker.co[0] + (min(xs) + max(xs)) / 2.0
    cv = marker.co[1] + (min(ys) + max(ys)) / 2.0
    x, y_up = three_de.uv_to_px(cu, cv, w, h)
    return (x, (h - 1.0) - y_up,
            (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h)


def marker_search_px(marker, w, h):
    """Search box width in plate pixels -- the artist's statement of how far this feature
    is allowed to move between frames."""
    return max((marker.search_max[0] - marker.search_min[0]) * w,
               (marker.search_max[1] - marker.search_min[1]) * h)


def live_frames(track):
    return sorted(m.frame for m in track.markers if not m.mute)


def dead_tracks(tracks, n_frames, tail=2):
    """Tracks whose last real marker is short of the end. `tail` ignores the last frames,
    where 'died' and 'finished' are the same thing."""
    out = []
    for tr in tracks:
        fr = live_frames(tr)
        if not fr:
            continue
        if fr[-1] < n_frames - tail:
            out.append((tr, fr[0], fr[-1]))
    return out


class CLIP_OT_btr_assist_track(bpy.types.Operator):
    bl_idname = "clip.btr_assist_track"
    bl_label = "Track selected + re-acquire"
    bl_description = ("Track the selected markers with Blender; where one dies, ask "
                      "CoTracker where it went and resume it there for you to confirm")

    rounds: IntProperty(
        name="Re-acquire rounds", default=3, min=0, max=10,
        description="A resumed track can die again. Each round tracks, then re-acquires "
                    "whatever is still short of the end")
    gap: IntProperty(
        name="Gap (frames)", default=3, min=1, max=60,
        description="How far past the failure to resume. The frames in between stay empty "
                    "-- a legal gap in 3DE, not a deletion")
    backward: BoolProperty(
        name="Track backwards too", default=True,
        description="A marker placed at frame 40 covers the head of the shot only if it is "
                    "also tracked backwards")
    tail: IntProperty(
        name="End tolerance", default=2, min=0, max=50,
        description="A track ending within this many frames of the end has finished, not died")
    min_resume_len: IntProperty(
        name="Give up under (frames)", default=12, min=0, max=200,
        description="If a resumed segment survives fewer frames than this, stop "
                    "re-acquiring that track. Measured: a point that died to blur or low "
                    "contrast rather than to an occluder gets re-acquired successfully and "
                    "then dies again immediately, so the loop crawls forward a few frames "
                    "per round forever. The feature is bad; that needs a hand, not a retry")
    verify_pattern: BoolProperty(
        name="Must match your pattern", default=True,
        description="Correlate YOUR pattern box -- the patch shown in the Track panel "
                    "preview -- against every candidate resume at full plate resolution, "
                    "and refuse the ones that are not the same feature. Off restores the "
                    "old behaviour: CoTracker's first visible frame is planted unchecked")
    min_match: bpy.props.FloatProperty(
        name="Minimum match", default=0.60, min=0.0, max=1.0, subtype="FACTOR",
        description="Normalised correlation a candidate must reach against your pattern. "
                    "Below it the track is left dead rather than resumed on the wrong "
                    "thing. Raise it if wrong resumes still get through; lower it if a "
                    "feature that legitimately changed appearance is being refused -- the "
                    "report prints the scores it saw either way")

    _timer = None
    _phase = ""
    _root = ""
    _job = ""
    _gen = None
    _stats = None
    _records = None
    _ctx = None
    _proxy = None
    _round = 0
    _opts = None
    _clip = None
    _seeds_px = None          # id -> (frame, x, y) in image px, the artist's own points
    _patterns = None          # id -> the artist's pattern box, captured before tracking
    _search_px = None         # id -> the artist's search box width, in plate px
    _scores = None            # id -> [(frame, score)], what each resume actually matched
    _misses = None            # id -> reason, for the report
    _gave_up = None           # ids the loop has stopped re-asking about
    _pending = None           # resumes awaiting insertion
    _resumed = None           # id -> [resume frames], for the report
    _n_frames = 0

    # ---------------------------------------------------------------- setup

    def invoke(self, context, event):
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        p = prefs.get(context)
        self._root = bpy.path.abspath(p.assist_root) if p else ""
        if not self._root:
            self.report({"ERROR"}, "set the Blender_assitant folder in Preferences")
            return {"CANCELLED"}

        tracks = [t for t in three_de.active_tracks(clip) if t.select]
        if not tracks:
            self.report({"ERROR"}, "select the markers you want tracked first")
            return {"CANCELLED"}

        w, h = clip.size
        self._clip = clip
        self._n_frames = clip.frame_duration
        self._opts = track_core.Opts(
            leash=0.0,          # no AI guide during tracking: Blender tracks on its own
            backwards=self.backward)
        # Explicit, never inherited -- an artist's scene has no --factory-startup behind it.
        track_core.apply_settings(clip, self._opts)

        # The artist's own seed position, captured BEFORE anything is tracked. This is what
        # CoTracker gets queried with, so it follows the feature that was chosen rather than
        # wherever the track later drifted to.
        self._seeds_px = {}
        self._patterns = {}
        self._search_px = {}
        self._records = []
        for tr in tracks:
            fr = live_frames(tr)
            if not fr:
                continue
            m = tr.markers.find_frame(fr[0], exact=True)
            if m is None:
                continue
            x, y = marker_to_image_px(m, w, h)
            self._seeds_px[tr.name] = (fr[0], x, y)
            # The pattern patch is captured HERE, before a single frame is tracked, because
            # it is the identity of the feature the artist chose. Reading it later would
            # read wherever the track had drifted to by then, which is the thing the check
            # exists to catch.
            cx, cy, pw, ph = marker_pattern_box(m, w, h)
            self._patterns[tr.name] = {"frame": fr[0], "cx": cx, "cy": cy,
                                       "w": pw, "h": ph}
            self._search_px[tr.name] = marker_search_px(m, w, h)
            self._records.append({"t": tr, "id": tr.name, "kind": "", "alive": True,
                                  "w": w, "h": h, "seed_frame": fr[0]})
        if not self._records:
            self.report({"ERROR"}, "the selected tracks have no usable markers")
            return {"CANCELLED"}

        space = context.space_data
        self._proxy = track_core.FullResolution(
            clip, space, enabled=bool(p.force_full_res) if p else True)
        self._proxy.__enter__()

        win, area, region = three_de.clip_editor(context, clip)
        self._ctx = (win, area, region, clip, context.scene)
        self._resumed = {}
        self._scores = {}
        self._misses = {}
        self._gave_up = set()
        self._round = 0
        self._start_tracking(context)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    # ---------------------------------------------------------------- phases

    def _start_tracking(self, context):
        self._stats = {}
        self._gen = track_core.track_job(self._ctx, self._records, self._n_frames,
                                         {}, self._opts, stats=self._stats)
        self._phase = "tracking"
        self._status(context, "round %d: tracking %d marker(s)"
                     % (self._round + 1, len(self._records)))

    def _status(self, context, msg):
        context.workspace.status_text_set("Assist: %s" % msg)
        for area in context.window.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.tag_redraw()

    def _finish(self, context, level, msg):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._proxy is not None:
            self._proxy.__exit__(None, None, None)
            self._proxy = None
        context.workspace.status_text_set(None)
        self.report({level}, msg)
        return {"FINISHED"} if level != "ERROR" else {"CANCELLED"}

    def modal(self, context, event):
        if event.type == "ESC":
            if self._phase == "waiting":
                client.cancel(self._root, self._job)
            return self._done(context, "cancelled")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            if self._phase == "tracking":
                return self._tick_tracking(context)
            if self._phase == "waiting":
                return self._tick_waiting(context)
        except client.SidecarError as exc:
            return self._finish(context, "ERROR", str(exc))
        except Exception as exc:                     # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._finish(context, "ERROR",
                                "%s: %s" % (type(exc).__name__, str(exc)[:120]))
        return {"RUNNING_MODAL"}

    def _tick_tracking(self, context):
        import time
        deadline = time.time() + TICK_SECONDS
        done = False
        while time.time() < deadline:
            try:
                next(self._gen)
            except StopIteration:
                done = True
                break
        st = self._stats
        self._status(context, "round %d: frame %d/%d   %d live   %d dead"
                     % (self._round + 1, st.get("frame", 0), st.get("total", 0),
                        st.get("alive", 0), st.get("deaths", 0)))
        if not (done or st.get("done")):
            return {"RUNNING_MODAL"}

        if self.backward and self._round == 0:
            self._status(context, "backward pass (not cancellable) ...")
            track_core.track_backward_pass(self._ctx, self._records, self._opts)

        if self._round >= self.rounds:
            return self._done(context, "finished")
        return self._ask_cotracker(context)

    def _ask_cotracker(self, context):
        clip = self._clip
        w, h = clip.size
        reqs = []
        for tr, f0, f1 in dead_tracks([r["t"] for r in self._records],
                                      self._n_frames, self.tail):
            seed = self._seeds_px.get(tr.name)
            if seed is None or tr.name in self._gave_up:
                continue
            # Did the LAST resume actually buy anything? A point that died to blur rather
            # than to an occluder re-acquires fine and dies again straight away, so without
            # this the loop grinds forward a few frames per round on a feature that is
            # simply not trackable.
            prev = (self._resumed or {}).get(tr.name)
            if prev and self.min_resume_len and (f1 - prev[-1]) < self.min_resume_len:
                self._gave_up.add(tr.name)
                continue
            m = tr.markers.find_frame(f1, exact=True)
            if m is None:
                continue
            lx, ly = marker_to_image_px(m, w, h)
            reqs.append({"id": tr.name,
                         "query_frame": seed[0], "query_x": seed[1], "query_y": seed[2],
                         "last_good_frame": f1, "last_good_x": lx, "last_good_y": ly,
                         "gap": self.gap,
                         # The artist's own pattern, so the sidecar can refuse a resume that
                         # is not this feature. Sent as a box, not as pixels -- the sidecar
                         # reads the plate off disk itself and nothing but JSON crosses.
                         "pattern": self._patterns.get(tr.name),
                         "search_px": self._search_px.get(tr.name, 0.0)})
        if not reqs:
            return self._done(context, "nothing died -- every track reaches the end")

        p = prefs.get(context)
        client.ensure(self._root, bpy.path.abspath(p.python_exe) if p else "",
                      p.port if p else 0)
        r = client.start_reacquire(self._root, clip_info(context, clip), reqs,
                                   {"frame_hi": self._n_frames,
                                    "verify_pattern": bool(self.verify_pattern),
                                    "min_match": float(self.min_match)})
        self._job = r["id"]
        self._phase = "waiting"
        self._status(context, "round %d: asking CoTracker about %d dead track(s) ..."
                     % (self._round + 1, len(reqs)))
        return {"RUNNING_MODAL"}

    def _tick_waiting(self, context):
        st = client.poll(self._root, self._job)
        if st["state"] in ("queued", "running"):
            self._status(context, "%s  (%.0fs)" % (st.get("stage") or "working",
                                                   st.get("seconds", 0)))
            return {"RUNNING_MODAL"}
        if st["state"] == "error":
            return self._finish(context, "ERROR", st["error"]["message"])
        if st["state"] == "cancelled":
            return self._done(context, "cancelled")

        data = st["result"]
        for miss in data.get("misses") or []:
            self._misses[miss["id"]] = miss.get("reason", "")
            print("[assist] %s not resumed: %s" % (miss["id"], miss.get("reason", "")))
            # A refused track has not moved: next round asks the same question about the
            # same death frame with the same pattern and gets the same answer, at the cost
            # of a whole CoTracker pass. Ask once.
            self._gave_up.add(miss["id"])
        n = self._insert_resumes(context, data.get("resumes") or [])
        if not n:
            return self._done(context, "no candidate matched the pattern you set"
                              if (data.get("misses") and self.verify_pattern)
                              else "CoTracker found no way back for any track")
        self._round += 1
        self._start_tracking(context)
        return {"RUNNING_MODAL"}

    def _insert_resumes(self, context, resumes):
        """Plant each resume MUTED, widen its search box, and mark it alive again."""
        clip = self._clip
        w, h = clip.size
        by_name = {r["id"]: r for r in self._records}
        n = 0
        for res in resumes:
            rec = by_name.get(res["id"])
            if rec is None:
                continue
            tr = rec["t"]
            u, v = image_px_to_uv(float(res["x"]), float(res["y"]), w, h)
            m = tr.markers.insert_frame(int(res["frame"]), co=(u, v))
            m.mute = False        # it must be live for Blender to track FROM it
            # The search box still gets room: with the pattern check on, the position is a
            # full-res correlation peak rather than a guess, but Blender's first step from
            # it is across a gap of `gap_frames`, which is further than one frame of motion.
            # The pattern stays the size it was -- that size IS the artist's feature.
            old = tr.markers.find_frame(int(res["last_good_frame"]), exact=True)
            if old is not None:
                sx = (old.search_max[0] - old.search_min[0]) * 1.0
                sy = (old.search_max[1] - old.search_min[1]) * 1.0
                m.pattern_corners = old.pattern_corners
                m.search_min = (-sx, -sy)
                m.search_max = (sx, sy)
            rec["alive"] = True
            rec["seed_frame"] = int(res["frame"])
            self._resumed.setdefault(res["id"], []).append(int(res["frame"]))
            sc = res.get("match_score")
            self._scores.setdefault(res["id"], []).append((int(res["frame"]), sc))
            print("[assist] %s resumed at %d  match %s  tried %s%s"
                  % (res["id"], int(res["frame"]),
                     "n/a" if sc is None else "%.2f" % sc,
                     res.get("tried") or [],
                     ("  (%s)" % res["match_note"]) if res.get("match_note") else ""))
            n += 1
        return n

    def _done(self, context, why):
        """Mute every resumed segment, so nothing enters the export unconfirmed."""
        muted = 0
        for name, frames in (self._resumed or {}).items():
            rec = next((r for r in self._records if r["id"] == name), None)
            if rec is None:
                continue
            tr = rec["t"]
            for f0 in frames:
                for m in tr.markers:
                    if m.frame >= f0:
                        m.mute = True
                        muted += 1
        spans = sorted(len(live_frames(r["t"])) for r in self._records)
        median = spans[len(spans) // 2] if spans else 0
        n_res = sum(len(v) for v in (self._resumed or {}).values())
        # The match score is the part of this report worth reading: it is the only number in
        # the loop that says a resume is the artist's feature rather than merely a feature.
        sc = [s for v in (self._scores or {}).values() for _, s in v if s is not None]
        tail = ""
        if sc:
            tail = ", match %.2f-%.2f (min %.2f)" % (min(sc), max(sc), self.min_match)
        elif self.verify_pattern:
            tail = ", pattern check found nothing to check"
        n_ref = len(self._misses or {})
        if n_ref:
            tail += ", %d refused (see console)" % n_ref
        return self._finish(
            context, "INFO",
            "%s: %d track(s), median span %d/%d, %d resume(s) muted for review%s"
            % (why, len(self._records), median, self._n_frames, n_res, tail))


class CLIP_OT_btr_confirm_resumes(bpy.types.Operator):
    bl_idname = "clip.btr_confirm_resumes"
    bl_label = "Confirm resumes"
    bl_description = ("Keep or drop the muted resumed segments on the selected tracks. "
                      "Look at the plate first -- only 26-47% land on the right feature")

    action: bpy.props.EnumProperty(
        items=[("KEEP", "Keep", "Un-mute the resumed segment"),
               ("DROP", "Drop", "Delete the resumed segment")],
        default="KEEP")

    def execute(self, context):
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        sel = [t for t in three_de.active_tracks(clip) if t.select]
        if not sel:
            self.report({"ERROR"}, "select the tracks to confirm")
            return {"CANCELLED"}
        n = 0
        for tr in sel:
            muted = [m for m in tr.markers if m.mute]
            if not muted:
                continue
            if self.action == "KEEP":
                first = min(m.frame for m in muted)
                for m in muted:
                    # The resume frame itself is the guide's estimate, not a measurement.
                    # The frame after it is the first one Blender matched.
                    if m.frame > first:
                        m.mute = False
                        n += 1
            else:
                for m in muted:
                    tr.markers.delete_frame(m.frame)
                    n += 1
        self.report({"INFO"}, "%s %d marker(s) on %d track(s)"
                    % ("kept" if self.action == "KEEP" else "dropped", n, len(sel)))
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_assist_track, CLIP_OT_btr_confirm_resumes)
