"""The artist's loop: you place the seed, Blender tracks it, CoTracker gets it back.

    select your markers
      -> Blender tracks them, dies where it loses correlation
      -> CoTracker predicts where the point goes on EVERY frame after that
      -> YOUR pattern box is correlated against every one of those frames at full plate
         resolution, and the FIRST frame it matches on is where the feature came back
      -> the clip jumps there with the marker snapped onto it and the run STOPS: you look,
         then Enter to track on, D to drop, A to accept the rest, Esc to stop
      -> Blender tracks on from there; repeat for N rounds
      -> a search that runs out of window continues from where it stopped next round, so an
         occlusion longer than one window is crossed in stages instead of ending the track

Running underneath all of that, once the seed is placed: LOCATION AND SCALE ARE BOTH
ANIMATED, and the box size is watched frame by frame. A tracker that has started sliding
off its feature does not announce it -- the position stays plausible and the track stays
alive -- but the pattern box swells, because correlation is being satisfied by the
surroundings rather than by the feature. So a box that grows or shrinks unusually fast, or
that wanders far from the size the artist set, stops the track on the spot; the patch from
the seed frame is correlated against the plate there at its own size AND resized by exactly
how much the box grew; and the two scores say which of four things happened
(`sidecar/patmatch.classify_drift`):

    the feature really did approach camera  -> keep the box, re-place on the scaled peak
    the box swelled, the feature did not    -> put the artist's box back, drop the frames
                                               measured while it was growing, re-place on
                                               the artist's own patch
    grain, a lighting step, a false alarm   -> carry on, nothing touched
    the feature is not there at any size    -> cut back to the last frame measured by the
                                               artist's box and hand it to re-acquire

`tests/test_scale_watch.py` fixes when it stops; `tests/test_scale_drift.py` fixes what the
stop means. Neither needs Blender.

Blender does every per-frame measurement. CoTracker is only consulted at a death, and only
to answer "where did this go" -- it never contributes a tracked position to the export.

Why nothing resumes silently: measured across three shots, a re-acquired point lands on the
RIGHT feature 26-47 % of the time, and a wrong one tracks perfectly well afterwards.
Surviving proves a seed was trackable, not that it was the thing you asked for. So every
resume is a proposal shown to the artist on its own frame before tracking continues. With
`confirm_resumes` off it is not shown, and then nothing un-mutes itself: the batch Keep/Drop
pass in the panel is the review instead.

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

from . import client, overlay, prefs, three_de, track_core
from .ops_seed import clip_info

TICK_SECONDS = 0.05

#: Design displacement as a multiple of the cell's measured p95 motion. A box has to survive
#: the fastest moment the feature lives through, not the typical one; p95 is already a
#: high-water mark and this is headroom for the frames above it. Measured on SH013, the worst
#: single-frame motion ran about 1.4x the cell p95.
MOTION_HEADROOM = 1.5

#: Never grow a search box past this fraction of the plate width. A huge box is slow AND
#: starts matching things that merely look similar -- the failure the tight built-in table
#: was chosen to avoid. Widening is a floor being lifted, not a licence.
MAX_SEARCH_FRAC = 0.25

#: Plate motion, px/frame, above which a seed is worth warning about. Measured on SH013: at
#: 43-48 px/frame the foreground gives 7-23 frame tracks, at 17-32 it gives 32-154, and at
#: 0-8 it gives 128-206. The number is where that falls off, not a tracking limit -- nothing
#: in the tracker moved it.
FAST_SEED_PX = 35.0

#: Pattern-box standard deviation below which a seed is worth warning about. Measured on
#: SH013 over a 30-seed grid: std >= 8 ran a median 113 frames, 4-8 a median 23, under 4 a
#: median 26, and every seed dying before frame 25 had a median std of 4.4. Artist-placed
#: features on SH004 measure 40-64, so this is a low bar deliberately -- it catches "there
#: is nothing here", not "this is not a great feature".
SOFT_SEED_STD = 8.0

#: The only events the confirm phase consumes. EVERYTHING else passes through to the clip
#: editor, because the question it asks -- "is that your feature?" -- is answered by zooming
#: in, panning, and scrubbing, and swallowing every event to protect four keys made Blender
#: read as HUNG: no zoom, no pan, no click, no playhead, with the only hint in the status
#: bar at the bottom edge of the window. Reported from a real session; the artist killed
#: Blender rather than press a key they could not see.
#:
#: SPACE used to accept, and is deliberately gone: it is playback. Now that navigation
#: passes through, an artist pressing it to watch the motion would have silently accepted
#: the proposal instead of looking at it.
#:
#: N cycles to the next candidate. One answer is not enough: measured on SH006 the top match
#: was frame 17, the artist confirmed by eye that it was the wrong feature entirely, and the
#: real reappearance was frame 25 -- which the sweep had already scored and nothing kept.
ANSWER_KEYS = frozenset(("RET", "NUMPAD_ENTER", "D", "A", "N"))


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
    confirm_resumes: BoolProperty(
        name="Confirm each re-acquire", default=True,
        description="When a feature is found again, jump the clip to that frame with the "
                    "marker snapped onto it and WAIT: you look at it, then press Enter to "
                    "track on, D to drop it, or Esc to stop. Off runs every round straight "
                    "through and leaves the whole batch muted for review at the end")
    confirm_only_occluded: BoolProperty(
        name="Only ask when it was hidden", default=True,
        description="Confirm only the resumes where CoTracker says the feature was actually "
                    "OCCLUDED, and accept the rest without stopping. Measured on SH004 "
                    "against an independent Lucas-Kanade reference: over 10 autonomous "
                    "resumes with no occlusion, the worst single frame was 6.66 px and not "
                    "one landed on a different feature -- so stopping to ask about those "
                    "buys nothing. A resume that crosses a real occlusion has NOT been "
                    "measured that way and still stops. Turn this off to be asked about "
                    "every resume")
    hold_feature: BoolProperty(
        name="Cut where it stops being your feature", default=True,
        description="After tracking, check every frame against the pattern YOU seeded and "
                    "cut the track where it stopped being that feature. Blender matches each "
                    "frame against the one before it, so an occluder slides in over a few "
                    "frames without any single step looking wrong -- the track never dies, "
                    "never asks for a re-acquire, and the drift is written to the file as if "
                    "it were data. Measured on an occluded seed: your pattern scored 0.91 at "
                    "frame 14 and 0.22 at frame 15, and the track carried on to frame 22")
    stop_at_frame_edge: BoolProperty(
        name="Stop when the pattern leaves frame", default=True,
        description="End a track as soon as its pattern box reaches the edge of the plate. "
                    "The pattern is what correlates, so once part of it is off the frame "
                    "there is nothing there to match -- and Blender does not stop, it "
                    "solves a SMALLER box and keeps returning positions while the track "
                    "walks off the feature. The search box is not tested: it routinely "
                    "hangs off the plate and Blender copes")
    fit_search_box: BoolProperty(
        name="Fit search box to the plate", default=True,
        description="Measure how far the plate actually moves between frames and widen any "
                    "search box too small to reach that far. The built-in sizes were tuned "
                    "on a slow shot and carry no motion term -- measured on a 59.94 fps "
                    "chase plate the near-ground moves 21-53 px per frame while the default "
                    "corner box reaches 13, so every foreground marker died on its FIRST "
                    "step. Boxes are only ever made bigger, never smaller")
    animate_scale: BoolProperty(
        name="Animate location + scale", default=True,
        description="Track with the LocScale motion model, so Blender solves a SIZE for the "
                    "pattern box on every frame as well as a position. That size is the "
                    "signal the drift watch reads -- under plain Loc the box never changes "
                    "and there is nothing to watch")
    watch_scale: BoolProperty(
        name="Watch the pattern box", default=True,
        description="Stop a track the moment its pattern box grows or shrinks unusually "
                    "fast, or drifts far from the size you set, and check the patch you "
                    "seeded against the plate there before letting it write another frame")
    scale_rate: bpy.props.FloatProperty(
        name="Max change per frame", default=0.10, min=0.0, max=2.0, subtype="FACTOR",
        description="Fractional size change from one frame to the next that counts as "
                    "unusual. A feature approaching camera grows smoothly; a box that "
                    "jumps has jumped onto something else")
    scale_ratio: bpy.props.FloatProperty(
        name="Max size vs your box", default=1.6, min=1.05, max=6.0,
        description="Cumulative size against the box you set, either way round. A slow "
                    "creep never trips the per-frame limit -- 4% a frame for fifteen frames "
                    "is a doubling, and every step looked reasonable")
    drift_fixes: IntProperty(
        name="Repairs per track", default=2, min=0, max=10,
        description="How many times one track may be stopped and repaired before the watch "
                    "leaves it alone. A feature that keeps swelling after two corrections "
                    "is not a tracking accident")
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
    _continue_from = None     # id -> (frame, x, y): resume the SEARCH here, not a position
    _awaiting = None          # resumes snapped and waiting for the artist to say yes
    _auto_kept = 0            # resumes taken without asking, because nothing was occluded
    _motion = None            # per-cell plate motion, px/frame, measured by the sidecar
    _seed_motion = None       # id -> px/frame where that seed sits
    _widened = None           # [(id, from_px, to_px)] for the report
    _hold_done = False        # the identity check has run for this pass
    _cut = None               # [(id, frame, n)] tracks cut off a lost feature
    _pending = None           # resumes awaiting insertion
    _fixes = None             # id -> how many times the scale watch has repaired it
    _drift = None             # [(id, frame, verdict, ...)] for the report
    _backward_done = False
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
            backwards=self.backward,
            # Scale is animated so that the box SIZE becomes a per-frame measurement, which
            # is what the watch reads. The two are one decision: watching under `Loc`
            # watches a constant, and animating scale without watching it is how a box
            # quietly swells onto the background for forty frames.
            motion_model="LocScale" if self.animate_scale else "",
            edge_stop=bool(self.stop_at_frame_edge),
            # The watch's own "too far from your box" number, reused as a hard bound so the
            # box cannot reach a degenerate size in the first place. One number, two uses.
            scale_clamp=(float(self.scale_ratio) if self.animate_scale else 0.0),
            watch_scale=bool(self.watch_scale and self.animate_scale),
            scale_rate=float(self.scale_rate),
            scale_ratio=float(self.scale_ratio))
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
            # Every tracking setting, per track -- not only as a clip default. This is the
            # same argument `apply_settings` makes and it was only half applied: the clip's
            # `default_*` settings are read when a track is CREATED, so a marker the artist
            # placed earlier keeps whatever it was made with and the addon's measured
            # configuration never reaches it.
            #
            # Found in a diagnostic report from a real session: a track carrying
            # `pattern_match = KEYFRAME` while the clip default said PREV_FRAME. This
            # project's own measurement (see `track_core.Opts`) is that KEYFRAME dies
            # **2.6-2.9x more often** on real plates, because it matches the seed patch
            # forever while appearance drifts. That track died at f158 with the feature 82 px
            # inside the frame, and nothing in the addon had touched the setting responsible.
            changed = []
            if self.animate_scale and tr.motion_model != "LocScale":
                # A `Loc` track would silently opt out of the whole scale watch.
                changed.append("motion_model %s->LocScale" % tr.motion_model)
                tr.motion_model = "LocScale"
            if tr.pattern_match != self._opts.pattern_match:
                changed.append("pattern_match %s->%s"
                               % (tr.pattern_match, self._opts.pattern_match))
                tr.pattern_match = self._opts.pattern_match
            if abs(tr.correlation_min - self._opts.correlation) > 1e-4:
                changed.append("correlation %.2f->%.2f"
                               % (tr.correlation_min, self._opts.correlation))
                tr.correlation_min = self._opts.correlation
            if not tr.use_brute:
                changed.append("brute on")
                tr.use_brute = True
            if not tr.use_normalization:
                changed.append("normalization on")
                tr.use_normalization = True
            if tr.frames_limit:
                changed.append("frames_limit %d->0" % tr.frames_limit)
                tr.frames_limit = 0
            if changed:
                # Say what was taken over. Changing an artist's settings silently is its own
                # kind of wrong, even when the change is right.
                print("[assist] %s: %s" % (tr.name, ", ".join(changed)))
            self._records.append({"t": tr, "id": tr.name, "kind": "", "alive": True,
                                  "w": w, "h": h, "seed_frame": fr[0],
                                  "seed_pat": (float(pw), float(ph))})
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
        self._continue_from = {}
        self._awaiting = []
        self._auto_kept = 0
        self._motion = None
        self._seed_motion = {}
        self._widened = []
        self._hold_done = False
        self._cut = []
        self._fixes = {}
        self._drift = []
        self._backward_done = False
        self._round = 0
        if not self._start_motion(context):
            self._start_tracking(context)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    # ---------------------------------------------------------------- phases

    def _start_motion(self, context):
        """Ask the sidecar how far this plate moves. False means just start tracking.

        A modal PHASE, not a call inside `invoke`: this is several optical-flow passes and
        takes seconds, and blocking the main thread for seconds is the exact behaviour that
        made this operator read as hung once already.
        """
        if not self.fit_search_box:
            return False
        p = prefs.get(context)
        try:
            client.ensure(self._root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            # The artist's own boxes ride along, so one pre-flight answers both "can the
            # box reach?" and "is there anything in it to hold?".
            seeds = [{"id": k, "frame": v["frame"], "cx": v["cx"], "cy": v["cy"],
                      "w": v["w"], "h": v["h"]}
                     for k, v in (self._patterns or {}).items()]
            r = client.start_motion(self._root, clip_info(context, self._clip),
                                    seeds=seeds)
        except client.SidecarError as exc:
            # Not fatal. A plate whose motion cannot be measured still tracks with the
            # built-in sizes, which is exactly what happened before this existed.
            print("[assist] could not measure plate motion (%s); using the built-in "
                  "search boxes" % exc)
            return False
        self._job = r["id"]
        self._phase = "motion"
        self._status(context, "measuring how far the plate moves ...")
        return True

    def _tick_motion(self, context):
        st = client.poll(self._root, self._job)
        if st["state"] in ("queued", "running"):
            self._status(context, "%s  (%.0fs)" % (st.get("stage") or "measuring",
                                                   st.get("seconds", 0)))
            return {"RUNNING_MODAL"}
        if st["state"] == "cancelled":
            return self._done(context, "cancelled")
        if st["state"] == "error":
            print("[assist] motion measurement failed: %s" % st["error"]["message"])
        else:
            self._motion = st.get("result")
            # The loop re-fits per frame from this; `_widen_boxes` only sets the box the
            # first step will use.
            self._opts.motion = self._motion
            self._opts.motion_headroom = MOTION_HEADROOM
            self._opts.motion_cap_frac = MAX_SEARCH_FRAC
            self._widen_boxes(context)
            self._report_seeds(context)
            self._warn_fast_seeds(context)
            self._warn_soft_seeds(context)
        self._start_tracking(context)
        return {"RUNNING_MODAL"}

    def _widen_boxes(self, context):
        """Enlarge any search box too small to reach the motion measured where it sits.

        Only ever ENLARGES. A box the artist deliberately made small is a statement about how
        far that feature is allowed to move. A box too small to reach the feature at all is
        not a statement, it is a track that dies on its first step.
        """
        mo = self._motion
        if not mo:
            return
        clip = self._clip
        w, h = clip.size
        gx, gy = mo["grid"]
        cap = w * MAX_SEARCH_FRAC
        for rec in self._records:
            tr = rec["t"]
            m = tr.markers.find_frame(int(rec["seed_frame"]), exact=True)
            if m is None:
                continue
            x, y = marker_to_image_px(m, w, h)
            i = min(gx - 1, max(0, int(x / max(1.0, float(w)) * gx)))
            j = min(gy - 1, max(0, int(y / max(1.0, float(h)) * gy)))
            p95 = float(mo["p95"][j][i])
            self._seed_motion[rec["id"]] = p95          # every seed, widened or not
            pat_w = (max(c[0] for c in m.pattern_corners)
                     - min(c[0] for c in m.pattern_corners)) * w
            pat_h = (max(c[1] for c in m.pattern_corners)
                     - min(c[1] for c in m.pattern_corners)) * h
            have = (m.search_max[0] - m.search_min[0]) * w
            # Blender finds the feature while it stays inside (search_half - pattern_half),
            # so the reach that matters is the DIFFERENCE, not the box width.
            want = 2.0 * (p95 * MOTION_HEADROOM + max(pat_w, pat_h) / 2.0)
            want = min(want, cap)
            if want <= have + 1.0:
                continue
            sx, sy = want / 2.0 / w, want / 2.0 / h
            m.search_min = (-sx, -sy)
            m.search_max = (sx, sy)
            self._widened.append((rec["id"], have, want))
            print("[assist] %s: plate moves %.0f px/frame here, search box %.0f -> %.0f px"
                  % (rec["id"], p95, have, want))

    def _report_seeds(self, context):
        """Print what was measured at every seed, not only the ones that trip a threshold.

        Both thresholds missed the case that prompted this. A seed on SH013 measured 33
        px/frame (under the 35 warning) with a patch std of 9.8 (over the 8 warning) and
        died at frame 11 -- neither number was extreme, and their combination was still
        hopeless. Two binary warnings tuned on one shot cannot express that, so the numbers
        themselves go in the log and the artist can move a seed on the evidence.

        For scale when reading these: artist-placed features on SH004 measure std 40-64 and
        sit in cells moving 0-8 px/frame. SH013 is uniform brown dirt at 59.94 fps and
        nothing on it reads above std 15.
        """
        con = (self._motion or {}).get("contrast") or {}
        mot = self._seed_motion or {}
        if not con and not mot:
            return
        for rec in self._records:
            name = rec["id"]
            print("[assist] %s: plate %.0f px/frame here, patch contrast std %.1f"
                  % (name, mot.get(name, 0.0), con.get(name, 0.0)))

    def _warn_soft_seeds(self, context):
        """Say which seeds have too little contrast to hold, before the take is spent.

        Measured on SH013 over a 30-seed grid: std >= 8 ran a median 113 frames, under 8 a
        median of ~25, and the seeds that died before frame 25 had a median std of 4.4. For
        scale, artist-placed features on SH004 measure std 40-64 -- this plate is uniform
        brown dirt and there is very little anywhere on it for correlation to hold.

        Not a gate. An artist may have a reason to track a soft feature, and refusing would
        be the tool overruling them on their own plate. It is the difference between "the
        tracker is broken" and "there is nothing there to hold".
        """
        con = (self._motion or {}).get("contrast") or {}
        soft = sorted((v, k) for k, v in con.items() if v < SOFT_SEED_STD)
        if not soft:
            return
        for std, name in soft:
            print("[assist] %s has almost no contrast (std %.1f) -- tracks on patches this "
                  "soft ran a median 25 frames where std 8+ ran 113" % (name, std))
        self.report({"WARNING"},
                    "%d seed(s) have very little contrast (down to std %.1f); expect short "
                    "tracks there whatever the settings" % (len(soft), soft[0][0]))

    def _warn_fast_seeds(self, context):
        """Tell the artist which seeds sit where the plate will not hold a long track.

        Measured on SH013 with everything else already fixed -- box re-fitted per frame,
        LocScale, 0.75 floor -- spans by region, out of 303 frames:

            foreground  (47 px/frame)   13  16  10   7  23
            midground   (17-32)         32  47 154  57  63
            background  (0-8)          206 128  20  39 157

        Nothing in the tracker closes that gap: bigger boxes changed nothing, Affine and
        Perspective were worse than LocScale, and a bigger pattern was worse still. On the
        frame each foreground track stopped, its own patch scored 0.53-0.72 ANYWHERE within
        160 px -- the feature genuinely stops looking like itself in one frame. Short
        foreground tracks are what that footage has to give, and saying so beforehand is
        worth more than another round of tuning.
        """
        fast = sorted(((v, k) for k, v in (self._seed_motion or {}).items()
                       if v >= FAST_SEED_PX), reverse=True)
        if not fast:
            return
        for p95, name in fast:
            print("[assist] %s sits where the plate moves %.0f px/frame -- expect a short "
                  "track here, and place more of them rather than fewer" % (name, p95))
        self.report({"WARNING"},
                    "%d seed(s) are in fast-moving parts of the plate (up to %.0f px/frame); "
                    "tracks there will be short whatever the settings"
                    % (len(fast), fast[0][0]))

    def _start_tracking(self, context):
        overlay.hide()
        self._hold_done = False
        self._stats = {}
        # Re-baselined every pass. A track that restarts from a repair or a resume starts
        # with a different box, and a watch still holding the old baseline would flag its
        # first frame.
        track_core.attach_watch(self._records, self._opts)
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
        overlay.hide()
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
            if self._phase in ("waiting", "drift", "motion", "hold"):
                client.cancel(self._root, self._job)
            return self._done(context, "cancelled")
        try:
            # The confirm phase runs on keys rather than the timer. It consumes ONLY the
            # four answer keys (`ANSWER_KEYS`) and passes everything else through -- see
            # the comment there for what swallowing the rest cost.
            if self._phase == "confirm":
                if event.type == "TIMER":
                    return {"RUNNING_MODAL"}
                return self._tick_confirm(context, event)
            if event.type != "TIMER":
                return {"PASS_THROUGH"}
            if self._phase == "motion":
                return self._tick_motion(context)
            if self._phase == "hold":
                return self._tick_hold(context)
            if self._phase == "tracking":
                return self._tick_tracking(context)
            if self._phase == "waiting":
                return self._tick_waiting(context)
            if self._phase == "drift":
                return self._tick_drift(context)
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
        self._status(context,
                     "round %d: frame %d/%d   %d live   %d dead   %d flagged   %d at edge"
                     % (self._round + 1, st.get("frame", 0), st.get("total", 0),
                        st.get("alive", 0), st.get("deaths", 0), st.get("flagged", 0),
                        st.get("edge", 0)))
        if not (done or st.get("done")):
            return {"RUNNING_MODAL"}

        # A flagged track is stopped, not dead: its box changed size in a way that says it
        # may no longer be on the feature. That is answered against the plate BEFORE
        # anything else happens -- before the backward pass, before re-acquire -- because
        # every other step downstream treats the track's last marker as trustworthy.
        flagged = [r for r in self._records if r.get("scale_flag")]
        if flagged:
            return self._ask_drift(context, flagged)

        # Before anything trusts these markers: are they still the artist's feature? This
        # runs on the whole pass at once, and what it finds is removed rather than reported,
        # because a drifted marker in a track file is worse than a gap -- 3DE will happily
        # solve to it.
        if self.hold_feature and not self._hold_done:
            if self._start_hold(context):
                return {"RUNNING_MODAL"}
            self._hold_done = True

        return self._after_tracking(context)

    def _after_tracking(self, context):
        if self.backward and not self._backward_done:
            self._status(context, "backward pass (not cancellable) ...")
            # Anchored on each track's ORIGINAL head, never on `seed_frame`: a repair or a
            # resume moves `seed_frame` forward, and tracking backwards from there would
            # re-write frames that are already measured and good.
            heads = [dict(r, seed_frame=self._seeds_px.get(r["id"], (r["seed_frame"],))[0])
                     for r in self._records]
            track_core.track_backward_pass(self._ctx, heads, self._opts)
            self._backward_done = True

        if self._round >= self.rounds:
            return self._done(context, "finished")
        return self._ask_cotracker(context)

    # ---------------------------------------------------------------- holding the feature

    def _start_hold(self, context):
        """Ask the plate whether each track is still on the feature the artist chose.

        Returns False to carry straight on -- a failure here must not cost a run.
        """
        clip = self._clip
        w, h = clip.size
        reqs = []
        for rec in self._records:
            pat = self._patterns.get(rec["id"])
            if not pat:
                continue
            tr = rec["t"]
            path = []
            for m in tr.markers:
                if m.mute:
                    continue
                # Only frames from this track's own head onward, in frame order.
                path.append([int(m.frame), m.co[0] * w, (1.0 - m.co[1]) * h])
            path.sort()
            if len(path) < 4:
                continue
            reqs.append({"id": rec["id"], "pattern": pat, "path": path})
        if not reqs:
            return False
        p = prefs.get(context)
        try:
            client.ensure(self._root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            r = client.start_hold(self._root, clip_info(context, clip), reqs)
        except client.SidecarError as exc:
            print("[assist] could not check the tracks against your pattern (%s)" % exc)
            return False
        self._job = r["id"]
        self._phase = "hold"
        self._status(context, "checking %d track(s) are still your feature ..." % len(reqs))
        return True

    def _tick_hold(self, context):
        st = client.poll(self._root, self._job)
        if st["state"] in ("queued", "running"):
            self._status(context, "%s  (%.0fs)" % (st.get("stage") or "checking",
                                                   st.get("seconds", 0)))
            return {"RUNNING_MODAL"}
        self._hold_done = True
        if st["state"] == "cancelled":
            return self._done(context, "cancelled")
        if st["state"] == "error":
            print("[assist] feature check failed: %s" % st["error"]["message"])
        else:
            self._apply_hold(context, (st.get("result") or {}).get("tracks") or [])
        return self._after_tracking(context)

    def _apply_hold(self, context, results):
        """Cut each track at the frame it stopped being the artist's feature.

        This DELETES markers, which the addon otherwise refuses to do -- and the difference
        from the verdict that was removed for deleting good work is evidence. There, a patch
        that could not be found anywhere was treated as proof the track was lost; it is not,
        and on low-contrast footage a healthy track reads the same. Here the patch is scored
        AT the position the track claims, frame by frame, and the finding is a fall: 0.91 at
        one frame, 0.22 at the next, against a baseline the track itself set. A score that
        was never high cannot fall, so a plate where nothing correlates well produces no
        cuts -- checked in `patmatch.first_loss`.

        The cut frames are drift. Leaving them muted would still put them in front of the
        artist as something to judge; leaving them live would put them in the track file.
        """
        by_id = {r["id"]: r for r in self._records}
        for res in results:
            lost = res.get("lost_at")
            if not lost:
                continue
            rec = by_id.get(res["id"])
            if rec is None:
                continue
            tr = rec["t"]
            gone = [m.frame for m in tr.markers if m.frame >= int(lost)]
            for f in gone:
                if len(tr.markers) <= 1:
                    break
                tr.markers.delete_frame(f)
            # Dead, and anchored on the last frame that WAS the feature -- which is exactly
            # what re-acquire needs to search from.
            rec["alive"] = False
            self._cut.append((res["id"], int(lost), len(gone)))
            print("[assist] %s stopped being your feature at f%d (was %.2f, became %.2f) "
                  "-- %d frame(s) of drift removed, re-acquire takes it from here"
                  % (res["id"], lost, res.get("score_first") or 0.0,
                     next((sc for f, sc in (res.get("scores") or [])
                           if f == lost and sc is not None), 0.0), len(gone)))

    # ---------------------------------------------------------------- scale drift

    def _ask_drift(self, context, flagged):
        """Ask the plate what a swollen pattern box means, for every flagged track at once.

        The question is deliberately narrow: take the patch the artist seeded, and look for
        it on the frame the track was stopped -- at its own size, and resized by exactly how
        much the box grew. Which of those two wins is the difference between a feature that
        got bigger and a box that got bigger, and neither can be told from the position
        alone, which is why the track was still alive and still moving plausibly.
        """
        clip = self._clip
        w, h = clip.size
        reqs = []
        for r in flagged:
            flag = r["scale_flag"]
            pat = self._patterns.get(r["id"])
            m = r["t"].markers.find_frame(int(flag["frame"]), exact=True)
            if pat is None or m is None:
                # Nothing to judge it against. Stop watching rather than stop the track:
                # an unanswerable question must not cost the artist a track. It carries on
                # from where it stopped, not from its seed.
                r["seed_frame"] = int(flag["frame"])
                r.pop("scale_flag", None)
                r["no_watch"] = True
                r["alive"] = True
                continue
            # The BOX centre, not the marker: the box is what correlates, and the two
            # differ whenever the artist dragged one corner.
            cx, cy, pw, ph = marker_pattern_box(m, w, h)
            reqs.append({"id": r["id"], "frame": int(flag["frame"]),
                         "pattern": pat,
                         "current": {"cx": cx, "cy": cy, "w": pw, "h": ph},
                         # The box is where the tracker put it; this only has to cover how
                         # far the correlation peak sits from that.
                         "radius": max(8.0, min(64.0,
                                                self._search_px.get(r["id"], 48.0) / 2.0))})
        if not reqs:
            self._start_tracking(context)
            return {"RUNNING_MODAL"}

        p = prefs.get(context)
        try:
            client.ensure(self._root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            r = client.start_patcheck(self._root, clip_info(context, clip), reqs,
                                      {"min_match": float(self.min_match)})
        except client.SidecarError as exc:
            # No sidecar means no pixels to judge with. That is a reason to stop WATCHING,
            # never a reason to end a tracking run the artist is already half way through --
            # the tracks keep whatever they have measured and the run continues unwatched.
            self._opts.watch_scale = False
            for rec in flagged:
                fl = rec.pop("scale_flag", None) or {}
                if fl.get("frame"):
                    rec["seed_frame"] = int(fl["frame"])
                rec["no_watch"], rec["alive"] = True, True
            print("[assist] pattern check unavailable, watch off: %s" % exc)
            self._start_tracking(context)
            return {"RUNNING_MODAL"}
        self._job = r["id"]
        self._phase = "drift"
        self._status(context, "checking %d pattern box(es) against the patch you seeded ..."
                     % len(reqs))
        return {"RUNNING_MODAL"}

    def _tick_drift(self, context):
        st = client.poll(self._root, self._job)
        if st["state"] in ("queued", "running"):
            self._status(context, "%s  (%.0fs)" % (st.get("stage") or "checking patterns",
                                                   st.get("seconds", 0)))
            return {"RUNNING_MODAL"}
        if st["state"] == "error":
            return self._finish(context, "ERROR", st["error"]["message"])
        if st["state"] == "cancelled":
            return self._done(context, "cancelled")
        self._apply_drift(context, st["result"] or {})
        self._start_tracking(context)
        return {"RUNNING_MODAL"}

    def _apply_drift(self, context, data):
        """Act on each verdict. Four answers, and the track continues on three of them."""
        clip = self._clip
        w, h = clip.size
        by_name = {r["id"]: r for r in self._records}
        for chk in data.get("checks") or []:
            rec = by_name.get(chk.get("id"))
            if rec is None:
                continue
            flag = rec.pop("scale_flag", None) or {}
            verdict = chk.get("verdict")
            f = int(chk.get("frame") or flag.get("frame") or 0)
            onset = int(flag.get("onset") or f)
            tr = rec["t"]
            note = ""

            if not chk.get("ok"):
                # Could not be judged (box off-plate, unreadable frame). The track keeps
                # what it has and stops being watched -- refusing to answer is not evidence
                # against the track.
                rec["alive"], rec["no_watch"] = True, True
                note = chk.get("reason", "not checked")
            elif verdict == "unknown":
                # The patch is not findable at any size. That reads as proof the track is
                # lost and is not: the tracker sliding off and the feature ceasing to look
                # like its seed frame are indistinguishable from here. This used to delete
                # every frame back to the onset, and on a 59.94 fps chase plate it cut a
                # track that otherwise ran the whole 303-frame shot down to FIVE markers.
                # Same rule as a question that cannot be asked: keep what was measured,
                # stop watching this track, carry on from where it stopped.
                rec["alive"], rec["no_watch"] = True, True
                rec["seed_frame"] = f
                note = ("no match at any size -- cannot tell drift from a change in "
                        "appearance, so nothing was dropped and the watch is off here")
            elif verdict == "grown":
                # The feature really did change size. The box is right and only the baseline
                # was wrong, so keep the box and carry on from here.
                rec["alive"], rec["seed_frame"] = True, f
                note = "feature really is %.2fx (scaled match %.2f vs %.2f)" % (
                    chk.get("scale", 1.0), chk.get("score_scaled") or 0.0,
                    chk.get("score_ref") or 0.0)
            elif verdict == "bad-box":
                # The patch is there at the size the artist set and the box is not: it has
                # taken in the surroundings. Put the artist's box back and track on from the
                # position Blender measured.
                #
                # The position is NOT touched, and that is a measurement, not caution: over
                # 36 LocScale tracks on SH004, healthy tracks sit p50 4.2 px / p90 25.0 px
                # from where their own seed patch correlates best 150 frames later. Snapping
                # to that peak would move good tracks tens of pixels. The box is what the
                # evidence supports fixing; the position stays Blender's.
                m = tr.markers.find_frame(f, exact=True)
                seed_f = (self._seeds_px.get(rec["id"]) or (None,))[0]
                seed_m = tr.markers.find_frame(int(seed_f), exact=True) if seed_f else None
                if m is not None and seed_m is not None:
                    m.pattern_corners = seed_m.pattern_corners
                    m.search_min = seed_m.search_min
                    m.search_max = seed_m.search_max
                rec["alive"], rec["seed_frame"] = True, f
                note = ("box was %.2fx yours and the feature was not (match %.2f) -- box "
                        "reset, %.1f px from the patch peak"
                        % (chk.get("scale", 1.0), chk.get("score_ref") or 0.0,
                           chk.get("offset_px") or 0.0))
            else:                                    # "clean"
                rec["alive"], rec["seed_frame"] = True, f
                note = "false alarm (match %.2f, box %.2fx)" % (
                    chk.get("score_ref") or 0.0, chk.get("scale", 1.0))

            n = self._fixes.get(rec["id"], 0) + 1
            self._fixes[rec["id"]] = n
            if n >= self.drift_fixes:
                # Stop asking. A feature whose box keeps swelling after this many
                # corrections is not having tracking accidents.
                rec["no_watch"] = True
            self._drift.append((rec["id"], f, verdict))
            print("[assist] %s f%d %s: %s (%s, %.2fx)"
                  % (rec["id"], f, verdict, note, flag.get("text", ""),
                     chk.get("scale", 1.0)))

        # Anything the sidecar did not answer for keeps its flag, and a kept flag sends the
        # next tracking pass straight back here -- forever. An unanswered track is released
        # unwatched instead, which is the same rule as an unanswerable one.
        for rec in self._records:
            fl = rec.pop("scale_flag", None)
            if fl is not None:
                if fl.get("frame"):
                    rec["seed_frame"] = int(fl["frame"])
                rec["no_watch"], rec["alive"] = True, True
                print("[assist] %s: no answer from the pattern check, watch off"
                      % rec["id"])

    def _ask_cotracker(self, context):
        clip = self._clip
        w, h = clip.size
        reqs = []
        for tr, f0, f1 in dead_tracks([r["t"] for r in self._records],
                                      self._n_frames, self.tail):
            seed = self._seeds_px.get(tr.name)
            if seed is None or tr.name in self._gave_up:
                continue
            rec = next((r for r in self._records if r["id"] == tr.name), None)
            if rec is not None and rec.get("edge_stopped"):
                # It did not fail, it left. Every sweep for it would be looking off-plate.
                self._gave_up.add(tr.name)
                print("[assist] %s reached the edge of frame at f%d -- finished, not lost"
                      % (tr.name, rec["edge_stopped"]))
                continue
            # Did the LAST resume actually buy anything? A point that died to blur rather
            # than to an occluder re-acquires fine and dies again straight away, so without
            # this the loop grinds forward a few frames per round on a feature that is
            # simply not trackable.
            prev = (self._resumed or {}).get(tr.name)
            if prev and self.min_resume_len and (f1 - prev[-1]) < self.min_resume_len:
                self._gave_up.add(tr.name)
                continue
            # Normally the search starts from where Blender died. But if the previous round
            # swept its whole window without finding the feature, it handed back where the
            # guide thought the point was at the end of that window: start there instead and
            # sweep the NEXT window, so an occlusion longer than one window is crossed in
            # stages rather than ending the track. Nothing from that continuation is planted
            # -- it is a place to look from, and the pattern check still rules on whatever
            # the look finds.
            cont = self._continue_from.get(tr.name)
            last_box = None
            if cont is not None and cont[0] > f1:
                lf, lx, ly = int(cont[0]), float(cont[1]), float(cont[2])
                gap = 1
            else:
                m = tr.markers.find_frame(f1, exact=True)
                if m is None:
                    continue
                lf = f1
                lx, ly = marker_to_image_px(m, w, h)
                gap = self.gap
                # The box the track was CARRYING when it died, on the frame it died on.
                # Under LocScale that is a per-frame measurement of what the feature looks
                # like now -- which after 250 frames is a different thing from what it
                # looked like when it was seeded. The sidecar localises with this and still
                # checks identity against the seed box below.
                bcx, bcy, bpw, bph = marker_pattern_box(m, w, h)
                last_box = {"frame": lf, "cx": bcx, "cy": bcy, "w": bpw, "h": bph}
            reqs.append({"id": tr.name,
                         "query_frame": seed[0], "query_x": seed[1], "query_y": seed[2],
                         "last_good_frame": lf, "last_good_x": lx, "last_good_y": ly,
                         "gap": gap, "last_box": last_box,
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
            # Two different failures, two different answers. A window that simply ran out
            # before the feature came back is worth another pass further along -- that is
            # what a long occlusion looks like from here, and abandoning the track there is
            # exactly the "it just stops" behaviour. Anything else (no feature in the box,
            # the shot ended) will answer the same way forever, so ask once.
            if miss.get("retry") and miss.get("tail_frame"):
                self._continue_from[miss["id"]] = (int(miss["tail_frame"]),
                                                   float(miss["tail_x"]),
                                                   float(miss["tail_y"]))
            else:
                self._gave_up.add(miss["id"])
        n = self._insert_resumes(context, data.get("resumes") or [])
        if not n:
            if self._continue_from and self._round < self.rounds:
                # Nothing found yet, but the search can go further. Skip the tracking pass
                # -- there is nothing new to track -- and sweep the next window.
                self._round += 1
                return self._ask_cotracker(context)
            return self._done(context, "no candidate matched the pattern you set"
                              if (data.get("misses") and self.verify_pattern)
                              else "CoTracker found no way back for any track")
        self._round += 1
        if self.confirm_resumes and self._awaiting:
            return self._start_confirm(context)
        self._start_tracking(context)
        return {"RUNNING_MODAL"}

    # ---------------------------------------------------------------- confirm

    def _start_confirm(self, context):
        """Show the artist what was found, at the frame it was found on, and wait.

        The point of the whole loop is that a re-acquire is a PROPOSAL. Muting it and
        letting the run continue makes the artist audit a batch afterwards, out of context,
        with no memory of which frame mattered. Stopping here instead puts them on the
        reappearance frame with the marker already snapped -- the one moment where "is that
        my feature?" is a two-second question.
        """
        self._awaiting.sort(key=lambda a: a["frame"])
        return self._show_current(context)

    def _show_current(self, context):
        a = self._awaiting[0]
        clip = self._clip
        win, area, region, _clip_, scene = self._ctx
        scene.frame_set(int(a["frame"]))
        space = area.spaces.active
        space.clip_user.frame_current = int(a["frame"])
        for tr in three_de.active_tracks(clip):
            on = tr.name == a["id"]
            tr.select = on
            tr.select_anchor = on
            tr.select_pattern = on
            tr.select_search = on
        self._phase = "confirm"
        score = "n/a" if a["score"] is None else "%.2f" % a["score"]
        alts = a.get("alts") or []
        nth = ("  [%d/%d]" % (a.get("alt_i", 0) + 1, len(alts))) if len(alts) > 1 else ""
        kind = "NOT VERIFIED, your call" if a.get("unverified") else "match %s" % score
        msg = ("%s found again at frame %d (%s)%s -- ENTER track on   N next match   "
               "D drop   A accept all %d   ESC stop"
               % (a["id"], a["frame"], kind, nth, len(self._awaiting)))
        self._status(context, msg)
        # ...and in the editor itself. The status bar alone is why this phase was mistaken
        # for a hang. Navigation still works while this is up.
        overlay.show(["%s found again at frame %d   (%s)"
                      % (a["id"], a["frame"],
                         ("pattern only reached %s -- NOT verified, your call" % score)
                         if a.get("unverified") else "match %s" % score),
                      ("ENTER  track on      N  next match%s      D  drop      "
                       "A  accept all %d      ESC  stop") % (nth, len(self._awaiting)),
                      "zoom and pan as normal -- nothing is frozen"])
        return {"RUNNING_MODAL"}

    def _place_candidate(self, a):
        """Move this track's proposed resume onto candidate `alt_i`, keeping the artist's box.

        Re-planted rather than nudged: a candidate is a different FRAME as well as a different
        position, so the old marker has to go or the track carries two heads.
        """
        rec = next((r for r in self._records if r["id"] == a["id"]), None)
        if rec is None:
            return
        tr = rec["t"]
        w, h = self._clip.size
        old_m = tr.markers.find_frame(int(a["frame"]), exact=True)
        geom = None
        if old_m is not None:
            geom = (tuple(tuple(c) for c in old_m.pattern_corners),
                    tuple(old_m.search_min), tuple(old_m.search_max))
            if len(tr.markers) > 1:
                tr.markers.delete_frame(int(a["frame"]))
        cand = a["alts"][a["alt_i"]]
        u, v = image_px_to_uv(float(cand["x"]), float(cand["y"]), w, h)
        m = tr.markers.insert_frame(int(cand["frame"]), co=(u, v))
        m.mute = False
        if geom is not None:
            m.pattern_corners, m.search_min, m.search_max = geom
        a["frame"] = int(cand["frame"])
        a["score"] = cand.get("score")
        rec["seed_frame"] = int(cand["frame"])
        # Keep the report honest: this track resumed HERE, not where it was first proposed.
        fr = self._resumed.get(a["id"])
        if fr:
            fr[-1] = int(cand["frame"])

    def _drop_resume(self, name, frame):
        """Delete the proposed segment and stop re-acquiring that track."""
        rec = next((r for r in self._records if r["id"] == name), None)
        if rec is None:
            return
        tr = rec["t"]
        for m in [m for m in tr.markers if m.frame >= int(frame)]:
            tr.markers.delete_frame(m.frame)
        rec["alive"] = False
        self._gave_up.add(name)
        frames = self._resumed.get(name) or []
        self._resumed[name] = [f for f in frames if f != int(frame)]
        if not self._resumed[name]:
            self._resumed.pop(name, None)

    def _tick_confirm(self, context, event):
        key = event.type
        if key not in ANSWER_KEYS:
            return {"PASS_THROUGH"}
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}     # swallow the release of a key we consumed
        if key in ("RET", "NUMPAD_ENTER"):
            self._awaiting.pop(0)
        elif key == "N":
            # Next candidate for THIS track. Wraps, so cycling past the right one comes back
            # to it rather than costing the track to a mis-key.
            a = self._awaiting[0]
            alts = a.get("alts") or []
            if len(alts) < 2:
                self.report({"INFO"}, "no other candidate for %s" % a["id"])
                return {"RUNNING_MODAL"}
            a["alt_i"] = (a["alt_i"] + 1) % len(alts)
            self._place_candidate(a)
            return self._show_current(context)
        elif key == "D":
            a = self._awaiting.pop(0)
            self._drop_resume(a["id"], a["frame"])
        else:                            # A -- accept the remaining proposals unread
            self._awaiting = []
        if self._awaiting:
            return self._show_current(context)
        # Everything answered. If the artist dropped every one there is nothing to track.
        if not any(r["alive"] for r in self._records):
            return self._done(context, "every proposal dropped")
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
            # `last_good_frame` is where the SEARCH started, which after a continuation is a
            # frame the track has no marker on. Fall back to its real last live marker --
            # the geometry has to come from a marker that exists, or the resume inherits
            # Blender's default box instead of the artist's.
            old = tr.markers.find_frame(int(res["last_good_frame"]), exact=True)
            if old is None or old.mute:
                live = [f for f in live_frames(tr) if f < int(res["frame"])]
                old = tr.markers.find_frame(live[-1], exact=True) if live else None
            if old is not None:
                sx = (old.search_max[0] - old.search_min[0]) * 1.0
                sy = (old.search_max[1] - old.search_min[1]) * 1.0
                m.pattern_corners = old.pattern_corners
                m.search_min = (-sx, -sy)
                m.search_max = (sx, sy)
            rec["alive"] = True
            rec["seed_frame"] = int(res["frame"])
            self._continue_from.pop(res["id"], None)
            # Ask only about the case that has not been measured. A resume across frames the
            # guide calls VISIBLE is the case the numbers cover; one across an occlusion is
            # the case the confirm phase was built for, and it still stops.
            occluded = int(res.get("occluded_frames") or 0)
            # An UNVERIFIED resume is always asked about. The whole reason it exists is that
            # the pattern check could not vouch for it, so the one thing that must not happen
            # is it being taken on the artist's behalf.
            unverified = res.get("verified") is False
            # Alternatives, so a wrong landing costs a keypress instead of the track.
            # The proposal the sidecar chose goes FIRST, then the alternatives by score.
            # N should walk away from what is already on screen, not jump somewhere else on
            # the first press.
            planted = {"frame": int(res["frame"]), "x": float(res["x"]),
                       "y": float(res["y"]), "score": res.get("match_score")}
            alts = [planted] + [c for c in (res.get("candidates") or [])
                                if int(c.get("frame", -1)) != planted["frame"]]
            if not alts:
                alts = [{"frame": int(res["frame"]), "x": float(res["x"]),
                         "y": float(res["y"]), "score": res.get("match_score")}]
            if unverified:
                self._awaiting.append({"id": res["id"], "frame": int(res["frame"]),
                                       "score": res.get("match_score"),
                                       "occluded": occluded, "unverified": True,
                                       "alts": alts, "alt_i": 0})
            elif self.confirm_only_occluded and occluded <= 0:
                self._auto_kept += 1
            else:
                self._awaiting.append({"id": res["id"], "frame": int(res["frame"]),
                                       "score": res.get("match_score"),
                                       "occluded": occluded, "unverified": False,
                                       "alts": alts, "alt_i": 0})
            self._resumed.setdefault(res["id"], []).append(int(res["frame"]))
            sc = res.get("match_score")
            self._scores.setdefault(res["id"], []).append((int(res["frame"]), sc))
            if len(alts) > 1:
                print("[assist] %s: %d candidate(s) to cycle with N -- %s"
                      % (res["id"], len(alts),
                         ", ".join("f%d(%.2f)" % (c["frame"], c["score"] or 0.0)
                                   for c in alts[:6])))
            print("[assist] %s died f%s -> back at f%d (first over the line f%s, "
                  "%d frame(s) swept)  match %s%s"
                  % (res["id"], res.get("last_good_frame"), int(res["frame"]),
                     res.get("first_match_frame"), int(res.get("scanned") or 0),
                     "n/a" if sc is None else "%.2f" % sc,
                     ("  (%s)" % res["match_note"]) if res.get("match_note") else ""))
            n += 1
        return n

    def _done(self, context, why):
        """Settle the resumed segments.

        With `confirm_resumes` OFF nothing has been looked at, so everything from each
        resume onwards is muted and the batch Keep/Drop pass in the panel is the review.

        With it ON the artist has already answered for each one, on its own reappearance
        frame, with the marker snapped in front of them -- so re-muting it would ask the
        same question a second time out of context, which is the thing the confirm phase
        exists to avoid. The resume FRAME ITSELF is deleted instead: it is the estimate of
        where the feature went, not a measurement of it, and the frame after it is the first
        one Blender actually matched. That is exactly what `Keep` does by hand, and deleting
        rather than muting leaves no stray muted marker for the panel to report as unread.
        """
        muted = 0
        for name, frames in (self._resumed or {}).items():
            rec = next((r for r in self._records if r["id"] == name), None)
            if rec is None:
                continue
            tr = rec["t"]
            for f0 in frames:
                if self.confirm_resumes:
                    if len(tr.markers) > 1 and tr.markers.find_frame(f0, exact=True):
                        tr.markers.delete_frame(f0)
                    continue
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
        n_unver = sum(1 for v in (self._scores or {}).values()
                      for _f, sc in v if sc is not None and sc < self.min_match)
        if n_unver:
            tail += ", %d unverified (you confirmed them)" % n_unver
        if self._cut:
            tail += (", %d track(s) cut where they left your feature (%d frame(s) of drift)"
                     % (len(self._cut), sum(n for _i, _f, n in self._cut)))
        if self._widened:
            tail += (", %d search box(es) widened to %.0f px for plate motion"
                     % (len(self._widened), max(t for _, _, t in self._widened)))
        if self._auto_kept:
            tail += (", %d resume(s) taken without asking (nothing occluded)"
                     % self._auto_kept)
        n_ref = len(self._misses or {})
        if n_ref:
            tail += ", %d refused (see console)" % n_ref
        # The scale watch reports by VERDICT, not as one count. "3 repairs" says nothing;
        # "2 boxes reset, 1 track cut back" says what happened to the track file.
        if self._drift:
            kinds = {}
            for _, _, verdict in self._drift:
                kinds[verdict] = kinds.get(verdict, 0) + 1
            tail += ", scale watch: " + ", ".join(
                "%d %s" % (n, k) for k, n in sorted(kinds.items()))
        return self._finish(
            context, "INFO",
            "%s: %d track(s), median span %d/%d, %d resume(s) %s%s"
            % (why, len(self._records), median, self._n_frames, n_res,
               "confirmed by you" if self.confirm_resumes else "muted for review", tail))


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
            # Muted markers BELOW the track's first live frame are not resumes -- they are
            # the `sequence=False` artefact recorded in `track_core.track_backward_pass`: a
            # track seeded at frame N comes back with a marker at N-1. Measured live on a
            # seed at frame 1, which produces one at frame 0. It matters here and nowhere
            # else: KEEP deliberately leaves the FIRST muted frame muted, because a resume
            # frame is the guide's estimate rather than a measurement -- and with the
            # artefact in the list, `first` is the artefact, so the estimate it meant to
            # discard gets un-muted instead.
            live = live_frames(tr)
            floor = live[0] if live else 0
            muted = [m for m in tr.markers if m.mute and m.frame > floor]
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
