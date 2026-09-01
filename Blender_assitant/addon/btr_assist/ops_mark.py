"""Mark mode: you say where the feature is visible, the assistant only tracks.

Measured, on this footage, the assistant cannot decide when a feature goes behind something
or when it comes back. The artist's own pattern scores **0.86-0.96 on frames where the
feature is definitively hidden** -- higher than on frames where it is genuinely visible --
because the plate's texture repeats. No threshold separates those two sets, CoTracker's
visibility head agrees with the artist on 56 % of frames, and an occluder mask reached 67 %
before broadening the prompts made it worse (the feature is on a pole, so masking poles masks
the feature).

And knowing WHEN it comes back is not enough on its own. Given the artist's exact resume
frames, a search still landed on the wrong feature in two of their three gaps, scoring 0.89
and 0.94 -- confident, and wrong, with nothing downstream able to tell.

So this mode stops guessing WHEN. The artist marks the frames -- the last one where the
feature is visible, the first one where it is back -- and each mark records WHICH of the two
it is, so the pairing is something they stated rather than something inferred from the order
they happened to press the button in.

From there each engine does the one thing it is measured to be good at:

  * **CoTracker finds it** at the first frame of every later run. It is asked WHERE, never
    WHEN, and never on a frame the artist did not name.
  * **Blender tracks the runs**, with Blender's own settings, so a run is the track the
    artist would have got by hand.

Measured on their `track3` Track.001 -- one seed at f1, six frame numbers, nothing dragged --
44 of their 47 frames land on the feature, worst 2.5 px, with nothing planted inside an
occlusion. The two re-acquisitions came back 2.5 px and 1.3 px off.

The end mark also makes drift MEASURABLE for the first time. A run either arrives where the
artist said it would or it does not, and the closing error is reported per run instead of
being inferred from a correlation score.
"""

import os
import time

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)

from . import client, prefs
from .ops_assist import (clip_info, image_px_to_uv, live_frames, marker_pattern_box,
                         marker_search_px, marker_to_image_px)


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def target(clip):
    """The track the artist means, which is NOT always `tracks.active`.

    `active` does not follow selection. Measured directly: with two tracks A and B, setting
    `A.select = True` and `B.select = False` leaves `tracks.active` pointing at B. Reading
    `active` therefore showed one track's marks while the artist was looking at another, and
    the Track button acted on the wrong one -- which is exactly the trap this reports as
    "the old frames won't let me".

    So: the active track only when it is actually selected, otherwise the single selected
    track, otherwise fall back to active so the panel still has something to say.
    """
    if clip is None:
        return None
    tracks = (clip.tracking.objects.active.tracks
              if clip.tracking.objects.active else clip.tracking.tracks)
    act = tracks.active
    sel = [t for t in tracks if t.select]
    if act is not None and act.select:
        return act
    if len(sel) == 1:
        return sel[0]
    return act


def stale_marks(scene, clip):
    """Marks whose track no longer exists -- a rename or a delete leaves them behind.

    Not purged automatically: a rename would silently destroy work the artist did by hand.
    Shown instead, with a button, so removing them is their decision.
    """
    if clip is None:
        return []
    tracks = (clip.tracking.objects.active.tracks
              if clip.tracking.objects.active else clip.tracking.tracks)
    have = {t.name for t in tracks}
    return sorted({m.track for m in scene.btr_marks if m.track not in have})


#: How many times Blender and CoTracker may take turns inside one run. Each round is one
#: Blender pass plus at most one CoTracker chain, and a round that adds nothing ends the run
#: -- so this is a guard against a pathological plate, not a normal working limit. A run that
#: needs more than this is telling the artist their marks disagree with the plate.
MAX_ROUNDS = 24

#: What a mark means. `AUTO` is not a choice the artist can make -- it is what a mark saved
#: before marks had a kind reads back as, and it is resolved by position so those scenes keep
#: behaving exactly as they did.
KIND_ITEMS = (
    ("AUTO", "Auto", "Saved before marks had a kind; read as start/end by position"),
    ("START", "Appears", "The FIRST frame the feature is visible again -- a run begins here"),
    ("END", "Last visible", "The LAST frame the feature is visible -- a run ends here"),
)


class BtrMark(bpy.types.PropertyGroup):
    """One frame the artist marked, on one track, and what that frame MEANS.

    Kept on the SCENE rather than the track because `MovieTrackingTrack` accepts no ID
    properties -- the same limitation `track_core` records for its own per-track state.

    `kind` exists because a bare list of frames cannot be read back. Marks were paired by
    POSITION -- first with second, third with fourth -- so a single mark placed out of order,
    or one the artist forgot, silently shifted every run after it: the tool would then track
    across the occlusion and stop inside the visible stretch, with nothing on screen saying
    which frame it had understood as which.
    """

    track: StringProperty(name="Track")
    frame: IntProperty(name="Frame")
    kind: EnumProperty(name="Means", items=KIND_ITEMS, default="AUTO")


def marks_for(scene, name):
    return sorted((m.frame for m in scene.btr_marks if m.track == name))


def marked(scene, name):
    """Every mark on one track as (frame, kind), in frame order, with AUTO resolved.

    An AUTO mark is resolved the way marks were read before they had a kind: by its position
    among that track's marks, even = start, odd = end. Mixed data is therefore stable rather
    than clever -- an old scene reads as it always did, and any mark the artist has since set
    explicitly keeps what they said.
    """
    ms = sorted((m for m in scene.btr_marks if m.track == name), key=lambda m: m.frame)
    out = []
    for i, m in enumerate(ms):
        k = m.kind
        out.append((int(m.frame), k if k in ("START", "END")
                    else ("START" if i % 2 == 0 else "END")))
    return out


def runs(scene, name):
    """Marks paired into runs, plus every reason a pair could not be made.

    A START opens a run and the next END closes it. Not adjacent pairs: mark 1 to mark 2 is a
    visible stretch, mark 2 to mark 3 is the OCCLUSION, mark 3 to mark 4 is visible again.
    Pairing every neighbour would tell the assistant to track through the very gaps the marks
    exist to declare.

    Problems are returned rather than raised, and named by frame, because the artist is
    usually mid-way through a deliberate two-step and needs to know which step is missing --
    not that their marks are "invalid".
    """
    pairs, problems, open_at = [], [], None
    for f, kind in marked(scene, name):
        if kind == "START":
            if open_at is not None:
                problems.append("f%d and f%d are both starts -- the stretch that began at "
                                "f%d was never ended" % (open_at, f, open_at))
            open_at = f
        else:
            if open_at is None:
                problems.append("f%d ends a stretch that never started" % f)
                continue
            if f <= open_at:
                problems.append("f%d ends before it starts (f%d)" % (f, open_at))
                open_at = None
                continue
            pairs.append((open_at, f))
            open_at = None
    if open_at is not None:
        problems.append("f%d starts a stretch with no end marked" % open_at)
    return pairs, problems


def runs_from(scene, track, w, h):
    """`runs`, restricted to pairs whose ends both carry a marker, with their positions."""
    pairs, _problems = runs(scene, track.name)
    out = []
    for a, b in pairs:
        ma = track.markers.find_frame(a, exact=True)
        mb = track.markers.find_frame(b, exact=True)
        if ma is None or mb is None:
            continue
        ax, ay = marker_to_image_px(ma, w, h)
        bx, by = marker_to_image_px(mb, w, h)
        out.append({"start": int(a), "end": int(b),
                    "start_x": ax, "start_y": ay, "end_x": bx, "end_y": by})
    return out, marks_for(scene, track.name)


class CLIP_OT_btr_mark(bpy.types.Operator):
    bl_idname = "clip.btr_mark"
    bl_label = "Mark this frame"
    bl_description = ("Mark the current frame on the selected track as the START or the END "
                      "of a visible stretch. Which one is recorded WITH the mark, so the "
                      "panel can show you what each frame means instead of leaving you to "
                      "count them in pairs")
    bl_options = {"REGISTER", "UNDO"}

    action: EnumProperty(
        items=(("START", "Appears", "The FIRST frame it is visible again -- start a run"),
               ("END", "Last visible", "The LAST frame it is visible -- end the run"),
               ("ADD", "Mark", "Mark the current frame, taking start or end from what is "
                               "already marked"),
               ("DROP", "Unmark", "Remove the mark on the current frame"),
               ("GOTO", "Go to", "Jump to a marked frame"),
               ("DROPAT", "Remove", "Remove one mark by frame, without going to it"),
               ("CLEAR", "Clear", "Remove every mark on this track"),
               ("STALE", "Clear stale", "Remove marks whose track no longer exists")),
        default="ADD")
    frame: IntProperty(
        name="Frame", default=0,
        description="Which frame GOTO and DROPAT act on. 0 means the current frame, which "
                    "is what every other action uses")

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        return clip is not None and target(clip) is not None

    def execute(self, context):
        clip = _clip(context)
        tr = target(clip)
        scene = context.scene
        f = int(scene.frame_current)
        w, h = clip.size

        if self.action == "STALE":
            gone = stale_marks(scene, clip)
            for i in reversed([i for i, m in enumerate(scene.btr_marks)
                               if m.track in gone]):
                scene.btr_marks.remove(i)
            self.report({"INFO"}, "cleared marks for %d track(s) that no longer exist"
                        % len(gone))
            return {"FINISHED"}

        if self.action == "CLEAR":
            for i in reversed([i for i, m in enumerate(scene.btr_marks)
                               if m.track == tr.name]):
                scene.btr_marks.remove(i)
            self.report({"INFO"}, "cleared every mark on %s" % tr.name)
            return {"FINISHED"}

        if self.action == "GOTO":
            # Jump to a marked frame from the list. `frame_set` alone moves the scene, but
            # the clip editor draws from the SPACE's own frame -- `track_core.track_group`
            # records the same thing for the same reason -- so a jump that only set the
            # scene left the viewport on the old frame in some layouts.
            scene.frame_set(int(self.frame or f))
            sp = getattr(context, "space_data", None)
            cu = getattr(sp, "clip_user", None)
            if cu is not None:
                cu.frame_current = int(self.frame or f)
            return {"FINISHED"}

        if self.action == "DROPAT":
            g = int(self.frame or f)
            hit = [i for i, m in enumerate(scene.btr_marks)
                   if m.track == tr.name and m.frame == g]
            for i in reversed(hit):
                scene.btr_marks.remove(i)
            self.report({"INFO"} if hit else {"WARNING"},
                        "removed the mark on f%d" % g if hit
                        else "nothing marked on f%d" % g)
            return {"FINISHED"}

        if self.action == "DROP":
            hit = [i for i, m in enumerate(scene.btr_marks)
                   if m.track == tr.name and m.frame == f]
            for i in reversed(hit):
                scene.btr_marks.remove(i)
            self.report({"INFO"} if hit else {"WARNING"},
                        "unmarked f%d" % f if hit else "nothing marked on f%d" % f)
            return {"FINISHED"}

        have = marked(scene, tr.name)
        clash = [k for g, k in have if g == f]
        if clash:
            self.report({"WARNING"}, "f%d is already marked as %s"
                        % (f, "the start" if clash[0] == "START" else "the end"))
            return {"CANCELLED"}

        if self.action == "ADD":
            # No kind asked for. Whatever the marks BEFORE this frame leave open decides it,
            # which is what the artist means by pressing one button twice -- and unlike
            # counting the whole list in pairs, inserting a mark in the middle now gets the
            # kind the surrounding marks imply rather than flipping every mark after it.
            before = [k for g, k in have if g < f]
            kind = "END" if (before and before[-1] == "START") else "START"
        else:
            kind = self.action

        m = tr.markers.find_frame(f, exact=True)
        if m is None:
            # A marker so the frame is visible in the dope sheet at all. Its POSITION is not
            # a claim and nothing reads it: a run that starts here is placed by CoTracker,
            # and one that ends here is tracked into by Blender. The artist may drag it, and
            # need not.
            fs = live_frames(tr)
            near = min(fs, key=lambda g: abs(g - f)) if fs else None
            src = tr.markers.find_frame(near, exact=True) if near is not None else None
            if src is None:
                self.report({"ERROR"}, "%s has no marker to start from" % tr.name)
                return {"CANCELLED"}
            x, y = marker_to_image_px(src, w, h)
            m = tr.markers.insert_frame(f, co=image_px_to_uv(x, y, w, h))
            m.mute = False
            m.pattern_corners = src.pattern_corners
            m.search_min, m.search_max = src.search_min, src.search_max

        mk = scene.btr_marks.add()
        mk.track, mk.frame, mk.kind = tr.name, f, kind
        pairs, problems = runs(scene, tr.name)
        word = "APPEARS at" if kind == "START" else "LAST VISIBLE on"
        self.report({"INFO"},
                    "%s %s f%d -- %d stretch(es)%s"
                    % (tr.name, word, f, len(pairs),
                       "; " + problems[-1] if problems else ""))
        return {"FINISHED"}


class CLIP_OT_btr_track_runs(bpy.types.Operator):
    """Seed once, mark the frames, and let each engine do what it is good at.

    The artist seeds ONE marker and marks the frames the feature disappears and comes back --
    numbers only, no dragging. Then:

      * **CoTracker finds it** at the first frame of every later run, anchored on the last
        frame the previous run reached. Crossing an occlusion is the one thing it is measured
        to be good at, and the artist's frames mean it is never asked WHEN, only WHERE.
      * **Blender tracks the runs.** Its frame-to-frame matching is smooth, and smoothness is
        the property that matters here: an independent per-frame match has uncorrelated
        error, which lands as jitter -- exactly what made the pin worse than doing nothing
        until its correction was averaged (0.683 -> 0.764 jitter on a hand track).

    Nothing decides when the feature is hidden, because that is undecidable on this footage:
    the artist's own pattern scores 0.86-0.96 on frames where it is definitively covered.

    And the marked range is finished. Blender stops where correlation gives out, which is a
    frame or two before the occluder actually arrives -- but the artist already said the
    feature is visible on those frames, so `finish_runs` finds them: CoTracker predicts, the
    feature as the track saw it ONE FRAME AGO localises, and the artist's seed patch scores
    the result without gating it.

    Measured end to end on the artist's `track3` Track.001 -- one seed marker at f1 and their
    six frame numbers, nothing dragged:

        run        landed        Blender     finished
        f1-f14     seeded        f1-f14      --
        f25-f32    2.5 px off    f25-f31     +f32
        f41-f65    1.3 px off    f41-f63     +f64, f65

    47 of 47 frames on the feature, worst 2.5 px, NOTHING inside an occlusion.

    A wrong landing is silent: every frame after it is wrong and nothing downstream can tell.
    So each guessed run reports what the pattern scored where it started, for the artist to
    look at.
    """

    bl_idname = "clip.btr_track_runs"
    bl_label = "Track the marked runs"
    bl_description = ("Track only between the frames you marked. CoTracker finds the feature "
                      "at each reappearance, Blender tracks each run. Nothing decides WHEN "
                      "the feature is hidden -- you already said. Each run reports what your "
                      "pattern scores where it started, so a bad landing is visible rather "
                      "than silent")
    bl_options = {"REGISTER", "UNDO"}

    finish_runs: BoolProperty(
        name="Finish every marked run", default=True,
        description="You marked the frame the feature is last visible, so it IS visible "
                    "there. Blender stops a frame or two early -- correlation drops as the "
                    "occluder arrives -- and this finds the rest rather than leaving your "
                    "range unfilled. CoTracker predicts, the feature as the track last saw "
                    "it localises, and your own pattern scores the result")
    fill_identity: FloatProperty(
        name="Stop filling below", default=0.0, min=0.0, max=1.0,
        description="How well a filled frame must still match YOUR pattern before the fill "
                    "gives up. OFF by default, and that is measured: at the artist's own "
                    "hand-tracked positions this shot's seed patch scores 0.66 at f32, 0.60 "
                    "at f64 and 0.33 at f65 -- the feature turns, so any useful threshold "
                    "throws away correct frames. The score is reported per run instead. Set "
                    "it only if your feature does NOT change appearance")
    backward: BoolProperty(
        name="Also track backwards", default=True,
        description="Within the run holding your seed, track back to the run's first frame "
                    "as well as forward. Blender's backward stepping is unsafe one frame at "
                    "a time on 5.2, so it runs to completion and anything before the run is "
                    "removed afterwards")

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        if clip is None:
            return False
        tr = target(clip)
        return tr is not None and bool(runs(context.scene, tr.name)[0])

    def _guess(self, context, clip, tr, root, pattern, anchor_f, anchor_px, frame):
        """CoTracker's best position for `frame`. None if it cannot offer one."""
        try:
            job = client.start_guess(
                root, clip_info(context, clip),
                {"pattern": pattern, "anchor_frame": int(anchor_f),
                 "anchor_x": float(anchor_px[0]), "anchor_y": float(anchor_px[1]),
                 "frame": int(frame)})
        except Exception as exc:                                      # noqa: BLE001
            print("[runs] guess failed: %s" % exc)
            return None
        waited = 0.0
        while waited < 600.0:
            st = client.poll(root, job["id"])
            if st["state"] == "done":
                cands = (st["result"] or {}).get("candidates") or []
                return cands[0] if cands else None
            if st["state"] == "error":
                print("[runs] guess failed: %s" % st["error"]["message"])
                return None
            time.sleep(0.15)
            waited += 0.15
        return None

    def _fill(self, context, clip, root, pattern, anchor_f, anchor_px, frames):
        """Positions for frames Blender did not reach. [] if it cannot offer any."""
        try:
            job = client.start_fill(
                root, clip_info(context, clip),
                {"pattern": pattern, "anchor_frame": int(anchor_f),
                 "anchor_x": float(anchor_px[0]), "anchor_y": float(anchor_px[1]),
                 "frames": [int(f) for f in frames]},
                {"min_identity": float(self.fill_identity)})
        except Exception as exc:                                      # noqa: BLE001
            print("[runs] fill failed: %s" % exc)
            return [], "the sidecar refused: %s" % exc
        waited = 0.0
        while waited < 900.0:
            st = client.poll(root, job["id"])
            if st["state"] == "done":
                r = st["result"] or {}
                return (r.get("placed") or []), r.get("stopped")
            if st["state"] == "error":
                print("[runs] fill failed: %s" % st["error"]["message"])
                return [], st["error"]["message"]
            time.sleep(0.15)
            waited += 0.15
        return [], "timed out"

    def execute(self, context):
        from . import track_core                                      # noqa: PLC0415
        clip = _clip(context)
        tr = target(clip)
        scene = context.scene
        w, h = clip.size
        pairs, problems = runs(scene, tr.name)
        if not pairs:
            self.report({"WARNING"}, "%s: %s" % (
                tr.name, problems[0] if problems else "nothing marked"))
            return {"CANCELLED"}
        if problems:
            # Tracked anyway, because the pairs that ARE complete are worth having -- but
            # said out loud, since an unpaired mark is a stretch that will not be tracked and
            # that is invisible in the result.
            self.report({"WARNING"}, "%s (tracking the %d complete stretch(es))"
                        % (problems[0], len(pairs)))

        live = live_frames(tr)
        if not live:
            self.report({"ERROR"}, "%s has no seed marker" % tr.name)
            return {"CANCELLED"}
        seed_f = live[0]
        seed_m = tr.markers.find_frame(seed_f, exact=True)
        cx, cy, pw, ph = marker_pattern_box(seed_m, w, h)
        pattern = {"frame": int(seed_f), "cx": cx, "cy": cy, "w": pw, "h": ph}
        # Read as PLAIN NUMBERS now, once. Two reasons, both measured:
        #   * `markers.insert_frame` returns a marker whose search box is ZERO-sized
        #     (x[0,0] y[0,0]). Blender then has nothing to search in and the run dies on its
        #     first step -- it looks exactly like a lost feature and is not one.
        #   * a marker reference goes stale once tracking has grown the marker array, so
        #     copying `seed_m.search_min` after the first run reads a dangling marker.
        pat_corners = [tuple(c) for c in seed_m.pattern_corners]
        s_min, s_max = tuple(seed_m.search_min), tuple(seed_m.search_max)
        if (s_max[0] - s_min[0]) * w < 2.0 or (s_max[1] - s_min[1]) * h < 2.0:
            sw, sh = pw * 1.5 / w, ph * 1.5 / h
            s_min, s_max = (-sw, -sh), (sw, sh)

        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p and p.assist_root else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            client.ensure(root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
        except Exception as exc:                                      # noqa: BLE001
            self.report({"ERROR"}, "sidecar: %s" % exc)
            return {"CANCELLED"}

        win = context.window
        area = next((a for a in win.screen.areas if a.type == "CLIP_EDITOR"), None)
        region = next((r for r in area.regions if r.type == "WINDOW"), None) if area else None
        ctx = (win, area, region, clip, scene)
        # Blender's tracker, untouched. Mark mode's whole claim is that the runs are what
        # Blender would have produced and CoTracker only says WHERE a run restarts.
        opts = track_core.Opts(leash=0.0, motion_model="", scale_clamp=0.0,
                               edge_stop=True, blender_defaults=True)
        track_core.apply_settings(clip, opts)

        notes, started, filled = [], [], []
        worst_id = 1.0
        no_engine = ""   # set when the sidecar or CoTracker is the reason
        for i, (a, b) in enumerate(pairs):
            if b <= a:
                continue
            m = tr.markers.find_frame(a, exact=True)
            if m is None or a != seed_f:
                # Not the seeded run: CoTracker has to find the feature here. Anchored on the
                # last frame the track actually reached, which is the newest position anything
                # has verified.
                # The nearest verified position, on EITHER side. Preferring an earlier one
                # keeps the common case unchanged; falling back to a later one is what makes
                # a run marked before the seed work at all, instead of coming back empty.
                have = [f for f in live_frames(tr) if f < a]
                after = [f for f in live_frames(tr) if f > a]
                anchor_f = have[-1] if have else (after[0] if after else seed_f)
                am = tr.markers.find_frame(anchor_f, exact=True)
                if am is None:
                    notes.append("f%d-f%d: nothing before it to look from" % (a, b))
                    continue
                cand = self._guess(context, clip, tr, root, pattern,
                                   anchor_f, marker_to_image_px(am, w, h), a)
                if cand is None:
                    notes.append("f%d-f%d: CoTracker offered nothing" % (a, b))
                    continue
                m = tr.markers.find_frame(a, exact=True) or tr.markers.insert_frame(a)
                m.co = image_px_to_uv(float(cand["x"]), float(cand["y"]), w, h)
                m.pattern_corners = pat_corners
                m.search_min, m.search_max = s_min, s_max
                started.append((a, cand.get("score")))
            m.mute = False

            # Blender tracks, CoTracker rescues, and they take turns until the artist's end
            # mark is reached. One pass of each was not enough: a run marked f1-f200 came
            # back 22 frames, because Blender stopped at f22 and the single fill that
            # followed stopped at the first frame it could not match -- with 178 frames of
            # the artist's own range left, and nothing trying again.
            reached, added, blocked = int(a), 0, None
            for _round in range(MAX_ROUNDS):
                rec = [{"t": tr, "id": tr.name, "kind": "", "alive": True, "w": w, "h": h,
                        "seed_frame": int(reached), "seed_pat": (pw, ph)}]
                # n_frames = b, so the last step is b-1 -> b. Passing b+1 would carry the
                # track one frame INTO the occlusion the artist just told us about.
                for _ in track_core.track_job(ctx, rec, int(b), {}, opts):
                    pass
                got_to = max([f for f in live_frames(tr) if a <= f <= b] or [reached])
                if got_to >= b:
                    reached = got_to
                    break
                if not self.finish_runs:
                    reached = got_to
                    break

                # A frame Blender FAILED on still carries a marker -- a muted one. Testing
                # for absence found nothing to fill and the pass was a silent no-op.
                want = []
                for f in range(got_to + 1, b + 1):
                    mf = tr.markers.find_frame(f, exact=True)
                    if mf is None or mf.mute:
                        want.append(f)
                rm = tr.markers.find_frame(got_to, exact=True)
                if rm is None or not want:
                    reached = got_to
                    break
                hits, why = self._fill(context, clip, root, pattern, got_to,
                                       marker_to_image_px(rm, w, h), want)
                if why and not hits and not no_engine:
                    low_why = why.lower()
                    if "sidecar" in low_why or "cotracker" in low_why:
                        no_engine = why
                for hit in hits:
                    mk = (tr.markers.find_frame(int(hit["frame"]), exact=True)
                          or tr.markers.insert_frame(int(hit["frame"])))
                    mk.co = image_px_to_uv(float(hit["x"]), float(hit["y"]), w, h)
                    mk.pattern_corners = pat_corners
                    mk.search_min, mk.search_max = s_min, s_max
                    mk.mute = False
                added += len(hits)
                if hits:
                    worst_id = min(h2.get("identity") or 1.0 for h2 in hits)
                if not hits:
                    # Neither engine can cross this frame. The artist marked it visible, so
                    # say which frame stopped it rather than returning a short run silently.
                    blocked = why or "f%d could not be matched" % (want[0],)
                    reached = got_to
                    break
                reached = max([f for f in live_frames(tr) if a <= f <= b] or [got_to])
            if added:
                filled.append((a, b, added, worst_id))
            if blocked:
                notes.append("f%d-f%d: %s" % (a, b, blocked))
            notes.append("f%d-f%d: reached f%d%s"
                         % (a, b, reached, "" if reached >= b else " of f%d" % b))

            if self.backward and a < seed_f <= b:
                # Only the seeded run needs it, and only Blender's whole-sequence mode is
                # safe backwards on 5.2 -- so it runs past the start and is trimmed.
                track_core.track_group(ctx, [tr], int(seed_f), backwards=True, sequence=True)
                for f in [x for x in live_frames(tr) if x < a]:
                    if len(tr.markers) > 1:
                        tr.markers.delete_frame(f)

        # Anything outside a marked stretch is a frame the artist said is not there.
        keep = set()
        for a, b in pairs:
            keep.update(range(a, b + 1))
        removed = 0
        # Every marker, not just the live ones: a failed step leaves a MUTED marker behind,
        # and one of those sitting inside an occlusion is still a frame the artist said the
        # feature is not on.
        for f in [int(m.frame) for m in tr.markers]:
            if f not in keep and len(tr.markers) > 1:
                tr.markers.delete_frame(f)
                removed += 1

        msg = "; ".join(notes)
        if filled:
            msg += " | finished " + ", ".join(
                "f%d-f%d (+%d, your pattern %.2f)" % (a, b, n, ident)
                for a, b, n, ident in filled)
        if started:
            msg += " | CoTracker started " + ", ".join(
                "f%d (%s)" % (f, "no score" if sc is None else "%.2f" % sc)
                for f, sc in started)
        print("[runs] %s: %s" % (tr.name, msg))

        # A run that came back short is the thing the artist most needs told, and it used to
        # be buried in a note behind whatever else happened. They marked that range: the
        # frames are theirs, and a hole in it is not a detail.
        short = [(a, b, got) for a, b, got in
                 [(a, b, len([f for f in live_frames(tr) if a <= f <= b]))
                  for a, b in pairs]
                 if got < (b - a + 1)]
        low = [f for f, sc in started if sc is not None and sc < 0.6]
        if short:
            if no_engine:
                # The single most likely reason a run stops dead, and it is invisible from
                # the viewport: without the sidecar there is nothing to re-acquire WITH, so
                # a run ends wherever Blender's correlation happened to give out.
                self.report({"ERROR"},
                            "%d run(s) came back short and CoTracker never ran (%s) -- "
                            "Blender tracked alone. %s"
                            % (len(short), no_engine, msg))
            else:
                self.report({"WARNING"},
                            "short: %s -- %s"
                            % (", ".join("f%d-f%d got %d of %d" % (a, b, got, b - a + 1)
                                         for a, b, got in short), msg))
        elif low:
            self.report({"WARNING"},
                        "CoTracker's landing looks wrong at %s -- check it, or drag the mark "
                        "and track again" % ", ".join("f%d" % f for f in low))
        else:
            self.report({"INFO"}, "%d run(s) complete, %d frame(s) outside them removed -- %s"
                        % (len(pairs), removed, msg))
        return {"FINISHED"}


class CLIP_OT_btr_mark_guess(bpy.types.Operator):
    """Move the mark at the current frame to where CoTracker thinks the feature went.

    OFFERS, never decides, and that is measured rather than cautious. On the artist's three
    gaps a right candidate existed in two, but nothing available can tell which is right: the
    closing error against their own end mark is dominated by the run's drift, so it ranked a
    52 px WRONG candidate above two that were 1-3 px correct. Accepting automatically would
    be confidently wrong about a third of the time.

    So each press moves the mark to the next candidate and says what it scored. The artist
    looks, and either keeps it or drags it themselves -- which is one keypress instead of a
    drag when it is right, and no worse than today when it is not.
    """

    bl_idname = "clip.btr_mark_guess"
    bl_label = "Guess where it went"
    bl_description = ("Move the mark on this frame to where CoTracker thinks the feature "
                      "went, using the previous mark as the anchor. Press again to see the "
                      "next candidate. It OFFERS -- measured on three real gaps it found a "
                      "correct position in two, and could not tell which was correct, so you "
                      "decide. Check it before tracking")
    bl_options = {"REGISTER", "UNDO"}

    #: Candidates for the last (track, frame) asked about, so repeat presses cycle instead of
    #: re-running CoTracker for an answer already in hand.
    _cache = {}

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        if clip is None:
            return False
        tr = target(clip)
        if tr is None:
            return False
        f = int(context.scene.frame_current)
        fs = marks_for(context.scene, tr.name)
        return f in fs and fs.index(f) > 0

    def execute(self, context):
        clip = _clip(context)
        tr = target(clip)
        scene = context.scene
        w, h = clip.size
        f = int(scene.frame_current)
        fs = marks_for(scene, tr.name)
        if f not in fs or fs.index(f) == 0:
            self.report({"WARNING"}, "stand on a mark that has another mark before it")
            return {"CANCELLED"}

        key = (tr.name, f)
        cached = self._cache.get(key)
        if cached is None:
            anchor_f = fs[fs.index(f) - 1]
            ma = tr.markers.find_frame(anchor_f, exact=True)
            m0 = tr.markers.find_frame(fs[0], exact=True)
            if ma is None or m0 is None:
                self.report({"ERROR"}, "the mark before this one has no marker")
                return {"CANCELLED"}
            axp, ayp = marker_to_image_px(ma, w, h)
            cx, cy, pw, ph = marker_pattern_box(m0, w, h)
            p = prefs.get(context)
            root = bpy.path.abspath(p.assist_root) if p and p.assist_root else                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                client.ensure(root, bpy.path.abspath(p.python_exe) if p else "",
                              p.port if p else 0)
                job = client.start_guess(
                    root, clip_info(context, clip),
                    {"pattern": {"frame": int(fs[0]), "cx": cx, "cy": cy, "w": pw, "h": ph},
                     "anchor_frame": int(anchor_f), "anchor_x": axp, "anchor_y": ayp,
                     "frame": f})
            except Exception as exc:                                  # noqa: BLE001
                self.report({"ERROR"}, "sidecar: %s" % exc)
                return {"CANCELLED"}
            res, waited = None, 0.0
            while waited < 600.0:
                st = client.poll(root, job["id"])
                if st["state"] == "done":
                    res = st["result"]
                    break
                if st["state"] == "error":
                    self.report({"ERROR"}, st["error"]["message"])
                    return {"CANCELLED"}
                time.sleep(0.15)
                waited += 0.15
            if res is None:
                self.report({"ERROR"}, "the guess did not finish")
                return {"CANCELLED"}
            cached = {"cands": res.get("candidates") or [], "i": -1}
            if not cached["cands"]:
                self.report({"WARNING"}, res.get("reason") or "nothing to offer here")
                return {"CANCELLED"}
            self._cache[key] = cached

        cached["i"] = (cached["i"] + 1) % len(cached["cands"])
        c = cached["cands"][cached["i"]]
        m = tr.markers.find_frame(f, exact=True)
        if m is None:
            self.report({"ERROR"}, "no marker on f%d" % f)
            return {"CANCELLED"}
        m.co = image_px_to_uv(float(c["x"]), float(c["y"]), w, h)
        m.mute = False
        self.report({"INFO"},
                    "candidate %d of %d at f%d -- %s, %.0f px from CoTracker's own guess. "
                    "Press again for the next; check it before tracking"
                    % (cached["i"] + 1, len(cached["cands"]), f,
                       "no pattern score (this is CoTracker's raw position)"
                       if c["score"] is None else "your pattern scores %.2f" % c["score"],
                       c["from_guide_px"]))
        return {"FINISHED"}


CLASSES = (BtrMark, CLIP_OT_btr_mark, CLIP_OT_btr_mark_guess,
           CLIP_OT_btr_track_runs)
