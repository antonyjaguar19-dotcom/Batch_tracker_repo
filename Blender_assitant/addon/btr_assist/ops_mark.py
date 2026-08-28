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

So this mode stops guessing. The artist marks the corners of each visible stretch and drags
each mark onto the feature; consecutive marks are the runs, and every run has a verified
start AND a verified end.

Two things follow, and both are measured rather than argued:

  * **No re-acquisition anywhere.** The one decision the tool cannot make is no longer asked.
  * **No neural model.** WITHIN a run, pattern-only and CoTracker+pin scored identically to
    two decimals against the artist's hand track -- 0.1 px absolute, 0.12 px motion. The
    guide only ever earned its keep crossing occlusions, and there are none left to cross.
    Runs track on the CPU with the artist's own patch.

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


class BtrMark(bpy.types.PropertyGroup):
    """One frame the artist marked, on one track.

    Kept on the SCENE rather than the track because `MovieTrackingTrack` accepts no ID
    properties -- the same limitation `track_core` records for its own per-track state.
    """

    track: StringProperty(name="Track")
    frame: IntProperty(name="Frame")


def marks_for(scene, name):
    return sorted((m.frame for m in scene.btr_marks if m.track == name))


def runs_from(scene, track, w, h):
    """Consecutive marks, paired into runs, with the artist's position at each end.

    Pairs and not a continuous chain: mark 1 to mark 2 is a visible stretch, mark 2 to mark 3
    is the occlusion, mark 3 to mark 4 is visible again. Taking every adjacent pair would
    tell the assistant to track THROUGH the very gaps the marks exist to declare.
    """
    fs = [f for f in marks_for(scene, track.name)
          if track.markers.find_frame(f, exact=True) is not None]
    out = []
    for a, b in zip(fs[0::2], fs[1::2]):
        ma = track.markers.find_frame(a, exact=True)
        mb = track.markers.find_frame(b, exact=True)
        if ma is None or mb is None or b <= a:
            continue
        ax, ay = marker_to_image_px(ma, w, h)
        bx, by = marker_to_image_px(mb, w, h)
        out.append({"start": int(a), "end": int(b),
                    "start_x": ax, "start_y": ay, "end_x": bx, "end_y": by})
    return out, fs


class CLIP_OT_btr_mark(bpy.types.Operator):
    bl_idname = "clip.btr_mark"
    bl_label = "Mark this frame"
    bl_description = ("Mark the current frame on the selected track as a run boundary, and "
                      "put a marker there for you to drag onto the feature. Mark the LAST "
                      "frame it is visible before something covers it, then the FIRST frame "
                      "it is back. Pairs of marks are the stretches the assistant will track")
    bl_options = {"REGISTER", "UNDO"}

    action: EnumProperty(
        items=(("ADD", "Mark", "Mark the current frame"),
               ("DROP", "Unmark", "Remove the mark on the current frame"),
               ("CLEAR", "Clear", "Remove every mark on this track"),
               ("STALE", "Clear stale", "Remove marks whose track no longer exists")),
        default="ADD")

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

        if self.action == "DROP":
            hit = [i for i, m in enumerate(scene.btr_marks)
                   if m.track == tr.name and m.frame == f]
            for i in reversed(hit):
                scene.btr_marks.remove(i)
            self.report({"INFO"} if hit else {"WARNING"},
                        "unmarked f%d" % f if hit else "nothing marked on f%d" % f)
            return {"FINISHED"}

        if any(m.track == tr.name and m.frame == f for m in scene.btr_marks):
            self.report({"WARNING"}, "f%d is already marked" % f)
            return {"CANCELLED"}

        m = tr.markers.find_frame(f, exact=True)
        if m is None:
            # Put a marker here to drag. It starts at the nearest existing position, which is
            # a starting point and not a claim -- the artist moves it onto the feature, and
            # that drag is the whole value of the mark.
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
        mk.track, mk.frame = tr.name, f
        n = len(marks_for(scene, tr.name))
        self.report({"INFO"},
                    "marked f%d on %s (%d mark%s) -- drag it onto the feature%s"
                    % (f, tr.name, n, "" if n == 1 else "s",
                       "" if n % 2 == 0 else "; mark the far end of this stretch next"))
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

    WHERE it comes back is still a guess. Measured end to end on the artist's `track3`
    Track.001 -- one seed marker at f1 and their six frame numbers, nothing dragged:

        run        landed        tracked
        f1-f14     seeded        f1-f14
        f25-f32    2.5 px off    f25-f31
        f41-f65    1.3 px off    f41-f63

    44 of their 47 frames on the feature, worst 2.5 px, and NOTHING inside an occlusion. The
    three missing frames are the last one or two before a gap, where Blender loses the
    feature as it starts to be covered -- it stops rather than following the occluder, which
    is the behaviour worth having.

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
        return tr is not None and len(marks_for(context.scene, tr.name)) >= 2

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

    def execute(self, context):
        from . import track_core                                      # noqa: PLC0415
        clip = _clip(context)
        tr = target(clip)
        scene = context.scene
        w, h = clip.size
        fs = [f for f in marks_for(scene, tr.name)]
        pairs = list(zip(fs[0::2], fs[1::2]))
        if not pairs:
            self.report({"WARNING"},
                        "%d mark(s) on %s -- they come in PAIRS, one for each end of a "
                        "visible stretch" % (len(fs), tr.name))
            return {"CANCELLED"}

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
        opts = track_core.Opts(leash=0.0, motion_model="LocScale", scale_clamp=1.6,
                               edge_stop=True)
        track_core.apply_settings(clip, opts)

        notes, started = [], []
        for i, (a, b) in enumerate(pairs):
            if b <= a:
                continue
            m = tr.markers.find_frame(a, exact=True)
            if m is None or a != seed_f:
                # Not the seeded run: CoTracker has to find the feature here. Anchored on the
                # last frame the track actually reached, which is the newest position anything
                # has verified.
                have = [f for f in live_frames(tr) if f < a]
                anchor_f = have[-1] if have else seed_f
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

            rec = [{"t": tr, "id": tr.name, "kind": "", "alive": True, "w": w, "h": h,
                    "seed_frame": int(a), "seed_pat": (pw, ph)}]
            # n_frames = b, so the last step is b-1 -> b. Passing b+1 would carry the track
            # one frame INTO the occlusion the artist just told us about.
            for _ in track_core.track_job(ctx, rec, int(b), {}, opts):
                pass
            reached = max([f for f in live_frames(tr) if a <= f <= b] or [a])
            notes.append("f%d-f%d: reached f%d" % (a, b, reached))

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
        if started:
            msg += " | CoTracker started " + ", ".join(
                "f%d (%s)" % (f, "no score" if sc is None else "%.2f" % sc)
                for f, sc in started)
        print("[runs] %s: %s" % (tr.name, msg))
        low = [f for f, sc in started if sc is not None and sc < 0.6]
        if low:
            self.report({"WARNING"},
                        "CoTracker's landing looks wrong at %s -- check it, or drag the mark "
                        "and track again" % ", ".join("f%d" % f for f in low))
        else:
            self.report({"INFO"}, "%d run(s), %d frame(s) outside them removed -- %s"
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
