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
from bpy.props import EnumProperty, FloatProperty, IntProperty, StringProperty

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
    bl_idname = "clip.btr_track_runs"
    bl_label = "Track the marked runs"
    bl_description = ("Track only between the frames you marked, from each verified start to "
                      "each verified end. NOTHING RE-ACQUIRES and CoTracker is never called: "
                      "your click at the frame it comes back IS the re-acquisition, which is "
                      "the point -- given exact resume frames an automatic search still landed "
                      "on the wrong feature in two of three gaps, scoring 0.89 and 0.94. Each "
                      "mark is checked against your pattern first, because a mark you forgot "
                      "to drag onto the feature starts the run in the wrong place and reads "
                      "exactly like a failed re-acquisition")
    bl_options = {"REGISTER", "UNDO"}

    min_match: FloatProperty(
        name="Minimum match", default=0.60, min=0.1, max=0.99,
        description="Below this the run stops rather than carrying on into whatever matched")
    radius: FloatProperty(
        name="May move by", default=0.0, min=0.0, max=200.0, subtype="PIXEL",
        description="Plate pixels a marker may move between frames. ZERO takes it from your "
                    "own search box, which is the right answer far more often than a fixed "
                    "number: measured on this plate the feature moves up to 32 px in one "
                    "frame, so a 12 px cap stopped two runs of three within a frame or two "
                    "of their start. Set it only to override the box you drew")

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        if clip is None:
            return False
        tr = target(clip)
        return tr is not None and len(marks_for(context.scene, tr.name)) >= 2

    def execute(self, context):
        clip = _clip(context)
        tr = target(clip)
        w, h = clip.size
        runs, fs = runs_from(context.scene, tr, w, h)
        if not runs:
            self.report({"WARNING"},
                        "%d mark(s) on %s -- they come in PAIRS, one for each end of a "
                        "visible stretch" % (len(fs), tr.name))
            return {"CANCELLED"}

        seed = tr.markers.find_frame(fs[0], exact=True)
        cx, cy, pw, ph = marker_pattern_box(seed, w, h)
        req = [{"id": tr.name,
                "pattern": {"frame": int(fs[0]), "cx": cx, "cy": cy, "w": pw, "h": ph},
                "runs": runs}]

        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p and p.assist_root else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            client.ensure(root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            # The artist's own search box is their statement of how far this feature moves,
            # and `fit_search_box` sizes it from measured plate motion when they let it. A
            # fixed cap ignores both. Halved because the box spans both directions.
            radius = float(self.radius) or max(12.0, marker_search_px(seed, w, h) / 2.0)
            job = client.start_runs(root, clip_info(context, clip), req,
                                    {"min_match": float(self.min_match),
                                     "radius": radius})
        except Exception as exc:                                      # noqa: BLE001
            self.report({"ERROR"}, "sidecar: %s" % exc)
            return {"CANCELLED"}

        res, waited = None, 0.0
        while waited < 900.0:
            st = client.poll(root, job["id"])
            if st["state"] == "done":
                res = st["result"]
                break
            if st["state"] == "error":
                self.report({"ERROR"}, st["error"]["message"])
                return {"CANCELLED"}
            time.sleep(0.1)
            waited += 0.1
        if res is None:
            self.report({"ERROR"}, "the run did not finish")
            return {"CANCELLED"}

        got = (res.get("tracks") or {}).get(tr.name) or []
        marked = set(fs)
        planted = 0
        for g in got:
            f = int(g["frame"])
            if f in marked:
                # Never move a frame the artist placed by hand. It is the reference the run
                # was tracked from, and overwriting it would quietly replace their judgement
                # with the tool's.
                continue
            m = tr.markers.find_frame(f, exact=True)
            if m is None:
                m = tr.markers.insert_frame(f, co=image_px_to_uv(g["x"], g["y"], w, h))
                m.pattern_corners = seed.pattern_corners
                m.search_min, m.search_max = seed.search_min, seed.search_max
            else:
                m.co = image_px_to_uv(g["x"], g["y"], w, h)
            m.mute = False
            planted += 1

        note = (res.get("notes") or {}).get(tr.name, "")
        print("[runs] %s: %s" % (tr.name, note))
        if "MARKS NOT ON YOUR FEATURE" in note:
            # The commonest way this goes wrong, and it looks like a tracking failure rather
            # than a missed drag. Reported as a WARNING so it interrupts.
            self.report({"WARNING"}, note.split(";")[0])
        else:
            self.report({"INFO"}, "%d run(s), %d frame(s) tracked -- %s"
                        % (len(runs), planted, note or "see console"))
        return {"FINISHED"}


CLASSES = (BtrMark, CLIP_OT_btr_mark, CLIP_OT_btr_track_runs)
