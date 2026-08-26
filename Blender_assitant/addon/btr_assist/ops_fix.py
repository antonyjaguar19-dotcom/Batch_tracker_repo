"""Find where a finished track slid off the feature, and put it back.

The artist's report: *"tracking is going good till occlusion and once the track is obstructed
by something, the track just slides."* Nothing downstream notices, because every frame still
correlates with the frame before it -- which is what a slide IS.

Their proposal was to compare the last frame's pattern against the seed. That is the natural
test, and **on this plate it does not work**: the artist's own seed patch scores 1.00 at a
position 33.8 px off the feature, because the texture repeats. Appearance cannot separate the
feature from its neighbour here, and no threshold on a score ever will -- which is also why
the existing end-of-track check passes a slid track without hesitating.

What does work is the same idea with a different reference. CoTracker is following the real
feature, so a track that slides moves DIFFERENTLY from it, and over ten frames that difference
is far larger than the guide's own noise. Measured against the artist's hand track with a 77 px
slide built into a copy of it:

    detector                     slide found     false flags on the correct hand track
    5-frame window               f100            2
    10-frame window              f108            0
    15-frame window              f109            0

Ten frames is the earliest window that never accuses a correct track. Repairing from there --
the guide for WHERE the feature went, the artist's pattern box for exactly where within a few
pixels of that -- took the slid copy from 216 to **249 of 250 frames on the feature**, and left
the artist's own hand track completely untouched.

The clip is only edited by this operator, from what the sidecar reports, so a repair is one
undo step like any other edit.
"""

import os
import time

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty

from . import client, prefs
from .ops_assist import (clip_info, image_px_to_uv, live_frames, marker_pattern_box,
                         marker_to_image_px)


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def build_requests(clip, tracks):
    """One request per track: the artist's pattern box, and every live frame."""
    w, h = clip.size
    out = []
    for tr in tracks:
        fs = live_frames(tr)
        if len(fs) < 12:
            continue
        m0 = tr.markers.find_frame(fs[0], exact=True)
        if m0 is None:
            continue
        cx, cy, pw, ph = marker_pattern_box(m0, w, h)
        path = []
        for f in fs:
            mm = tr.markers.find_frame(f, exact=True)
            if mm is None:
                continue
            x, y = marker_to_image_px(mm, w, h)
            path.append([int(f), float(x), float(y)])
        out.append({"id": tr.name,
                    "pattern": {"frame": int(fs[0]), "cx": cx, "cy": cy, "w": pw, "h": ph},
                    "path": path})
    return out


def apply_actions(clip, tracks_by_name, results, do_cut=True):
    """Snap what can be put back, remove what cannot. Returns (snapped, cut)."""
    w, h = clip.size
    snapped = cut = 0
    for tid, acts in (results or {}).items():
        tr = tracks_by_name.get(tid)
        if tr is None:
            continue
        for a in acts:
            f = int(a["frame"])
            mm = tr.markers.find_frame(f, exact=True)
            if mm is None:
                continue
            if a["action"] == "snap":
                mm.co = image_px_to_uv(float(a["x"]), float(a["y"]), w, h)
                snapped += 1
            elif a["action"] == "cut" and do_cut:
                # A gap is honest and a slid marker is not -- 3DE will solve to the second.
                if len(tr.markers) > 1:
                    tr.markers.delete_frame(f)
                    cut += 1
    return snapped, cut


class CLIP_OT_btr_fix_slides(bpy.types.Operator):
    bl_idname = "clip.btr_fix_slides"
    bl_label = "Find and fix slides"
    bl_description = ("Check a finished track against the feature it was seeded on and put "
                      "back the frames where it slid off -- typically after an occluder "
                      "dragged it. Compares the track's MOTION against CoTracker rather than "
                      "its appearance, because on repeating texture a slid marker still "
                      "matches your pattern perfectly (measured: 1.00 at 34 px off the "
                      "feature). Frames it cannot put back are removed, because a gap is "
                      "honest and a slid marker is not")
    bl_options = {"REGISTER", "UNDO"}

    selected_only: BoolProperty(
        name="Selected tracks only", default=True,
        description="Off checks every track in the clip. Each track costs a CoTracker pass")
    cut_unfixable: BoolProperty(
        name="Remove what cannot be put back", default=True,
        description="A frame the pattern cannot be found on is a frame the track should not "
                    "have. Off leaves it where it slid to, which 3DE will solve to")
    min_match: FloatProperty(
        name="Minimum match", default=0.60, min=0.1, max=0.99,
        description="How well your pattern must match before a repaired position is accepted")
    lookback: IntProperty(
        name="Measured over", default=10, min=3, max=60,
        description="How many frames a slide is measured across. A slide is slow by nature, "
                    "so one frame of it hides inside the guide's own noise. Measured on a "
                    "hand track with a known slide: 5 frames raised two false alarms on a "
                    "CORRECT track, 10 raised none and still caught it")

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        return clip is not None and len(clip.tracking.tracks) > 0

    def execute(self, context):
        clip = _clip(context)
        tracks = [t for t in clip.tracking.tracks if (t.select or not self.selected_only)]
        if not tracks:
            self.report({"WARNING"}, "no tracks selected")
            return {"CANCELLED"}
        reqs = build_requests(clip, tracks)
        if not reqs:
            self.report({"WARNING"}, "no track is long enough to judge (needs 12+ frames)")
            return {"CANCELLED"}

        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p and p.assist_root else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            client.ensure(root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            job = client.start_fix(root, clip_info(context, clip), reqs,
                                   {"min_match": float(self.min_match),
                                    "lookback": int(self.lookback)})
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
            time.sleep(0.2)
            waited += 0.2
        if res is None:
            self.report({"ERROR"}, "the check did not finish")
            return {"CANCELLED"}

        for tid, note in (res.get("notes") or {}).items():
            print("[fix] %s: %s" % (tid, note))
        by_name = {t.name: t for t in tracks}
        snapped, cut = apply_actions(clip, by_name, res.get("tracks"),
                                     do_cut=bool(self.cut_unfixable))
        touched = sum(1 for acts in (res.get("tracks") or {}).values() if acts)
        if not snapped and not cut:
            self.report({"INFO"}, "no track has slid off its feature (see console)")
        else:
            self.report({"INFO"}, "%d track(s) slid: %d frame(s) put back, %d removed"
                                  % (touched, snapped, cut))
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_fix_slides,)
