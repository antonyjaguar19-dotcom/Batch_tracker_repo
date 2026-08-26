"""Pin every frame onto the pattern the artist drew.

Blender matches each frame against the one BEFORE it. That is what gives it sub-pixel
precision, and it is also why a small error survives into the next frame, and the next.
Nothing ever pulls it back to the thing the artist actually pointed at.

Measured on the artist's own 250-frame reference, once the false cuts were gone and the track
ran f96-f250 without a single re-acquire: it drifted **8.8 px** by the end. Right feature,
wrong sub-position -- and every re-anchor that used to reset it had been a cut firing for the
wrong reason. Removing bad cuts made the drift worse, which is the clearest possible statement
that the drift was never being fixed, only interrupted.

So the anchor becomes the one thing that does not move: the pattern box drawn at the seed.
Every frame is registered against THAT, allowing the patch to warp, so the position answers to
the artist's feature rather than to the previous frame.

    reference 2 (250 frames)     p50 4.9 -> 2.1 px    offset 0.8 -> 0.3 px
    reference 1 (47 frames)      p50 4.7 -> 3.6 px    p90 5.6 -> 6.5 px

Both stayed at 98 % on the feature with nothing off it. The long track is where it pays; on a
47-frame one there is little accumulated drift to take out and the tail moves either way.

Positions only. Nothing is deleted, muted or added, and a frame whose pattern cannot be found
is left exactly where it was -- a pass that removes work is not a pin.
"""

import os
import time

import bpy
from bpy.props import BoolProperty, FloatProperty

from . import client, prefs
from .ops_assist import (clip_info, image_px_to_uv, live_frames, marker_pattern_box,
                         marker_to_image_px)


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def build_requests(clip, tracks, seed_frames=None):
    """One request per track: its artist pattern box, and every live frame it has.

    The pattern is taken from the track's FIRST live frame, which is the frame the artist
    placed and sized it on. Taking it from the current frame would pin the track to wherever
    it has already got to, which is the drift asking to be kept.
    """
    w, h = clip.size
    out = []
    for tr in tracks:
        fs = live_frames(tr)
        if len(fs) < 2:
            continue
        seed_f = int((seed_frames or {}).get(tr.name, fs[0]))
        if seed_f not in fs:
            seed_f = fs[0]
        m0 = tr.markers.find_frame(seed_f, exact=True)
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
        if len(path) < 2:
            continue
        out.append({"id": tr.name,
                    "pattern": {"frame": seed_f, "cx": cx, "cy": cy, "w": pw, "h": ph},
                    "path": path})
    return out


def apply_moves(clip, tracks_by_name, moved):
    """Write the corrected positions back. Returns how many markers moved."""
    w, h = clip.size
    n = 0
    for tid, got in (moved or {}).items():
        tr = tracks_by_name.get(tid)
        if tr is None:
            continue
        for mv in got:
            mm = tr.markers.find_frame(int(mv["frame"]), exact=True)
            if mm is None:
                continue
            mm.co = image_px_to_uv(float(mv["x"]), float(mv["y"]), w, h)
            n += 1
    return n


def run(context, clip, tracks, root, seed_frames=None, radius=12.0, min_match=0.60,
        say=None):
    """Do the pin. Returns (markers_moved, summary) or raises."""
    say = say or (lambda m: None)
    reqs = build_requests(clip, tracks, seed_frames)
    if not reqs:
        return 0, {}
    p = prefs.get(context)
    client.ensure(root, bpy.path.abspath(p.python_exe) if p else "", p.port if p else 0)
    job = client.start_pin(root, clip_info(context, clip), reqs,
                           {"pin_radius": float(radius), "min_match": float(min_match)})
    waited = 0.0
    while waited < 600.0:
        st = client.poll(root, job["id"])
        if st["state"] == "done":
            res = st["result"]
            by_name = {t.name: t for t in tracks}
            n = apply_moves(clip, by_name, res.get("moved"))
            return n, res.get("summary") or {}
        if st["state"] == "error":
            raise RuntimeError(st["error"]["message"])
        time.sleep(0.1)
        waited += 0.1
    raise RuntimeError("the pin did not finish")


class CLIP_OT_btr_pin(bpy.types.Operator):
    bl_idname = "clip.btr_pin"
    bl_label = "Pin to my pattern"
    bl_description = ("Register every frame of the selected tracks against the pattern box "
                      "you drew, letting the patch warp so a feature turning towards the "
                      "camera still matches. Blender tracks against the previous frame, so "
                      "error accumulates with nothing to pull it back -- on a 250-frame "
                      "reference that reached 8.8 px. Moves positions only: nothing is "
                      "deleted, muted or added, and a frame whose pattern cannot be found is "
                      "left where it is")
    bl_options = {"REGISTER", "UNDO"}

    selected_only: BoolProperty(
        name="Selected tracks only", default=True,
        description="Off pins every track in the clip")
    radius: FloatProperty(
        name="How far it may move a marker", default=12.0, min=2.0, max=48.0,
        subtype="PIXEL",
        description="In plate pixels, per frame. It has to cover accumulated drift -- 8.8 px "
                    "on the reference -- without being able to step onto a neighbouring "
                    "feature. Measured on that reference: 8 px leaves p90 at 7.1, 12 px "
                    "brings it to 4.5, and neither puts a single frame off the feature")
    min_match: FloatProperty(
        name="Minimum match", default=0.60, min=0.1, max=0.99,
        description="Below this the pattern was not found and the frame is left alone. "
                    "Measured on the reference, dropping this to 0.45 pinned one extra frame "
                    "out of 250 and changed no number -- so it buys nothing and loosens the "
                    "one gate standing between a pin and a lookalike")

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        return clip is not None and len(clip.tracking.tracks) > 0

    def execute(self, context):
        clip = _clip(context)
        tracks = [t for t in clip.tracking.tracks
                  if (t.select or not self.selected_only)]
        if not tracks:
            self.report({"WARNING"}, "no tracks selected")
            return {"CANCELLED"}
        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p and p.assist_root else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            n, summary = run(context, clip, tracks, root, radius=self.radius,
                             min_match=self.min_match, say=print)
        except Exception as exc:                                      # noqa: BLE001
            self.report({"ERROR"}, "pin: %s" % exc)
            return {"CANCELLED"}
        left = sum(s.get("left", 0) for s in summary.values())
        med = max([s.get("median_move_px", 0.0) for s in summary.values()] or [0.0])
        self.report({"INFO"},
                    "pinned %d marker(s) across %d track(s); %d frame(s) left as they were, "
                    "worst median move %.2f px" % (n, len(summary), left, med))
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_pin,)
