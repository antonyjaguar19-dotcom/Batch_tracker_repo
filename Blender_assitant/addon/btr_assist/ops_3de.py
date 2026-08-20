"""3DE import/export/check operators.

Operator ids are prefixed `clip.btr_*`. The standalone
`experiments/blender_track/addon_3de_io.py` registers `clip.export_3de`; if an artist has
both enabled, a colliding `bl_idname` makes the second `register_class` raise and the whole
addon fails to enable. Prefixing costs nothing and removes that failure entirely.
"""

import os

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import three_de


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


class CLIP_OT_btr_export_3de(bpy.types.Operator, ExportHelper):
    bl_idname = "clip.btr_export_3de"
    bl_label = "Export 3DE tracks"
    bl_description = "Write the clip's tracks as classic 3DE 2D-track ASCII"
    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={"HIDDEN"})

    selected_only: BoolProperty(
        name="Selected tracks only", default=False,
        description="Export only the selected tracks")
    skip_muted: BoolProperty(
        name="Skip disabled markers", default=True,
        description="Leave muted markers out, so they become gaps rather than positions "
                    "you did not mean to publish")
    frame_offset: IntProperty(
        name="Frame offset", default=0,
        description="Added to every exported frame number. Use when the plate's first "
                    "frame is not Blender's frame 1")

    def execute(self, context):
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        tracks = three_de.collect(clip, self.selected_only, self.skip_muted,
                                  self.frame_offset)
        if not tracks:
            self.report({"ERROR"}, "nothing to export (no tracks, or none selected)")
            return {"CANCELLED"}
        try:
            three_de.write_3de(self.filepath, tracks)
        except OSError as e:
            self.report({"ERROR"}, "could not write: %s" % e)
            return {"CANCELLED"}
        n = sum(len(p) for _, p in tracks)
        self.report({"INFO"}, "%d tracks, %d samples -> %s"
                    % (len(tracks), n, os.path.basename(self.filepath)))
        return {"FINISHED"}


class CLIP_OT_btr_import_3de(bpy.types.Operator, ImportHelper):
    bl_idname = "clip.btr_import_3de"
    bl_label = "Import 3DE tracks"
    bl_description = "Load a 3DE 2D-track ASCII file onto this clip as tracks"
    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt", options={"HIDDEN"})

    prefix: StringProperty(
        name="Name prefix", default="",
        description="Prepended to every imported track name, so a second file loaded over "
                    "the first stays tellable apart")
    frame_offset: IntProperty(
        name="Frame offset", default=0,
        description="Added to every frame number read from the file. The inverse of the "
                    "offset used on export")
    clear_existing: BoolProperty(
        name="Delete existing tracks", default=False,
        description="Remove the clip's current tracks first")

    def execute(self, context):
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        try:
            data = three_de.read_3de(self.filepath)
        except (OSError, ValueError, IndexError) as e:
            self.report({"ERROR"}, "could not read this as 3DE ASCII: %s" % e)
            return {"CANCELLED"}

        w, h = clip.size
        tracks = three_de.active_tracks(clip)
        if self.clear_existing:
            try:
                three_de.delete_all_tracks(context, clip)
            except RuntimeError as e:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}

        n_pts = out_of_frame = 0
        for name, pts in data:
            if not pts:
                continue
            pts = sorted(pts)
            f0, x0, y0 = pts[0]
            tr = tracks.new(name=self.prefix + name, frame=f0 + self.frame_offset)
            # tracks.new() already made a marker on that frame; move it rather than adding
            # a second one on top of it.
            u, v = three_de.px_to_uv(x0, y0, w, h)
            tr.markers[0].co = (u, v)
            for fr, x, y in pts[1:]:
                u, v = three_de.px_to_uv(x, y, w, h)
                tr.markers.insert_frame(fr + self.frame_offset, co=(u, v))
            n_pts += len(pts)
            out_of_frame += sum(1 for _, x, y in pts if not (0 <= x < w and 0 <= y < h))

        msg = "%d tracks, %d samples from %s" % (len(data), n_pts,
                                                 os.path.basename(self.filepath))
        if out_of_frame:
            # Almost always a resolution mismatch: the file was made against a different
            # plate size, so everything is scaled. Worth saying out loud -- on screen it
            # just looks like bad tracking.
            msg += "  [%d samples fall outside %dx%d -- is this the right plate?]" % (
                out_of_frame, w, h)
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, msg)
        return {"FINISHED"}


class CLIP_OT_btr_check_3de(bpy.types.Operator):
    bl_idname = "clip.btr_check_3de"
    bl_label = "Check tracks"
    bl_description = ("Report what is actually on this clip: track count, frame span, "
                      "gaps and markers outside the frame")

    def execute(self, context):
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        w, h = clip.size
        n_tr = n_pts = n_gaps = n_out = n_mute = 0
        lo, hi = None, None
        for tr in three_de.active_tracks(clip):
            frames = sorted(m.frame for m in tr.markers if not m.mute)
            n_mute += sum(1 for m in tr.markers if m.mute)
            if not frames:
                continue
            n_tr += 1
            n_pts += len(frames)
            n_gaps += sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
            lo = frames[0] if lo is None else min(lo, frames[0])
            hi = frames[-1] if hi is None else max(hi, frames[-1])
            for m in tr.markers:
                x, y = three_de.uv_to_px(m.co[0], m.co[1], w, h)
                if not (0 <= x < w and 0 <= y < h):
                    n_out += 1
        if not n_tr:
            self.report({"WARNING"}, "no tracks with usable markers on this clip")
            return {"FINISHED"}
        msg = ("%d tracks, %d samples, frames %d-%d, %d gaps, %d muted markers"
               % (n_tr, n_pts, lo, hi, n_gaps, n_mute))
        if n_out:
            msg += ", %d outside %dx%d" % (n_out, w, h)
        self.report({"INFO"}, msg)
        print("[btr] " + msg)
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_export_3de, CLIP_OT_btr_import_3de, CLIP_OT_btr_check_3de)
