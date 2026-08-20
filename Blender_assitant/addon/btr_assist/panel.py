"""Panels: one "Assist" tab in the Movie Clip Editor sidebar.

The warnings box is not decoration. Two of the three warnings it draws are conditions that
produce a plausible-looking track file that is quietly wrong -- a proxy halves precision
invisibly, and a resolution mismatch puts every marker on the wrong feature while the
viewport still looks fine.
"""

import bpy

from . import prefs, three_de, track_core


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def warnings_for(context, clip):
    """Conditions that silently corrupt a result, in the order they bite."""
    out = []
    sd = context.space_data
    # 'FULL' is the original footage. 'PROXY_100' is a rendered 100%-size proxy file, which
    # is a re-encode -- so it is NOT the safe value, despite the name.
    size = getattr(sd.clip_user, "proxy_render_size", track_core.FULL_RES)
    if clip.use_proxy and size != track_core.FULL_RES:
        p = prefs.get(context)
        if p is None or p.force_full_res:   # matches the preference default
            out.append(("INFO", "Proxy %s is on; jobs will run at full res" % size))
        else:
            out.append(("ERROR", "Proxy %s -- tracking precision is reduced" % size))
    if clip.filepath.startswith("//"):
        out.append(("INFO", "Clip path is relative to the .blend"))
    if clip.frame_duration <= 1:
        out.append(("ERROR", "Clip reports %d frame -- is this a sequence?"
                    % clip.frame_duration))
    return out


class CLIP_PT_btr_main(bpy.types.Panel):
    bl_label = "Tracking Assistant"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Assist"

    def draw(self, context):
        layout = self.layout
        clip = _clip(context)
        if clip is None:
            layout.label(text="Load a clip first", icon="ERROR")
            return

        box = layout.box()
        box.label(text=clip.name, icon="SEQUENCE")
        box.label(text="%d x %d, %d frames"
                  % (clip.size[0], clip.size[1], clip.frame_duration))
        box.label(text="%d tracks" % len(three_de.active_tracks(clip)))
        # Version and clip kind, on screen, because both have already cost a debugging
        # round. Blender keeps a disabled addon's submodules in sys.modules, so installing
        # a new zip and re-enabling can leave the OLD code running until a restart -- with
        # nothing anywhere to say so. And SEQUENCE vs MOVIE decides how the plate path is
        # sent to the sidecar; getting it wrong sent a folder instead of a file.
        from . import bl_info                                          # noqa: PLC0415
        box.label(text="v%d.%d.%d   source: %s"
                  % (bl_info["version"] + (clip.source,)))

        for level, msg in warnings_for(context, clip):
            layout.label(text=msg, icon="ERROR" if level == "ERROR" else "INFO")


class CLIP_PT_btr_3de(bpy.types.Panel):
    bl_label = "3DE 2D tracks"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Assist"

    @classmethod
    def poll(cls, context):
        return _clip(context) is not None

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("clip.btr_import_3de", text="Import 3DE file", icon="IMPORT")
        col.operator("clip.btr_export_3de", text="Export 3DE file", icon="EXPORT")
        layout.operator("clip.btr_check_3de", text="Check tracks", icon="INFO")
        layout.label(text="Gaps are kept in both directions.")





class CLIP_PT_btr_seed(bpy.types.Panel):
    bl_label = "Auto-seed"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Assist"

    @classmethod
    def poll(cls, context):
        return _clip(context) is not None

    def draw(self, context):
        layout = self.layout
        p = prefs.get(context)

        row = layout.row(align=True)
        row.operator("clip.btr_sidecar", text="Start", icon="PLAY").action = "START"
        row.operator("clip.btr_sidecar", text="Check", icon="INFO").action = "CHECK"
        row.operator("clip.btr_sidecar", text="Stop", icon="X").action = "STOP"
        if p is not None and not p.python_exe:
            layout.label(text="Run bootstrap.bat first", icon="ERROR")

        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("clip.btr_autoseed", text="Auto-seed and track", icon="TRACKER")
        op = col.operator("clip.btr_autoseed", text="Place seeds only", icon="TRACKER_DATA")
        op.track_after = False

        box = layout.box()
        box.label(text="Blender measures every frame;", icon="INFO")
        box.label(text="the model only chooses where.")
        box.label(text="Measured 2.20 px vs hand tracks.")





class CLIP_PT_btr_assist(bpy.types.Panel):
    bl_label = "Track + re-acquire"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Assist"

    @classmethod
    def poll(cls, context):
        return _clip(context) is not None

    def draw(self, context):
        layout = self.layout
        clip = _clip(context)
        n_sel = sum(1 for t in three_de.active_tracks(clip) if t.select)
        n_muted = sum(1 for t in three_de.active_tracks(clip) if t.select
                      for m in t.markers if m.mute)

        col = layout.column(align=True)
        col.scale_y = 1.3
        col.enabled = n_sel > 0
        col.operator("clip.btr_assist_track",
                     text="Track %d selected + re-acquire" % n_sel, icon="TRACKING")
        if not n_sel:
            layout.label(text="Select your markers first", icon="INFO")

        box = layout.box()
        box.label(text="Place your own seeds. Blender", icon="INFO")
        box.label(text="tracks; CoTracker only says")
        box.label(text="where a dead one went.")

        if n_muted:
            rev = layout.box()
            rev.label(text="%d marker(s) awaiting review" % n_muted, icon="ERROR")
            rev.label(text="Look at the plate at the resume")
            rev.label(text="frame. 26-47%% land correctly.")
            row = rev.row(align=True)
            row.operator("clip.btr_confirm_resumes", text="Keep", icon="CHECKMARK").action = "KEEP"
            row.operator("clip.btr_confirm_resumes", text="Drop", icon="X").action = "DROP"


CLASSES = (CLIP_PT_btr_main, CLIP_PT_btr_assist, CLIP_PT_btr_seed, CLIP_PT_btr_3de)
