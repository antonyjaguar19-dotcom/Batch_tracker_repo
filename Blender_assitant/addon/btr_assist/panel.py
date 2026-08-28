"""Panels: one "Assist" tab in the Movie Clip Editor sidebar.

Shaped around what an artist does, not around what the addon can do. The tab opens on two
buttons -- track what is selected, or seed the shot -- and everything else is a drawer that
starts closed. Nine settings in a flat list is a specification, not an interface; the same
nine grouped under "While tracking" and "When a track dies" say when each one matters.

Blender 5.2 has no `layout.panel()`, so the drawers are real sub-panels (`bl_parent_id` plus
`DEFAULT_CLOSED`), which is also what gives them a memory of being opened.

The warnings box is not decoration. Two of the three warnings it draws are conditions that
produce a plausible-looking track file that is quietly wrong -- a proxy halves precision
invisibly, and a resolution mismatch puts every marker on the wrong feature while the
viewport still looks fine.

No panel here may raise. A draw() that throws does so on EVERY redraw, buries the console,
and makes the whole addon look broken when only a label is. `tests/test_panels_draw.py`
calls each of these against the installed build.
"""

import bpy

from . import prefs, three_de, track_core


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def _version_str():
    """Never raise inside draw(). A panel that throws does so on EVERY redraw, which buries
    the console and makes the addon look broken even when only the label is."""
    import sys
    pkg = sys.modules.get(__package__)
    v = getattr(pkg, "VERSION", None)
    if not v:
        v = getattr(pkg, "bl_info", {}).get("version") if pkg else None
    return ".".join(str(int(x)) for x in v) if v else "?"


def _build_str():
    """Never raise inside draw() -- same rule as `_version_str`."""
    import sys
    pkg = sys.modules.get(__package__)
    return str(getattr(pkg, "BUILD", "?") if pkg else "?")


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
    # A scene range shorter than the clip is not a tracking limit here -- the operator uses
    # `clip.frame_duration` -- but it decides what the artist can SEE and scrub, so a track
    # that ran to frame 300 looks like it stopped at the end of the range. Seen in a real
    # diagnostic report: scene end 250 against a 328-frame clip.
    if context.scene.frame_end < clip.frame_duration:
        out.append(("INFO", "Scene ends at %d but the clip is %d frames"
                    % (context.scene.frame_end, clip.frame_duration)))
    if clip.frame_duration <= 1:
        out.append(("ERROR", "Clip reports %d frame -- is this a sequence?"
                    % clip.frame_duration))
    return out


def _counts(clip):
    """(tracks, selected, selected tracks carrying muted markers)."""
    tracks = list(three_de.active_tracks(clip))
    sel = [t for t in tracks if t.select]
    unread = sum(1 for t in sel if any(m.mute for m in t.markers))
    return len(tracks), len(sel), unread


class _Base(bpy.types.Panel):
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Assist"

    @classmethod
    def poll(cls, context):
        return _clip(context) is not None


class CLIP_PT_btr_main(_Base):
    """The whole tab in one screen: what is loaded, what is wrong, and the two buttons."""

    bl_label = "Tracking Assistant"

    def draw(self, context):
        layout = self.layout
        clip = _clip(context)
        if clip is None:
            layout.label(text="Load a clip first", icon="ERROR")
            return
        p = prefs.get(context)
        n_tracks, n_sel, n_unread = _counts(clip)

        # One line, not a box. The detail that used to live here is in the Setup drawer;
        # what an artist needs at a glance is the plate size and how many tracks exist.
        row = layout.row()
        row.label(text="%d x %d  ·  %d frames  ·  %d tracks"
                       % (clip.size[0], clip.size[1], clip.frame_duration, n_tracks),
                  icon="SEQUENCE")

        for level, msg in warnings_for(context, clip):
            layout.label(text=msg, icon="ERROR" if level == "ERROR" else "INFO")

        # The primary action. Big, first, and disabled with a reason rather than silently
        # doing nothing when nothing is selected.
        col = layout.column(align=True)
        col.scale_y = 1.6
        col.enabled = n_sel > 0
        op = col.operator("clip.btr_assist_track",
                          text=("Track %d selected" % n_sel) if n_sel else "Track selected",
                          icon="TRACKING")
        # The prefs hold the artist's choice; the operator holds the property. Copying here
        # keeps one source of truth without giving a modal operator a redo panel it cannot
        # have.
        if p is not None:
            op.verify_pattern = p.verify_pattern
            op.fill_gaps = p.fill_gaps
            op.track_engine = p.track_engine
            op.pin_to_pattern = p.pin_to_pattern
            op.pin_radius = p.pin_radius
            op.min_match = p.min_match
            op.confirm_resumes = p.confirm_resumes
            op.fit_search_box = p.fit_search_box
            op.stop_at_frame_edge = p.stop_at_frame_edge
            op.hold_feature = p.hold_feature
            op.rounds = p.rounds
            op.confirm_only_occluded = p.confirm_only_occluded
            op.animate_scale = p.animate_scale
            op.watch_scale = p.watch_scale
            op.scale_ratio = p.scale_ratio
        if not n_sel:
            layout.label(text="Select the markers you want tracked", icon="INFO")

        col = layout.column(align=True)
        col.scale_y = 1.2
        col.operator("clip.btr_autoseed", text="Auto-seed and track", icon="TRACKER")

        # Only shown when there is something to answer. A Keep/Drop pair with nothing muted
        # is a button that does nothing, and an artist cannot tell that by looking.
        if n_unread:
            box = layout.box()
            box.label(text="%d selected track(s) have unread resumes" % n_unread,
                      icon="QUESTION")
            row = box.row(align=True)
            row.scale_y = 1.2
            row.operator("clip.btr_confirm_resumes", text="Keep",
                         icon="CHECKMARK").action = "KEEP"
            row.operator("clip.btr_confirm_resumes", text="Drop",
                         icon="X").action = "DROP"

        # The question about the SHOT rather than about a track. It belongs on the main panel
        # and not in a closed sub-panel: every other check here judges one track at a time,
        # and a set made entirely of good tracks can still be unsolvable -- all on one plane,
        # all in one corner, or eleven frames in the middle held up by four of them. None of
        # that is visible in the viewport, and an export and a solve is an expensive way to
        # find out.
        clip = getattr(context.space_data, "clip", None)
        if clip is not None and len(clip.tracking.tracks) > 1:
            layout.separator()
            row = layout.row()
            row.scale_y = 1.2
            row.operator("clip.btr_shot_report", text="Will this shot solve?",
                         icon="CAMERA_DATA")


class CLIP_PT_btr_opts(_Base):
    """Every knob, grouped by WHEN it applies. Closed until someone wants it."""

    bl_label = "Options"
    bl_parent_id = "CLIP_PT_btr_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        p = prefs.get(context)
        if p is None:
            layout.label(text="Preferences unavailable", icon="ERROR")
            return
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(heading="Quality", align=True)
        col.prop(p, "hold_feature", text="Cut where it leaves my feature")
        col.prop(p, "rounds", text="Re-acquire rounds")

        col = layout.column(heading="While tracking", align=True)
        col.prop(p, "fit_search_box", text="Fit box to plate motion")
        col.prop(p, "stop_at_frame_edge", text="Stop at frame edge")
        col.prop(p, "animate_scale", text="Track scale too")
        sub = col.column(align=True)
        sub.enabled = p.animate_scale
        sub.prop(p, "watch_scale", text="Watch the box")
        row = sub.row()
        row.enabled = p.watch_scale
        row.prop(p, "scale_ratio", text="Box size limit")

        row = layout.row()
        row.alignment = "RIGHT"
        row.operator("clip.btr_reset_prefs", text="Reset to defaults", icon="LOOP_BACK")

        col = layout.column(align=True)
        col.prop(p, "track_engine", text="Between occlusions")

        col = layout.column(heading="Pinning", align=True)
        col.prop(p, "pin_to_pattern", text="Pin every frame to my pattern")
        row = col.row()
        row.enabled = p.pin_to_pattern
        row.prop(p, "pin_radius", text="May move by")

        col = layout.column(heading="New markers", align=True)
        col.prop(p, "constant_box", text="Same size on screen at any zoom")
        row = col.row()
        row.enabled = p.constant_box
        row.prop(p, "box_screen_px", text="On-screen size")
        # A live readout, because "it still looks small" and "it is not running" are the same
        # picture. The three numbers say which: the zoom it is reading, the plate size it
        # asked for, and what that comes to on screen. If the last one is not what the
        # setting above says, the reconciler is not reaching this clip.
        clip = getattr(context.space_data, "clip", None)
        zoom = getattr(context.space_data, "zoom_percentage", 0.0) or 0.0
        if clip is not None and zoom > 0:
            pat = clip.tracking.settings.default_pattern_size
            col.label(text="now: %d px at %d%% zoom  ->  %d px on screen"
                           % (pat, round(zoom), round(pat * zoom / 100.0)),
                      icon="ZOOM_IN")

        col = layout.column(heading="When a track dies", align=True)
        col.prop(p, "verify_pattern", text="Must match my pattern")
        row = col.row()
        row.enabled = p.verify_pattern
        row.prop(p, "min_match", text="Minimum match")
        col.prop(p, "fill_gaps", text="Bridge the gap it crossed")
        col.prop(p, "confirm_resumes", text="Ask me about each one")
        row = col.row()
        row.enabled = p.confirm_resumes
        row.prop(p, "confirm_only_occluded", text="Only when hidden")


class CLIP_PT_btr_mark(_Base):
    """Mark mode: the artist says where the feature is visible, the tool only tracks.

    Its own panel rather than a row in Options, because it is a WORKFLOW and not a setting --
    marking, dragging and tracking happen in that order and the panel has to show where you
    are in it.
    """

    bl_label = "Mark mode"
    bl_parent_id = "CLIP_PT_btr_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from . import ops_mark
        layout = self.layout
        clip = getattr(context.space_data, "clip", None)
        tr = clip.tracking.tracks.active if clip else None
        if tr is None:
            layout.label(text="Select a track first", icon="INFO")
            return

        fs = ops_mark.marks_for(context.scene, tr.name)
        col = layout.column(align=True)
        col.scale_y = 1.2
        col.operator("clip.btr_mark", text="Mark this frame",
                     icon="MARKER_HLT").action = "ADD"
        row = layout.row(align=True)
        row.operator("clip.btr_mark", text="Unmark").action = "DROP"
        row.operator("clip.btr_mark", text="Clear").action = "CLEAR"

        if not fs:
            layout.label(text="Mark the LAST visible frame, then the FIRST frame it is back.")
            layout.label(text="Drag each mark onto the feature.")
            return
        box = layout.box()
        box.label(text="%s: %s" % (tr.name, ", ".join("f%d" % f for f in fs)))
        if len(fs) % 2:
            # Saying which half of the pair is missing beats "invalid": the artist is mid-way
            # through a deliberate two-step and needs to know which step.
            box.label(text="one end of a stretch is unmarked", icon="ERROR")
        else:
            box.label(text="%d stretch(es) to track" % (len(fs) // 2), icon="CHECKMARK")
        row = layout.row()
        row.scale_y = 1.3
        row.enabled = len(fs) >= 2 and len(fs) % 2 == 0
        row.operator("clip.btr_track_runs", text="Track the marked runs", icon="TRACKING")


class CLIP_PT_btr_3de(_Base):
    bl_label = "3DE tracks"
    bl_parent_id = "CLIP_PT_btr_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.scale_y = 1.2
        col.operator("clip.btr_import_3de", text="Import", icon="IMPORT")
        col.operator("clip.btr_export_3de", text="Export", icon="EXPORT")
        layout.operator("clip.btr_check_3de", text="Check tracks", icon="INFO")
        # The QC an artist actually asks about a finished track: is it still on the thing
        # they picked? Lives here rather than in Options because it is a thing you DO.
        layout.operator("clip.btr_qc_ends", text="Check ends on my pattern", icon="CHECKMARK")
        # Works on any track, including ones tracked by hand or imported -- the pin does not
        # care how a track was made, only that its first frame is where the artist framed it.
        layout.operator("clip.btr_pin", text="Pin to my pattern", icon="PINNED")
        # The one that answers "the track slid after the occluder". It is separate from the
        # pin because the pin corrects a few pixels and this re-finds the feature entirely.
        layout.operator("clip.btr_fix_slides", text="Find and fix slides", icon="TRACKING_REFINE_FORWARDS")
        layout.label(text="Gaps are kept in both directions.")


class CLIP_PT_btr_setup(_Base):
    """Sidecar, seeding without tracking, and the two strings that identify a build."""

    bl_label = "Setup"
    bl_parent_id = "CLIP_PT_btr_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        clip = _clip(context)
        p = prefs.get(context)

        layout.label(text="Sidecar")
        row = layout.row(align=True)
        row.operator("clip.btr_sidecar", text="Start", icon="PLAY").action = "START"
        row.operator("clip.btr_sidecar", text="Check", icon="INFO").action = "CHECK"
        row.operator("clip.btr_sidecar", text="Stop", icon="X").action = "STOP"
        if p is not None and not p.python_exe:
            layout.label(text="Run bootstrap.bat first", icon="ERROR")

        op = layout.operator("clip.btr_autoseed", text="Place seeds only",
                             icon="TRACKER_DATA")
        op.track_after = False
        layout.operator("clip.btr_diagnose", text="Write diagnostic report", icon="TEXT")

        # Version AND build, because both have cost a debugging round. Blender keeps a
        # disabled addon's submodules in sys.modules, so installing a new zip can leave the
        # OLD code running until a restart with nothing anywhere to say so -- and a version
        # that only moves on release cannot tell that apart from a fix that did not work.
        # `source` decides how the plate path is sent to the sidecar; getting it wrong sent
        # a folder instead of a file.
        box = layout.box()
        box.label(text="v%s   build %s" % (_version_str(), _build_str()))
        if clip is not None:
            box.label(text="clip source: %s" % clip.source)


CLASSES = (CLIP_PT_btr_main, CLIP_PT_btr_opts, CLIP_PT_btr_mark, CLIP_PT_btr_3de,
           CLIP_PT_btr_setup)
