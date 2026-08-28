"""AI-assisted 2D tracking inside Blender's Movie Clip Editor.

What this is, and what it is not.

Blender's own tracker is the most precise thing this project has measured on real footage
-- 2.20 px against artist hand tracks, sub-pixel on the features it holds, at full plate
resolution. What it is bad at is *robustness*: it dies 1.0-2.3 times per track where an AI
guide dies 0.3. So the split here is deliberate. A model (TAPNext++, Apache-2.0, in a
separate process) decides WHERE to put trackers and roughly how the plate moves; Blender
measures every frame.

Re-acquisition after an occlusion is NOT solved, and this addon does not pretend it is.
Autonomous re-acquire measured 315.73 px against hand tracks, with 3 of 5 resumes landing
on a different feature. So repair proposes candidates and shows the evidence; an artist
confirms every one. That is the feature, not a limitation waiting to be engineered away.

Licence note: this file imports bpy, so the addon is GPL-2.0-or-later. Everything with a
different licence -- torch, TAPNext, and anything proprietary -- lives in the sidecar
process and is reached over localhost, never imported here.
"""

bl_info = {
    "name": "Tracking Assistant",
    "author": "batch_tracker",
    "version": (0, 16, 1),
    "blender": (4, 2, 0),
    "location": "Movie Clip Editor > Sidebar (N) > Assist",
    "description": "AI-assisted 2D tracking: auto-seed, repair, 3DE import/export",
    "category": "Tracking",
}

# Blender 4.2+ extensions are described by blender_manifest.toml, and the extension loader
# STRIPS `bl_info` off the module after import -- so `from . import bl_info` raises
# ImportError at draw() time, once per redraw. This copy is taken while the module is still
# executing and survives. The literal above stays literal so build.py's version_from_source
# regex keeps the manifest and the source from disagreeing.
VERSION = bl_info["version"]

#: Stamped by build.py at package time. The version number alone could not answer "am I
#: running the fix?" -- three separate times in one session a change looked ineffective
#: because Blender or the sidecar was still holding the old code, and a version that only
#: moves on release cannot distinguish those. This moves on every build.
BUILD = "dev"

from . import (click_size, ops_3de, ops_assist, ops_diag, ops_fix, ops_mark,
               ops_pin, ops_qc, ops_report, ops_seed, overlay, panel, prefs)

MODULES = (prefs, ops_3de, ops_seed, ops_assist, ops_diag, ops_fix, ops_mark,
           ops_pin, ops_qc, ops_report, panel)


def register():
    import bpy
    for mod in MODULES:
        for cls in mod.CLASSES:
            bpy.utils.register_class(cls)
    # After the classes: the pointer needs BtrMark to exist. Marks live on the Scene because
    # MovieTrackingTrack accepts no ID properties.
    bpy.types.Scene.btr_marks = bpy.props.CollectionProperty(type=ops_mark.BtrMark)
    # After the classes, because it reads the addon preferences.
    click_size.start()


def unregister():
    import bpy
    # A draw handler outlives unregister(): disabling the addon while the confirm prompt is
    # up would leave a callback pointing into a dead module, firing on every redraw.
    overlay.hide()
    # A timer outlives unregister() the same way a draw handler does, and this one also
    # holds the artist's own default box sizes -- stopping without restoring would leave
    # their preference overwritten by a disabled addon.
    click_size.stop()
    if hasattr(bpy.types.Scene, "btr_marks"):
        del bpy.types.Scene.btr_marks
    for mod in reversed(MODULES):
        for cls in reversed(mod.CLASSES):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
