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
    "version": (0, 1, 3),
    "blender": (4, 2, 0),
    "location": "Movie Clip Editor > Sidebar (N) > Assist",
    "description": "AI-assisted 2D tracking: auto-seed, repair, 3DE import/export",
    "category": "Tracking",
}

from . import ops_3de, ops_seed, panel, prefs

MODULES = (prefs, ops_3de, ops_seed, panel)


def register():
    for mod in MODULES:
        for cls in mod.CLASSES:
            import bpy
            bpy.utils.register_class(cls)


def unregister():
    import bpy
    for mod in reversed(MODULES):
        for cls in reversed(mod.CLASSES):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
