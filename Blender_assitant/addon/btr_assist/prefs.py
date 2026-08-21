"""Addon preferences: where the sidecar lives, and what it needs.

Nothing here is guessed at runtime. `bootstrap.py` writes `config/paths.json` next to the
addon's install source, and these fields default from it on first register -- but an artist
can always override, and an override always wins.
"""

import json
import os

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty

DEFAULTS_CACHE = {}


def _repo_default():
    """`Blender_assitant/` sits inside the repo, but the addon is installed OUT of it.

    Once Blender copies the zip into its extensions folder, walking up from __file__ lands
    in Blender's config, not the repo. So the path is baked at build time into
    `paths.json`, and this only reads it.
    """
    if DEFAULTS_CACHE:
        return DEFAULTS_CACHE
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "paths.json"),
                 os.path.join(here, "..", "..", "config", "paths.json")):
        try:
            with open(os.path.abspath(cand), encoding="utf-8") as fh:
                DEFAULTS_CACHE.update(json.load(fh))
                break
        except (OSError, ValueError):
            continue
    return DEFAULTS_CACHE


class BtrAssistPrefs(bpy.types.AddonPreferences):
    # Must match the addon package name. For an extension that is the full dotted id, so
    # __package__ is the only thing that is right in both install shapes.
    bl_idname = __package__

    assist_root: StringProperty(
        name="Blender_assitant folder", subtype="DIR_PATH",
        default=_repo_default().get("assist_root", ""),
        description="The folder holding sidecar/ and runtime/. Written by bootstrap")
    python_exe: StringProperty(
        name="Sidecar Python", subtype="FILE_PATH",
        default=_repo_default().get("python_exe", ""),
        description="Python 3.11 with torch+CUDA. Written by bootstrap; Blender's own "
                    "Python is 3.13 and cannot load this project's torch")
    port: IntProperty(
        name="Port", default=0, min=0, max=65535,
        description="0 lets the sidecar pick a free port and write it to logs/sidecar.json")
    autostart: BoolProperty(
        name="Start sidecar on first use", default=True,
        description="Spawn the sidecar automatically instead of requiring a button press")
    force_full_res: BoolProperty(
        name="Force full resolution while tracking", default=True,
        description="A 50% proxy halves tracking precision invisibly. With this on, the "
                    "addon uses the original footage for the duration of a job and "
                    "restores your setting afterwards")
    # These two live here rather than only on the operator because they are a judgement the
    # artist makes once per plate, not per run, and a modal operator has no redo panel to
    # adjust them in. The operator still owns the properties -- the panel copies these into
    # it -- so scripting the operator directly is unaffected.
    verify_pattern: BoolProperty(
        name="Re-acquire must match your pattern", default=True,
        description="Correlate the pattern box you set -- the patch shown in the Track "
                    "panel preview -- against every candidate resume, at full plate "
                    "resolution, and refuse the ones that are not the same feature")
    confirm_resumes: BoolProperty(
        name="Confirm each re-acquire", default=True,
        description="When a feature is found again, jump to that frame with the marker "
                    "snapped onto it and wait for you: Enter tracks on, D drops it, A "
                    "accepts the rest, Esc stops. Off runs straight through and leaves the "
                    "batch muted for review at the end")
    min_match: FloatProperty(
        name="Minimum match", default=0.60, min=0.0, max=1.0, subtype="FACTOR",
        description="Correlation a candidate must reach against your pattern before it is "
                    "planted. Below it the track is left dead rather than resumed on the "
                    "wrong thing. The run reports the scores it saw, so tune from those")

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "assist_root")
        col.prop(self, "python_exe")
        row = layout.row()
        row.prop(self, "port")
        row.prop(self, "autostart")
        layout.prop(self, "force_full_res")
        layout.prop(self, "confirm_resumes")
        layout.prop(self, "verify_pattern")
        sub = layout.row()
        sub.enabled = self.verify_pattern
        sub.prop(self, "min_match")
        if not self.python_exe:
            box = layout.box()
            box.label(text="Run bootstrap.bat in Blender_assitant to fill these in.",
                      icon="ERROR")
            box.label(text="3DE import/export works without it; auto-seed does not.")


def get(context):
    try:
        return context.preferences.addons[__package__].preferences
    except KeyError:
        return None


CLASSES = (BtrAssistPrefs,)
