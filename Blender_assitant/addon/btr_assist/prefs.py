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
    fill_gaps: BoolProperty(
        name="Bridge the gap it crossed", default=True,
        description="After a re-acquire, fill the frames between the cut and the resume "
                    "where the guide can be shown to have stayed on your feature. A gap is "
                    "not automatically an occlusion -- on a 250-frame hand-tracked "
                    "reference, 5 of the 7 frames left empty were frames the artist HAS, "
                    "because the loop cut at f91 as a precaution and re-acquired at f96. "
                    "Costs one extra CoTracker pass per group of deaths")
    confirm_resumes: BoolProperty(
        name="Confirm each re-acquire", default=True,
        description="When a feature is found again, jump to that frame with the marker "
                    "snapped onto it and wait for you: Enter tracks on, D drops it, A "
                    "accepts the rest, Esc stops. Off runs straight through and leaves the "
                    "batch muted for review at the end")
    confirm_only_occluded: BoolProperty(
        name="Only ask when it was hidden", default=True,
        description="Confirm only the resumes where the feature was actually OCCLUDED, and "
                    "take the rest without stopping. Measured on SH004 against an "
                    "independent Lucas-Kanade reference: over 10 autonomous resumes with "
                    "nothing occluded the worst single frame was 6.66 px and none landed on "
                    "a different feature. Crossing a real occlusion has not been measured "
                    "that way and still stops for you")
    rounds: IntProperty(
        name="Re-acquire rounds", default=8, min=0, max=50,
        description="How many times a track may be re-acquired. One occlusion needs one; a "
                    "shot can easily have several")
    hold_feature: BoolProperty(
        name="Cut where it stops being your feature", default=True,
        description="After tracking, check every frame against the pattern YOU seeded and "
                    "cut the track where it stopped being that feature. An occluder captures "
                    "a track without any single frame looking wrong, so it never dies and "
                    "never asks for a re-acquire -- the drift just gets written to the file")
    stop_at_frame_edge: BoolProperty(
        name="Stop when the pattern leaves frame", default=True,
        description="End a track as soon as its pattern box reaches the edge of the plate, "
                    "rather than letting Blender shrink the box and keep going while the "
                    "track drifts off the feature")
    fit_search_box: BoolProperty(
        name="Fit search box to the plate", default=True,
        description="Measure how far the plate moves between frames and widen any search box "
                    "too small to reach that far. The built-in sizes carry no motion term: on "
                    "a 59.94 fps chase plate the near-ground moves 21-53 px per frame while "
                    "the default corner box reaches 13, and every foreground marker died on "
                    "its first step. Boxes are only made bigger, never smaller")
    min_match: FloatProperty(
        name="Minimum match", default=0.60, min=0.0, max=1.0, subtype="FACTOR",
        description="Correlation a candidate must reach against your pattern before it is "
                    "planted. Below it the track is left dead rather than resumed on the "
                    "wrong thing. The run reports the scores it saw, so tune from those")

    animate_scale: BoolProperty(
        name="Animate location + scale", default=True,
        description="Track with the LocScale motion model so Blender solves a size for the "
                    "pattern box on every frame. That size is what the drift watch reads")
    watch_scale: BoolProperty(
        name="Watch the pattern box", default=True,
        description="Stop a track whose pattern box grows or shrinks unusually fast, or "
                    "drifts far from the size you set, and check the patch you seeded "
                    "against the plate there before it writes another frame")
    scale_ratio: FloatProperty(
        name="Max size vs your box", default=1.6, min=1.05, max=6.0,
        description="Cumulative pattern-box size against the box you set, either way "
                    "round, before the track is stopped and checked")

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "assist_root")
        col.prop(self, "python_exe")
        row = layout.row()
        row.prop(self, "port")
        row.prop(self, "autostart")
        layout.prop(self, "force_full_res")
        layout.prop(self, "rounds")
        layout.prop(self, "hold_feature")
        layout.prop(self, "stop_at_frame_edge")
        layout.prop(self, "fit_search_box")
        layout.prop(self, "confirm_resumes")
        sub_c = layout.row()
        sub_c.enabled = self.confirm_resumes
        sub_c.prop(self, "confirm_only_occluded")
        layout.prop(self, "verify_pattern")
        sub = layout.row()
        sub.enabled = self.verify_pattern
        sub.prop(self, "min_match")
        layout.prop(self, "animate_scale")
        row = layout.row()
        row.enabled = self.animate_scale
        row.prop(self, "watch_scale")
        row = layout.row()
        row.enabled = self.animate_scale and self.watch_scale
        row.prop(self, "scale_ratio")
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
