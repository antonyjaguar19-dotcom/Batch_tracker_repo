"""Addon preferences: where the sidecar lives, and what it needs.

Nothing here is guessed at runtime. `bootstrap.py` writes `config/paths.json` next to the
addon's install source, and these fields default from it on first register -- but an artist
can always override, and an override always wins.
"""

import json
import os

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       StringProperty)

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
    track_engine: EnumProperty(
        name="Engine between occlusions", default="BLENDER",
        items=(("BLENDER", "Blender",
                "Blender's own tracker, matching each frame against the one before it"),
               ("COTRACKER", "CoTracker",
                "CoTracker picks the neighbourhood on every frame, your pattern box picks "
                "the pixel. Measured against two hand tracks: 247/250 against 246 on the "
                "long one, level on the occluded one, tail slightly worse. Falls back to "
                "Blender if the sidecar or CoTracker is unavailable")),
        description="Which engine carries a track BETWEEN occlusions. Crossing an occlusion "
                    "is CoTracker either way")
    blender_tracking: BoolProperty(
        name="Track with Blender's own settings", default=True,
        description="Track with the same settings Blender itself uses, so a track the assistant makes is the track you would have made by hand. Turning this OFF applies this addon's own measured configuration -- PREV_FRAME matching with normalization, which survives 2.6-2.9x longer on real plates but does NOT reproduce Blender: measured 10.36 px apart over 14 frames of SH006 from the same seed")
    pin_to_pattern: BoolProperty(
        name="Pin every frame to my pattern", default=False,
        description="When a run finishes, register every frame against the pattern box you "
                    "drew instead of leaving it answerable to the previous frame. Blender "
                    "tracks against the frame before, so error accumulates with nothing to "
                    "pull it back -- on a 250-frame reference it reached 8.8 px. Pinning took "
                    "the median from 4.9 px to 2.1 and the constant offset from 0.8 px to "
                    "0.3, with nothing moved off its feature. Positions only. OFF by default: it "
                    "MOVES frames Blender placed, so the result stops being Blender's "
                    "track -- turn it on when you want the drift removed and are ready to "
                    "check the result against your own eye")
    pin_radius: FloatProperty(
        name="Pin may move a marker by", default=12.0, min=2.0, max=48.0, subtype="PIXEL",
        description="Plate pixels, per frame. Has to cover accumulated drift without being "
                    "able to reach a neighbouring feature. Measured: 8 px leaves p90 at "
                    "7.1 px, 12 px brings it to 4.5, neither puts a frame off the feature")
    constant_box: BoolProperty(
        name="New markers keep their on-screen size", default=True,
        description="Blender sizes a new track in PLATE pixels, so the same setting that "
                    "looks right on an HD plate draws a five-pixel box on a 4K one at "
                    "fit-to-window zoom -- too small to see what you seeded. With this on, "
                    "the size Ctrl-click uses is kept in step with your zoom so the box is "
                    "always the same size in the viewport. Your own default pattern size is "
                    "remembered and put back when this is switched off",
        update=lambda self, ctx: _constant_box_changed(self))
    box_screen_px: IntProperty(
        name="On-screen box size", default=40, min=12, max=200, subtype="PIXEL",
        description="How big a new marker's pattern box should look, in SCREEN pixels, "
                    "whatever the zoom. Clamped in plate pixels at both ends: never under "
                    "16, because a smaller patch holds too little texture to correlate "
                    "however big it looks, and never over a quarter of the short edge")
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
        layout.prop(self, "track_engine")
        layout.prop(self, "pin_to_pattern")
        row = layout.row()
        row.enabled = self.pin_to_pattern
        row.prop(self, "pin_radius")
        layout.prop(self, "constant_box")
        row = layout.row()
        row.enabled = self.constant_box
        row.prop(self, "box_screen_px")
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
        layout.separator()
        layout.operator("clip.btr_reset_prefs", icon="LOOP_BACK")
        if not self.python_exe:
            box = layout.box()
            box.label(text="Run bootstrap.bat in Blender_assitant to fill these in.",
                      icon="ERROR")
            box.label(text="3DE import/export works without it; auto-seed does not.")


def _constant_box_changed(self):
    """Switching this off must hand the artist's own default box size back at once."""
    from . import click_size                                          # noqa: PLC0415
    if self.constant_box:
        click_size.start()
    else:
        click_size.stop()



#: Left alone by the reset. These are facts about THIS MACHINE, not preferences -- putting
#: them back to a shipped default turns a settings tidy-up into a broken sidecar, and the
#: artist finds out by running bootstrap again.
#:
#: Module-level and not a class attribute: `self.KEEP` inside a bpy Operator does not resolve
#: to the class attribute the way plain Python would -- bpy_struct's attribute lookup shadows
#: it -- so the guard silently matched nothing and the reset cleared the paths it was written
#: to protect. The class attribute was plainly visible from the type the whole time.
RESET_KEEP = ("assist_root", "python_exe", "port")


class CLIP_OT_btr_reset_prefs(bpy.types.Operator):
    """Put every assist setting back to the value it ships with.

    Worth having because almost every default here is a MEASURED number rather than a taste:
    `min_match` at 0.66, `pin_radius` at 12, the closure trust at 25 px. Each was fitted
    against hand tracks, and a session spent nudging them to chase one awkward shot leaves
    the tool quietly worse on the next one with no record of what moved.

    Deliberately does NOT touch the two path settings. `assist_root` and `python_exe` are
    facts about this machine, not preferences -- resetting them turns a settings tidy-up into
    a broken sidecar, and finding that out means running bootstrap again.
    """

    bl_idname = "clip.btr_reset_prefs"
    bl_label = "Reset assist settings"
    bl_description = ("Put every assist setting back to its shipped default. Most of them are "
                      "measured against hand tracks rather than chosen, so this is the way "
                      "back after tuning for one awkward shot. Leaves the sidecar paths alone "
                      "-- those describe this machine, not a preference")
    # REGISTER only. UNDO on an operator that writes AddonPreferences makes Blender reload
    # preferences from userpref.blend as part of the undo push, which quietly reverts any
    # unsaved change -- including the sidecar paths this is careful not to touch. It looked
    # exactly like the guard failing, and the guard was fine.
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        # `get`, not `prefs.get` -- this class lives IN prefs.py, and the qualified name is a
        # NameError that surfaces as "poll() failed, context is incorrect", which reads like a
        # UI problem and is not one.
        return get(context) is not None

    def execute(self, context):
        p = get(context)
        if p is None:
            self.report({"ERROR"}, "preferences unavailable")
            return {"CANCELLED"}
        changed = []
        for key, prop in p.bl_rna.properties.items():
            # `bl_*` and `rna_type` are registration metadata, not settings, and writing one
            # is not a harmless no-op: setting `bl_idname` back to its "default" (empty)
            # rebinds the AddonPreferences and restores every value from userpref.blend --
            # including the sidecar paths this is careful to skip. It looked for a long time
            # like the skip list was broken, and the skip list was correct throughout.
            if (key in RESET_KEEP or prop.is_readonly
                    or key.startswith("bl_") or key in ("rna_type", "name")):
                continue
            default = getattr(prop, "default", None)
            if default is None:
                continue
            try:
                now = getattr(p, key)
            except AttributeError:
                continue
            if now == default:
                continue
            try:
                setattr(p, key, default)
            except (AttributeError, TypeError, ValueError):
                # A property that refuses its own advertised default is worth knowing about
                # rather than swallowing, but it must not stop the rest being reset.
                print("[assist] could not reset %r" % key)
                continue
            changed.append("%s %r -> %r" % (key, now, default))
        for line in changed:
            print("[assist] reset %s" % line)
        if not changed:
            self.report({"INFO"}, "already at the shipped defaults")
        else:
            self.report({"INFO"}, "%d setting(s) back to default (see console); sidecar "
                                  "paths left alone" % len(changed))
        return {"FINISHED"}


def get(context):
    try:
        return context.preferences.addons[__package__].preferences
    except KeyError:
        return None


CLASSES = (BtrAssistPrefs, CLIP_OT_btr_reset_prefs)
