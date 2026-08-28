"""The assistant's own operator must get past its setup.

Written after it did not. A refactor put a `continue` in the loop that builds the record
list, one line above the `append` -- so with the new default on, every selected track was
skipped, the list came back empty, and the operator refused with "the selected tracks have
no usable markers" on every shot.

The whole suite passed while that was true. Twenty-two tests exercised `track_core`, the
sidecar, the panels, the overlay and the mark-mode operator, and not one of them ran the
setup of `clip.btr_assist_track` -- the thing the artist actually presses.

`bpy.ops.clip.btr_assist_track('INVOKE_DEFAULT')` cannot do it: a modal operator invoked
under `--background` returns {'PASS_THROUGH'} without running its body, so it reports success
on code that is completely broken. The operator is therefore driven the way
`test_confirm_keys` drives its methods -- a real instance, its properties filled from the
RNA defaults, `invoke` called directly. `modal_handler_add` at the end fails without a real
window, and that is fine: everything this test cares about has happened by then.

Both settings of `blender_tracking` are run, because the bug lived in one branch of an `if`.

    blender.exe --background -noaudio --python tests/test_assist_start.py -- \
        --plate D:/Jefrin/IN/SH006.mp4
"""

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
EXT = "bl_ext.user_default.btr_assist"
PAT = 41.2

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[st] %-58s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


class Fake:
    """Stands in for the operator so `invoke` can be called outside a modal context.

    A registered operator class cannot be instantiated from Python (`bpy_struct.__new__`
    refuses), so this holds the properties at their registered defaults and forwards every
    method lookup to the real class, bound to itself. Same device as `test_confirm_keys`.
    """

    def __init__(self, cls):
        object.__setattr__(self, "_cls", cls)
        self.reported = None
        rna = bpy.ops.clip.btr_assist_track.get_rna_type()
        for prop in rna.properties:
            if prop.identifier == "rna_type" or prop.identifier.startswith("bl_"):
                continue
            try:
                setattr(self, prop.identifier, prop.default)
            except (AttributeError, TypeError):
                pass

    def __getattr__(self, name):
        v = getattr(object.__getattribute__(self, "_cls"), name)
        return v.__get__(self, type(self)) if callable(v) and not isinstance(v, type) else v

    def report(self, level, msg):
        self.reported = (set(level), msg)


def make_op(cls):
    return Fake(cls)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""
    if not plate or not os.path.exists(plate):
        print("[st] need --plate")
        return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    oa = importlib.import_module(EXT + ".ops_assist")
    tc = importlib.import_module(EXT + ".track_core")
    prefs = importlib.import_module(EXT + ".prefs")

    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    w, h = clip.size
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    sp = area.spaces.active
    sp.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")

    tr = clip.tracking.tracks.new(name="SEED", frame=1)
    tc._select(tr, True)
    tr.markers[0].mute = False
    tc.set_geom(tr.markers[0], PAT, PAT * 3.0, w, h)

    for blender_tracking in (True, False):
        label = "on " if blender_tracking else "off"
        op = make_op(oa.CLIP_OT_btr_assist_track)
        op.blender_tracking = blender_tracking
        with bpy.context.temp_override(window=win, area=area, region=region,
                                       space_data=sp, edit_movieclip=clip):
            p = prefs.get(bpy.context)
            if p is not None and not p.assist_root:
                p.assist_root = ASSIST
            try:
                res = op.invoke(bpy.context, None)
            except Exception as exc:                                  # noqa: BLE001
                # A raise past `modal_handler_add` still means setup ran; anything earlier
                # is the failure this test exists for, so the records are what decides.
                res = {"RAISED: %s" % exc}
        recs = getattr(op, "_records", None) or []
        print("[st] blender_tracking %s -> %r, %d record(s)%s"
              % (label, res, len(recs),
                 "" if not op.reported else "  reported %s" % (op.reported[1],)))
        check("with Blender's settings %s the selected track is usable" % label,
              len(recs), 1)
        check("  and setup did not refuse", "CANCELLED" in res, False)

    print("")
    if FAILED:
        print("ASSIST START: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[st] ASSIST START: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
