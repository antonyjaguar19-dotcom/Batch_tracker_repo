"""The engine choice actually routes, and the fallback actually falls back.

The measurement that CoTracker-primary is worth having lives in `eval_track_vs_manual.py
--engine cotracker`, against hand tracks. This is the other question, and it is the one a
unit test can answer: does picking the engine change which code runs, and does an
unavailable sidecar quietly go back to Blender rather than failing the run.

That second half matters more than it looks. A secondary engine that only works when the
primary already does is not a fallback, and the failure would show up as a run that stops
with no tracks -- on a shot, in front of an artist, not here.

    blender.exe --background -noaudio --python tests/test_engine.py -- \\
        --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "bl_ext.user_default.btr_assist"
FAILED = []


def check(name, got, want):
    ok = got == want
    print("[eng] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def truthy(name, got):
    ok = bool(got)
    print("[eng] %-56s %s" % (name, "ok" if ok else "FAIL  got %r" % (got,)))
    if not ok:
        FAILED.append(name)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""
    if not plate or not os.path.exists(plate):
        print("[eng] need --plate")
        return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    mod = importlib.import_module(EXT)
    ops_assist = importlib.import_module(EXT + ".ops_assist")
    print("[eng] installed %s build %s" % (mod.VERSION, mod.BUILD))

    # The REGISTERED operator, not the class object. `bpy.types.CLIP_OT_...` and the class in
    # the module both carry the annotations and neither carries the registered RNA, so
    # reading properties off them reports every property missing -- which is what this test
    # did first, and it failed loudly for entirely the wrong reason.
    props = bpy.ops.clip.btr_assist_track.get_rna_type().properties
    truthy("the operator carries an engine choice", "track_engine" in props)
    items = [i.identifier for i in props["track_engine"].enum_items]
    check("both engines are offered", sorted(items), ["BLENDER", "COTRACKER"])
    check("Blender is the default, so nothing changes unasked",
          props["track_engine"].default, "BLENDER")

    p = bpy.context.preferences.addons[EXT].preferences
    truthy("and the preference exists to make it stick", hasattr(p, "track_engine"))

    cls = ops_assist.CLIP_OT_btr_assist_track
    truthy("there is a CoTracker phase to route into", hasattr(cls, "_start_ctrack"))
    truthy("  and a tick for it", hasattr(cls, "_tick_ctrack"))
    # A phase nothing ticks is a hang, not an error -- so the router has to know the name.
    import inspect
    src = inspect.getsource(cls.modal)
    truthy("  and modal() routes the phase by name", "ctracking" in src)

    print("\n[eng] the fallback, with the sidecar deliberately unreachable")
    # The whole point of a secondary engine. `_start_ctrack` must answer False -- not raise,
    # not leave the operator in a phase nothing ticks -- so `_start_tracking` carries on into
    # Blender exactly as it would have without the choice ever being offered.
    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    area.spaces.active.clip = clip

    # An Operator cannot be instantiated from Python -- bpy_struct.__new__ refuses -- so the
    # method is called unbound against a stand-in carrying only what it reads. That is enough
    # here because the fallback path is deliberately the part that touches nothing else.
    class Stub(object):
        track_engine = "COTRACKER"
        min_match = 0.66
        pin_radius = 12.0

        def _status(self, *a, **k):
            pass

    fake = Stub()
    fake._clip = clip
    fake._records = []
    fake._patterns = {}
    fake._root = os.path.join(HERE, "no-such-root")
    fake._n_frames = clip.frame_duration
    start = ops_assist.CLIP_OT_btr_assist_track._start_ctrack
    try:
        got = start(fake, bpy.context)
        check("no tracks to send -> falls through to Blender", got, False)
    except Exception as exc:                                          # noqa: BLE001
        check("no tracks to send -> falls through to Blender", "raised %r" % (exc,), False)

    # And with a track that HAS a pattern but no reachable sidecar.
    tr = clip.tracking.tracks.new(name="E", frame=1)
    fake._records = [{"id": "E", "t": tr, "alive": True}]
    fake._patterns = {"E": {"frame": 1, "cx": 100.0, "cy": 100.0, "w": 40.0, "h": 40.0}}
    try:
        got = start(fake, bpy.context)
        check("unreachable sidecar -> falls through to Blender", got, False)
    except Exception as exc:                                          # noqa: BLE001
        check("unreachable sidecar -> falls through to Blender", "raised %r" % (exc,), False)

    print("")
    if FAILED:
        print("ENGINE: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[eng] ENGINE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
