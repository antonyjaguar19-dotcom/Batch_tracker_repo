"""What the confirm phase is allowed to swallow, against the REAL installed extension.

The confirm phase used to consume EVERY event to protect four keys. The artist gets asked
"is that your feature?", and then cannot zoom, pan, click, or move the playhead to answer
it -- with the only hint in the status bar at the bottom edge of the window. It reads as a
hung Blender, and was reported as one.

So the rule is now narrow and worth pinning: consume the four answer keys, pass everything
else through. `SPACE` is in the pass-through list on purpose -- it used to ACCEPT, and now
that navigation reaches the editor an artist pressing it to play the shot would otherwise
silently accept the proposal they were about to look at.

`_tick_confirm` is called UNBOUND with a stand-in `self`, the same way `test_panels_draw`
calls `draw()`: `bpy_struct.__new__` refuses to make an Operator instance, and the method
touches only plain attributes.

    blender.exe --background -noaudio --python tests\test_confirm_keys.py
"""

import sys
import traceback

import bpy

EXT = "bl_ext.user_default.btr_assist"

#: Everything an artist does to LOOK at a marker, plus the keys that used to be stolen.
PASS_THROUGH = ("MOUSEMOVE", "INBETWEEN_MOUSEMOVE", "MIDDLEMOUSE", "LEFTMOUSE",
                "RIGHTMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN",
                "TRACKPADZOOM", "NUMPAD_PLUS", "NUMPAD_MINUS", "LEFT_ARROW",
                "RIGHT_ARROW", "UP_ARROW", "DOWN_ARROW", "SPACE", "G", "S", "TAB")

FAILURES = []


def log(m):
    print("[confirm] %s" % m, flush=True)


def fail(m):
    FAILURES.append(m)
    log("FAIL %s" % m)


class Event:
    def __init__(self, type_, value="PRESS"):
        self.type = type_
        self.value = value


class FakeSelf:
    """Only what `_tick_confirm` reaches for. Records which exit it took."""

    def __init__(self, op, awaiting):
        self._op = op
        self._awaiting = list(awaiting)
        self._records = [{"id": a["id"], "alive": True} for a in awaiting]
        self.went = None
        self.dropped = []

    def _show_current(self, context):
        self.went = "show"
        return {"RUNNING_MODAL"}

    def _start_tracking(self, context):
        self.went = "track"

    def _done(self, context, why):
        self.went = "done:%s" % why
        return {"FINISHED"}

    def _drop_resume(self, name, frame):
        self.dropped.append((name, frame))
        for r in self._records:
            if r["id"] == name:
                r["alive"] = False


def main():
    bpy.ops.preferences.addon_enable(module=EXT)
    ops_assist = sys.modules["%s.ops_assist" % EXT]
    tick = ops_assist.CLIP_OT_btr_assist_track._tick_confirm

    two = [{"id": "a", "frame": 10, "score": 0.8},
           {"id": "b", "frame": 20, "score": 0.7}]

    # 1. Everything that is not an answer key reaches the clip editor.
    for key in PASS_THROUGH:
        for value in ("PRESS", "RELEASE", "NOTHING"):
            s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, two)
            try:
                r = tick(s, None, Event(key, value))
            except Exception as exc:                # noqa: BLE001
                traceback.print_exc()
                fail("%s/%s raised %s" % (key, value, type(exc).__name__))
                continue
            if r != {"PASS_THROUGH"}:
                fail("%s/%s returned %r, not PASS_THROUGH" % (key, value, r))
            if len(s._awaiting) != 2 or s.went is not None:
                fail("%s/%s changed state" % (key, value))
    log("%-30s ok (%d keys x 3 values)" % ("navigation passes through", len(PASS_THROUGH)))

    # 2. ENTER accepts one and moves to the next.
    s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, two)
    r = tick(s, None, Event("RET"))
    if r != {"RUNNING_MODAL"} or s.went != "show" or [a["id"] for a in s._awaiting] != ["b"]:
        fail("RET: %r went=%s left=%r" % (r, s.went, [a["id"] for a in s._awaiting]))
    else:
        log("%-30s ok" % "ENTER accepts one")

    # 3. A key we consume must swallow its RELEASE too, or it leaks into the editor.
    s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, two)
    if tick(s, None, Event("RET", "RELEASE")) != {"RUNNING_MODAL"} or len(s._awaiting) != 2:
        fail("RET/RELEASE was not swallowed cleanly")
    else:
        log("%-30s ok" % "answer key release swallowed")

    # 4. D drops the proposal it is showing, and only that one.
    s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, two)
    tick(s, None, Event("D"))
    if s.dropped != [("a", 10)] or [a["id"] for a in s._awaiting] != ["b"]:
        fail("D dropped %r, left %r" % (s.dropped, [a["id"] for a in s._awaiting]))
    else:
        log("%-30s ok" % "D drops the shown one")

    # 5. A accepts the rest unread and goes back to tracking.
    s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, two)
    tick(s, None, Event("A"))
    if s._awaiting or s.went != "track":
        fail("A: left=%r went=%s" % (s._awaiting, s.went))
    else:
        log("%-30s ok" % "A accepts all")

    # 6. Dropping every proposal leaves nothing to track, and must not start a pass.
    s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, [two[0]])
    tick(s, None, Event("D"))
    if s.went != "done:every proposal dropped":
        fail("dropping the last one went to %r" % (s.went,))
    else:
        log("%-30s ok" % "last drop ends the run")

    # 7. SPACE must NOT accept. Pinned separately because it used to.
    s = FakeSelf(ops_assist.CLIP_OT_btr_assist_track, two)
    if tick(s, None, Event("SPACE")) != {"PASS_THROUGH"} or len(s._awaiting) != 2:
        fail("SPACE still accepts -- it is playback")
    else:
        log("%-30s ok" % "SPACE is playback, not accept")

    if ops_assist.ANSWER_KEYS != frozenset(("RET", "NUMPAD_ENTER", "D", "A")):
        fail("ANSWER_KEYS changed: %r" % (ops_assist.ANSWER_KEYS,))

    log("CONFIRM KEYS: %s" % ("FAIL" if FAILURES else "PASS"))
    sys.exit(1 if FAILURES else 0)


main()
