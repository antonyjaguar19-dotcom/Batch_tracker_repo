"""Drive the confirm overlay's draw callback against the REAL installed extension.

A draw handler that raises is worse than a panel that raises: it fires on every redraw and
there is nothing to collapse to stop it. It is registered from a modal operator, so the
exception arrives while the artist is already mid-question.

`blf` is STUBBED. Calling the real one outside a GPU draw context does not raise, it takes
the process down with EXCEPTION_ACCESS_VIOLATION -- measured, headless, on 5.2. So this
tests the PYTHON in `_draw` (the indexing, the colour choice, the scale lookup, the
stringifying), which is where a crash would come from, and does not pretend to test
Blender's text renderer. The stub also records its calls, so the arguments are checked
rather than merely survived.

What IS checked against the real module: that `blf` still has the four functions `_draw`
calls, so an API change is caught here instead of on an artist's screen.

    blender.exe --background -noaudio --python tests\test_overlay_draw.py
"""

import sys
import traceback

import bpy

EXT = "bl_ext.user_default.btr_assist"

FAILURES = []


def log(m):
    print("[overlay] %s" % m, flush=True)


def fail(m):
    FAILURES.append(m)
    log("FAIL %s" % m)


class StubBlf:
    """Records calls and returns None, like the real module."""

    def __init__(self):
        self.calls = []

    def size(self, font, size):
        self.calls.append(("size", font, size))

    def color(self, font, r, g, b, a):
        self.calls.append(("color", font, r, g, b, a))

    def position(self, font, x, y, z):
        self.calls.append(("position", font, x, y, z))

    def draw(self, font, text):
        if not isinstance(text, str):
            raise TypeError("blf.draw got %r, not a str" % type(text).__name__)
        self.calls.append(("draw", font, text))


def check(name, fn):
    try:
        fn()
    except Exception as exc:                        # noqa: BLE001
        traceback.print_exc()
        fail("%s: %s: %s" % (name, type(exc).__name__, exc))
        return False
    log("%-26s ok" % name)
    return True


def main():
    bpy.ops.preferences.addon_enable(module=EXT)
    overlay = sys.modules["%s.overlay" % EXT]

    import blf as real_blf
    for fn in ("size", "color", "position", "draw"):
        if not hasattr(real_blf, fn):
            fail("blf has no %s() -- _draw calls it" % fn)
    log("%-26s ok" % "blf api present")

    stub = StubBlf()
    overlay.blf = stub

    def drawn():
        return [c[2] for c in stub.calls if c[0] == "draw"]

    # Never shown: no lines and no handler. Must be a no-op, not an IndexError.
    check("draw before show", overlay._draw)
    if drawn():
        fail("drew %r before anything was shown" % (drawn(),))

    check("show", lambda: overlay.show(["head", "keys", "hint"]))
    if overlay._HANDLE is None:
        fail("show did not register the handler")
    stub.calls = []
    check("draw while shown", overlay._draw)
    if drawn() != ["head", "keys", "hint"]:
        fail("drew %r, expected the three lines in order" % (drawn(),))
    # The question is highlighted and the rest is not -- that is the whole point of having
    # three lines rather than one.
    colours = [c[1:] for c in stub.calls if c[0] == "color"]
    if len(colours) != 3 or colours[0] == colours[1] or colours[1] != colours[2]:
        fail("colour pattern wrong: %r" % (colours,))
    # Bottom-left upwards: first line highest, and every line on screen (y > 0).
    ys = [c[3] for c in stub.calls if c[0] == "position"]
    if not (len(ys) == 3 and ys[0] > ys[1] > ys[2] > 0):
        fail("lines not stacked upward off the bottom edge: %r" % (ys,))

    # Non-str content: `show` stringifies, so a caller passing a number must not surface as
    # a TypeError inside blf forty redraws later, pointing at the wrong place.
    check("show non-str", lambda: overlay.show([1, None, 2.5]))
    stub.calls = []
    check("draw non-str", overlay._draw)
    if drawn() != ["1", "None", "2.5"]:
        fail("did not stringify: %r" % (drawn(),))

    # ---- the STATUS mode, which is a different colour and carries a heartbeat ----------
    # It exists because working, idle and stuck looked identical to the artist. The heartbeat
    # indexes a string from the clock inside _draw, which is exactly the sort of thing that
    # works until it does not -- and a draw callback that throws does so on EVERY redraw.
    stub.calls = []
    check("status", lambda: overlay.status(["tracking in Blender  (3s)", "round 1"]))
    check("draw while status is up", overlay._draw)
    got = drawn()
    if len(got) != 2 or "tracking in Blender" not in got[0]:
        fail("status drew %r" % (got,))
    if got[0] == "tracking in Blender  (3s)":
        fail("no heartbeat was prepended: %r" % (got[0],))
    stub.calls = []
    check("status with a stall note",
          lambda: overlay.status(["looking for the feature again  (48s)",
                                  "CoTracker: tracking 1 point(s)",
                                  "no change for 31s -- still working, Esc stops it"]))
    check("draw with three status lines", overlay._draw)
    if len(drawn()) != 3:
        fail("stall status drew %r" % (drawn(),))
    check("status with no lines at all", lambda: overlay.status([]))
    check("draw with an empty status", overlay._draw)
    check("a prompt after a status", lambda: overlay.show(["a question", "ENTER  yes"]))
    check("draw the prompt again", overlay._draw)

    check("hide", overlay.hide)
    if overlay._HANDLE is not None:
        fail("hide left the handler registered")
    stub.calls = []
    check("draw after hide", overlay._draw)
    if drawn():
        fail("drew %r after hide" % (drawn(),))

    # Double hide is the NORMAL case: `_finish` hides, and `_start_tracking` may already
    # have hidden on the way there.
    check("hide twice", overlay.hide)
    check("draw after double hide", overlay._draw)

    # Re-show must register a NEW handler, or the second confirm round is invisible --
    # exactly the failure this overlay exists to fix.
    check("re-show", lambda: overlay.show(["again"]))
    if overlay._HANDLE is None:
        fail("re-show did not re-register the handler")
    stub.calls = []
    check("draw after re-show", overlay._draw)
    if drawn() != ["again"]:
        fail("re-show drew %r" % (drawn(),))
    overlay.hide()

    overlay.blf = real_blf
    log("OVERLAY DRAW: %s" % ("FAIL" if FAILURES else "PASS"))
    sys.exit(1 if FAILURES else 0)


main()
