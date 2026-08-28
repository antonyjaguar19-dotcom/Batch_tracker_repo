"""The confirm prompt AND the live status, drawn IN the clip editor.

A modal operator that waits for a keypress and announces it only in the status bar reads as
a frozen Blender. The status line is at the bottom edge of the window; the artist is looking
at the marker in the middle of it, and until they press the right key nothing they click
responds. This puts the question where they are already looking.

It also carries the running status, because "working, idle or stuck" was impossible to tell
apart: the status strip is at the bottom edge of the window, the artist is looking at the
marker in the middle, and a long sidecar job looks exactly like a hung one. Three things fix
that and all three are drawn here -- what it is doing, how long it has been doing it, and a
HEARTBEAT that moves on every redraw. A frozen heartbeat means the operator itself is gone; a
moving heartbeat with a stalled elapsed time means it is waiting on something and says so.

`_draw` must never raise. A draw callback that throws does so on EVERY redraw, buries the
console, and unlike a panel there is nothing to collapse -- the handler outlives the
exception and keeps firing. `tests/test_overlay_draw.py` calls it with the module in every
state it can be in.
"""

import blf
import bpy

_HANDLE = None
_LINES = []
#: "prompt" (a question, gold) or "status" (progress, blue). Different colours because they
#: demand different things: one needs an answer, the other only needs to be believed.
_MODE = "prompt"
#: Frames of the heartbeat. Drawn from the clock rather than a counter so it keeps moving
#: even when nothing else changes -- which is the whole point of it.
_BEAT = "|/-\\"

#: The first line is the question, the rest are the keys. Only the question is highlighted.
_HEAD_RGBA = (1.0, 0.85, 0.25, 1.0)
_BODY_RGBA = (0.90, 0.90, 0.90, 1.0)
_STAT_RGBA = (0.45, 0.80, 1.00, 1.0)
#: A stall this long is worth saying out loud rather than leaving the artist to guess.
STALL_SECONDS = 15.0


def show(lines, mode="prompt"):
    """Put `lines` on every clip editor, adding the handler on first use."""
    global _HANDLE, _MODE
    _MODE = mode
    _LINES[:] = [str(x) for x in lines]
    if _HANDLE is None:
        _HANDLE = bpy.types.SpaceClipEditor.draw_handler_add(
            _draw, (), "WINDOW", "POST_PIXEL")
    tag_redraw()


def status(lines):
    """The running status: same surface as the prompt, different colour, with a heartbeat."""
    show(lines, mode="status")


def hide():
    """Remove the handler. Safe to call when it was never added, and twice."""
    global _HANDLE
    _LINES[:] = []
    if _HANDLE is not None:
        try:
            bpy.types.SpaceClipEditor.draw_handler_remove(_HANDLE, "WINDOW")
        except (ValueError, RuntimeError, TypeError):
            pass
        _HANDLE = None
    tag_redraw()


def tag_redraw():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == "CLIP_EDITOR":
                    area.tag_redraw()
    except (AttributeError, RuntimeError):
        # No window manager (headless), or context not available during a handler.
        pass


def _draw():
    try:
        if not _LINES:
            return
        font = 0
        try:
            ui = bpy.context.preferences.system.ui_scale
        except AttributeError:
            ui = 1.0
        size = 13.0 * ui
        pad = 12.0 * ui
        line_h = size * 1.6
        blf.size(font, size)
        import time                                                   # noqa: PLC0415
        head = _LINES[0]
        if _MODE == "status":
            # Appended at DRAW time, not when the status was set, so it turns whenever
            # Blender redraws. If this stops turning, the operator is not running.
            head = "%s  %s" % (_BEAT[int(time.time() * 4) % len(_BEAT)], head)
        # Drawn from the BOTTOM-left of the region upwards, so the block never covers the
        # marker being judged -- which is centred, because `_show_current` framed it.
        y = pad + line_h * (len(_LINES) - 1)
        for i, text in enumerate([head] + _LINES[1:]):
            blf.color(font, *((_STAT_RGBA if _MODE == "status" else _HEAD_RGBA)
                              if i == 0 else _BODY_RGBA))
            blf.position(font, pad, y - i * line_h, 0.0)
            blf.draw(font, text)
    except Exception:                    # noqa: BLE001 -- see the module docstring
        pass
