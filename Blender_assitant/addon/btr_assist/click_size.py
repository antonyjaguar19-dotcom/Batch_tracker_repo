"""Make a Ctrl-clicked marker the same size ON SCREEN whatever the zoom.

Blender sizes a new track from `clip.tracking.settings.default_pattern_size`, which is a
count of PLATE pixels. That is a fixed number, and the plate is not: 21 px is a sixth of the
width of a 128-px proxy and 0.55 % of a 3840-px plate. Fit a 4K plate into a clip editor and
the zoom is around 26 %, so the box Blender draws for a new marker is **five or six screen
pixels** -- too small to see what you seeded, let alone judge it. The artist's workaround was
to zoom in, drag the box bigger, and zoom back out, once per track.

What is wanted is the opposite constant: a box that is always the same size in the VIEWPORT,
so it is worth looking at at any zoom. Plate pixels and screen pixels are related by the zoom
alone --

    plate_px = screen_px / (zoom_percentage / 100)

-- so the setting Blender already uses can simply be kept up to date. Nothing here
intercepts the click, replaces `clip.add_marker`, or touches the keymap: Ctrl-click stays
Blender's own operator with its own drag-to-place behaviour, and it reads a default that
happens to be correct by the time it runs.

**Why a timer and not a draw handler.** The zoom is only known from a region, and the
obvious place to read one is a draw callback -- which runs on every zoom, for free. But a
draw callback must not write data, and this writes two RNA properties. A timer runs in a
context where that is allowed. The cost of the timer is a loop over open areas at 5 Hz that
usually decides nothing has changed.

**The setting belongs to the artist.** `default_pattern_size` is a real preference someone
may have set deliberately, and this overwrites it. So the value in force when a clip is
first touched is remembered and put back when the option is switched off or the addon is
unregistered -- taking a setting over for the session is defensible, keeping it afterwards
is not.
"""

import bpy

#: How often to reconcile the defaults with the zoom. The number only has to beat the gap
#: between the artist stopping a zoom and clicking, so this is a latency budget rather than a
#: sampling rate: at 0.2 s a click that lands within 200 ms of a scroll can still get the
#: previous size. Faster costs a no-op loop over open areas, which is why it is not 0.01.
INTERVAL = 0.2

#: Below this many plate pixels a pattern holds too little texture to correlate, whatever it
#: looks like on screen -- at a 400 % zoom the screen-size rule alone would ask for 10.
MIN_PLATE_PX = 16.0

#: And a pattern may not eat the plate. A box a quarter of the short edge is already far past
#: anything an artist would place by hand; the clamp exists for the zoomed-way-out case where
#: the arithmetic would otherwise ask for a box bigger than the frame.
MAX_PLATE_FRAC = 0.25

#: Search box as a multiple of the pattern. Blender's own defaults are 21 and 71, and this
#: keeps that proportion rather than inventing one -- the search box that actually matters is
#: the one `track_core.refit_search` computes from measured plate motion at track time, not
#: the one a fresh marker is born with.
SEARCH_RATIO = 71.0 / 21.0

#: clip name -> (pattern, search) as the artist had them before this started.
_saved = {}
_running = False


def _clip_editors():
    """Every clip editor currently on screen, with the region the zoom applies to."""
    wm = bpy.context.window_manager
    for win in getattr(wm, "windows", ()) or ():
        scr = getattr(win, "screen", None)
        if scr is None:
            continue
        for area in scr.areas:
            if area.type != "CLIP_EDITOR":
                continue
            space = area.spaces.active
            if space is None or space.clip is None:
                continue
            yield space


def plate_px_for(screen_px, zoom_percentage, clip_size):
    """Plate pixels that occupy `screen_px` on screen at this zoom, clamped to be usable.

    Split out from the timer so it can be tested without a window, a clip or a running
    Blender -- the arithmetic is the part that can be wrong, and it is one line surrounded by
    two clamps that each exist for a measured reason.
    """
    zoom = float(zoom_percentage or 0.0) / 100.0
    if zoom <= 0.0:
        return None
    want = float(screen_px) / zoom
    short = float(min(clip_size)) if clip_size else 0.0
    hi = short * MAX_PLATE_FRAC if short > 0 else want
    return max(MIN_PLATE_PX, min(hi, want))


def artist_default(clip):
    """The new-track box size as the ARTIST set it, ignoring anything this module did.

    Exists because `default_pattern_size` stopped being a preference the moment this started
    writing it, and one thing still reads it as one: importing 3DE tracks creates them from
    the default, and the QC pass then correlates using that box. Without this, a track
    imported while zoomed out would carry a 540 px pattern and QC would be reading half a
    building. Returns (pattern, search) in plate pixels.
    """
    st = clip.tracking.settings
    return _saved.get(clip.name, (st.default_pattern_size, st.default_search_size))


def _apply():
    from . import prefs                                               # noqa: PLC0415
    p = prefs.get(bpy.context)
    if p is None or not getattr(p, "constant_box", False):
        # Keep the timer alive rather than tearing it down: the option can be switched back
        # on, and a tick that decides nothing costs a dictionary lookup.
        return INTERVAL

    want_screen = float(getattr(p, "box_screen_px", 40))
    for space in _clip_editors():
        clip = space.clip
        st = clip.tracking.settings
        _saved.setdefault(clip.name, (st.default_pattern_size, st.default_search_size))
        px = plate_px_for(want_screen, getattr(space, "zoom_percentage", 0.0), clip.size)
        if px is None:
            continue
        pat = int(round(px))
        sea = int(round(px * SEARCH_RATIO))
        # Only on a real change. Writing the same value back every tick tags the ID as
        # modified, which marks the .blend dirty and asks the artist to save work that did
        # not happen.
        if st.default_pattern_size != pat:
            st.default_pattern_size = pat
        if st.default_search_size != sea:
            st.default_search_size = sea
    return INTERVAL


def start():
    global _running
    if _running:
        return
    _running = True
    bpy.app.timers.register(_apply, first_interval=INTERVAL, persistent=True)


def stop(restore=True):
    """Stop reconciling, and hand the artist's own settings back."""
    global _running
    _running = False
    try:
        bpy.app.timers.unregister(_apply)
    except (ValueError, TypeError):
        pass
    if not restore:
        _saved.clear()
        return
    for name, (pat, sea) in list(_saved.items()):
        clip = bpy.data.movieclips.get(name)
        if clip is None:
            continue
        st = clip.tracking.settings
        if st.default_pattern_size != pat:
            st.default_pattern_size = pat
        if st.default_search_size != sea:
            st.default_search_size = sea
    _saved.clear()


CLASSES = ()
