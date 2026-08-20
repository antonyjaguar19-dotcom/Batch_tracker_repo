"""3DE 2D-track ASCII, and the coordinate conventions that decide whether it is right.

Lifted from `experiments/blender_track/addon_3de_io.py`, which is a working addon in its
own right. Kept as plain functions here so the operators, the tests and any future code can
share one copy.

Two conventions, both easy to get backwards:

  * **Both formats are y-up.** Blender's normalised clip coordinates put (0,0) at the
    BOTTOM-left, and so does 3DE. So there is NO vertical flip here. Image files are
    y-down, which is why code elsewhere in this project that touches pixels does flip --
    that is a different conversion, not this one.
  * **Half-pixel centres.** A pixel's centre, not its corner, is the position:
    `x = u * width - 0.5`, and back `u = (x + 0.5) / width`. Dropping this puts every track
    half a pixel out, which is invisible on screen and ruins a reference.

Gaps are legal in 3DE -- a track carries its own frame numbers, so an occluded feature
survives as a hole rather than being deleted. Both directions preserve that.
"""

import bpy


def active_tracks(clip):
    """The track collection to work on -- the active tracking object's, not the clip's.

    A clip can hold several tracking objects (the camera solve plus one per moving object).
    `clip.tracking.tracks` is only the camera's, so on an object track it would silently
    read and write the wrong set.
    """
    obj = clip.tracking.objects.active
    return obj.tracks if obj is not None else clip.tracking.tracks


def uv_to_px(u, v, w, h):
    return u * float(w) - 0.5, v * float(h) - 0.5


def px_to_uv(x, y, w, h):
    return (x + 0.5) / float(w), (y + 0.5) / float(h)


def safe_name(name):
    """3DE ASCII is whitespace-delimited, so a name with a space in it corrupts the file.

    The reader would take "my track" as a name followed by a colour id, and every field
    after it shifts by one -- a file that parses successfully and is entirely wrong.
    """
    return "_".join(str(name).split()) or "track"


def read_3de(path):
    """<count>, then per track: name, colour id, sample count, then `frame x y` lines."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        tok = fh.read().split()
    if not tok:
        raise ValueError("file is empty")
    i = 0
    n = int(tok[i]); i += 1
    out = []
    for _ in range(n):
        name = tok[i]; i += 1
        i += 1                                        # colour id, unused
        count = int(tok[i]); i += 1
        pts = []
        for _ in range(count):
            pts.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2])))
            i += 3
        out.append((name, pts))
    return out


def write_3de(path, tracks):
    """`tracks` is [(name, [(frame, x, y), ...]), ...]. Matches the bot's own writer."""
    def sort_key(item):
        try:
            return (0, int(item[0]), "")
        except ValueError:
            return (1, 0, str(item[0]))

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("%d\n" % len(tracks))
        for name, pts in sorted(tracks, key=sort_key):
            fh.write("%s\n0\n%d\n" % (safe_name(name), len(pts)))
            for fr, x, y in pts:
                fh.write("%d %.12f %.12f\n" % (int(fr), float(x), float(y)))


def collect(clip, selected_only=False, skip_muted=True, offset=0):
    """Blender markers -> 3DE samples.

    Iterates the marker collection directly rather than asking for a marker per frame.
    `markers.find_frame(f)` without `exact=True` returns the NEAREST marker instead of
    nothing, so a frame-by-frame loop would invent samples across every gap and produce a
    track with no holes that never existed.
    """
    w, h = clip.size
    out = []
    for tr in active_tracks(clip):
        if selected_only and not tr.select:
            continue
        pts = []
        for m in tr.markers:
            if skip_muted and m.mute:
                continue
            x, y = uv_to_px(m.co[0], m.co[1], w, h)
            pts.append((m.frame + offset, x, y))
        if pts:
            pts.sort()
            out.append((tr.name, pts))
    return out


def delete_all_tracks(context, clip):
    """`tracks.remove()` does not exist.

    `MovieTrackingTracks` is add-only from RNA on 5.2 -- `tracks.new()` is there,
    `tracks.remove()` raises `AttributeError: bpy_prop_collection: attribute "remove" not
    found`. Measured 2026-08-20; the original addon_3de_io.py's "Delete existing tracks"
    option hits exactly this. Deletion is the operator, which needs a CLIP_EDITOR context.
    """
    tracks = active_tracks(clip)
    if not len(tracks):
        return 0
    n = len(tracks)
    for tr in tracks:
        tr.select = True
        tr.select_anchor = True
        tr.select_pattern = True
        tr.select_search = True
    win, area, region = clip_editor(context, clip)
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=area.spaces.active,
                                   edit_movieclip=clip, scene=context.scene):
        bpy.ops.clip.delete_track()
    return n


def clip_editor(context, clip):
    """A CLIP_EDITOR window/area/region to override operator calls with.

    Prefers the area the artist is actually looking at. Unlike the headless path
    (`bl_track.clip_editor`) this must NOT retype somebody's area out from under them --
    if there is no clip editor open, that is an error to report, not a layout to rearrange.
    """
    win = context.window
    area = getattr(context, "area", None)
    if area is None or area.type != "CLIP_EDITOR":
        area = None
        for w in context.window_manager.windows:
            for a in w.screen.areas:
                if a.type == "CLIP_EDITOR" and a.spaces.active.clip == clip:
                    win, area = w, a
                    break
            if area is not None:
                break
    if area is None:
        raise RuntimeError("no Movie Clip Editor showing this clip")
    region = None
    for r in area.regions:
        if r.type == "WINDOW":
            region = r
    if region is None:
        raise RuntimeError("clip editor has no window region")
    return win, area, region
