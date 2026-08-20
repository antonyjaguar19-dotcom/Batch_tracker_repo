"""Seeding and tracking, as a resumable generator.

This is `experiments/blender_track/bl_track.py`'s tracking body with one change: the
per-frame loop `yield`s instead of running to completion. Two drivers share it --

    headless:   for _ in track_job(...): pass
    foreground: a modal operator drains it for ~50 ms per timer tick

-- so the interactive path and the batch path are the same code, and the measured numbers
(2.20 px vs hand tracks, 1.0-2.3 deaths/track) carry over rather than needing to be earned
again.

Why a generator and not `frames_limit` chunking: chunking moves the leash checkpoint from
every frame to every N frames, which is a behaviour change with no measurement behind it.
Yielding changes only where Python gets control back. (`frames_limit` does work as a lever
-- measured in M0, `logs/m0_spike.json` test C -- it is simply not the right one.)

M0 also settled the operator question: `bpy.ops.clip.track_markers()` called from Python is
SYNCHRONOUS in a windowed Blender (FINISHED, 11.0 s, full 160-frame span). The modal
behaviour lives in the operator's `invoke`, which is what Blender's own Track button runs;
Python defaults to `EXEC_DEFAULT`. So this runs in the artist's Blender, in-process.
"""

import math

import bpy

# ---------------------------------------------------------------- geometry tables
# Copied from bl_track.py:43-86 rather than imported: this file runs inside Blender, where
# the repo is not importable. `tests/test_track_core_parity.py` is what keeps the two
# copies honest -- it runs both over the same seeds and diffs the output.
#
# (pattern_px, search_px, motion_model). A corner is pinned in both axes and needs no extra
# freedom; a blob may breathe with defocus; an edge point can only be located ACROSS the
# edge, so it gets a search box wide enough to survive sliding along it.
KIND_GEOM = {
    "corner":       (21, 41, "Loc"),
    "blob":         (31, 51, "LocScale"),
    "edge":         (25, 61, "Loc"),
    "dense-corner": (21, 41, "Loc"),
    "dense-edge":   (25, 61, "Loc"),
    "dense":        (25, 45, "Loc"),
    "":             (25, 45, "Loc"),
}
FLAT_GEOM = KIND_GEOM[""]

# Per-seed sizing from the seed's own measured scale. Kept available, OFF by default:
# `--flat-geom` tied it on the bench and the bot's own constants made Blender worse
# (2.02 deaths/track against 1.91), because every measurement here says Blender wants
# SMALLER boxes than the bot's native-res NCC does.
SCALE_RULE = {
    "corner": (2.0, 11, 27),
    "blob":   (2.5, 17, 37),
    "edge":   (2.0, 15, 31),
}
SEARCH_MARGIN = {"corner": 20, "blob": 20, "edge": 36, "": 20}
DENSE_MARGIN_CAP = 20


class Opts:
    """Every knob, with the value that measured best. Changing one needs a number.

    A 10-config sweep against exact ground truth put Loc/LocScale, brute, corr 0.75 on top
    (0.050 px mean / 0.080 p90 / 0.100 worst). Affine 0.300 worst, Perspective 0.360,
    bigger patterns 0.210. `experiments/blender_track/FINDINGS.md:179-185`.

    `pattern_match` is the one place where the bench and real footage disagree, and real
    footage wins. On precision alone KEYFRAME is better (0.050 px vs 0.060 px) and that is
    what the sweep reported -- but on real plates KEYFRAME dies **2.6-2.9x more often**,
    because it matches against the seed patch forever and appearance drift eventually
    breaks correlation. PREV_FRAME always compares against a nearly identical image; its
    cost is accumulated drift, which is exactly what the leash bounds (ungoverned it pushed
    drift p90 from 58 px to 79 px; with the leash it is capped at the leash). So the
    shipped default is PREV_FRAME **with `leash > 0`** -- the two are one decision, and
    setting `leash = 0` without also moving to KEYFRAME is the worst of both.
    """

    def __init__(self, **kw):
        self.correlation = 0.75
        self.pattern_match = "PREV_FRAME"   # see the class docstring before changing this
        self.motion_model = ""              # "" = per-class from KIND_GEOM
        self.flat_geom = False
        self.scale_geom = False             # ties at best; off keeps one fewer variable
        self.pattern_scale = 1.0
        self.search_scale = 1.0
        self.leash = 20.0                   # px from the guide; 0 disables the leash
        self.backwards = True
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown option %r" % k)
            setattr(self, k, v)


def _odd(v, lo, hi):
    """Pattern boxes are centred on the point, so they must be odd."""
    n = int(round(v))
    n = max(lo, min(hi, n))
    return n if n % 2 == 1 else n + 1


def geom_for(kind, scale_px=0.0, flat=False):
    if flat:
        return FLAT_GEOM
    pat, srch, model = KIND_GEOM.get(kind, KIND_GEOM[""])
    core = kind.replace("dense-", "")
    if scale_px > 0.0 and core in SCALE_RULE:
        mult, lo, hi = SCALE_RULE[core]
        pat = _odd(scale_px * mult, lo, hi)
        margin = SEARCH_MARGIN.get(core, 20)
        if kind.startswith("dense"):
            margin = min(margin, DENSE_MARGIN_CAP)
        srch = pat + margin
    return pat, srch, model


def set_geom(marker, pattern_px, search_px, w, h):
    hu, hv = pattern_px / 2.0 / w, pattern_px / 2.0 / h
    su, sv = search_px / 2.0 / w, search_px / 2.0 / h
    # Blender's corner order is bottom-left, bottom-right, top-right, top-left, and the
    # values are offsets from the marker, not absolute clip coords.
    marker.pattern_corners = ((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv))
    marker.search_min = (-su, -sv)
    marker.search_max = (su, sv)


def _select(track, on):
    track.select = on
    track.select_anchor = on
    track.select_pattern = on
    track.select_search = on


# ---------------------------------------------------------------- proxy guard

class FullResolution:
    """Force PROXY_100 for the duration of a job, then put the artist's setting back.

    A 50%% proxy may run tracking against half-resolution pixels, which halves precision
    invisibly -- the viewport looks identical and the exported numbers are quietly worse.
    No experiment in this project has ever run with a proxy, so rather than measure the
    damage the addon simply refuses to work in that state. See FINDINGS.md, test F.
    """

    def __init__(self, clip, space, enabled=True):
        self.clip, self.space, self.enabled = clip, space, enabled
        self.was = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.was = (self.clip.use_proxy,
                    self.space.clip_user.proxy_render_size)
        self.clip.use_proxy = False
        self.space.clip_user.proxy_render_size = "PROXY_100"
        return self

    def __exit__(self, *exc):
        if self.was is not None:
            self.clip.use_proxy, self.space.clip_user.proxy_render_size = self.was
        return False


# ---------------------------------------------------------------- settings + seeding

def apply_settings(clip, opts):
    """Write every tracking setting explicitly.

    The headless path gets `--factory-startup` (`blio.run_blender():81`) precisely so a
    user's preferences cannot change results. In the artist's Blender there is no such
    protection, so nothing here may be inherited -- a scene whose defaults were left on
    Affine/PREV_FRAME would silently produce the sweep's WORST configuration.
    """
    st = clip.tracking.settings
    st.use_default_brute = True             # survives fast motion; the search box is the cost
    st.use_default_normalization = True     # exposure/grain changes must not read as failure
    st.default_correlation_min = opts.correlation
    st.default_pattern_match = opts.pattern_match
    if opts.motion_model:
        st.default_motion_model = opts.motion_model


def seed_tracks(clip, seeds, opts, tracks=None):
    """Create one tracker per seed. Returns the record list the tracking loop wants.

    `MovieTrackingTrack` takes no ID properties and `tracks.new()` may uniquify a name, so
    the caller's id and geometry ride alongside the track object rather than on it.
    """
    w, h = clip.size
    if tracks is None:
        obj = clip.tracking.objects.active
        tracks = obj.tracks if obj is not None else clip.tracking.tracks
    made = []
    for s in seeds:
        kind = s.get("kind", "") or ""
        pat, srch, model = geom_for(
            kind, float(s.get("scale", 0.0)) if opts.scale_geom else 0.0, opts.flat_geom)
        pat = max(5, int(round(pat * opts.pattern_scale)))
        srch = max(pat + 4, int(round(srch * opts.search_scale)))
        model = opts.motion_model or model
        t = tracks.new(name=s["id"], frame=int(s["frame"]))
        t.motion_model = model
        t.use_brute = True
        t.use_normalization = True
        t.correlation_min = opts.correlation
        t.frames_limit = 0
        t.pattern_match = opts.pattern_match
        m = t.markers[0]
        m.co = (float(s["u"]), float(s["v"]))
        set_geom(m, pat, srch, w, h)
        made.append({"t": t, "id": s["id"], "kind": kind, "pat": pat, "srch": srch,
                     "alive": True, "w": w, "h": h, "seed_frame": int(s["frame"])})
    return made


def seed_roundtrip_error(records, seeds):
    """Max pixel distance between what was asked for and what the tracker actually holds.

    MANDATORY before believing any other number. Blender's marker lookup returns the
    NEAREST marker rather than failing, so a seed that failed to land is silently re-read
    from another frame and produces a full-length track that never touched its own feature
    -- and the synthetic bench scores that as a GOOD run, because its ground truth is
    anchored per track. Measured clean: 0.0001 px.
    """
    worst = 0.0
    by_id = {s["id"]: s for s in seeds}
    for r in records:
        s = by_id.get(r["id"])
        if s is None:
            continue
        m = r["t"].markers.find_frame(int(s["frame"]), exact=True)
        if m is None:
            return float("inf")
        dx = (m.co[0] - float(s["u"])) * r["w"]
        dy = (m.co[1] - float(s["v"])) * r["h"]
        worst = max(worst, math.hypot(dx, dy))
    return worst


# ---------------------------------------------------------------- the tracking loop

def track_group(ctx, group, frame, backwards=False, sequence=True):
    """One operator call. `sequence` tracks to failure or end; otherwise a single frame."""
    win, area, region, clip, scene = ctx
    obj = clip.tracking.objects.active
    all_tracks = obj.tracks if obj is not None else clip.tracking.tracks
    for t in all_tracks:
        _select(t, False)
    for t in group:
        _select(t, True)
    scene.frame_set(int(frame))
    # `scene.frame_set` alone IS enough in a windowed Blender -- measured in M0, test E.
    # It is NOT enough headless: with no redraw, `clip_user.frame_current` stays pinned at
    # 1 while the scene says 40, `track_markers` reads the SPACE's frame, and Blender's
    # nearest-marker fallback re-anchors the seed onto frame 1. This loop suppresses
    # redraws between frames, which is the same condition, so it is set here regardless.
    space = area.spaces.active
    space.clip_user.frame_current = int(frame)
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=space, edit_movieclip=clip, scene=scene):
        return bpy.ops.clip.track_markers(backwards=backwards, sequence=sequence)


def track_job(ctx, records, n_frames, guide, opts, backwards=False, stats=None):
    """Advance every live track one frame at a time, yielding between frames.

    `sequence=True` hands the whole shot to Blender and gets it back at the end, which is
    the fastest way to run and the reason drift goes unnoticed: nothing looks at the track
    until it has already died. Measured on SH004, by the time a track dies it sits a median
    6 px from the guide, p90 35 px, max 242 px -- tracking a different object, confidently,
    for many frames.

    Stepping costs one operator call per FRAME, not per track: every live track goes into
    the same call, so a 312-frame shot is 312 calls whatever the track count.

    With `opts.leash <= 0` this is a pure refactor of sequence mode and must reproduce its
    output; that equivalence is the correctness test for the loop itself.
    """
    win, area, region, clip, scene = ctx
    step = -1 if backwards else 1
    frames = list(range(n_frames, 1, -1) if backwards else range(1, n_frames))

    live = {}
    for r in records:
        live.setdefault(int(r.get("seed_frame", r["t"].markers[0].frame)), []).append(r)
    running = []
    st = stats if stats is not None else {}
    st.update({"deaths": 0, "clamped": 0, "entered": 0, "calls": 0,
               "frame": 0, "total": len(frames), "alive": 0, "done": False})

    for i, f in enumerate(frames):
        newly = live.pop(f, [])
        st["entered"] += len(newly)
        running.extend(newly)
        running = [r for r in running if r["alive"]]
        st["frame"] = i + 1
        st["alive"] = len(running)
        if not running:
            if not live:
                break
            yield st
            continue

        track_group(ctx, [r["t"] for r in running], f, backwards=backwards, sequence=False)
        st["calls"] += 1
        nxt = f + step

        for r in running:
            m = r["t"].markers.find_frame(nxt, exact=True)
            if m is None or m.mute:
                r["alive"] = False
                st["deaths"] += 1
                continue
            if opts.leash <= 0.0:
                continue
            g = guide.get(r["id"]) or {}
            gp = g.get(str(nxt))
            if gp is None:
                continue
            # Leash in PIXELS on a normalised coordinate system, so the two axes convert
            # separately -- a plate is not square, and one shared factor would make the
            # leash tighter vertically than horizontally.
            dx = (m.co[0] - gp[0]) * r["w"]
            dy = (m.co[1] - gp[1]) * r["h"]
            d = math.hypot(dx, dy)
            if d <= opts.leash:
                continue
            # Pull back TO the leash length rather than snapping onto the guide. The guide
            # is the coarse channel (2.71 px mean on the bench, worse on real plates), so
            # snapping would trade Blender's localisation for TAPNext's. Clamping keeps
            # every sub-pixel gain Blender still has while bounding the wander.
            k = opts.leash / d
            m.co = (gp[0] + dx * k / r["w"], gp[1] + dy * k / r["h"])
            st["clamped"] += 1

        st["alive"] = sum(1 for r in running if r["alive"])
        yield st

    st["done"] = True
    yield st


def track_backward_pass(ctx, records, opts):
    """The backward pass, one synchronous call per distinct seed frame.

    `sequence=False` is not a safe primitive backwards: measured on 5.2, a track seeded at
    frame 40 came back with markers at 39..44 after three single-frame calls -- including
    one BEFORE the seed. So the backward direction stays `sequence=True`, which means it
    runs to completion inside one call and cannot be cancelled or progress-reported. On
    staggered seeds that is a handful of calls, not one per frame.

    Anchors on `r["seed_frame"]`, recorded when the track was created -- NOT on
    `markers[0].frame`, which is no longer the seed by the time the forward pass has run.
    Measured on 5.2: a track seeded at frame 40 comes back from forward stepping with a
    marker at 39, below its own seed (the same `sequence=False` artefact recorded at
    `bl_track.py:415-424`). That marker is not a match, so tracking backward from it fails
    on the first frame and the whole backward pass silently does nothing -- 40 of 122 tracks
    lost their entire head, while every position that DID exist stayed pixel-exact. The
    original avoids this by capturing the seed frames before the forward pass rather than
    by knowing about it; using the recorded value is the same fix without the ordering trap.

    Returns the number of operator calls made, so the UI can say how many freezes to expect.
    """
    by_frame = {}
    for r in records:
        by_frame.setdefault(int(r.get("seed_frame", r["t"].markers[0].frame)), []).append(r)
    calls = 0
    for f in sorted(by_frame):
        if f <= 1:
            continue
        track_group(ctx, [r["t"] for r in by_frame[f]], f,
                    backwards=True, sequence=True)
        calls += 1
    return calls
