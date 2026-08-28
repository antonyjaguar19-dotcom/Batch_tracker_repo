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

try:
    from .scale_watch import ScaleWatch, size_of
except ImportError:                   # pragma: no cover -- see below
    # The parity harness (`tests/parity_run_core.py`) puts `addon/btr_assist` on sys.path
    # and imports this file as a TOP-LEVEL module, because the package's __init__ registers
    # operators it has no use for. A relative import is an ImportError there, and the gate
    # that proves this loop matches the original is not worth losing to an import style.
    from scale_watch import ScaleWatch, size_of

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
        # Pattern-box scale watch. Inert unless a record carries a watch (`attach_watch`),
        # so the headless path and the parity test are byte-identical with these present.
        # A watch only fires under a motion model that solves scale -- `Loc` never resizes
        # the box, so there is nothing for it to see.
        self.watch_scale = False
        # Per-cell plate motion from the sidecar, or None. With None this loop is
        # byte-identical to what it was -- which is what keeps the parity gate meaningful.
        # Stop a track when its PATTERN box reaches the edge of the plate. Off here so the
        # headless path and the parity gate are unchanged; the operator turns it on.
        self.edge_stop = False
        # Hard bounds on how far LocScale may take the pattern box from the size the artist
        # set, as a ratio either way. 0 disables. See `clamp_pattern`.
        self.scale_clamp = 0.0
        # Leave every tracking setting exactly as Blender has it: do not write the clip's
        # defaults, do not write a track's own properties. OFF here so the headless path and
        # the parity gate are unchanged; the interactive operators turn it ON.
        #
        # It exists because the two were measurably not the same tracker. Same seed, same
        # pattern box, 14 frames of SH006: a track carrying Blender's own defaults and one
        # carrying this file's (PREV_FRAME, normalization on) end up **10.36 px apart**, and
        # `apply_settings` wrote those into the artist's CLIP, so markers they made later in
        # Blender's own UI changed too (7.91 px). The settings below are better on this
        # project's own numbers, and that is not the point: an assistant that silently
        # tracks differently from the application it is assisting cannot be checked against
        # it, and the artist checks it constantly.
        self.blender_defaults = False
        self.motion = None
        self.motion_headroom = 1.5
        self.motion_cap_frac = 0.25
        self.scale_rate = 0.10              # fractional size change in ONE frame
        self.scale_ratio = 1.6              # cumulative size against the artist's box
        self.scale_onset = 1.06             # size band that counts as "still the same box"
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


def pattern_px(marker, w, h):
    """The marker's pattern box size in plate pixels, as (width, height).

    `pattern_corners` are four OFFSETS from the marker in normalised clip space. Under
    `LocScale` Blender rewrites them every frame, so this is a per-frame MEASUREMENT of how
    big the tracker currently thinks the feature is -- not a setting.
    """
    xs = [c[0] for c in marker.pattern_corners]
    ys = [c[1] for c in marker.pattern_corners]
    return (max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h


def attach_watch(records, opts, frame=None):
    """Give each record a `ScaleWatch` baselined on the box it starts this pass with.

    Called again before every pass: after a repair (or a resume) the track restarts from a
    new frame with a new box, and a watch still holding the old baseline would flag the very
    first frame. `watch=None` on a record opts it out permanently, which is how a track that
    has already used up its repairs stops being asked about.
    """
    for r in records:
        if not opts.watch_scale or r.get("no_watch"):
            r["watch"] = None
            continue
        f = int(frame if frame is not None else r.get("seed_frame",
                                                      r["t"].markers[0].frame))
        m = r["t"].markers.find_frame(f, exact=True)
        if m is None:
            r["watch"] = None
            continue
        r["watch"] = ScaleWatch(size_of(pattern_px(m, r["w"], r["h"])),
                                rate=opts.scale_rate, ratio=opts.scale_ratio,
                                onset=opts.scale_onset)


def clamp_pattern(marker, w, h, seed_w, seed_h, ratio):
    """Keep the pattern box within `ratio` of the size the artist set, both ways.

    Under `LocScale` Blender solves a size every frame, and on low-contrast footage that
    solve is not stable: measured over 30 seeds on SH013, seven tracks ended with a pattern
    box between **0.3 px and 2.0 px** -- 0.01x to 0.07x of the 28 px they were seeded with --
    and one ballooned to 237 px (8.5x). A sub-pixel pattern is not tracking anything, and the
    track does not die; it keeps returning positions, so the span looks excellent. Those
    tracks are the "302 of 303 frames" successes, and most of their frames are drift with a
    marker attached.

    Totals over the same 30 seeds: `Loc` produced 1660 tracked frames with no degenerate box;
    `LocScale` produced 2414, of which the extra came largely from boxes that had collapsed.
    Longer, and worth less.

    Position is NOT touched -- only the box. Same rule as the search-box re-fit and the
    watch's `bad-box` repair: geometry is what the evidence supports correcting, the position
    stays Blender's measurement.

    Returns the clamped width in px, or 0.0 if nothing changed.
    """
    if ratio <= 1.0 or seed_w <= 0.0 or seed_h <= 0.0:
        return 0.0
    xs = [c[0] for c in marker.pattern_corners]
    ys = [c[1] for c in marker.pattern_corners]
    pw = (max(xs) - min(xs)) * w
    ph = (max(ys) - min(ys)) * h
    if pw <= 0.0 or ph <= 0.0:
        pw, ph = seed_w, seed_h
    lo_w, hi_w = seed_w / ratio, seed_w * ratio
    lo_h, hi_h = seed_h / ratio, seed_h * ratio
    nw = min(hi_w, max(lo_w, pw))
    nh = min(hi_h, max(lo_h, ph))
    if abs(nw - pw) < 0.5 and abs(nh - ph) < 0.5:
        return 0.0
    hu, hv = nw / 2.0 / w, nh / 2.0 / h
    marker.pattern_corners = ((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv))
    return nw


def first_jump(path, k=3.0, floor=12.0, look=8, minimum=6):
    """The first frame where a track moves out of character with its own motion.

    Costs nothing and needs no plate: it reads the track's own positions. That matters
    because the appearance check cannot see this case -- an occluder that looks like the
    feature keeps the correlation happy while the marker is dragged somewhere it could not
    physically have gone.

    Measured against an artist's hand track and the assist output for the same feature
    (SH006). Their hand track never steps more than 16 px. The assist output:

        f15  step 39.5 px  against a recent median of 9    the occluder arriving
        f20-23  steps 28-34 px                             sliding on it
        f24  step 70.4 px
        f27  step 116.5 px                                 snapping back by luck

    Both of the artist's complaints -- "unwanted jumps" and "unwanted slides" -- are the
    same signal at different sizes, and both are invisible to a correlation score.

    Judged against the track's OWN recent median rather than an absolute speed, because a
    plate that moves 40 px a frame everywhere is not jumping. `minimum` samples must exist
    first: a track that starts slow and accelerates would otherwise be cut on its own
    acceleration -- measured, the hand track above trips at f5 without it.

    Steps ACROSS a gap are skipped. A resume is a new head, and the distance from the last
    frame before an occlusion means nothing.

    Returns (frame, step, median) or (None, None, None).
    """
    recent = []
    prev = None
    for f, x, y in path:
        if prev is not None and int(f) == int(prev[0]) + 1:
            step = math.hypot(x - prev[1], y - prev[2])
            if len(recent) >= minimum:
                med = sorted(recent)[len(recent) // 2]
                if step > floor and step > k * max(med, 1.0):
                    return int(f), step, med
            recent.append(step)
            if len(recent) > look:
                recent.pop(0)
        else:
            recent = []
        prev = (f, x, y)
    return None, None, None


def pattern_outside(marker, w, h):
    """Is any part of this marker's PATTERN box off the plate?

    The pattern is what correlates. Once part of it is outside the frame there is nothing
    there to match, and Blender does not stop -- under `LocScale` it solves a SMALLER box
    instead, keeps returning a position, and the track walks off the feature while still
    looking alive. Measured on SH013 a track ended 15 px from the frame edge with its pattern
    already shrunk from 28.0 px to 23.9, and the frames before it are the drift.

    The SEARCH box is deliberately not tested. It routinely hangs off the plate and Blender
    copes -- a track was measured running to 1 px from the edge with a 167 px search box --
    so stopping on that would cut every edge-bound track short for no reason.
    """
    xs = [c[0] for c in marker.pattern_corners]
    ys = [c[1] for c in marker.pattern_corners]
    x0 = (marker.co[0] + min(xs)) * w
    x1 = (marker.co[0] + max(xs)) * w
    y0 = (marker.co[1] + min(ys)) * h
    y1 = (marker.co[1] + max(ys)) * h
    return x0 < 0.0 or y0 < 0.0 or x1 > float(w) or y1 > float(h)


def motion_at(mo, x, y, w, h):
    """Measured p95 motion in px/frame at a plate position (image px, y-DOWN)."""
    gx, gy = mo["grid"]
    i = min(gx - 1, max(0, int(x / max(1.0, float(w)) * gx)))
    j = min(gy - 1, max(0, int(y / max(1.0, float(h)) * gy)))
    return float(mo["p95"][j][i])


def refit_search(marker, w, h, mo, headroom=1.5, cap_frac=0.25):
    """Widen this marker's search box if the plate moves further HERE than it reaches.

    Sizing once from the seed is not enough on a plate whose motion is not uniform. Measured
    on SH013 the p95 runs 22.7 px/frame across the middle of frame and 47.6 near the bottom,
    and a foreground feature crosses from one to the other in a few dozen frames: seeded at
    22.7 it gets a 96 px box, needs 171 by the time it has sped up, and dies at frame 42 with
    the feature still well inside the frame. That reads exactly like a tracking failure and
    is a box that stopped reaching.

    Returns the new width in px, or 0.0 when nothing changed. Only ever ENLARGES -- see
    `_widen_boxes` in the operator for why a small box is not always a mistake.
    """
    xs = [c[0] for c in marker.pattern_corners]
    ys = [c[1] for c in marker.pattern_corners]
    pat = max((max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h)
    # y-DOWN for the grid lookup: `marker.co` is y-UP clip space.
    x = marker.co[0] * w
    y = (1.0 - marker.co[1]) * h
    p95 = motion_at(mo, x, y, w, h)
    want = min(2.0 * (p95 * headroom + pat / 2.0), w * cap_frac)
    have = (marker.search_max[0] - marker.search_min[0]) * w
    if want <= have + 1.0:
        return 0.0
    sx, sy = want / 2.0 / w, want / 2.0 / h
    marker.search_min = (-sx, -sy)
    marker.search_max = (sx, sy)
    return want


def _select(track, on):
    track.select = on
    track.select_anchor = on
    track.select_pattern = on
    track.select_search = on


# ---------------------------------------------------------------- proxy guard

#: The original footage. NOT `PROXY_100` -- that is a rendered 100%-size PROXY FILE, which
#: is still a re-encode and may not even exist. Measured on 5.2 the enum is
#: ['PROXY_25','PROXY_50','PROXY_75','PROXY_100','FULL'] and the default is 'FULL'.
FULL_RES = "FULL"


class FullResolution:
    """Force the original footage for the duration of a job, then restore the artist's setting.

    A 50% proxy may run tracking against half-resolution pixels, which halves precision
    invisibly -- the viewport looks identical and the exported numbers are quietly worse.
    No experiment in this project has ever run with a proxy, so rather than measure the
    damage the addon simply refuses to work in that state. See FINDINGS.md, test F.

    `clip.use_proxy = False` is the decisive one; the render size is set as well so the
    state is unambiguous while the job runs.
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
        self.space.clip_user.proxy_render_size = FULL_RES
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
    if getattr(opts, "blender_defaults", False):
        # The artist asked for Blender's tracker. Writing clip defaults here would change
        # every track they create afterwards, in Blender's own UI, for the rest of the
        # session -- measured at 7.91 px over 14 frames.
        return
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
        if getattr(opts, "blender_defaults", False):
            # `tracks.new` has already copied the clip's own defaults onto the track. Leave
            # them: this seed must track identically to one the artist placed by hand. Only
            # the geometry below is ours, because the pattern box IS the artist's input.
            model = t.motion_model
        else:
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
                     "alive": True, "w": w, "h": h, "seed_frame": int(s["frame"]),
                     "seed_pat": (float(pat), float(pat))})
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
    st.update({"deaths": 0, "clamped": 0, "entered": 0, "calls": 0, "flagged": 0,
               "refit": 0, "edge": 0, "boxclamp": 0, "frame": 0, "total": len(frames),
               "alive": 0, "done": False})

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
            # Keep the pattern box a plausible size BEFORE anything reads it -- the edge
            # test, the watch and the next step all ask how big it is.
            if opts.scale_clamp > 1.0:
                sp = r.get("seed_pat")
                if sp and clamp_pattern(m, r["w"], r["h"], sp[0], sp[1], opts.scale_clamp):
                    st["boxclamp"] += 1

            # The pattern box has reached the edge of the plate. Stop -- but keep this
            # marker: the position Blender just measured is not wrong, it is the LAST one
            # measured by a box that was still fully on the plate's side of the line. What
            # comes after it is the drift, and nothing is deleted to establish that.
            if opts.edge_stop and pattern_outside(m, r["w"], r["h"]):
                r["alive"] = False
                r["edge_stopped"] = int(nxt)
                st["edge"] += 1
                continue

            # Re-fit the search box to where the feature has GOT to, not where it started.
            # Before the scale watch and the leash on purpose: both of those judge a marker
            # that has already been placed, while this decides whether the NEXT step can
            # reach at all.
            if opts.motion:
                if refit_search(m, r["w"], r["h"], opts.motion,
                                headroom=opts.motion_headroom,
                                cap_frac=opts.motion_cap_frac):
                    st["refit"] += 1

            # Scale watch. A flag STOPS the track where it is -- it does not delete
            # anything and it is not a death: the marker Blender just wrote stays, and the
            # caller decides what the swell meant by looking at the pixels. Checked before
            # the leash because a track that is no longer on its feature has nothing to
            # gain from being pulled towards the guide.
            watch = r.get("watch")
            if watch is not None:
                flag = watch.feed(nxt, size_of(pattern_px(m, r["w"], r["h"])))
                if flag:
                    r["alive"] = False
                    r["scale_flag"] = flag
                    st["flagged"] += 1
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
