"""M0 -- does Blender's tracker work from Python in a WINDOWED Blender?

Every tracking measurement in `experiments/blender_track/` was taken in `--background`,
where `bl_track.py:14-18` recorded that `bpy.ops.clip.track_markers` runs SYNCHRONOUSLY.
That docstring also warns that windowed, "the same operator is modal and would return
before the tracking finished". If that is true of a Python call, the whole addon has to
shell out to a second Blender and the interactive story dies.

It is probably NOT true of a Python call. `CLIP_OT_track_markers` registers three
callbacks -- exec, invoke, modal -- and the modal/job behaviour lives in `invoke`, which
is what the panel BUTTON runs. `bpy.ops.x.y()` from Python defaults to 'EXEC_DEFAULT',
i.e. the same exec path that background mode takes. If so the foreground problem is not
correctness but a frozen UI, which is a different and much smaller problem.

This script settles it by measurement rather than argument. Run it in a real window:

    blender.exe --factory-startup --python spike_foreground.py -- \
        --plate <folder of frames> --out results.json

`--factory-startup` is used here only so the spike is reproducible. The shipping addon
gets no such protection -- see test G.

Answers written to --out and printed. Exit code 0 whatever the answers are: a spike that
fails a test has still succeeded at being a spike.
"""

import json
import os
import sys
import time

import bpy


def log(msg):
    print("[spike] %s" % msg, flush=True)


# ---------------------------------------------------------------- scene setup

def load_clip(path):
    """Same rules as bl_track.load_clip: a folder becomes an image sequence."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in
                       (".exr", ".dpx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        if not names:
            raise RuntimeError("no frames in " + path)
        clip = bpy.data.movieclips.load(os.path.join(path, names[0]))
    else:
        clip = bpy.data.movieclips.load(path)
    return clip


def clip_editor(clip):
    """A CLIP_EDITOR area + region to override with. Lifted from bl_track.clip_editor."""
    wm = bpy.context.window_manager
    if not len(wm.windows):
        raise RuntimeError("no window at all -- cannot reach bpy.ops.clip")
    win = wm.windows[0]
    area = None
    for a in win.screen.areas:
        if a.type == "CLIP_EDITOR":
            area = a
            break
    if area is None:
        area = max(win.screen.areas, key=lambda x: x.width * x.height)
        area.type = "CLIP_EDITOR"
    area.spaces.active.clip = clip
    region = None
    for r in area.regions:
        if r.type == "WINDOW":
            region = r
    return win, area, region


def set_geom(marker, pattern_px, search_px, w, h):
    hu, hv = pattern_px / 2.0 / w, pattern_px / 2.0 / h
    su, sv = search_px / 2.0 / w, search_px / 2.0 / h
    marker.pattern_corners = ((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv))
    marker.search_min = (-su, -sv)
    marker.search_max = (su, sv)


def _select(track, on):
    track.select = on
    track.select_anchor = on
    track.select_pattern = on
    track.select_search = on


def clear_tracks(ctx):
    """`tracks.remove()` does not exist on 5.2 -- MovieTrackingTracks is add-only from RNA,
    so removal is the operator, which needs the same context override as tracking."""
    win, area, region, clip, scene = ctx
    tracks = clip.tracking.objects.active.tracks
    if not len(tracks):
        return
    for t in tracks:
        _select(t, True)
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=area.spaces.active,
                                   edit_movieclip=clip, scene=scene):
        bpy.ops.clip.delete_track()


def seed_grid(clip, frame, n, prefix):
    """n trackers on a grid, avoiding the frame edge. Returns [(track, u, v), ...]."""
    w, h = clip.size
    tracks = clip.tracking.objects.active.tracks
    made = []
    cols = int(n ** 0.5 + 0.999)
    rows = (n + cols - 1) // cols
    for i in range(n):
        cx, cy = i % cols, i // cols
        u = 0.15 + 0.70 * (cx + 0.5) / cols
        v = 0.15 + 0.70 * (cy + 0.5) / rows
        t = tracks.new(name="%s_%04d" % (prefix, i), frame=int(frame))
        t.motion_model = "Loc"
        t.use_brute = True
        t.use_normalization = True
        t.correlation_min = 0.75
        t.frames_limit = 0
        t.pattern_match = "KEYFRAME"
        m = t.markers[0]
        m.co = (u, v)
        set_geom(m, 21.0, 41.0, w, h)
        made.append((t, u, v))
    return made


def apply_settings(clip):
    """Written explicitly, never inherited -- see test G."""
    st = clip.tracking.settings
    st.use_default_brute = True
    st.use_default_normalization = True
    st.default_correlation_min = 0.75
    st.default_pattern_match = "KEYFRAME"
    st.default_motion_model = "Loc"


def track_call(ctx, group, frame, backwards=False, sequence=True, set_space_frame=True):
    """One operator call, with bl_track.track_group's context override."""
    win, area, region, clip, scene = ctx
    for t in clip.tracking.objects.active.tracks:
        _select(t, False)
    for t in group:
        _select(t, True)
    scene.frame_set(int(frame))
    space = area.spaces.active
    if set_space_frame:
        space.clip_user.frame_current = int(frame)
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=space, edit_movieclip=clip, scene=scene):
        return bpy.ops.clip.track_markers(backwards=backwards, sequence=sequence)


def spans(made):
    out = []
    for t, _, _ in made:
        fr = sorted(m.frame for m in t.markers if not m.mute)
        out.append((t.name, fr[0], fr[-1], len(fr)) if fr else (t.name, 0, 0, 0))
    return out


# ---------------------------------------------------------------- the tests

def test_B_exec_is_synchronous(ctx, clip, n_frames):
    """THE question. Does a Python track_markers() call block until the track is done?

    Synchronous  -> markers ~= n_frames and the call took real time  -> modal generator.
    Modal        -> markers == 1 and the call returned instantly     -> headless fallback.
    """
    clear_tracks(ctx)
    made = seed_grid(clip, 1, 20, "B")
    t0 = time.time()
    res = track_call(ctx, [t for t, _, _ in made], 1, sequence=True)
    dt = time.time() - t0
    sp = spans(made)
    counts = sorted(s[3] for s in sp)
    median = counts[len(counts) // 2]
    verdict = ("synchronous" if median > 1 and dt > 0.5 else
               "modal-or-instant" if median <= 1 else "unclear")
    return {
        "operator_result": list(res),
        "seconds": round(dt, 3),
        "n_frames": n_frames,
        "marker_counts": counts,
        "median_markers": median,
        "max_markers": counts[-1],
        "tracker_frames_per_s": round(sum(counts) / dt, 1) if dt > 0 else None,
        "verdict": verdict,
        "PASS": verdict == "synchronous",
    }


def test_C_frames_limit(ctx, clip):
    """Does `frames_limit` give a usable chunk lever? 8 forward from the seed -> 9 markers."""
    clear_tracks(ctx)
    made = seed_grid(clip, 1, 6, "C")
    for t, _, _ in made:
        t.frames_limit = 8
    t0 = time.time()
    track_call(ctx, [t for t, _, _ in made], 1, sequence=True)
    dt = time.time() - t0
    counts = sorted(s[3] for s in spans(made))
    # A track that dies at frame 4 yields 4 markers and that is the TRACKER failing, not
    # the limit. The lever works iff nothing ever exceeds the cap and something reaches it.
    return {
        "seconds": round(dt, 3),
        "marker_counts": counts,
        "cap": 9,
        "over_cap": [c for c in counts if c > 9],
        "reached_cap": counts.count(9),
        "PASS": not [c for c in counts if c > 9] and 9 in counts,
    }


def test_E_space_frame_anchor(ctx, clip):
    """bl_track.py:249-257 -- does scene.frame_set alone re-anchor a late seed onto frame 1?

    Background mode does (no redraw pins clip_user.frame_current at 1). Windowed the space
    frame may follow the scene, which would make the workaround unnecessary -- but the
    addon's modal loop deliberately does not redraw between frames, so the answer matters.
    """
    out = {}
    for label, set_space in (("with_space_frame", True), ("scene_frame_only", False)):
        clear_tracks(ctx)
        made = seed_grid(clip, 40, 6, "E_" + label)
        for t, _, _ in made:
            t.frames_limit = 5
        track_call(ctx, [t for t, _, _ in made], 40, sequence=True,
                   set_space_frame=set_space)
        sp = spans(made)
        out[label] = {"first_frames": sorted(s[1] for s in sp),
                      "last_frames": sorted(s[2] for s in sp)}
    anchored = all(f == 40 for f in out["scene_frame_only"]["first_frames"])
    out["scene_frame_alone_is_enough"] = anchored
    out["workaround_still_needed"] = not anchored
    out["PASS"] = all(f == 40 for f in out["with_space_frame"]["first_frames"])
    return out


def test_G_inherited_settings(clip):
    """No --factory-startup in an artist's Blender. Record what the scene defaults were,
    so the addon can prove it overwrites every one of them."""
    st = clip.tracking.settings
    return {
        "default_motion_model": st.default_motion_model,
        "default_pattern_match": st.default_pattern_match,
        "default_correlation_min": round(st.default_correlation_min, 4),
        "use_default_brute": st.use_default_brute,
        "use_default_normalization": st.use_default_normalization,
        "default_pattern_size": st.default_pattern_size,
        "default_search_size": st.default_search_size,
        "note": "addon must write all of these per job; it cannot inherit them",
    }


def test_F_proxy(ctx, clip, n_frames):
    """Does a 50% proxy silently halve tracking precision?

    No experiment in this repo has ever run with a proxy, and artists leave them on. If
    the two runs disagree, the addon must force PROXY_100 or refuse the job.
    """
    win, area, region, _, scene = ctx
    space = area.spaces.active
    out = {}
    for label, size in (("PROXY_100", "PROXY_100"), ("PROXY_50", "PROXY_50")):
        clear_tracks(ctx)
        made = seed_grid(clip, 1, 8, "F_" + label)
        for t, _, _ in made:
            t.frames_limit = 20
        space.clip_user.proxy_render_size = size
        clip.use_proxy = (size != "PROXY_100")
        track_call(ctx, [t for t, _, _ in made], 1, sequence=True)
        pos = {}
        for t, _, _ in made:
            for m in t.markers:
                if not m.mute:
                    pos.setdefault(t.name, {})[m.frame] = (m.co[0], m.co[1])
        out[label] = pos
    clip.use_proxy = False
    space.clip_user.proxy_render_size = "PROXY_100"

    w, h = clip.size
    worst, n = 0.0, 0
    a, b = out["PROXY_100"], out["PROXY_50"]
    for name in a:
        bn = b.get(name.replace("F_PROXY_100", "F_PROXY_50"), {})
        for f, (u, v) in a[name].items():
            if f in bn:
                du = (u - bn[f][0]) * w
                dv = (v - bn[f][1]) * h
                worst = max(worst, (du * du + dv * dv) ** 0.5)
                n += 1
    return {
        "compared_samples": n,
        "worst_disagreement_px": round(worst, 4),
        "proxy_is_safe": bool(n and worst < 0.01),
        "PASS": bool(n),
        "note": ("proxies must be forced to PROXY_100 by the addon"
                 if worst >= 0.01 else "no measurable difference"),
    }


# ---------------------------------------------------------------- main

def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = None
    out_path = "spike_results.json"
    do_proxy = False
    i = 0
    while i < len(argv):
        if argv[i] == "--plate":
            plate = argv[i + 1]
            i += 2
        elif argv[i] == "--out":
            out_path = argv[i + 1]
            i += 2
        elif argv[i] == "--proxy":
            do_proxy = True
            i += 1
        else:
            i += 1
    if not plate:
        raise SystemExit("--plate is required")

    results = {"blender": bpy.app.version_string, "background": bpy.app.background}
    log("blender %s  background=%s" % (results["blender"], results["background"]))
    if bpy.app.background:
        log("WARNING: running in --background. This spike only means something windowed.")

    clip = load_clip(plate)
    win, area, region = clip_editor(clip)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = clip.frame_duration
    ctx = (win, area, region, clip, scene)
    results["clip"] = {"size": list(clip.size), "frames": clip.frame_duration,
                       "source": clip.source}
    log("clip %dx%d, %d frames" % (clip.size[0], clip.size[1], clip.frame_duration))

    results["G_inherited_settings"] = test_G_inherited_settings(clip)
    apply_settings(clip)

    for name, fn in (("B_exec_is_synchronous",
                      lambda: test_B_exec_is_synchronous(ctx, clip, clip.frame_duration)),
                     ("C_frames_limit", lambda: test_C_frames_limit(ctx, clip)),
                     ("E_space_frame_anchor", lambda: test_E_space_frame_anchor(ctx, clip))):
        log("running %s ..." % name)
        try:
            results[name] = fn()
        except Exception as exc:  # a spike reports failures, it does not raise them
            results[name] = {"ERROR": "%s: %s" % (type(exc).__name__, exc), "PASS": False}
        log("%s -> %s" % (name, json.dumps(results[name])[:400]))

    if do_proxy:
        log("running F_proxy (builds proxies, slow) ...")
        try:
            results["F_proxy"] = test_F_proxy(ctx, clip, clip.frame_duration)
        except Exception as exc:
            results["F_proxy"] = {"ERROR": "%s: %s" % (type(exc).__name__, exc),
                                  "PASS": False}
        log("F_proxy -> %s" % json.dumps(results["F_proxy"])[:400])

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log("wrote %s" % os.path.abspath(out_path))

    b = results.get("B_exec_is_synchronous", {})
    log("=" * 68)
    log("ANSWER: track_markers from Python is %r" % b.get("verdict"))
    log("  -> %s" % ("Path A: modal generator, tracking runs in-process"
                     if b.get("PASS") else
                     "Path C: shell out to blender --background"))
    log("=" * 68)

    if not bpy.app.background:
        bpy.ops.wm.quit_blender()


main()
