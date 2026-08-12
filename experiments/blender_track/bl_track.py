"""Runs INSIDE Blender. Seeds handed in, tracked markers handed back.

    blender --background --factory-startup -noaudio --python bl_track.py -- \
        --clip <plate> --seeds seeds.json --out tracks.json

This file must not import anything from this repo: it executes under Blender's own
interpreter, which has bpy but not torch, cv2 or the bot. Everything it needs arrives as
JSON. Everything it says is prefixed `BL ` so the caller can separate it from Blender's
own start-up chatter.

Why background mode works at all
--------------------------------
`bpy.ops.clip.track_markers` polls for a CLIP_EDITOR space, which is why the usual advice
is to run Blender windowed. Measured on 5.2.0 LTS: `--background` still constructs one
window with a screen, so an existing area can be retyped to CLIP_EDITOR, given the clip,
and `track_markers(sequence=True)` runs to completion and returns FINISHED. It also runs
SYNCHRONOUSLY there -- windowed, the same operator is modal and would return before the
tracking finished, which is the trap that makes people think it did nothing.

Tracking model
--------------
`track_markers(sequence=True)` advances every SELECTED track from the current scene frame
until it either reaches the end or loses correlation. So a round is: select a group, set
the frame, one call. Replanting is therefore grouped by resume frame, not done per track
-- one operator call per distinct frame instead of one per track.
"""
import json
import os
import sys

import bpy

PREFIX = "BL "


def log(msg):
    print(PREFIX + str(msg))
    sys.stdout.flush()


# Blender motion models: Loc, LocRot, LocScale, LocRotScale, Affine, Perspective.
# The bot's own refine defaults to affine-with-translation-fallback (see
# app/pattern_refine.py), so the mapping below is the same idea in Blender's fields: a
# corner is pinned in both axes and needs no extra freedom, a blob may breathe with
# defocus, and an edge point can only be located ACROSS the edge -- it gets a search box
# wide enough to survive sliding along it without also inviting a jump.
# (pattern_px, search_px, motion_model)
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


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat-geom", action="store_true",
                    help="one pattern/search size for every seed: the control")
    ap.add_argument("--no-replant", action="store_true")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--replant-rounds", type=int, default=6)
    ap.add_argument("--replant-gap", type=int, default=3,
                    help="frames to skip past the failure before resuming")
    ap.add_argument("--correlation", type=float, default=0.75)
    ap.add_argument("--pattern-match", default="KEYFRAME", choices=["KEYFRAME", "PREV_FRAME"])
    # Sweepable so a tuning run is a command line, not an edit. `--motion-model` overrides
    # the per-class choice for every track; empty = keep the per-class mapping.
    ap.add_argument("--motion-model", default="",
                    choices=["", "Loc", "LocRot", "LocScale", "LocRotScale", "Affine",
                             "Perspective"])
    ap.add_argument("--pattern-scale", type=float, default=1.0)
    ap.add_argument("--search-scale", type=float, default=1.0)
    return ap.parse_args(argv)


def clip_editor(clip):
    """A CLIP_EDITOR area + region to override with, made out of whatever exists."""
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


def load_clip(path):
    """Movie file, or a folder of frames loaded as an image sequence."""
    # Blender resolves a relative path against the .blend, not the shell's cwd, so a
    # perfectly good relative plate path fails with "No such file or directory".
    path = os.path.abspath(path)
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in
                       (".exr", ".dpx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        if not names:
            raise RuntimeError("no frames in " + path)
        # `clip.source` and `clip.frame_duration` are both read-only: Blender decides them
        # at load time. Loading any numbered still gives source=SEQUENCE and a duration
        # scanned off the folder, so the only thing to do is load the first frame and check
        # what came back.
        clip = bpy.data.movieclips.load(os.path.join(path, names[0]))
        log("sequence: %d files on disk, clip reports source=%s duration=%d"
            % (len(names), clip.source, clip.frame_duration))
    else:
        clip = bpy.data.movieclips.load(path)
    return clip


def set_geom(marker, pattern_px, search_px, w, h):
    hu, hv = pattern_px / 2.0 / w, pattern_px / 2.0 / h
    su, sv = search_px / 2.0 / w, search_px / 2.0 / h
    # Blender's corner order is bottom-left, bottom-right, top-right, top-left, and the
    # values are offsets from the marker, not absolute clip coords.
    marker.pattern_corners = ((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv))
    marker.search_min = (-su, -sv)
    marker.search_max = (su, sv)


def live_frames(track):
    return sorted(m.frame for m in track.markers if not m.mute)


def _select(track, on):
    """Select/deselect a track for the next operator call.

    Blender counts a track as selected if ANY of its three flags is set (anchor, pattern,
    search), which is why all three are written here. Measured on 5.2: `track.select`
    alone is in fact enough -- its RNA setter clears the other two -- so this is belt and
    braces, not a fix for anything observed. Selection IS honoured in background mode: a
    deselected track is left with its single seed marker.
    """
    track.select = on
    track.select_anchor = on
    track.select_pattern = on
    track.select_search = on


def track_group(ctx, group, frame, backwards=False):
    """One operator call: advance every track in `group` from `frame` to failure or end."""
    win, area, region, clip, scene = ctx
    for t in clip.tracking.tracks:
        _select(t, False)
    for t in group:
        _select(t, True)
    scene.frame_set(int(frame))
    # `scene.frame_set` is NOT enough. The clip editor keeps its own frame in
    # `space.clip_user.frame_current`, which normally follows the scene through a UI
    # redraw -- and in background mode there is no redraw, so it stays pinned at 1 while
    # the scene says 25. track_markers reads the SPACE's frame, so every call started from
    # frame 1 regardless. Worse, Blender's marker lookup returns the NEAREST marker rather
    # than failing, so a track seeded at frame 67 was silently re-anchored onto frame 1 and
    # tracked forward from there: a full-length track that never touched its own seed.
    # Measured, both ways: space frame 1 -> span 1..100, space frame 25 -> span 25..100.
    space = area.spaces.active
    space.clip_user.frame_current = int(frame)
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=space, edit_movieclip=clip, scene=scene):
        bpy.ops.clip.track_markers(backwards=backwards, sequence=True)


def main():
    args = parse_args()
    spec = json.load(open(args.seeds, "r", encoding="utf-8"))
    seeds = spec["seeds"]
    w, h = int(spec["width"]), int(spec["height"])

    clip = load_clip(spec["plate"] if os.path.exists(spec["plate"]) else args.clip)
    n_frames = int(spec.get("frames") or clip.frame_duration)
    log("clip %s  %dx%d  %d frames" % (os.path.basename(clip.filepath), clip.size[0],
                                       clip.size[1], clip.frame_duration))
    if (clip.size[0], clip.size[1]) != (w, h):
        # The seeds are in the seeder's pixel space. If Blender reads a different size the
        # normalised coords still "work" and every tracker starts on the wrong feature, so
        # this is a hard stop rather than a warning.
        raise RuntimeError("clip is %dx%d but seeds are for %dx%d"
                           % (clip.size[0], clip.size[1], w, h))

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, n_frames
    win, area, region = clip_editor(clip)
    ctx = (win, area, region, clip, scene)

    st = clip.tracking.settings
    st.use_default_brute = True            # survives fast motion; the search box is the cost
    st.use_default_normalization = True    # exposure/grain changes must not read as failure
    st.default_correlation_min = args.correlation
    st.default_pattern_match = args.pattern_match

    # --------------------------------------------------------------- create the seeds
    made = []
    for s in seeds:
        kind = s.get("kind", "") or ""
        pat, srch, model = FLAT_GEOM if args.flat_geom else KIND_GEOM.get(kind, FLAT_GEOM)
        pat = max(5, int(round(pat * args.pattern_scale)))
        srch = max(pat + 4, int(round(srch * args.search_scale)))
        model = args.motion_model or model
        t = clip.tracking.tracks.new(name=s["id"], frame=int(s["frame"]))
        t.motion_model = model
        t.use_brute = True
        t.use_normalization = True
        t.correlation_min = args.correlation
        t.frames_limit = 0
        t.pattern_match = args.pattern_match
        m = t.markers[0]
        m.co = (float(s["u"]), float(s["v"]))
        set_geom(m, pat, srch, w, h)
        # MovieTrackingTrack takes no ID properties, and tracks.new() may uniquify the
        # name, so the caller's id and geometry are carried alongside the track object
        # rather than on it. Order is the identity here.
        made.append({"t": t, "id": s["id"], "kind": kind, "pat": pat, "srch": srch})
    log("seeded %d tracks  (model=%s match=%s corr=%.2f pat_x%.2f srch_x%.2f%s)"
        % (len(made), args.motion_model or "per-class", args.pattern_match,
           args.correlation, args.pattern_scale, args.search_scale,
           " flat" if args.flat_geom else ""))

    # --------------------------------------------------------------- forward / backward
    by_frame = {}
    for r in made:
        by_frame.setdefault(r["t"].markers[0].frame, []).append(r["t"])

    for fr in sorted(by_frame):
        track_group(ctx, by_frame[fr], fr, backwards=False)
    log("forward pass done (%d start frames)" % len(by_frame))

    if not args.no_backward:
        # Seeds are staggered, so a seed born at frame 90 has nothing before it. The
        # backward pass is what makes a late seed cover the head of the shot; it is the
        # same idea as the bot's own backward TAPNext passes.
        for fr in sorted(by_frame):
            if fr > 1:
                track_group(ctx, by_frame[fr], fr, backwards=True)
        log("backward pass done")

    # --------------------------------------------------------------- replant
    # A Blender tracker that loses correlation is finished -- it has no notion of the
    # feature coming back. TAPNext does: it carries a position through the occlusion. So
    # where the local tracker dies, the guide says where the feature went, a marker is
    # planted there, and the SAME track resumes. The frames in between stay empty, which
    # is a legal gap in 3DE ASCII, not a deletion.
    replants = 0
    if not args.no_replant:
        guide = {s["id"]: s.get("guide") or {} for s in seeds}
        for rnd in range(max(0, args.replant_rounds)):
            groups = {}
            for r in made:
                fr = live_frames(r["t"])
                if not fr or fr[-1] >= n_frames:
                    continue
                rf = fr[-1] + max(1, args.replant_gap)
                g = guide.get(r["id"]) or {}
                uv = g.get(str(rf))
                # Walk forward to the first guided frame at or after the resume point: the
                # guide track can itself be short, or have its own gap there.
                while uv is None and rf <= n_frames:
                    rf += 1
                    uv = g.get(str(rf))
                if uv is None or rf > n_frames:
                    continue
                groups.setdefault(rf, []).append(r)
            if not groups:
                break
            n_this = 0
            for rf in sorted(groups):
                grp = []
                for r in groups[rf]:
                    uv = (guide.get(r["id"]) or {})[str(rf)]
                    m = r["t"].markers.insert_frame(int(rf),
                                                    co=(float(uv[0]), float(uv[1])))
                    m.mute = False
                    set_geom(m, r["pat"], r["srch"], w, h)
                    grp.append(r["t"])
                track_group(ctx, grp, rf, backwards=False)
                n_this += len(grp)
            replants += n_this
            log("replant round %d: %d resumes" % (rnd + 1, n_this))

    # --------------------------------------------------------------- hand back
    out = {"width": w, "height": h, "frames": n_frames, "replants": replants, "tracks": []}
    for r in made:
        pts = [[int(m.frame), float(m.co[0]), float(m.co[1])]
               for m in r["t"].markers if not m.mute and 1 <= m.frame <= n_frames]
        pts.sort()
        out["tracks"].append({"id": r["id"], "kind": r["kind"], "pts": pts})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    lens = [len(t["pts"]) for t in out["tracks"]]
    full = sum(1 for n in lens if n >= n_frames)
    log("wrote %s: %d tracks, %d replants, mean length %.1f/%d, %d full-length"
        % (os.path.basename(args.out), len(lens), replants,
           (sum(lens) / len(lens)) if lens else 0.0, n_frames, full))
    log("DONE")


if __name__ == "__main__":
    main()
