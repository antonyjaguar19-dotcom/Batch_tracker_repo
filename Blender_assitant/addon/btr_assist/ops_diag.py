"""One button that writes down everything about the current tracking state.

Why this exists: a running Blender cannot be inspected from outside. `bpy` only exists inside
a Blender process and stock Blender has no IPC endpoint, so the sidecar link is one-way --
Blender calls out, nothing calls in. Adding a command listener to an artist's working session
would fix that and is not worth the risk of a daemon executing requests inside their scene.

So the artist presses a button and the scene describes itself into a file instead. It carries
what an investigation actually needs and cannot otherwise get: which frames each track holds,
where its gaps are, how big its boxes are, which markers are muted, and what the addon and
the clip are set to. That is the difference between reproducing an artist's case and guessing
at a seed position that merely resembles it.

Read-only. It writes one JSON file and touches nothing in the scene.
"""

import json
import os
import time

import bpy

from . import prefs, three_de


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def _runs(track, frames):
    """Frame numbers as contiguous runs: [[1, 12], [40, 57]]. A track's shape in one line --
    where it stops, and whether a hole is a legal 3DE gap or the end of the track."""
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def _box_state(sd, clip):
    """What a Ctrl-click would produce right now, and whether anything is keeping it honest."""
    zoom = float(getattr(sd, "zoom_percentage", 0.0) or 0.0)
    pat = int(clip.tracking.settings.default_pattern_size)
    out = {"zoom_percentage": round(zoom, 2),
           "default_pattern_size_px": pat,
           "default_search_size_px": int(clip.tracking.settings.default_search_size),
           "on_screen_px": round(pat * zoom / 100.0, 1) if zoom > 0 else None,
           "timer_running": None, "remembered": None}
    try:
        from . import click_size                                      # noqa: PLC0415
        out["timer_running"] = bool(bpy.app.timers.is_registered(click_size._apply))
        out["remembered"] = list(click_size._saved.get(clip.name, ()))
    except Exception as exc:                                          # noqa: BLE001
        out["timer_running"] = "unavailable: %s" % exc
    return out


def collect(context, clip):
    w, h = clip.size
    sd = context.space_data
    p = prefs.get(context)
    import sys
    pkg = sys.modules.get(__package__)

    tracks = []
    for tr in three_de.active_tracks(clip):
        live = sorted(m.frame for m in tr.markers if not m.mute)
        muted = sorted(m.frame for m in tr.markers if m.mute)
        first = tr.markers[0] if len(tr.markers) else None
        row = {
            "name": tr.name,
            "selected": bool(tr.select),
            "live_runs": _runs(tr, live),
            "muted_runs": _runs(tr, muted),
            "n_live": len(live),
            "n_muted": len(muted),
            "motion_model": tr.motion_model,
            "pattern_match": tr.pattern_match,
            "correlation_min": round(float(tr.correlation_min), 4),
            "use_brute": bool(tr.use_brute),
            "use_normalization": bool(tr.use_normalization),
            "frames_limit": int(tr.frames_limit),
        }
        # Geometry at the FIRST and LAST live frame: the pair says whether a box was widened
        # on the way, and whether the pattern box grew under LocScale.
        for label, f in (("first", live[0] if live else None),
                         ("last", live[-1] if live else None)):
            m = tr.markers.find_frame(int(f), exact=True) if f is not None else None
            if m is None:
                continue
            xs = [c[0] for c in m.pattern_corners]
            ys = [c[1] for c in m.pattern_corners]
            row[label] = {
                "frame": int(f),
                # Image px, y-DOWN -- the convention the sidecar and the logs use.
                "x": round(m.co[0] * w, 2),
                "y": round((1.0 - m.co[1]) * h, 2),
                "pattern_px": [round((max(xs) - min(xs)) * w, 1),
                               round((max(ys) - min(ys)) * h, 1)],
                "search_px": [round((m.search_max[0] - m.search_min[0]) * w, 1),
                              round((m.search_max[1] - m.search_min[1]) * h, 1)],
                "edge_px": round(min(m.co[0] * w, w - m.co[0] * w,
                                     (1.0 - m.co[1]) * h, h - (1.0 - m.co[1]) * h), 1),
            }
        tracks.append(row)

    return {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "addon": {"version": ".".join(str(x) for x in getattr(pkg, "VERSION", ()) or ()),
                  "build": str(getattr(pkg, "BUILD", "?"))},
        "clip": {
            "name": clip.name,
            "filepath": bpy.path.abspath(clip.filepath),
            "size": [w, h],
            "frames": clip.frame_duration,
            "source": clip.source,
            # A proxy silently halves precision, so it is the first thing worth ruling out.
            "use_proxy": bool(clip.use_proxy),
            "proxy_render_size": getattr(sd.clip_user, "proxy_render_size", "?"),
        },
        "scene": {"frame_start": context.scene.frame_start,
                  "frame_end": context.scene.frame_end,
                  "frame_current": context.scene.frame_current},
        "settings": {k: getattr(p, k) for k in (
            "fit_search_box", "confirm_resumes", "confirm_only_occluded", "verify_pattern",
            "min_match", "animate_scale", "watch_scale", "scale_ratio", "force_full_res",
            "fill_gaps", "constant_box", "box_screen_px")
            if p is not None and hasattr(p, k)},
        # "the new-marker box is still tiny" and "the reconciler is not running" look
        # identical from a screenshot. These four say which.
        "new_marker_box": _box_state(sd, clip),
        "defaults": {
            "default_motion_model": clip.tracking.settings.default_motion_model,
            "default_pattern_match": clip.tracking.settings.default_pattern_match,
            "default_correlation_min": round(
                float(clip.tracking.settings.default_correlation_min), 4),
        },
        "tracks": tracks,
    }


class CLIP_OT_btr_diagnose(bpy.types.Operator):
    bl_idname = "clip.btr_diagnose"
    bl_label = "Write diagnostic report"
    bl_description = ("Write everything about the current tracks to a file: which frames each "
                      "one holds, where the gaps are, box sizes, and every setting in play. "
                      "Read-only -- it changes nothing. Use it when a track behaves oddly and "
                      "someone needs to see exactly what you have")

    def execute(self, context):
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p else ""
        if not root:
            self.report({"ERROR"}, "set the Blender_assitant folder in Preferences")
            return {"CANCELLED"}
        out_dir = os.path.join(root, "logs", "diag")
        os.makedirs(out_dir, exist_ok=True)
        data = collect(context, clip)
        # A stable name so it can be read without being told the path, and a stamped copy so
        # a second run does not erase the one someone is still looking at.
        stamped = os.path.join(out_dir, time.strftime("diag_%Y%m%d-%H%M%S.json"))
        latest = os.path.join(out_dir, "latest.json")
        for path in (stamped, latest):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1)
        n = len(data["tracks"])
        print("[assist] diagnostic report: %d track(s) -> %s" % (n, latest), flush=True)
        self.report({"INFO"}, "wrote %d track(s) to logs/diag/latest.json" % n)
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_diagnose,)
