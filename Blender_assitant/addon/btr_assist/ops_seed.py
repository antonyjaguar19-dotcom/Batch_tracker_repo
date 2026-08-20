"""Auto-seed: ask the sidecar where the trackers go, then let Blender measure them.

Runs as a modal operator on a timer, in three phases:

  waiting   the sidecar is running TAPNext; the addon polls and the UI stays live
  tracking  the per-frame generator is drained for ~50 ms per tick, so the viewport
            redraws, the progress reads out and ESC works between frames
  backward  one blocking call per distinct seed frame -- `sequence=False` is not a safe
            primitive backwards (`track_core.track_backward_pass` explains why), so this
            phase genuinely freezes, and the status line says so rather than pretending

M0 measured that `bpy.ops.clip.track_markers()` from Python is synchronous in a windowed
Blender, which is why any of this works in-process at all.
"""

import os

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty

from . import client, prefs, three_de, track_core

TICK_SECONDS = 0.05          # how long to hold the main thread per timer tick


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def clip_info(context, clip):
    """What the sidecar needs to find the same pixels. A path and a range, never images."""
    path = bpy.path.abspath(clip.filepath)
    # Blender resolves `//plate/...` against the .blend, not against cwd. Sending the raw
    # string would give the sidecar a path that does not exist on any drive.
    if os.path.isfile(path):
        path = os.path.dirname(path)
    sd = context.space_data
    return {"path": path, "width": clip.size[0], "height": clip.size[1],
            "frames": clip.frame_duration, "frame_start": clip.frame_start,
            "use_proxy": bool(clip.use_proxy),
            "proxy_render_size": getattr(sd.clip_user, "proxy_render_size", "PROXY_100")}


class CLIP_OT_btr_sidecar(bpy.types.Operator):
    bl_idname = "clip.btr_sidecar"
    bl_label = "Sidecar"
    bl_description = "Start, check or stop the model process"

    action: bpy.props.EnumProperty(
        items=[("START", "Start", ""), ("CHECK", "Check", ""), ("STOP", "Stop", "")],
        default="CHECK")

    def execute(self, context):
        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p else ""
        if not root:
            self.report({"ERROR"}, "set the Blender_assitant folder in Preferences")
            return {"CANCELLED"}
        try:
            if self.action == "STOP":
                client.shutdown(root)
                self.report({"INFO"}, "sidecar stopped")
                return {"FINISHED"}
            if self.action == "START":
                _, h = client.ensure(root, bpy.path.abspath(p.python_exe), p.port)
            else:
                h = client.health(root)
                if h is None:
                    self.report({"WARNING"}, "sidecar is not running")
                    return {"FINISHED"}
        except client.SidecarError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, "sidecar %s  torch %s  cuda %s  %s MB free"
                    % (h.get("version"), h.get("torch"), h.get("cuda"),
                       h.get("vram_free_mb", "?")))
        return {"FINISHED"}


class CLIP_OT_btr_autoseed(bpy.types.Operator):
    bl_idname = "clip.btr_autoseed"
    bl_label = "Auto-seed"
    bl_description = ("Ask the model where the trackers should go, then track them with "
                      "Blender's own tracker")

    target: IntProperty(name="Target tracks", default=150, min=4, max=2000)
    spacing_px: FloatProperty(
        name="Spacing (px)", default=15.0, min=2.0, max=400.0,
        description="Minimum distance between seeds. This, not quality, is what caps the "
                    "count: one shot read '1278 past quality bar -> 28 after spacing' at "
                    "the default 60, and 122 at 15")
    reject_movers: BoolProperty(
        name="Reject movers", default=False,
        description="OFF until it has a measured reference set. See FINDINGS.md")
    track_after: BoolProperty(
        name="Track after seeding", default=True,
        description="Off = place the markers and stop, so you can look at where the model "
                    "chose before spending the tracking time")

    _timer = None
    _phase = ""
    _job = ""
    _root = ""
    _gen = None
    _stats = None
    _records = None
    _ctx = None
    _proxy = None
    _seeds = None

    # ---------------------------------------------------------------- lifecycle

    def invoke(self, context, event):
        p = prefs.get(context)
        clip = _clip(context)
        if clip is None:
            self.report({"ERROR"}, "no clip loaded")
            return {"CANCELLED"}
        self._root = bpy.path.abspath(p.assist_root) if p else ""
        if not self._root:
            self.report({"ERROR"}, "set the Blender_assitant folder in Preferences")
            return {"CANCELLED"}

        try:
            client.ensure(self._root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            r = client.start_seed(self._root, clip_info(context, clip),
                                  {"target": self.target,
                                   "spacing_px": self.spacing_px,
                                   "reject_movers": self.reject_movers})
        except client.SidecarError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self._job = r["id"]
        self._phase = "waiting"
        self._status(context, "asking the model where to track ...")
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _status(self, context, msg):
        context.workspace.status_text_set("Auto-seed: %s" % msg)
        for area in context.window.screen.areas:
            if area.type == "CLIP_EDITOR":
                area.tag_redraw()

    def _finish(self, context, level, msg):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if self._proxy is not None:
            self._proxy.__exit__(None, None, None)
            self._proxy = None
        context.workspace.status_text_set(None)
        self.report({level}, msg)
        return {"FINISHED"} if level != "ERROR" else {"CANCELLED"}

    # ---------------------------------------------------------------- modal

    def modal(self, context, event):
        if event.type in ("ESC",):
            if self._phase == "waiting":
                client.cancel(self._root, self._job)
            return self._finish(context, "WARNING", "cancelled")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            if self._phase == "waiting":
                return self._tick_waiting(context)
            if self._phase == "tracking":
                return self._tick_tracking(context)
        except client.SidecarError as exc:
            return self._finish(context, "ERROR", str(exc))
        except Exception as exc:                     # noqa: BLE001
            # Never let a traceback dialog interrupt an artist. The detail is in the log.
            print("[btr] auto-seed failed: %r" % (exc,))
            import traceback
            traceback.print_exc()
            return self._finish(context, "ERROR",
                                "%s: %s" % (type(exc).__name__, str(exc)[:120]))
        return {"RUNNING_MODAL"}

    def _tick_waiting(self, context):
        st = client.poll(self._root, self._job)
        if st["state"] in ("queued", "running"):
            self._status(context, "%s  (%.0fs)" % (st.get("stage") or "working",
                                                   st.get("seconds", 0)))
            return {"RUNNING_MODAL"}
        if st["state"] == "error":
            return self._finish(context, "ERROR", st["error"]["message"])
        if st["state"] == "cancelled":
            return self._finish(context, "WARNING", "cancelled")

        data = st["result"]
        self._seeds = data["seeds"]
        clip = _clip(context)
        if (clip.size[0], clip.size[1]) != (int(data["width"]), int(data["height"])):
            return self._finish(context, "ERROR",
                                "seeds are for %dx%d but this clip is %dx%d"
                                % (data["width"], data["height"],
                                   clip.size[0], clip.size[1]))

        opts = track_core.Opts()
        space = context.space_data
        p = prefs.get(context)
        self._proxy = track_core.FullResolution(
            clip, space, enabled=bool(p.force_full_res) if p else True)
        self._proxy.__enter__()

        track_core.apply_settings(clip, opts)
        self._records = track_core.seed_tracks(clip, self._seeds, opts)

        err = track_core.seed_roundtrip_error(self._records, self._seeds)
        if err >= 0.01:
            # Mandatory gate. A seed that failed to land is silently re-read from the
            # nearest marker, and the result is a full-length track that never touched its
            # own feature -- which every position metric scores as good.
            return self._finish(context, "ERROR",
                                "seed round-trip FAILED (%.4f px) -- markers did not land"
                                % err)

        if not self.track_after:
            return self._finish(context, "INFO",
                                "%d seeds placed, not tracked (round-trip %.4f px)"
                                % (len(self._records), err))

        win, area, region = three_de.clip_editor(context, clip)
        scene = context.scene
        self._ctx = (win, area, region, clip, scene)
        guide = {s["id"]: s.get("guide") or {} for s in self._seeds}
        self._stats = {}
        self._gen = track_core.track_job(self._ctx, self._records,
                                         int(data["frames"]), guide, opts,
                                         stats=self._stats)
        self._opts = opts
        self._phase = "tracking"
        self._status(context, "tracking %d markers ..." % len(self._records))
        return {"RUNNING_MODAL"}

    def _tick_tracking(self, context):
        import time
        deadline = time.time() + TICK_SECONDS
        done = False
        while time.time() < deadline:
            try:
                next(self._gen)
            except StopIteration:
                done = True
                break
        st = self._stats
        self._status(context, "frame %d/%d   %d live   %d dead   %d clamped"
                     % (st.get("frame", 0), st.get("total", 0), st.get("alive", 0),
                        st.get("deaths", 0), st.get("clamped", 0)))
        if not (done or st.get("done")):
            return {"RUNNING_MODAL"}

        self._status(context, "backward pass (not cancellable) ...")
        calls = track_core.track_backward_pass(self._ctx, self._records, self._opts)
        n = len(self._records)
        spans = [len([m for m in r["t"].markers if not m.mute]) for r in self._records]
        spans.sort()
        median = spans[len(spans) // 2] if spans else 0
        return self._finish(context, "INFO",
                            "%d tracks, median span %d frames, %d deaths, %d backward calls"
                            % (n, median, st.get("deaths", 0), calls))


CLASSES = (CLIP_OT_btr_sidecar, CLIP_OT_btr_autoseed)
