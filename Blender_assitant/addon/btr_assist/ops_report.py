"""Will this shot solve? Ask before exporting, not after.

Every other check in this addon judges ONE track: is it still on its feature, did it jump,
did its box run away. An artist does not deliver one track, they deliver a set, and a set can
be made entirely of good tracks and still be useless -- all on one plane, all in one corner,
eleven frames in the middle held up by four of them, six of the strongest sitting on a truck.

None of that is visible in the viewport. It is visible after an export, a solve and a look at
the error, which is an expensive way to find out the shot was a nodal pan.

The arithmetic is in `sidecar/coverage.py` and is proved against cameras whose answer is
known by construction. This is the part that collects the tracks, asks, and puts the answer
somewhere an artist can act on it.

Read-only about the tracks themselves. It can SELECT the ones it wants looked at, which is
navigation rather than editing -- no marker is moved, muted or deleted -- and that is a
switch the artist can turn off.
"""

import os
import time

import bpy
from bpy.props import BoolProperty, IntProperty

from . import client, prefs
from .ops_assist import clip_info, live_frames, marker_to_image_px

#: Name of the text datablock the full report is written to. Reused rather than accumulating
#: "report.001", "report.002" -- an artist wants the current answer, and Blender's text editor
#: is the one place in this app that can hold more than a status line.
TEXT_NAME = "BTR shot report"


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


def collect(clip, selected_only=False):
    """Every live marker of every track, in image pixels, y-DOWN.

    Muted markers are left out on purpose: a muted marker is one the artist has already said
    is not to be trusted, and counting it would report coverage the export will not have.
    """
    w, h = clip.size
    out = {}
    for tr in clip.tracking.objects.active.tracks if clip.tracking.objects.active \
            else clip.tracking.tracks:
        if selected_only and not tr.select:
            continue
        pts = {}
        for f in live_frames(tr):
            m = tr.markers.find_frame(f, exact=True)
            if m is None:
                continue
            x, y = marker_to_image_px(m, w, h)
            pts[str(int(f))] = [float(x), float(y)]
        if len(pts) >= 2:
            out[tr.name] = pts
    return out


def render(rep):
    """The report as lines an artist reads top to bottom, worst news first."""
    out = []
    if not rep.get("tracks"):
        return ["no tracks to judge"]
    if rep.get("size_warning"):
        out.append("!! " + rep["size_warning"])
        out.append("")

    lo, hi = rep["frames"]
    out.append("%d tracks over f%d-f%d, median span %d frames"
               % (rep["tracks"], lo, hi, rep["median_span"]))
    out.append("live at once: median %d, fewest %d" % (rep["median_live"], rep["min_live"]))
    out.append("")

    par = rep["parallax"]
    out.append("PARALLAX: %s" % par["verdict"].upper())
    out.append("  %s" % par["reason"])
    out.append("")

    thin = rep["thin_runs"]
    if thin:
        out.append("THIN -- under %d tracks, so the solve has little to hold here:"
                   % rep["floor"])
        for a, b, worst in thin:
            out.append("  f%d-f%d   as few as %d" % (a, b, worst))
    else:
        out.append("No stretch falls under %d simultaneous tracks." % rep["floor"])
    out.append("")

    bad = rep["suspect"]
    if bad:
        out.append("%d track(s) disagree with the camera motion the rest agree on." % len(bad))
        out.append("Each is on something that moves, or has slid off its feature:")
        for s in bad:
            out.append("  %-18s failed %d of %d checks" % (s["id"], s["failed"], s["tested"]))
    else:
        out.append("Nothing disagrees with the camera motion the rest agree on.")
        out.append("  That is not the same as 'no movers': a point travelling ALONG its own")
        out.append("  epipolar line looks exactly like a static point at another depth, and")
        out.append("  no amount of checking sees it. Measured: a mover 0.6 deg off the")
        out.append("  epipolar direction was missed on every pair, one at 2.4 deg was caught")
        out.append("  on every pair.")
    out.append("")

    q = rep.get("quality") or {}
    grades = rep.get("grades") or {}
    if q.get("tracks"):
        v = q.get("verdicts") or {}
        out.append("QUALITY OF THE TRACKS THEMSELVES")
        out.append("  %d good, %d worth a look, %d poor"
                   % (v.get("good", 0), v.get("check", 0), v.get("poor", 0)))
        out.append("  steadiness: median %s px, worst tenth %s px, worst %s px"
                   % (q.get("jitter_p50"), q.get("jitter_p90"), q.get("jitter_max")))
        bad = sorted(((tid, g) for tid, g in grades.items() if g["verdict"] != "good"),
                     key=lambda kv: -(kv[1].get("jitter_px") or 0))
        for tid, g in bad[:12]:
            out.append("  %-18s %-6s %s" % (tid, g["verdict"], "; ".join(g["why"])))
        if len(bad) > 12:
            out.append("  ... and %d more" % (len(bad) - 12))
        out.append("  This does NOT include drift. A track that slid off its feature is still")
        out.append("  smooth, so it grades well here -- use 'Find and fix slides' for that.")
        out.append("")

    bare = rep["bare_cells"]
    if bare:
        out.append("Regions with little coverage across the shot:")
        for c in bare:
            out.append("  %-16s holds a track on %d%% of frames"
                       % (c["name"], round(c["present_share"] * 100)))
    else:
        out.append("Every ninth of the frame holds a track for most of the shot.")
    return out


class CLIP_OT_btr_shot_report(bpy.types.Operator):
    bl_idname = "clip.btr_shot_report"
    bl_label = "Will this shot solve?"
    bl_description = ("Judge the tracks as a SET rather than one at a time: how many are live "
                      "on each frame, which regions of the frame have none, whether there is "
                      "any parallax to solve with at all, and which tracks disagree with the "
                      "camera motion the rest agree on. Answers before an export and a solve "
                      "do. Reads the tracks and changes none of them")
    bl_options = {"REGISTER"}

    selected_only: BoolProperty(
        name="Selected tracks only", default=False,
        description="Judge only the selected tracks. Off judges everything the export would "
                    "contain, which is usually the question being asked")
    select_suspect: BoolProperty(
        name="Select the ones to look at", default=True,
        description="Leave the disagreeing tracks selected when the report finishes, so they "
                    "can be stepped through immediately. This selects; it moves, mutes and "
                    "deletes nothing")
    floor: IntProperty(
        name="Tracks a frame needs", default=8, min=3, max=60,
        description="Below this many simultaneous tracks a frame is called thin. Eight is a "
                    "working number, not a measured one: a solve wants comfortably more than "
                    "the degrees of freedom it is estimating, with margin for the tracks that "
                    "turn out to be wrong")

    @classmethod
    def poll(cls, context):
        clip = _clip(context)
        return clip is not None and len(clip.tracking.tracks) > 0

    def execute(self, context):
        clip = _clip(context)
        tracks = collect(clip, self.selected_only)
        if len(tracks) < 2:
            self.report({"WARNING"}, "need at least two tracks with two frames each")
            return {"CANCELLED"}

        p = prefs.get(context)
        root = bpy.path.abspath(p.assist_root) if p and p.assist_root else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            client.ensure(root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            job = client.start_report(root, clip_info(context, clip), tracks,
                                      {"floor": int(self.floor)})
        except Exception as exc:                                      # noqa: BLE001
            self.report({"ERROR"}, "sidecar: %s" % exc)
            return {"CANCELLED"}

        # Blocking, deliberately. This is CPU-only track arithmetic that finishes in well
        # under a second on a few hundred tracks -- a modal operator and a poll loop would be
        # more machinery than the job, and unlike tracking there is no GPU to hand over.
        rep, waited = None, 0.0
        while waited < 120.0:
            st = client.poll(root, job["id"])
            if st["state"] == "done":
                rep = st["result"]
                break
            if st["state"] == "error":
                self.report({"ERROR"}, st["error"]["message"])
                return {"CANCELLED"}
            time.sleep(0.1)
            waited += 0.1
        if rep is None:
            self.report({"ERROR"}, "the report did not finish")
            return {"CANCELLED"}

        lines = render(rep)
        txt = bpy.data.texts.get(TEXT_NAME) or bpy.data.texts.new(TEXT_NAME)
        txt.clear()
        txt.write("%s   %s\n\n" % (TEXT_NAME, time.strftime("%Y-%m-%d %H:%M:%S")))
        txt.write("\n".join(lines) + "\n")
        for ln in lines:
            print("[report] %s" % ln)

        if self.select_suspect:
            # Everything worth an artist's eye: tracks that disagree with the camera motion,
            # and tracks that are poor in their own right. One selection, because the artist
            # is about to step through them and does not care which check objected.
            want = set(s["id"] for s in rep["suspect"])
            want |= set(tid for tid, g in (rep.get("grades") or {}).items()
                        if g["verdict"] == "poor")
            for tr in clip.tracking.tracks:
                tr.select = tr.name in want

        par = rep["parallax"]["verdict"]
        bad = len(rep["suspect"])
        thin = len(rep["thin_runs"])
        poor = sum(1 for g in (rep.get("grades") or {}).values() if g["verdict"] == "poor")
        head = ("parallax %s; %d poor track(s); %d suspect; %d thin stretch(es)"
                % (par, poor, bad, thin))
        # A degenerate shot is the finding that changes what the artist does next, so it is
        # the one that gets to interrupt.
        self.report({"WARNING"} if par == "degenerate" else {"INFO"},
                    "%s -- full report in the Text editor as '%s'" % (head, TEXT_NAME))
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_shot_report,)
