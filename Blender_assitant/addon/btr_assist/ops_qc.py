"""Does each track still end on the pattern the artist seeded it with?

The one question an artist asks about a finished track, and the addon had no way to answer
it. Everything else here judges a track WHILE it is being made -- the scale watch, the hold
check, the jump check -- and all of them run inside the assist loop. A track imported from
3DE, tracked by hand, or made before any of this existed got nothing.

The check is the simplest one there is: take the pattern box on the track's FIRST live
frame, correlate it at the position the track claims on its LAST live frame, and compare
that score against what the same patch scored over the track's own opening frames.

Judged against the track's OWN opening, not an absolute number, for the reason measured
repeatedly on this footage: a patch on low-contrast dirt scores 0.53-0.72 against the very
next frame while tracking perfectly well, and a patch on a hard corner scores 0.99. An
absolute pass mark condemns the first and waves the second through. A score that was never
high cannot fall.

Read-only. It reports and changes nothing -- a QC pass that edits the thing it is checking
is not a QC pass.
"""

import os
import time

import bpy
from bpy.props import BoolProperty

from . import client, prefs, three_de
from .ops_assist import (clip_info, live_frames, marker_pattern_box)


def _clip(context):
    sd = getattr(context, "space_data", None)
    return getattr(sd, "clip", None)


#: How much better a nearby match has to be before the track is judged to be OFF its feature.
#:
#: This, and not the raw score, is the verdict. A long track's feature legitimately changes
#: appearance -- measured on the artist's own 250-frame hand track, its seed patch scores 0.73
#: at the end, which an absolute pass mark would condemn. But it is still the BEST match
#: available there. The assist track that drifted onto a lookalike scores 0.50 while 0.98 sits
#: 23 px away. Gain of 0.055 against 0.478 -- an order of magnitude apart, and the reading
#: survives the feature simply looking different.
PROBE_GAIN = 0.10

#: How far to look for that better match. Wide enough to find the neighbour a track slid onto,
#: narrow enough that it is still the same part of the plate.
PROBE_RADIUS = 60.0

#: The end-vs-opening ratio is REPORTED, never used as the verdict -- see PROBE_GAIN. It is
#: the number an artist reads to see how much the feature has changed.
END_RATIO = 0.75

#: How many opening frames define "what this track was holding". Enough to survive one bad
#: frame, short enough to stay inside the part anchored to the artist's own seed.
HEAD = 8


class CLIP_OT_btr_qc_ends(bpy.types.Operator):
    bl_idname = "clip.btr_qc_ends"
    bl_label = "Check tracks end on my pattern"
    bl_description = ("Correlate the pattern each track STARTED with against where it ENDS, "
                      "and report the ones that no longer sit on it. Works on any track -- "
                      "hand-tracked, imported, or from the assistant. Changes nothing")

    selected_only: BoolProperty(
        name="Selected tracks only", default=True,
        description="Off checks every track in the clip")

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

        w, h = clip.size
        reqs, meta = [], {}
        for tr in three_de.active_tracks(clip):
            if self.selected_only and not tr.select:
                continue
            fr = live_frames(tr)
            if len(fr) < 3:
                continue
            first, last = fr[0], fr[-1]
            m0 = tr.markers.find_frame(first, exact=True)
            if m0 is None:
                continue
            cx, cy, pw, ph = marker_pattern_box(m0, w, h)
            # The opening frames plus the final one. `hold_check` scores the seed patch at
            # every position given, so this is one request and one decode per frame.
            want = [f for f in fr[:HEAD]] + [last]
            path = []
            for f in want:
                m = tr.markers.find_frame(int(f), exact=True)
                if m is not None:
                    path.append([int(f), m.co[0] * w, (1.0 - m.co[1]) * h])
            if len(path) < 3:
                continue
            reqs.append({"id": tr.name,
                         "pattern": {"frame": first, "cx": cx, "cy": cy, "w": pw, "h": ph},
                         "path": sorted(path)})
            meta[tr.name] = (first, last, len(fr))

        if not reqs:
            self.report({"WARNING"}, "no tracks with enough frames to check")
            return {"CANCELLED"}

        try:
            client.ensure(root, bpy.path.abspath(p.python_exe) if p else "",
                          p.port if p else 0)
            r = client.start_hold(root, clip_info(context, clip), reqs,
                                  {"probe_radius": PROBE_RADIUS})
        except client.SidecarError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # Blocking, deliberately: this is a check the artist asked for and waits on, not a
        # phase of a long run. It is one correlation per track per opening frame.
        deadline = time.time() + 300.0
        st = None
        while time.time() < deadline:
            st = client.poll(root, r["id"])
            if st["state"] not in ("queued", "running"):
                break
            time.sleep(0.3)
        if st is None or st["state"] != "done":
            self.report({"ERROR"}, "the check did not finish")
            return {"CANCELLED"}

        bad, ok, unknown = [], [], []
        for res in (st["result"] or {}).get("tracks") or []:
            name = res["id"]
            first, last, n = meta.get(name, (0, 0, 0))
            rows = [r for r in (res.get("scores") or []) if r[1] is not None]
            if not rows:
                unknown.append((name, "nothing could be correlated"))
                continue
            end_row = next((r for r in rows if r[0] == last), None)
            head = [r[1] for r in rows if r[0] != last]
            if end_row is None or not head:
                unknown.append((name, "no score at the last frame"))
                continue
            end = end_row[1]
            gain = end_row[3] if len(end_row) > 3 else None
            dist = end_row[4] if len(end_row) > 4 else None
            base = sorted(head)[len(head) // 2]
            ratio = end / base if base > 0 else 0.0
            line = ("%s: f%d-%d, opened at %.2f, ends at %.2f (%.0f%% of its own)"
                    % (name, first, last, base, end, ratio * 100.0))
            if gain is None:
                unknown.append((name, "could not look for a better match nearby"))
                continue
            if gain > PROBE_GAIN:
                bad.append("%s -- %.2f sits %.0f px away" % (line, end + gain, dist or 0.0))
            else:
                ok.append("%s, and nothing better is nearby" % line)

        print("[assist] --- QC: does each track end on the pattern it started with? ---",
              flush=True)
        for line in bad:
            print("[assist]   NOT ON IT  %s" % line, flush=True)
        for line in ok:
            print("[assist]   on it      %s" % line, flush=True)
        for name, why in unknown:
            print("[assist]   unknown    %s: %s" % (name, why), flush=True)

        if bad:
            self.report({"WARNING"},
                        "%d of %d track(s) do NOT end on the pattern they started with "
                        "-- see the console" % (len(bad), len(bad) + len(ok)))
        else:
            self.report({"INFO"}, "all %d track(s) end on the pattern they started with"
                        % len(ok))
        return {"FINISHED"}


CLASSES = (CLIP_OT_btr_qc_ends,)
