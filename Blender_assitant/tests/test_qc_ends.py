"""The end-of-track QC, against the artist's own files.

"Always compare the pattern the track ends at with the pattern the user seeded" -- the one
question an artist asks about a finished track, and the only check here that works on a track
the assistant did not make.

Driven with the REAL tracks from SH006, imported into Blender from 3DE and checked through
the operator the artist presses:

  * `Track.002` -- the assist output that drifts onto a lookalike after the wire crosses. It
    must be reported as NOT on its pattern.
  * `Track.003` -- the artist's hand track of the SAME feature over the SAME frames, which
    sees the same plate ambiguity and stays on it. It must pass.

That pair is what makes this a test rather than a threshold: both tracks meet the same
ambiguous texture at the same frames, so anything that condemns one and not the other has to
be reading the track, not the plate.

    blender.exe --background -noaudio --python tests\\test_qc_ends.py -- \\
        --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import argparse
import os
import sys

import bpy

EXT = "bl_ext.user_default.btr_assist"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))

FAILURES = []


def log(m):
    print("[qc] %s" % m, flush=True)


def fail(m):
    FAILURES.append(m)
    log("FAIL %s" % m)


def read_3de(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 0
    n = int(tok[i])
    i += 1
    out = []
    for _ in range(n):
        name = tok[i]
        i += 1
        i += 1
        cnt = int(tok[i])
        i += 1
        pts = []
        for _ in range(cnt):
            pts.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2])))
            i += 3
        out.append((name, pts))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default=r"D:\Jefrin\IN\SH006.mp4")
    a = ap.parse_args(argv)

    bpy.ops.preferences.addon_enable(module=EXT)
    pr = bpy.context.preferences.addons[EXT].preferences
    pr.assist_root = ASSIST
    py = os.path.join(ASSIST, "runtime", "python311", "python.exe")
    if not os.path.isfile(py):
        py = r"D:\Jefrin\batch_tracker_v001_starter\runtime\python311\python.exe"
    pr.python_exe = py

    three_de = sys.modules["%s.three_de" % EXT]
    clip = bpy.data.movieclips.load(os.path.abspath(a.plate))
    w, h = clip.size
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda ar: ar.width * ar.height)
    area.type = "CLIP_EDITOR"
    area.spaces.active.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")

    wanted = {
        "Track.002": (os.path.join(HERE, "reacquretracke_assist tracked_v003.txt"),
                      "the assist output that drifts onto a lookalike", True),
        "Track.003": (os.path.join(HERE, "track2 manual tracked.txt"),
                      "the artist's hand track of the same feature", False),
    }
    tracks = three_de.active_tracks(clip)
    made = {}
    for name, (path, what, should_fail) in wanted.items():
        pts = dict(read_3de(path)).get(name)
        if pts is None:
            fail("%s not found in %s" % (name, os.path.basename(path)))
            continue
        # Same box the artist used, so the check sees what they saw.
        tr = tracks.new(name=name, frame=pts[0][0])
        tr.select = True
        for f, x, y in pts:
            m = tr.markers.find_frame(int(f), exact=True)
            if m is None:
                m = tr.markers.insert_frame(int(f))
            # 3DE is y-UP; marker.co is normalised y-UP too, so no flip -- only a scale.
            m.co = (x / w, y / h)
            m.mute = False
            hu, hv = 41.2 / 2.0 / w, 41.2 / 2.0 / h
            m.pattern_corners = ((-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv))
            su, sv = 113.1 / 2.0 / w, 113.1 / 2.0 / h
            m.search_min, m.search_max = (-su, -sv), (su, sv)
        made[name] = (tr, what, should_fail)
        log("loaded %-11s %-46s %d frames" % (name, what, len(pts)))

    if len(made) != 2:
        log("QC ENDS: FAIL -- could not load both tracks")
        sys.exit(1)

    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=area.spaces.active,
                                   edit_movieclip=clip, scene=bpy.context.scene):
        with redirect_stdout(buf):
            res = bpy.ops.clip.btr_qc_ends(selected_only=True)
    out = buf.getvalue()
    print(out, flush=True)
    if "FINISHED" not in str(res):
        fail("operator returned %r" % (res,))

    for name, (_tr, what, should_fail) in made.items():
        flagged = any(name in ln and "NOT ON IT" in ln for ln in out.splitlines())
        if should_fail and not flagged:
            fail("%s (%s) was NOT flagged -- it drifts onto a lookalike" % (name, what))
        elif not should_fail and flagged:
            fail("%s (%s) WAS flagged -- that is the artist's own correct track"
                 % (name, what))
        else:
            log("%-11s %-46s %s" % (name, what,
                                    "flagged, correctly" if flagged else "passed, correctly"))

    log("QC ENDS: %s" % ("FAIL" if FAILURES else "PASS"))
    sys.exit(1 if FAILURES else 0)


main()
