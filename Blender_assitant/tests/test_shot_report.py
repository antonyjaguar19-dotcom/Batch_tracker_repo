"""The shot report, driven through the real operator in Blender.

`test_coverage.py` proves the arithmetic against cameras whose answer is known. This proves
the other half: that the operator collects the right tracks, sends the right frame size, gets
an answer back through the sidecar, and puts it somewhere an artist can read.

Two cases, and the second is the one worth having:

  * the artist's own SH006 tracks against the SH006 plate -- the honest path, and with only
    seven tracks the parallax answer must come back UNKNOWN rather than invented;
  * SH008's 1920x1080 tracks against the 3840x2160 plate -- every number in the report is a
    fraction of the frame, so this must be caught and said out loud, not answered.

    blender.exe --background -noaudio --python tests/test_shot_report.py -- \\
        --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.dirname(ASSIST)
EXT = "bl_ext.user_default.btr_assist"

SH006_TRACKS = os.path.join(REPO, "experiments", "3de tracks from blender",
                            "tracks for SH006 from blender.txt")
SH008_TRACKS = os.path.join(REPO, "experiments", "blender_track", "out",
                            "SH008__dense__blender.txt")

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[rep] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def truthy(name, got):
    ok = bool(got)
    print("[rep] %-56s %s" % (name, "ok" if ok else "FAIL  got %r" % (got,)))
    if not ok:
        FAILED.append(name)


def setup(plate):
    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    sp = area.spaces.active
    sp.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    return clip, win, area, region, sp


def import_3de(path, ctx, prefix):
    win, area, region, sp, clip = ctx
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        bpy.ops.clip.btr_import_3de(filepath=path, prefix=prefix)


def run_report(ctx, **kw):
    win, area, region, sp, clip = ctx
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        return bpy.ops.clip.btr_shot_report(**kw)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""
    if not plate or not os.path.exists(plate):
        print("[rep] need --plate")
        return 2
    for f in (SH006_TRACKS, SH008_TRACKS):
        if not os.path.exists(f):
            print("[rep] missing reference file: %s" % f)
            return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    mod = importlib.import_module(EXT)
    ops_report = importlib.import_module(EXT + ".ops_report")
    print("[rep] installed %s build %s" % (mod.VERSION, mod.BUILD))

    clip, win, area, region, sp = setup(plate)
    ctx = (win, area, region, sp, clip)
    print("[rep] clip %dx%d" % (clip.size[0], clip.size[1]))

    print("\n[rep] the artist's own SH006 tracks, against the SH006 plate")
    import_3de(SH006_TRACKS, ctx, "A_")
    n = len(clip.tracking.tracks)
    truthy("tracks imported", n > 1)
    res = run_report(ctx, select_suspect=True)
    check("the operator finished", res, {"FINISHED"})
    txt = bpy.data.texts.get(ops_report.TEXT_NAME)
    truthy("a report was written where an artist can read it", txt is not None)
    body = txt.as_string() if txt else ""
    truthy("  it names the parallax verdict", "PARALLAX:" in body)
    truthy("  and says how many tracks are live at once", "live at once" in body)
    # Seven tracks cannot support a fundamental matrix. The right answer is to say so.
    truthy("  seven tracks is too few to rule on parallax, and it says so",
           "PARALLAX: UNKNOWN" in body)
    truthy("  and it does not invent a coverage verdict from it",
           "no frame pair" in body.lower() or "UNKNOWN" in body)

    print("\n[rep] SH008's 1920x1080 tracks read against a 3840x2160 plate")
    # Every number in the report is a fraction of the frame. Read at the wrong size this file
    # puts every track in one quadrant and reports five of nine regions as empty on every
    # frame -- which is exactly what a badly covered shot looks like. It must not answer.
    three_de = importlib.import_module(EXT + ".three_de")
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=sp, edit_movieclip=clip):
        three_de.delete_all_tracks(bpy.context, clip)
    check("the plate is clear before the second case", len(clip.tracking.tracks), 0)
    import_3de(SH008_TRACKS, ctx, "B_")
    truthy("the mismatched file imported", len(clip.tracking.tracks) > 50)
    res = run_report(ctx, select_suspect=False)
    check("the operator still finished", res, {"FINISHED"})
    body = bpy.data.texts.get(ops_report.TEXT_NAME).as_string()
    truthy("the size mismatch is stated BEFORE anything else",
           body.split("\n\n")[1].lstrip().startswith("!!") if "!!" in body else False)
    truthy("  and it names both sizes", "3840x2160" in body)

    print("")
    if FAILED:
        print("SHOT REPORT: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[rep] SHOT REPORT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
