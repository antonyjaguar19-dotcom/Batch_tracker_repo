"""A new marker keeps its on-screen size, and gives the artist's setting back.

Drives the real reconciler against a real clip editor at real zooms, and then adds a marker
through Blender's own path to check the size actually lands on the track -- the arithmetic
being right is not the same claim as Ctrl-click producing that box, and only the second one
is what was asked for.

    blender.exe --background -noaudio --python tests/test_click_size.py -- \\
        --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "addon"))

from btr_assist import click_size                                     # noqa: E402

#: The installed extension, whose preferences the reconciler actually reads.
EXT = "bl_ext.user_default.btr_assist"

FAILED = []


def log(m):
    print("[box] %s" % m)


def check(name, got, want):
    ok = got == want
    print("[box] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def near(name, got, want, tol):
    ok = got is not None and abs(got - want) <= tol
    print("[box] %-56s %s" % (name, "ok  %.1f" % got if ok
                              else "FAIL  got %r want %r +-%r" % (got, want, tol)))
    if not ok:
        FAILED.append(name)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""

    print("[box] the arithmetic, with no Blender state involved")
    uhd = (3840, 2160)
    # The complaint, in numbers: fit a 4K plate to a window and the zoom is about 26 %.
    near("40 screen px at 26 % zoom -> ~154 plate px",
         click_size.plate_px_for(40, 26.0, uhd), 153.8, 1.0)
    near("the same 40 px at 100 % zoom is just 40",
         click_size.plate_px_for(40, 100.0, uhd), 40.0, 0.01)
    near("and at 400 % zoom the floor takes over, not 10 px",
         click_size.plate_px_for(40, 400.0, uhd), click_size.MIN_PLATE_PX, 0.01)
    near("zoomed far out, the box may not eat the plate",
         click_size.plate_px_for(40, 1.0, uhd), 2160 * click_size.MAX_PLATE_FRAC, 0.01)
    check("a zoom of zero is refused rather than dividing by it",
          click_size.plate_px_for(40, 0.0, uhd), None)
    check("  and so is a negative one", click_size.plate_px_for(40, -5.0, uhd), None)
    # Same screen size, two plates: the whole point is that the PLATE number moves.
    hd = click_size.plate_px_for(40, 100.0, (1920, 1080))
    k4 = click_size.plate_px_for(40, 100.0, uhd)
    check("at equal zoom the plate size does not change the answer", hd, k4)

    if not plate or not os.path.exists(plate):
        print("[box] no --plate given; skipping the part that needs a clip editor")
        return 1 if FAILED else 0

    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda ar: ar.width * ar.height)
    area.type = "CLIP_EDITOR"
    space = area.spaces.active
    space.clip = clip
    st = clip.tracking.settings
    st.default_pattern_size, st.default_search_size = 21, 71
    mine = (st.default_pattern_size, st.default_search_size)

    # The INSTALLED extension, not the copy on sys.path. The reconciler reads addon
    # preferences, and those only exist for a registered addon -- driving the loose module
    # would test the arithmetic a second time and never run `_apply` at all, which is the
    # part that has to find the clip editor, read its zoom and write to the right clip.
    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    live = importlib.import_module(EXT + ".click_size")
    p = bpy.context.preferences.addons[EXT].preferences
    p.constant_box = True
    p.box_screen_px = 40
    log("installed %s build %s" % (importlib.import_module(EXT).VERSION,
                                   importlib.import_module(EXT).BUILD))
    # register() has to have started the timer. Everything below drives `_apply` by hand,
    # which would pass just as well if nothing were ever scheduled to call it.
    check("register() scheduled the reconciler",
          bpy.app.timers.is_registered(live._apply), True)

    print("\n[box] a real clip editor, at real zooms, through the real reconciler")
    for zoom, want in ((25.0, 160.0), (50.0, 80.0), (100.0, 40.0)):
        space.zoom_percentage = zoom
        live._apply()
        near("zoom %3d %% -> pattern %d plate px" % (zoom, st.default_pattern_size),
             float(st.default_pattern_size), want, 1.0)

        # The claim that matters: a marker created NOW is that size.
        tr = clip.tracking.tracks.new(name="Z%d" % zoom, frame=1)
        m = tr.markers[0]
        w, h = clip.size
        pc = [tuple(c) for c in m.pattern_corners]
        near("  the marker Blender creates is that size on the plate",
             (pc[2][0] - pc[0][0]) * w, float(st.default_pattern_size), 1.0)
        near("  and that size on SCREEN, which is the thing asked for",
             (pc[2][0] - pc[0][0]) * w * (zoom / 100.0), 40.0, 1.5)

    print("\n[box] reading a file off disk must not depend on how zoomed in you are")
    # The side effect this change had to not have: 3DE import creates tracks from the
    # new-marker default, and the QC pass then correlates using that box. With the zoom at
    # 25 % the default is 160 px, and an imported track picking that up would have QC
    # reading a patch six times the size of the feature.
    space.zoom_percentage = 25.0
    live._apply()
    ref = os.path.join(HERE, "reacquretracke_manual.txt")
    before = len(clip.tracking.tracks)
    region = next(r for r in area.regions if r.type == "WINDOW")
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=space, edit_movieclip=clip):
        bpy.ops.clip.btr_import_3de(filepath=ref, prefix="IMP_")
    got = [t for t in clip.tracking.tracks if t.name.startswith("IMP_")]
    check("the file imported", len(clip.tracking.tracks) > before, True)
    if got:
        m0 = got[0].markers[0]
        w, h = clip.size
        pc = [tuple(c) for c in m0.pattern_corners]
        imported = (pc[2][0] - pc[0][0]) * w
        near("imported box is the artist's %d px, not the zoom's %d"
             % (mine[0], st.default_pattern_size), imported, float(mine[0]), 1.0)

    print("\n[box] the artist's own setting is theirs again afterwards")
    # Through the LIVE module, and without planting anything: `_apply` had to have noticed
    # 21/71 on its first tick by itself. Stuffing `_saved` here and then asserting it came
    # back would be a check that cannot fail -- it would pass with the remembering removed.
    check("the reconciler remembered what it found, unprompted",
          live._saved.get(clip.name), mine)
    check("  and it really had changed them", st.default_pattern_size != mine[0], True)
    p.constant_box = False          # the update callback restores
    check("pattern restored", st.default_pattern_size, mine[0])
    check("search restored", st.default_search_size, mine[1])
    check("and nothing is held after the restore", live._saved, {})

    print("")
    if FAILED:
        print("CLICK SIZE: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[box] CLICK SIZE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
