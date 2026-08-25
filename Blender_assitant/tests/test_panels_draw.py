"""Call every panel's draw() against the REAL installed extension.

Written after an ImportError that only fired inside `draw()`: `bl_info` is stripped off the
module by Blender's extension loader, so `from . import bl_info` raised once per redraw.
Registration succeeded, operators existed, and the earlier check passed anyway -- because it
never drew anything.

Two things this exists to enforce:

  * a panel that raises does so on EVERY redraw, burying the console and making the whole
    addon look broken when only a label is
  * the installed copy is what the artist runs. Testing the source tree tests a different
    file, and has already cost two rounds of chasing a fix that was never installed.

    blender.exe --background -noaudio --python tests\\test_panels_draw.py -- \\
        --plate <mp4 or folder>
"""

import os
import sys
import traceback

import bpy

EXT = "bl_ext.user_default.btr_assist"


def log(m):
    print("[draw] %s" % m, flush=True)


class StubLayout:
    """Records calls and returns itself, so a real draw() body runs to completion.

    `cls()` cannot be used to make a Panel instance -- `bpy_struct.__new__` refuses -- and
    Blender will not hand out a UILayout headless. So `draw()` is called unbound with a
    stand-in. This tests the PYTHON in draw(), which is where the ImportError lived; it does
    not test Blender's layout engine, and does not pretend to.
    """

    def __init__(self):
        self.calls = 0
        self.scale_y = 1.0
        self.enabled = True
        self.alignment = "EXPAND"
        self.use_property_split = False

    #: Every icon Blender actually has, read from the real enum. A stub that accepts any
    #: keyword verifies that draw() RUNS, not that it runs in Blender -- and an icon name
    #: that does not exist raises only in a live session. That is exactly how
    #: `SEQUENCE_COLOR_03` reached an artist's screen with this test green.
    ICONS = None

    @classmethod
    def _icons(cls):
        if cls.ICONS is None:
            import bpy
            params = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"]
            cls.ICONS = {i.identifier for i in params.enum_items}
        return cls.ICONS

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls += 1
            icon = kw.get("icon")
            if icon is not None and icon not in self._icons():
                raise ValueError("icon %r does not exist in this Blender "
                                 "(UILayout.%s)" % (icon, name))
            return self
        return call


class FakePanel:
    def __init__(self, layout):
        self.layout = layout


def load_clip(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in
                       (".exr", ".dpx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        return bpy.data.movieclips.load(os.path.join(path, names[0]))
    return bpy.data.movieclips.load(path)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    mod = importlib.import_module(EXT)
    log("installed version: %s" % (getattr(mod, "VERSION", None),))

    clip = load_clip(plate) if plate else None
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    if clip is not None:
        area.spaces.active.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")

    # Walk the whole subclass TREE, not `Panel.__subclasses__()`, which returns direct
    # subclasses only. The moment the panels grew a shared base class they vanished from
    # that list and this test went green having drawn nothing at all.
    def descend(cls, seen):
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                descend(sub, seen)
        return seen

    panels = sorted((c for c in descend(bpy.types.Panel, set())
                     if c.__name__.startswith("CLIP_PT_btr")),
                    key=lambda c: c.__name__)
    log("panels found: %s" % [p.__name__ for p in panels])
    # A discovery that finds nothing must FAIL. Reporting PASS for zero panels is how this
    # test hid a rewrite of every panel in the addon.
    if not panels:
        log("PANEL DRAW: FAIL -- no panels discovered")
        sys.exit(1)

    # Tracks, because a panel's interesting branches are the CONDITIONAL ones and an empty
    # clip reaches none of them. The clip is loaded fresh, so before this every state below
    # was identical: 0 tracks, 0 selected, 0 unread -- and the branch that draws the
    # Keep/Drop box was never executed by this test at all. That is how an icon name which
    # does not exist in Blender 5.2 shipped to an artist with this file green.
    def fixture(state):
        obj = clip.tracking.objects.active
        # `tracks.remove()` does not exist on this collection; the addon already carries the
        # supported way to clear them.
        three_de = sys.modules["%s.three_de" % EXT]
        three_de.delete_all_tracks(bpy.context, clip)
        if state == "empty":
            return
        tr = obj.tracks.new(name="T_plain", frame=1)
        tr.markers[0].co = (0.5, 0.5)
        tr.select = state in ("selected", "unread")
        if state == "unread":
            # A muted marker ahead of the seed is what a proposed resume looks like, and it
            # is the only thing that makes the Keep/Drop box draw.
            m = tr.markers.insert_frame(10, co=(0.55, 0.55))
            m.mute = True

    STATES = ("empty", "unselected", "selected", "unread")

    failures = []
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=area.spaces.active,
                                   edit_movieclip=clip, scene=bpy.context.scene):
        for cls in panels:
            for label in STATES:
                fixture(label)
                try:
                    if hasattr(cls, "poll") and not cls.poll(bpy.context):
                        log("%-24s %-15s poll() False, skipped" % (cls.__name__, label))
                        continue
                    lay = StubLayout()
                    cls.draw(FakePanel(lay), bpy.context)
                    log("%-24s %-15s OK  (%d layout calls)"
                        % (cls.__name__, label, lay.calls))
                except Exception as exc:                 # noqa: BLE001
                    failures.append((cls.__name__, label, traceback.format_exc()))
                    log("%-24s %-15s FAIL %s: %s"
                        % (cls.__name__, label, type(exc).__name__, exc))

    log("=" * 60)
    if failures:
        for name, label, tb in failures:
            log("FAILED %s (%s):\n%s" % (name, label, tb))
        log("PANEL DRAW: FAIL (%d)" % len(failures))
        raise SystemExit(1)
    log("PANEL DRAW: PASS")


main()
