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

    def __getattr__(self, name):
        def call(*a, **kw):
            self.calls += 1
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

    panels = [c for c in bpy.types.Panel.__subclasses__()
              if c.__name__.startswith("CLIP_PT_btr")]
    log("panels found: %s" % sorted(p.__name__ for p in panels))

    failures = []
    with bpy.context.temp_override(window=win, area=area, region=region,
                                   space_data=area.spaces.active,
                                   edit_movieclip=clip, scene=bpy.context.scene):
        for cls in panels:
            for label, sel in (("no selection", False), ("with selection", True)):
                if clip is not None:
                    for t in clip.tracking.objects.active.tracks:
                        t.select = sel
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
