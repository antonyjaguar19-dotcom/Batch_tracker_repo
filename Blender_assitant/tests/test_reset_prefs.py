"""Reset puts the measured defaults back, and leaves this machine's paths alone.

Almost every default in this addon is a number fitted against hand tracks rather than a
taste -- `min_match` 0.66, `pin_radius` 12, the closure trust at 25 px. A session spent
nudging them to chase one awkward shot leaves the tool quietly worse on the next, with no
record of what moved. So there has to be a way back.

The half worth testing hardest is what it must NOT do. `assist_root` and `python_exe`
describe this machine; resetting them turns a settings tidy-up into a broken sidecar, and
the artist finds out by running bootstrap again.

    blender.exe --background -noaudio --python tests/test_reset_prefs.py
"""

import sys

import bpy

EXT = "bl_ext.user_default.btr_assist"
FAILED = []


def check(name, got, want):
    ok = got == want
    print("[rst] %-56s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def truthy(name, got):
    ok = bool(got)
    print("[rst] %-56s %s" % (name, "ok" if ok else "FAIL  got %r" % (got,)))
    if not ok:
        FAILED.append(name)


def main():
    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    mod = importlib.import_module(EXT)
    print("[rst] installed %s build %s" % (mod.VERSION, mod.BUILD))
    p = bpy.context.preferences.addons[EXT].preferences

    rna = p.bl_rna.properties
    defaults = {k: rna[k].default for k in
                ("min_match", "pin_radius", "rounds", "scale_ratio", "box_screen_px")
                if k in rna}
    truthy("there are measured defaults to restore", len(defaults) >= 4)

    # Move them somewhere they are definitely not, including the booleans and the enum, so
    # the reset has something to do on every kind of property this addon uses.
    p.min_match = 0.11
    p.pin_radius = 47.0
    p.rounds = 2
    p.box_screen_px = 199
    p.pin_to_pattern = not rna["pin_to_pattern"].default
    p.track_engine = "COTRACKER" if rna["track_engine"].default == "BLENDER" else "BLENDER"

    # And set the paths to something recognisable, to prove they survive.
    p.assist_root = "//KEEP-THIS-ROOT"
    p.python_exe = "//KEEP-THIS-PYTHON"

    # Blender stores floats as float32, so 0.11 comes back 0.10999999940395355. Comparing
    # exactly here would fail on the storage rather than on the behaviour.
    check("a float really moved", round(p.min_match, 4), 0.11)
    check("an int really moved", p.box_screen_px, 199)
    check("the enum really moved", p.track_engine != rna["track_engine"].default, True)

    print("")
    res = bpy.ops.clip.btr_reset_prefs()
    check("the operator finished", res, {"FINISHED"})

    for k, want in defaults.items():
        got = getattr(p, k)
        if isinstance(want, float):
            check("%s back to its default" % k, round(got, 6), round(want, 6))
        else:
            check("%s back to its default" % k, got, want)
    check("the boolean is back", p.pin_to_pattern, rna["pin_to_pattern"].default)
    check("the enum is back", p.track_engine, rna["track_engine"].default)

    print("")
    check("assist_root is LEFT ALONE -- it is this machine, not a preference",
          p.assist_root, "//KEEP-THIS-ROOT")
    check("python_exe is left alone too", p.python_exe, "//KEEP-THIS-PYTHON")

    # Running it twice must be a no-op rather than an error.
    res = bpy.ops.clip.btr_reset_prefs()
    check("resetting again is harmless", res, {"FINISHED"})
    check("  and the paths still survive it", p.assist_root, "//KEEP-THIS-ROOT")

    print("")
    if FAILED:
        print("RESET: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[rst] RESET: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
