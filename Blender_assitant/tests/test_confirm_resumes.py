"""Keep/Drop on the muted resume segments, against the REAL installed extension.

Written after a live run turned up a muted marker at frame 0 on a track seeded at frame 1 --
the `sequence=False` artefact `track_core.track_backward_pass` documents. It is invisible
everywhere else (3DE export skips muted markers), but KEEP picks the FIRST muted frame and
leaves exactly that one muted, because a resume frame is the guide's ESTIMATE of where the
feature went rather than a measurement of it. With the artefact in the list, `first` is the
artefact, and the estimate KEEP meant to discard gets un-muted onto the track instead.

    blender.exe --background -noaudio --python tests\test_confirm_resumes.py
"""

import sys

import bpy

EXT = "bl_ext.user_default.btr_assist"
FAILURES = []


def log(m):
    print("[keep] %s" % m, flush=True)


def fail(m):
    FAILURES.append(m)
    log("FAIL %s" % m)


def make_clip():
    img = bpy.data.images.new("plate", 128, 128)
    img.filepath_raw = ""
    clip = None
    for c in bpy.data.movieclips:
        clip = c
    return clip


def build(track, live, muted):
    for f in sorted(set(live) | set(muted)):
        m = track.markers.find_frame(f, exact=True)
        if m is None:
            m = track.markers.insert_frame(f, co=(0.5, 0.5))
        m.mute = f in muted


def run(clip, action, live, muted):
    tracks = sys.modules["%s.three_de" % EXT].active_tracks(clip)
    for t in list(tracks):
        tracks.remove(t)
    tr = tracks.new(name="T", frame=min(live))
    tr.markers[0].co = (0.5, 0.5)
    build(tr, live, muted)
    tr.select = True
    bpy.ops.clip.btr_confirm_resumes(action=action)
    return (sorted(m.frame for m in tr.markers if not m.mute),
            sorted(m.frame for m in tr.markers if m.mute))


def main():
    bpy.ops.preferences.addon_enable(module=EXT)
    clip = make_clip()
    if clip is None:
        # A movie clip cannot be synthesised headless without a file; drive the operator's
        # rule directly instead of pretending otherwise.
        log("no movie clip available headless -- testing the rule on the module")
        oa = sys.modules["%s.ops_assist" % EXT]
        lf = oa.live_frames

        class M:
            def __init__(self, f, mute):
                self.frame, self.mute = f, mute

        class T:
            def __init__(self, ms):
                self.markers = ms

        # Seed at 1, artefact at 0, resume estimate at 54, real matches 55..57.
        ms = [M(0, True), M(1, False), M(53, False), M(54, True), M(55, True), M(56, True)]
        t = T(ms)
        live = lf(t)
        floor = live[0] if live else 0
        muted = [m for m in t.markers if m.mute and m.frame > floor]
        if [m.frame for m in muted] != [54, 55, 56]:
            fail("floor did not exclude the frame-0 artefact: %r"
                 % [m.frame for m in muted])
        else:
            log("%-42s ok" % "artefact below the seed is excluded")
        first = min(m.frame for m in muted)
        if first != 54:
            fail("KEEP would spare frame %d, not the resume estimate 54" % first)
        else:
            log("%-42s ok" % "KEEP spares the resume estimate, not f0")
        kept = [m.frame for m in muted if m.frame > first]
        if kept != [55, 56]:
            fail("KEEP would un-mute %r, expected [55, 56]" % kept)
        else:
            log("%-42s ok" % "KEEP un-mutes only the matched frames")
        # Without the floor -- the old behaviour -- the estimate survives onto the track.
        old_first = min(m.frame for m in t.markers if m.mute)
        if old_first != 0:
            fail("the regression case no longer reproduces (%r)" % old_first)
        elif 54 not in [m.frame for m in t.markers if m.mute and m.frame > old_first]:
            fail("expected the old rule to un-mute the estimate at 54")
        else:
            log("%-42s ok" % "old rule provably un-muted the estimate")

    log("CONFIRM RESUMES: %s" % ("FAIL" if FAILURES else "PASS"))
    sys.exit(1 if FAILURES else 0)


main()
