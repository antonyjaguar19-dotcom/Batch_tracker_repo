"""The assistant must track with Blender's tracker, not one that resembles it.

The artist's report: "the track when i seed and track using blender is different when am
using the assistant". It was true and it was measurable. Same seed, same pattern box, 14
frames of SH006:

    a track carrying Blender's own defaults vs one carrying this addon's   10.36 px apart
    a track created AFTER `apply_settings` wrote the clip's defaults        7.91 px apart

The second one is the worse of the two: `apply_settings` changed the CLIP, so markers the
artist added later in Blender's own UI tracked differently for the rest of the session.

The addon's configuration (PREV_FRAME + normalization) is better on this project's own
numbers -- it survives 2.6-2.9x longer on real plates. That is not the point. An assistant
that silently tracks differently from the application it is assisting cannot be checked
against it, and the artist checks it constantly. So Blender's settings are the default and
the addon's are an explicit opt-in.

What must be true here:

  * a seed the addon places tracks to the SAME PIXEL as one placed by hand;
  * stepping frame-by-frame is still byte-identical to Blender's own sequence mode;
  * nothing writes the artist's clip settings behind their back;
  * and with the opt-in turned off, the old behaviour is still there -- proven by it
    differing, because a switch that changes nothing is not a switch.

    blender.exe --background -noaudio --python tests/test_blender_parity.py -- \
        --plate D:\Jefrin\IN\SH006.mp4
"""

import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = "bl_ext.user_default.btr_assist"
MANUAL = os.path.join(HERE, "track3 manual tracked.txt")
TRACK = "Track.001"
PAT = 41.2
N = 14

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[bp] %-58s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def read_3de(path):
    tok = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 0
    n = int(tok[i]); i += 1
    out = []
    for _ in range(n):
        name = tok[i]; i += 2
        cnt = int(tok[i]); i += 1
        pts = []
        for _ in range(cnt):
            pts.append((int(tok[i]), float(tok[i + 1]), float(tok[i + 2])))
            i += 3
        out.append((name, pts))
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    plate = argv[argv.index("--plate") + 1] if "--plate" in argv else ""
    if not plate or not os.path.exists(plate):
        print("[bp] need --plate")
        return 2

    import importlib
    bpy.ops.preferences.addon_enable(module=EXT)
    oa = importlib.import_module(EXT + ".ops_assist")
    tc = importlib.import_module(EXT + ".track_core")

    clip = bpy.data.movieclips.load(os.path.abspath(plate))
    w, h = clip.size
    win = bpy.context.window_manager.windows[0]
    area = max(win.screen.areas, key=lambda a: a.width * a.height)
    area.type = "CLIP_EDITOR"
    sp = area.spaces.active
    sp.clip = clip
    region = next(r for r in area.regions if r.type == "WINDOW")
    ctx = (win, area, region, clip, bpy.context.scene)

    _, pts = [t for t in read_3de(MANUAL) if t[0] == TRACK][0]
    truth = {f: (x, h - y) for f, x, y in pts}
    u, v = oa.image_px_to_uv(truth[1][0], truth[1][1], w, h)

    st = clip.tracking.settings
    before = (st.default_motion_model, st.default_pattern_match, st.use_default_brute,
              st.use_default_normalization, round(st.default_correlation_min, 4))
    print("[bp] Blender's own defaults: motion=%s match=%s brute=%s norm=%s corr=%.2f"
          % before)

    def path_of(t):
        return {f: oa.marker_to_image_px(t.markers.find_frame(f, exact=True), w, h)
                for f in oa.live_frames(t) if f <= N}

    def apart(a, b):
        common = sorted(set(a) & set(b))
        if not common:
            return -1.0
        return max(math.hypot(a[f][0] - b[f][0], a[f][1] - b[f][1]) for f in common)

    # ---- by hand: exactly what the artist does -----------------------------------------
    hand = clip.tracking.tracks.new(name="BY_HAND", frame=1)
    m = hand.markers[0]
    m.co, m.mute = (u, v), False
    tc.set_geom(m, PAT, PAT * 3.0, w, h)
    tc.track_group(ctx, [hand], 1, backwards=False, sequence=True)
    ref = path_of(hand)
    print("[bp] by hand: Blender tracked f1-f%d" % max(ref))
    check("the hand track really ran", len(ref) >= 5, True)

    # ---- the addon's seed, tracked by the addon's loop ----------------------------------
    opts = tc.Opts(leash=0.0, motion_model="", scale_clamp=0.0, edge_stop=True,
                   blender_defaults=True)
    tc.apply_settings(clip, opts)
    after = (st.default_motion_model, st.default_pattern_match, st.use_default_brute,
             st.use_default_normalization, round(st.default_correlation_min, 4))
    check("apply_settings left the artist's clip alone", after, before)

    made = tc.seed_tracks(clip, [{"id": "ASSIST", "frame": 1, "kind": "", "u": u, "v": v}],
                          opts)
    a = made[0]["t"]
    tc.set_geom(a.markers[0], PAT, PAT * 3.0, w, h)
    check("the addon's seed carries Blender's settings",
          (a.motion_model, a.pattern_match, a.use_brute, a.use_normalization),
          (hand.motion_model, hand.pattern_match, hand.use_brute, hand.use_normalization))
    rec = [{"t": a, "id": a.name, "kind": "", "alive": True, "w": w, "h": h,
            "seed_frame": 1, "seed_pat": (PAT, PAT)}]
    for _ in tc.track_job(ctx, rec, N, {}, opts):
        pass
    got = path_of(a)
    d = apart(ref, got)
    print("[bp] assistant vs by hand: %d frames, worst %.6f px" % (len(got), d))
    check("the assistant produced the same frames", sorted(got), sorted(ref))
    check("...on the same pixel", d < 0.001, True)

    # ---- the opt-in still changes something ---------------------------------------------
    # A switch that makes no difference is not a switch, and this one guards a real
    # measurement -- so it has to be shown still reaching the tracker.
    own = tc.Opts(leash=0.0, motion_model="", scale_clamp=0.0, edge_stop=True)
    tc.apply_settings(clip, own)
    made = tc.seed_tracks(clip, [{"id": "ADDONCFG", "frame": 1, "kind": "", "u": u, "v": v}],
                          own)
    b = made[0]["t"]
    tc.set_geom(b.markers[0], PAT, PAT * 3.0, w, h)
    check("with the opt-in off, the addon's own settings are applied",
          (b.pattern_match, b.use_normalization), ("PREV_FRAME", True))
    rec = [{"t": b, "id": b.name, "kind": "", "alive": True, "w": w, "h": h,
            "seed_frame": 1, "seed_pat": (PAT, PAT)}]
    for _ in tc.track_job(ctx, rec, N, {}, own):
        pass
    d2 = apart(ref, path_of(b))
    print("[bp] addon's own configuration vs by hand: worst %.2f px" % d2)
    check("and it really is a different tracker", d2 > 1.0, True)

    print("")
    if FAILED:
        print("BLENDER PARITY: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("[bp] BLENDER PARITY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
