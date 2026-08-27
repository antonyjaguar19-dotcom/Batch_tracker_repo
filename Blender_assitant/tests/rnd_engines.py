"""R&D: every tracking method we can build, scored against the artist's own tracks.

The question is whether matching a hand track exactly is achievable or a waste of time, so
the first thing this file records is what "exactly" can even mean. The artist tracked ONE
feature by hand TWICE (`reacquretracke_manual.txt` and `track3` Track.001, 46 shared frames):

    constant offset between the two attempts   0.43 px
    scatter about that offset                  p50 1.78 px, max 3.00 px
    frame-to-frame MOTION disagreement         p50 0.05 px, max 1.39 px
    frames identical                           0 of 46

So a hand track is not reproducible in ABSOLUTE position even by the person who made it --
two attempts wander up to 3 px apart. What IS reproducible is the motion, to a twentieth of
a pixel, and motion is what a solve consumes: a constant offset is a different but equally
valid point on the same feature and costs a solve nothing.

Every method here is therefore scored on three things, and only the first two are worth
chasing:

  * RE-ACQUIRED -- after each gap the artist left, did the method come back on the feature?
    Binary, per occlusion event. This one CAN be 100 % and is the real target.
  * MOTION -- frame-to-frame agreement with the hand track. The artist's own floor is
    0.05 px; anything near that is as good as re-tracking it by hand.
  * ABSOLUTE -- distance from the hand track. Reported because it is what people look at,
    and bounded below by the artist's own 1.78 px scatter. Chasing it under that is chasing
    the reference's noise.

    runtime\\python311\\python.exe tests\\rnd_engines.py ^
        --manual tests\\track3 manual tracked.txt --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ASSIST, os.path.join(ASSIST, "sidecar"), HERE]

import leash                                                          # noqa: E402
import patmatch                                                       # noqa: E402
import repo                                                           # noqa: E402
from eval_cotracker_direct import read_3de                            # noqa: E402

WRONG_PX = 25.0
PAT = 41.2


def log(m):
    print("[rnd] %s" % m, flush=True)


def runs_of(fs):
    out = [[fs[0], fs[0]]]
    for f in fs[1:]:
        if f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


# ---------------------------------------------------------------- the methods

def m_cotracker(plate, truth, fs, hi, patch, offset, guide):
    """The guide's own path, anchored on the seed. No pattern involved at all."""
    return {f: guide[f] for f in guide if f <= hi}


def m_pattern_only(plate, truth, fs, hi, patch, offset, guide):
    """No guide. The artist's patch, previous-frame prior, allowed to warp.

    A classical tracker with the seed as its permanent template -- the thing Blender would
    do if it matched every frame against the KEYFRAME instead of the frame before it.
    """
    out, cur = {}, truth[fs[0]]
    out[fs[0]] = cur
    for f in range(fs[0] + 1, hi + 1):
        img = patmatch._gray(plate.frame(f - 1))
        if img is None:
            break
        got = patmatch.match_pinned(img, patch, cur[0], cur[1], radius=12.0, offset=offset)
        if got is None or got[2] < 0.60:
            break
        out[f] = (got[0], got[1])
        cur = (got[0], got[1])
    return out


def m_cotracker_pinned(plate, truth, fs, hi, patch, offset, guide):
    """The guide picks the neighbourhood, the artist's patch picks the pixel.

    Stops where the pattern cannot be found, exactly as the shipped engine does -- so the
    frames it does NOT produce are as much a part of the result as the ones it does.
    """
    out = {fs[0]: truth[fs[0]]}
    a_f, a_px = fs[0], truth[fs[0]]
    for f in range(fs[0] + 1, hi + 1):
        p = leash.predict(guide, a_f, a_px, f)
        if p is None:
            break
        img = patmatch._gray(plate.frame(f - 1))
        if img is None:
            break
        got = patmatch.match_pinned(img, patch, p[0], p[1], radius=12.0, offset=offset)
        if got is None or got[2] < 0.60:
            break
        out[f] = (got[0], got[1])
        a_f, a_px = f, (got[0], got[1])
    return out


def m_pinned_reacquire(plate, truth, fs, hi, patch, offset, guide):
    """As above, but a frame the pattern fails on does not end the track.

    It keeps walking on the guide alone and re-tests the pattern every frame, which is the
    whole re-acquisition idea reduced to its simplest form: the guide carries the point
    across the occluder, the artist's pattern decides when it is back.
    """
    out = {fs[0]: truth[fs[0]]}
    a_f, a_px = fs[0], truth[fs[0]]
    for f in range(fs[0] + 1, hi + 1):
        p = leash.predict(guide, a_f, a_px, f)
        if p is None:
            break
        img = patmatch._gray(plate.frame(f - 1))
        if img is None:
            break
        got = patmatch.match_pinned(img, patch, p[0], p[1], radius=12.0, offset=offset)
        if got is not None and got[2] >= 0.60:
            out[f] = (got[0], got[1])
            a_f, a_px = f, (got[0], got[1])
        # else: leave the frame empty, keep the anchor where it was, and let the guide
        # carry the prediction forward. The gap stays a gap, which is the honest answer.
    return out


def m_sweep(plate, truth, fs, hi, patch, offset, guide, radius=56.0, settle=2):
    """The method that actually ships: when the pattern fails, KEEP SWEEPING, and widen.

    The three above all look 12 px around the guide, which is right while the feature is
    there and hopeless once an occluder has carried the guide away -- measured on this very
    track, the guide ends 1723 px from the feature. The shipped re-acquire searches the
    artist's search box instead, about 56 px, at every frame along the guide, and takes the
    first run of frames that pass. That width is the whole difference between coming back
    and not.

    `settle` frames in a row must pass before a resume is believed. One frame can be a
    lookalike; two in a row on a moving plate is the feature.
    """
    out = {fs[0]: truth[fs[0]]}
    a_f, a_px = fs[0], truth[fs[0]]
    run = []
    for f in range(fs[0] + 1, hi + 1):
        p = leash.predict(guide, a_f, a_px, f)
        if p is None:
            break
        img = patmatch._gray(plate.frame(f - 1))
        if img is None:
            break
        near = patmatch.match_pinned(img, patch, p[0], p[1], radius=12.0, offset=offset)
        if near is not None and near[2] >= 0.60:
            out[f] = (near[0], near[1])
            a_f, a_px = f, (near[0], near[1])
            run = []
            continue
        wide = patmatch.match_pinned(img, patch, a_px[0], a_px[1], radius=radius,
                                     offset=offset)
        if wide is not None and wide[2] >= 0.60:
            run.append((f, wide[0], wide[1], wide[2]))
            if len(run) >= settle:
                for (ff, xx, yy, _s) in run:
                    out[ff] = (xx, yy)
                a_f, a_px = run[-1][0], (run[-1][1], run[-1][2])
                run = []
        else:
            run = []
    return out


def m_requery(plate, truth, fs, hi, patch, offset, guide, radius=56.0, settle=2):
    """Sweep, and RE-QUERY the guide from the last good frame instead of extrapolating.

    This is the one that ships, and the difference is not a detail. A guide queried at the
    seed is asked to predict across an occlusion it has already walked into -- measured on
    this track it ends 1723 px from the feature, so no search radius rescues it. A guide
    re-queried at the last frame the pattern confirmed starts from a position that is known
    good and only has to cross the occluder itself.

    Costs one CoTracker pass per failure rather than one per track. That is why the shipped
    re-acquire groups deaths by frame and runs a short window per group.
    """
    out = {fs[0]: truth[fs[0]]}
    a_f, a_px = fs[0], truth[fs[0]]
    g = guide
    run = []
    f = fs[0] + 1
    while f <= hi:
        p = leash.predict(g, a_f, a_px, f)
        img = patmatch._gray(plate.frame(f - 1)) if p is not None else None
        near = (patmatch.match_pinned(img, patch, p[0], p[1], radius=12.0, offset=offset)
                if img is not None else None)
        if near is not None and near[2] >= 0.60:
            out[f] = (near[0], near[1])
            a_f, a_px = f, (near[0], near[1])
            run = []
            f += 1
            continue
        if img is not None:
            base = leash.predict(g, a_f, a_px, f)
            bx, by = (base[0], base[1]) if base else a_px
            wide = patmatch.match_pinned(img, patch, bx, by, radius=radius, offset=offset)
            if wide is not None and wide[2] >= 0.60:
                run.append((f, wide[0], wide[1]))
                if len(run) >= settle:
                    for (ff, xx, yy) in run:
                        out[ff] = (xx, yy)
                    a_f, a_px = run[-1][0], (run[-1][1], run[-1][2])
                    # Fresh guide from the frame just confirmed, exactly as the sidecar does.
                    g = leash._chain(plate, a_f, a_px, a_f, hi, 768, 120, None, +1) or g
                    run = []
                f += 1
                continue
            run = []
        # The pattern is not here. Re-query the guide from the last confirmed frame ONCE, so
        # the sweep ahead is predicted from known-good rather than from a drifting path.
        if f == a_f + 1:
            g = leash._chain(plate, a_f, a_px, a_f, hi, 768, 120, None, +1) or g
        f += 1
    return out


METHODS = [
    ("cotracker raw", m_cotracker, True),
    ("pattern only", m_pattern_only, False),
    ("cotracker+pin", m_cotracker_pinned, True),
    ("pin, keep looking", m_pinned_reacquire, True),
    ("sweep+settle (ships)", m_sweep, True),
    ("sweep+re-query", m_requery, True),
]


# ---------------------------------------------------------------- scoring

def score(pred, truth, rs):
    fs = sorted(truth)
    errs, n_ok = [], 0
    for f in fs:
        p = pred.get(f)
        if p is None:
            continue
        e = math.hypot(p[0] - truth[f][0], p[1] - truth[f][1])
        if e <= WRONG_PX:
            n_ok += 1
            errs.append(e)
    mot = []
    for f1, f2 in zip(fs, fs[1:]):
        if f2 != f1 + 1 or f1 not in pred or f2 not in pred:
            continue
        dt = (truth[f2][0] - truth[f1][0], truth[f2][1] - truth[f1][1])
        dp = (pred[f2][0] - pred[f1][0], pred[f2][1] - pred[f1][1])
        mot.append(math.hypot(dp[0] - dt[0], dp[1] - dt[1]))
    reac = []
    for i in range(1, len(rs)):
        lo, _hi = rs[i]
        # Judged on the first three frames the artist has after the gap: one frame could be
        # luck, and by the third a wrong landing has always shown itself.
        got = [f for f in range(lo, min(lo + 3, rs[i][1] + 1)) if f in pred]
        ok = [f for f in got
              if math.hypot(pred[f][0] - truth[f][0], pred[f][1] - truth[f][1]) <= WRONG_PX]
        reac.append((len(ok), len(range(lo, min(lo + 3, rs[i][1] + 1)))))

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")

    return {"on": n_ok, "of": len(fs), "abs_p50": q(errs, .5),
            "mot_p50": q(mot, .5), "mot_max": max(mot) if mot else float("nan"),
            "reac": reac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--methods", default="",
                    help="comma-separated substrings; empty runs them all. Use this to run "
                         "the CPU-only baseline while the GPU is busy training something")
    a = ap.parse_args()

    os.environ.setdefault("BTR_COTRACKER_MAX_FRAMES", "400")
    repo.require_repo()
    import blio                                                       # noqa: PLC0415
    out_dir = os.path.join(ASSIST, "logs", "rnd")
    os.makedirs(out_dir, exist_ok=True)
    plate = blio.Plate(os.path.abspath(a.plate), ifl_dir=out_dir)
    h = plate.h
    log("plate %dx%d, %d frames" % (plate.w, h, plate.count))

    totals = {name: {"on": 0, "of": 0, "abs": [], "mot": [], "reac_ok": 0, "reac_n": 0}
              for name, _fn, _g in METHODS}

    for name, pts in read_3de(a.manual):
        if a.only and name != a.only:
            continue
        truth = {int(f): (float(x), float(h - y)) for f, x, y in pts if int(f) >= 1}
        fs = sorted(truth)
        if len(fs) < 6:
            continue
        hi = min(fs[-1], int(plate.count))
        if a.max_frames:
            hi = min(hi, fs[0] + a.max_frames - 1)
        truth = {f: p for f, p in truth.items() if f <= hi}
        fs = sorted(truth)
        if len(fs) < 6:
            continue
        rs = runs_of(fs)
        ref = patmatch.reference_patch(plate, fs[0], truth[fs[0]][0], truth[fs[0]][1],
                                       PAT, PAT)
        if ref is None:
            log("%-12s skipped -- pattern box does not fit at the seed" % name)
            continue
        patch, offset = ref
        want = [m for m in METHODS
                if not a.methods or any(k.strip() in m[0] for k in a.methods.split(","))]
        # Only pay for a CoTracker pass if a method that was asked for actually needs one.
        guide = ({} if not any(m[2] for m in want)
                 else leash._chain(plate, fs[0], truth[fs[0]], fs[0], hi, 768, 120, None, +1))

        log("")
        log("%s  f%d-%d, %d run(s), %d gap(s)" % (name, fs[0], hi, len(rs), len(rs) - 1))
        for mname, fn, _needs in want:
            pred = fn(plate, truth, fs, hi, patch, offset, guide)
            s = score(pred, truth, rs)
            t = totals[mname]
            t["on"] += s["on"]
            t["of"] += s["of"]
            if s["abs_p50"] == s["abs_p50"]:
                t["abs"].append(s["abs_p50"])
            if s["mot_p50"] == s["mot_p50"]:
                t["mot"].append(s["mot_p50"])
            for ok, n in s["reac"]:
                t["reac_ok"] += ok
                t["reac_n"] += n
            log("   %-18s on %3d/%-3d  abs p50 %5.1f  motion p50 %5.2f max %5.2f  %s"
                % (mname, s["on"], s["of"], s["abs_p50"], s["mot_p50"], s["mot_max"],
                   "" if not s["reac"] else
                   "re-acquired " + " ".join("%d/%d" % r for r in s["reac"])))

    log("")
    log("=" * 78)
    log("%-18s %-14s %-11s %-13s %s"
        % ("method", "on feature", "abs p50", "motion p50", "re-acquired"))
    for mname, _fn, _g in METHODS:
        t = totals[mname]
        if not t["of"]:
            continue
        ab = sorted(t["abs"])
        mo = sorted(t["mot"])
        log("%-18s %4d/%-4d %3d%%  %6.1f px   %6.2f px     %s"
            % (mname, t["on"], t["of"], round(100.0 * t["on"] / max(1, t["of"])),
               ab[len(ab) // 2] if ab else float("nan"),
               mo[len(mo) // 2] if mo else float("nan"),
               "%d/%d" % (t["reac_ok"], t["reac_n"]) if t["reac_n"] else "no gaps"))
    log("")
    log("the artist against THEMSELVES, same feature twice by hand:")
    log("%-18s %-14s %-11s %-13s" % ("hand vs hand", "46/46 100%", "  2.1 px", "  0.05 px"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
