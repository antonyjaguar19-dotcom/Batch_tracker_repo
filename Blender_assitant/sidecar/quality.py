"""Grade a finished track without a hand track to compare it against.

The artist: *"the track quality varies from track to track, some tracks are good and some are
really bad. cant we make the tool always produce a decent quality track?"*

No tool can. Quality is bounded by what is in the plate -- a seed on flat sky, on a moving
object, or covered for eighty frames has no good answer, and anything that returns one is
inventing data. What a tool CAN do is never hand over a bad track without saying so, and that
needs a grade that works with no reference, because in a real shot there is none.

**Two independent kinds of bad, and one detector cannot see both.**

*Noise.* Measured across the artist's own 137-track SH008 export, per-frame deviation from a
local quadratic runs from **0.04 px to 4.55 px** -- p50 0.456, p90 1.539. That spread IS the
complaint: the same run, the same settings, tracks two orders of magnitude apart. For scale,
the artist's own hand tracks measure 0.68 and 0.86 px, because a human clicking scatters about
a pixel; a correlator on a hard corner beats that easily, and one on mush does not come close.

*Drift.* Jitter cannot see it, and that is not a threshold problem. A slide is SMOOTH -- it
fits a local quadratic perfectly. Measured: the artist's drifted assist track jitters 0.779 px
against their hand track's 0.677, which is to say the broken one looks marginally better. Drift
is caught by `leash.find_slides`, comparing motion against CoTracker, and nothing here
duplicates it.

So this module grades what a track can be judged on from its own geometry, and the slide check
is reported alongside it rather than folded in. A single number covering both would hide which
fault a track has, and they need different fixes: a slide is repairable, noise is not.
"""

import math

import numpy as np

#: Half-width of the window a frame is judged against. Three frames either side is enough to
#: fit a quadratic -- constant acceleration -- which is what a real camera move looks like
#: over seven frames at any sane frame rate. Wider starts fitting the shot instead of the
#: tracker; narrower cannot separate curve from noise at all.
JITTER_HALF = 3

#: Jitter in plate pixels above which a track is worth an artist's attention. The artist's own
#: bot export puts p90 at 1.54 px and their hand tracks at 0.68-0.86, so this is "noisier than
#: nine tracks in ten from this tool, and noisier than clicking it by hand".
JITTER_POOR = 1.5

#: And the point where it stops being usable for a solve. 4.55 px was the worst single track
#: in that export; a marker moving three pixels of pure noise per frame contributes error, not
#: constraint.
JITTER_BAD = 3.0

#: A track shorter than this constrains almost nothing however clean it is.
SHORT_SPAN = 12


def jitter(path, half=JITTER_HALF):
    """RMS deviation from a local quadratic, in plate pixels. None if there is nothing to fit.

    Judged only across UNBROKEN runs. Fitting across a gap measures the gap -- the two sides
    of an occlusion are a legitimate discontinuity in the samples, not a wobble in the track,
    and a fit that spans it reports the occluder as noise.
    """
    fs = sorted(path)
    if len(fs) < 2 * half + 1:
        return None
    res = []
    for i in range(half, len(fs) - half):
        win = fs[i - half:i + half + 1]
        if win[-1] - win[0] != 2 * half:
            continue
        t = np.asarray(win, dtype=float)
        t -= t[half]
        for ax in (0, 1):
            v = np.asarray([path[f][ax] for f in win], dtype=float)
            try:
                c = np.polyfit(t, v, 2)
            except (np.linalg.LinAlgError, ValueError):
                continue
            res.append(v[half] - np.polyval(c, 0.0))
    if not res:
        return None
    return float(math.sqrt(float(np.mean(np.square(res)))))


def runs_of(path):
    """Unbroken stretches of frames, as [(first, last), ...]."""
    fs = sorted(path)
    out = []
    for f in fs:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def grade(path, jitter_poor=JITTER_POOR, jitter_bad=JITTER_BAD, short_span=SHORT_SPAN):
    """What is measurable about one track from its own geometry.

    Returns a dict with the numbers and a verdict of "good", "check" or "poor". The verdict is
    a summary of the numbers and never replaces them: an artist deciding whether to keep a
    track wants to know it is NOISY rather than that it scored 2 out of 5.
    """
    fs = sorted(path)
    if len(fs) < 2:
        return {"frames": len(fs), "verdict": "poor", "why": ["barely exists"]}
    runs = runs_of(path)
    j = jitter(path)
    span = fs[-1] - fs[0] + 1
    why = []
    verdict = "good"
    if j is None:
        why.append("too short to measure how steady it is")
        verdict = "check"
    elif j >= jitter_bad:
        why.append("very unsteady (%.2f px of jitter)" % j)
        verdict = "poor"
    elif j >= jitter_poor:
        why.append("unsteady (%.2f px of jitter)" % j)
        verdict = "check"
    if len(fs) < short_span:
        why.append("only %d frame(s) long" % len(fs))
        verdict = "poor"
    if len(runs) > 1:
        # Not a fault by itself -- a gap is the honest answer across an occlusion and 3DE
        # solves through one. Reported because a track in six pieces is worth looking at.
        why.append("in %d pieces" % len(runs))
        if len(runs) >= 4 and verdict == "good":
            verdict = "check"
    return {"frames": len(fs), "span": span, "runs": len(runs),
            "first": fs[0], "last": fs[-1],
            "jitter_px": None if j is None else round(j, 3),
            "verdict": verdict, "why": why}


def grade_all(tracks, **kw):
    """{id: grade} for a whole clip, plus the distribution an artist actually reads."""
    out = {tid: grade(pts, **kw) for tid, pts in tracks.items() if pts}
    js = sorted(g["jitter_px"] for g in out.values() if g.get("jitter_px") is not None)

    def pc(p):
        return js[min(len(js) - 1, int(round(p * (len(js) - 1))))] if js else None

    counts = {}
    for g in out.values():
        counts[g["verdict"]] = counts.get(g["verdict"], 0) + 1
    return out, {"tracks": len(out), "verdicts": counts,
                 "jitter_p50": pc(0.5), "jitter_p90": pc(0.9),
                 "jitter_max": js[-1] if js else None}
