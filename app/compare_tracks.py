# -*- coding: utf-8 -*-
"""Compare a bot track against a hand-tracked reference, and say what is wrong.

Every tracking question in this project has come down to "is it better or worse?", answered
by eye in the 3DE viewport. This is the measuring stick: point it at a bot export and an
artist's track of the same feature and it reports where they part company, and how.

    python -m app.compare_tracks bot.txt manual.txt
    python -m app.compare_tracks bot.txt manual.txt --bot-track BWD_0010 --ref-track 658

What it reports and why each number matters:

  deviation        how far the bot sits from the artist. The headline error.
  first step       where the divergence is INTRODUCED. A large loss on the first frame after
                   a seed means the point started wrong and everything after inherited it --
                   a very different fault from slow drift, and a different fix.
  fast vs slow     under-shooting fast motion and over-shooting slow motion together mean the
                   position is being pulled toward where the point WAS instead of measured
                   fresh. Over-shoot on near-still frames is the visible wobble.
  smoothness       each track's own jerkiness, needing no reference. Compared against the
                   artist's, because a genuinely shaky plate makes BOTH jerky -- without that
                   comparison a bumpy shot reads as a bad tracker.
  worst frames     individual mistracks, as opposed to a systematic bias.

No torch, no cv2: this must run anywhere, including on a machine with no GPU.
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple

Track = Dict[int, Tuple[float, float]]


def load_tracks(path: str) -> Dict[str, Track]:
    """Read classic 3DE 2D-track ASCII: <N>, then per track name / colour / count / points."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tok = f.read().split()
    i = 0
    n = int(tok[i]); i += 1
    out: Dict[str, Track] = {}
    for _ in range(n):
        name = tok[i]; i += 1
        i += 1                                   # colour id, unused
        count = int(tok[i]); i += 1
        pts: Track = {}
        for _ in range(count):
            pts[int(tok[i])] = (float(tok[i + 1]), float(tok[i + 2]))
            i += 3
        out[name] = pts
    return out


def pick(tracks: Dict[str, Track], name: Optional[str], label: str) -> Tuple[str, Track]:
    if name:
        if name not in tracks:
            raise SystemExit(f"{label}: no track named {name!r}. Available: {', '.join(tracks)}")
        return name, tracks[name]
    # Default to the longest -- when a file holds one track this is it, and when it holds
    # many the longest is the one worth comparing.
    best = max(tracks.items(), key=lambda kv: len(kv[1]))
    return best[0], best[1]


def _fit(xs: List[float], ds: List[float]) -> Tuple[float, float, float]:
    """Least-squares d = a*x + b; returns (a, b, residual sd)."""
    n = len(xs)
    if n < 3:
        return 0.0, (st.mean(ds) if ds else 0.0), 0.0
    mx, md = st.mean(xs), st.mean(ds)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, md, st.pstdev(ds)
    a = sum((x - mx) * (d - md) for x, d in zip(xs, ds)) / sxx
    b = md - a * mx
    resid = [d - (a * x + b) for x, d in zip(xs, ds)]
    return a, b, st.pstdev(resid)


def compare(bot: Track, ref: Track, out=sys.stdout) -> dict:
    """Print the comparison and return the headline figures."""
    def p(s=""):
        print(s, file=out)

    common = sorted(set(bot) & set(ref))
    res: dict = {"common": len(common)}
    p(f"bot     frames {min(bot)}-{max(bot)}  ({len(bot)} points)")
    p(f"manual  frames {min(ref)}-{max(ref)}  ({len(ref)} points)")
    if len(common) < 3:
        p("\nToo few overlapping frames to compare.")
        return res
    p(f"common  frames {common[0]}-{common[-1]}  ({len(common)} points)")

    dx = [bot[f][0] - ref[f][0] for f in common]
    dy = [bot[f][1] - ref[f][1] for f in common]
    err = [math.hypot(a, b) for a, b in zip(dx, dy)]
    res.update(mean_err=st.mean(err), max_err=max(err))
    p()
    p("HOW FAR APART")
    p(f"  deviation from the artist   mean {st.mean(err):6.2f}px   "
      f"median {st.median(err):6.2f}px   worst {max(err):6.2f}px")
    p(f"  direction                   x {st.mean(dx):+6.2f}  y {st.mean(dy):+6.2f}")

    # Same path shifted, or a genuinely different path?
    mx, my = st.mean(dx), st.mean(dy)
    resid = [math.hypot(a - mx, b - my) for a, b in zip(dx, dy)]
    res["shift"] = math.hypot(mx, my)
    res["shape_err"] = st.mean(resid)
    p(f"  of which a constant shift   {math.hypot(mx, my):6.2f}px")
    p(f"  genuinely different path    {st.mean(resid):6.2f}px")
    ax, _bx, rx = _fit([ref[f][0] for f in common], dx)
    if abs(rx - st.pstdev(dx)) > 0.15 * max(1e-9, st.pstdev(dx)):
        p(f"  ! looks like a SCALE error, not a shift (x scale {1 + ax:.5f}) -- suspect a "
          f"resolution/proxy mismatch")

    # Where does the error get introduced?
    p()
    p("WHERE THE ERROR COMES IN")
    steps = []
    for k in range(len(common) - 1):
        f, g = common[k], common[k + 1]
        steps.append((f, g,
                      math.hypot(bot[g][0] - bot[f][0], bot[g][1] - bot[f][1]),
                      math.hypot(ref[g][0] - ref[f][0], ref[g][1] - ref[f][1])))
    # The anchor is wherever they agree best; the error is introduced walking away from it.
    anchor = min(common, key=lambda f: math.hypot(bot[f][0] - ref[f][0], bot[f][1] - ref[f][1]))
    ai = common.index(anchor)
    p(f"  the two tracks agree best at frame {anchor} "
      f"({math.hypot(bot[anchor][0] - ref[anchor][0], bot[anchor][1] - ref[anchor][1]):.2f}px apart)")
    if ai > 0:
        f0 = common[ai - 1]
        e0 = math.hypot(bot[f0][0] - ref[f0][0], bot[f0][1] - ref[f0][1])
        sb = math.hypot(bot[anchor][0] - bot[f0][0], bot[anchor][1] - bot[f0][1])
        sr = math.hypot(ref[anchor][0] - ref[f0][0], ref[anchor][1] - ref[f0][1])
        res["first_step_loss"] = e0
        p(f"  the very NEXT frame ({f0}) is already {e0:5.2f}px apart "
          f"-- bot moved {sb:.2f}px where the artist moved {sr:.2f}px")
        if e0 > 0.35 * st.mean(err):
            p("  ! most of the error appears in ONE step. The point starts wrong and never "
              "recovers, rather than drifting slowly.")

    # Damping: lazy on fast motion, twitchy on slow motion.
    p()
    p("MOTION: IS IT MEASURED FRESH, OR PULLED TOWARD WHERE IT WAS?")
    by_speed = sorted(steps, key=lambda s: s[3])
    q = max(1, len(by_speed) // 4)
    bands = (("slowest 25%", by_speed[:q]), ("middle 50%", by_speed[q:3 * q]),
             ("fastest 25%", by_speed[3 * q:]))
    ratios = {}
    for label, sl in bands:
        if not sl:
            continue
        mb, mr = st.mean([s[2] for s in sl]), st.mean([s[3] for s in sl])
        ratios[label] = (mb / mr) if mr else float("nan")
        p(f"  {label:<12} artist moves {mr:5.2f}px   bot moves {mb:5.2f}px   ({mb / mr if mr else 0:.2f}x)")
    res["fast_ratio"] = ratios.get("fastest 25%", float("nan"))
    res["slow_ratio"] = ratios.get("slowest 25%", float("nan"))
    if ratios.get("fastest 25%", 1) < 0.95 and ratios.get("slowest 25%", 1) > 1.05:
        p("  ! lazy on fast movement, twitchy on slow movement. The position is being pulled")
        p("    toward where the point was instead of measured fresh. The slow-frame excess is")
        p("    the wobble you see when zoomed in.")

    # Own smoothness -- needs no reference, but only means something next to the artist's.
    def accel(tr: Track) -> List[float]:
        a = []
        for k in range(1, len(common) - 1):
            f0, f1, f2 = common[k - 1], common[k], common[k + 1]
            a.append(math.hypot(tr[f2][0] - 2 * tr[f1][0] + tr[f0][0],
                                tr[f2][1] - 2 * tr[f1][1] + tr[f0][1]))
        return a

    ab, ar = accel(bot), accel(ref)
    if ab and ar:
        mb, mr = st.mean(ab), st.mean(ar)
        # A perfectly steady reference is real (a locked-off plate, or a synthetic test), so
        # the ratio has to cope with a zero denominator rather than skipping the section --
        # silently omitting the steadiness verdict is worse than reporting a degenerate one.
        if mr > 1e-9:
            ratio = mb / mr
        else:
            ratio = 1.0 if mb <= 1e-9 else float("inf")
        res["jerk_ratio"] = ratio
        p()
        p("STEADINESS (each track judged on its own)")
        p(f"  bot     {mb:5.2f}px per frame")
        p(f"  artist  {mr:5.2f}px per frame   <- a shaky plate makes BOTH jerky")
        if ratio == float("inf"):
            p("  the artist's track is perfectly steady and the bot's is not")
        else:
            p(f"  bot is {ratio:.1f}x jerkier than the artist")
        if ratio < 1.6:
            p("  -> steadiness is NOT the main fault here; being in the wrong place is.")
        if mb > 1e-9:
            worst = sorted(range(len(ab)), key=lambda k: -ab[k])[:6]
            p("  worst bot frames: "
              + ", ".join(f"{common[k + 1]} ({ab[k]:.1f}px)" for k in worst))
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare a bot track against a manual reference.")
    ap.add_argument("bot")
    ap.add_argument("manual")
    ap.add_argument("--bot-track", default=None, help="track name in the bot file")
    ap.add_argument("--ref-track", default=None, help="track name in the manual file")
    a = ap.parse_args(argv)

    bt, rt = load_tracks(a.bot), load_tracks(a.manual)
    bn, b = pick(bt, a.bot_track, "bot")
    rn, r = pick(rt, a.ref_track, "manual")
    print(f"comparing bot {bn!r} against manual {rn!r}\n")
    compare(b, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
