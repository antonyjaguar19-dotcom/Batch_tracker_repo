"""Does the leash know when to shut up?

`leash.compute` returns a guide path and a trust verdict from round-trip closure. The
verdict is the whole safety story: on a shot where CoTracker walks onto the occluder, a
leash built from its path demanded a cut at f27 of the artist's OWN hand track. Nothing
downstream can recover from that, so the gate has to fire before anything is steered.

Two shots with known answers, and they must land on opposite sides:

  * `reacquretracke_manual.txt` -- 2 real occlusions. The guide is wrong (true error p50
    152 px). MUST read NOT TRUSTED.
  * `track2 manual tracked.txt:Track.003` -- 250 frames, no true occlusion. The guide is
    right (250/250 on feature). MUST read TRUSTED.

It also walks each hand track against its own leash, because a trusted leash still has to
leave correct work alone: a breach on a hand track is a frame the addon would have cut.

    runtime\\python311\\python.exe tests\\eval_leash.py --plate D:\\Jefrin\\IN\\SH006.mp4
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [ASSIST, os.path.join(ASSIST, "sidecar"), HERE]

import leash                                                          # noqa: E402
import repo                                                           # noqa: E402
from eval_cotracker_direct import read_3de                            # noqa: E402

# file, track, must-be-trusted
CASES = [
    ("reacquretracke_manual.txt", "Track", False),
    ("track2 manual tracked.txt", "Track.003", True),
]


def log(m):
    print("[lh] %s" % m, flush=True)


def walk(path, guide, settle=3):
    """Walk a track against the leash, re-anchoring on every frame that agrees.

    A frame that DISAGREES does not re-anchor. If it did, a slide would be absorbed one
    frame at a time and never accumulate into anything visible -- the same trap as the
    recent-level pollution in `patmatch.first_loss`.

    Returns (first_breach_frame_or_None, deviations).
    """
    fs = sorted(path)
    anchor = fs[0]
    run, first, devs = 0, None, []
    for f in fs[1:]:
        p = leash.predict(guide, anchor, path[anchor], f)
        if p is None:
            continue
        dev = math.hypot(path[f][0] - p[0], path[f][1] - p[1])
        devs.append(dev)
        if dev <= p[2]:
            anchor, run = f, 0
        else:
            run += 1
            if run >= settle and first is None:
                first = f
    return first, devs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--chain", type=int, default=120)
    ap.add_argument("--span", type=int, default=120,
                    help="frames of the track to cover. The addon computes the leash over "
                         "the same window the re-acquire already uses, not the whole shot")
    a = ap.parse_args()

    os.environ.setdefault("BTR_COTRACKER_MAX_FRAMES", "400")
    repo.require_repo()
    import blio                                                       # noqa: PLC0415
    out_dir = os.path.join(ASSIST, "logs", "leash")
    os.makedirs(out_dir, exist_ok=True)
    plate = blio.Plate(os.path.abspath(a.plate), ifl_dir=out_dir)
    h = plate.h
    log("plate %dx%d, %d frames" % (plate.w, h, plate.count))

    bad = 0
    for fn, nm, want in CASES:
        full = os.path.join(HERE, fn)
        hit = [t for t in read_3de(full) if t[0] == nm]
        if not hit:
            log("no track %r in %s" % (nm, fn))
            sys.exit(2)
        name, pts = hit[0]
        truth = {f: (x, h - y) for f, x, y in pts}       # 3DE y-UP -> image y-DOWN
        fs = sorted(truth)
        lo, hi = fs[0], min(fs[-1], fs[0] + a.span)
        log("")
        log("%s (%s) f%d..f%d -- guide should be %s"
            % (name, fn, lo, hi, "TRUSTED" if want else "NOT trusted"))

        g = leash.compute(plate, lo, truth[lo], lo, hi,
                          max_side=a.max_side, chain=a.chain, on_status=log)
        err = [math.hypot(g["path"][f][0] - truth[f][0], g["path"][f][1] - truth[f][1])
               for f in truth if lo <= f <= hi and f in g["path"]]
        err.sort()
        log("  guide's TRUE error vs the hand track: p50 %.1f  max %.1f px"
            % (err[len(err) // 2], err[-1]))
        log("  %s" % g["reason"])

        if g["trusted"] != want:
            log("  !! wanted %s, got %s"
                % ("trusted" if want else "not trusted",
                   "trusted" if g["trusted"] else "not trusted"))
            bad += 1
            continue

        if g["trusted"]:
            # A trusted leash must still leave correct work alone. This track is correct by
            # construction, so any breach here is a frame the addon would have cut.
            inside = {f: p for f, p in truth.items() if lo <= f <= hi}
            first, devs = walk(inside, g["path"])
            devs.sort()
            log("  deviation on the hand track: p50 %.2f  p90 %.2f  max %.2f px"
                % (devs[len(devs) // 2], devs[min(len(devs) - 1, int(.9 * len(devs)))],
                   devs[-1]))
            if first is not None:
                log("  !! breached the artist's own hand track at f%d" % first)
                bad += 1
            else:
                log("  no breach on the hand track")
        else:
            log("  leash is off for this track -- the loop keeps its existing behaviour")

    print("VERDICT: %s" % ("FAIL" if bad else "PASS"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
