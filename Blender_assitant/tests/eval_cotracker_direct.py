"""CoTracker3 driving the track itself, scored against a hand track.

The assist loop already runs CoTracker on every re-acquire and keeps ONE number out of it:
where to restart. `track_points` returns a position for every frame, and all of it is
thrown away. So "CoTracker as its own mode" is not a new capability, it is a decision to
keep what is already computed -- and the only question that matters is whether that path is
good enough to hand an artist.

This answers it with the same scoring rules as `eval_track_vs_manual.py`, so the two
numbers can be put side by side:

  ON THE FEATURE -- within --wrong-px of the artist's click
  OFF            -- further out, or on a frame the artist deliberately left out (occluded)
  MISSED         -- frames the artist has that we do not

Run with the repo's own interpreter -- no Blender:

    runtime\python311\python.exe tests\eval_cotracker_direct.py ^
        --manual tests\reacquretracke_manual.txt --plate D:\Jefrin\IN\SH006.mp4
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ASSIST)
sys.path.insert(0, os.path.join(ASSIST, "sidecar"))

import cotrack                                                        # noqa: E402
import repo                                                           # noqa: E402


def log(m):
    print("[ct] %s" % m, flush=True)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True)
    ap.add_argument("--plate", required=True)
    ap.add_argument("--track", default="")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--budget", type=int, default=400,
                    help="frame cap inside cotrack.track_points; raised here deliberately "
                         "because a MODE has to cover the whole shot, not a resume window")
    ap.add_argument("--chain", type=int, default=0,
                    help="if >0, run in windows of this many frames, re-querying each one "
                         "at the previous window's last position. This is what a real mode "
                         "would have to do on a long shot; 0 = one window over everything")
    ap.add_argument("--wrong-px", type=float, default=25.0)
    ap.add_argument("--anchor", action="store_true",
                    help="score the guide's DISPLACEMENT from the seed applied to the "
                         "artist's own seed click, not the guide's absolute coordinates. "
                         "This is how the resume already uses it, and it is what a leash "
                         "would use -- it removes the constant offset between where a "
                         "human clicks and where the model centres")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--drift-table", action="store_true",
                    help="how far the guide's DISPLACEMENT over k frames disagrees with the "
                         "artist's, for each k. This is what sizes a leash: the guide is "
                         "re-anchored to the last frame the correlator was sure about, so "
                         "only the gap since that frame accrues error -- not the whole shot")
    a = ap.parse_args()

    os.environ["BTR_COTRACKER_MAX_FRAMES"] = str(a.budget)

    all_tracks = read_3de(a.manual)
    if a.track:
        hit = [t for t in all_tracks if t[0] == a.track]
        if not hit:
            log("no track %r -- file has %s" % (a.track, ", ".join(t[0] for t in all_tracks)))
            sys.exit(2)
        name, pts = hit[0]
    else:
        name, pts = all_tracks[0]

    repo.require_repo()
    import blio
    out_dir = os.path.join(ASSIST, "logs", "ctdirect")
    os.makedirs(out_dir, exist_ok=True)
    plate = blio.Plate(os.path.abspath(a.plate), ifl_dir=out_dir)
    w, h = plate.w, plate.h
    truth = {f: (x, h - y) for f, x, y in pts}        # 3DE y-UP -> image y-DOWN
    tf = sorted(truth)
    lo, hi = tf[0], tf[-1]

    runs = []
    for f in tf:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    log("hand track %s: %d samples, f%d..f%d, runs %s" % (name, len(pts), lo, hi, runs))
    log("plate %dx%d, %d frames" % (w, h, plate.count))

    sx, sy = truth[lo]
    ours = {}
    if a.chain <= 0:
        g = cotrack.track_points(plate, [(lo, sx, sy)], lo, hi,
                                 max_side=a.max_side, on_status=log)
        ours = dict(g["tracks"][0])
    else:
        f0, qx, qy = lo, sx, sy
        while f0 < hi:
            f1 = min(hi, f0 + a.chain - 1)
            log("window f%d..f%d, query at %.1f,%.1f" % (f0, f1, qx, qy))
            g = cotrack.track_points(plate, [(f0, qx, qy)], f0, f1,
                                     max_side=a.max_side, on_status=log)
            seg = g["tracks"][0]
            ours.update(seg)
            qx, qy = seg[f1]
            f0 = f1
            if f0 >= hi:
                break

    if a.anchor:
        g0 = ours.get(lo)
        if g0 is None:
            log("guide has no sample at the seed frame")
            sys.exit(2)
        ours = {f: (sx + (x - g0[0]), sy + (y - g0[1])) for f, (x, y) in ours.items()}

    errs, off_feature = [], []
    pairs = []
    for f, (x, y) in sorted(ours.items()):
        if f < lo or f > hi:
            continue
        if f not in truth:
            off_feature.append((f, None))
            continue
        tx, ty = truth[f]
        e = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        if e > a.wrong_px:
            off_feature.append((f, e))
        else:
            errs.append(e)
            pairs.append((x - tx, y - ty))
    missed = sum(1 for f in truth if f not in ours)

    if a.verbose:
        log("")
        log("%-6s %-18s %-18s %s" % ("frame", "cotracker", "hand track", "verdict"))
        for f in sorted(set(list(ours) + list(truth))):
            o, t = ours.get(f), truth.get(f)
            if o and t:
                e = ((o[0]-t[0])**2 + (o[1]-t[1])**2) ** 0.5
                v = "ok %.1f px" % e if e <= a.wrong_px else "OFF %.0f px" % e
            elif o:
                v = "OFF -- artist has no sample here (occluded)"
            else:
                v = "missed (gap)"
            log("%-6d %-18s %-18s %s" % (f, "%.0f,%.0f" % o if o else "-",
                                         "%.0f,%.0f" % t if t else "-", v))

    if a.drift_table:
        log("")
        log("guide displacement error vs the hand track, by gap length")
        log("%-6s %-7s %-8s %-8s %-8s %s" % ("gap", "n", "p50", "p90", "max", "px/frame @p90"))
        for k in (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120):
            d = []
            for f in truth:
                f0 = f - k
                if f0 not in truth or f not in ours or f0 not in ours:
                    continue
                gx = ours[f][0] - ours[f0][0]
                gy = ours[f][1] - ours[f0][1]
                tx = truth[f][0] - truth[f0][0]
                ty = truth[f][1] - truth[f0][1]
                d.append(((gx - tx) ** 2 + (gy - ty) ** 2) ** 0.5)
            if not d:
                continue
            d.sort()
            g = lambda pc: d[min(len(d) - 1, int(round(pc * (len(d) - 1))))]
            log("%-6d %-7d %-8.2f %-8.2f %-8.2f %.3f"
                % (k, len(d), g(.5), g(.9), d[-1], g(.9) / k))

    def q(v, pc):
        v = sorted(v)
        return v[min(len(v) - 1, int(round(pc * (len(v) - 1))))] if v else float("nan")

    log("")
    log("hand track frames          : %d" % len(truth))
    log("ON THE FEATURE             : %d  (%d%%)"
        % (len(errs), round(100.0 * len(errs) / max(1, len(truth)))))
    if errs:
        log("  precision vs hand track  : p50 %.1f px  p90 %.1f px  max %.1f px"
            % (q(errs, .5), q(errs, .9), max(errs)))
        mx = sum(p[0] for p in pairs) / len(pairs)
        my = sum(p[1] for p in pairs) / len(pairs)
        log("  constant offset          : %.1f px  (%.1f, %.1f)"
            % ((mx * mx + my * my) ** 0.5, mx, my))
    log("OFF THE FEATURE            : %d" % len(off_feature))
    if off_feature:
        log("  %s" % ", ".join("f%d%s" % (f, "" if e is None else " %.0fpx" % e)
                               for f, e in off_feature[:24]))
    log("MISSED (gap)               : %d" % missed)
    print("VERDICT: %s" % ("PASS" if not off_feature else "FAIL"))


if __name__ == "__main__":
    main()
