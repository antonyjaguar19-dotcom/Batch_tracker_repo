"""Did the tracks that disappeared come back on the SAME feature?

A gap in an exported track means the point went away and returned. Whether it returned
*correctly* is the one thing the span numbers cannot say: a track that resumes on the wrong
feature is longer, and wrong.

The measurement must not use the TAPNext guide, because the guide is where the replant
position came from -- checking it against itself would confirm anything. So this reads only
the plate:

  for each internal gap, last frame before it (fa, pa) -> first frame after it (fb, pb)
      test     NCC of the patch at (fa,pa) against the patch at (fb,pb)
      recheck  search a box around pb for the BEST match to the fa patch, and report how
               far that is from where the track actually resumed

`recheck` is the useful half. A high NCC at a 6px offset means the feature is there and the
replant missed it by 6px; a low NCC everywhere means the feature is not in that
neighbourhood at all.

### The control, and why there is one

An NCC of 0.62 across a gap means nothing on its own -- appearance drifts with lighting,
defocus and perspective, so some of that loss is the plate, not the tracker. Every gap is
therefore paired with a CONTROL measured on the same track, over the same number of frames,
on a stretch with NO gap in it. Same feature, same time separation, known-good: whatever
the control scores is what a correct reappearance should score. Both of the metric defects
this repo found in 2026-08 were metrics that looked plausible while measuring the plate
instead of the tracker (see bench/README.md), so the control is not optional.

    runtime\\python311\\python.exe experiments\\blender_track\\check_replants.py ^
        --plate experiments\\blender_track\\out\\SH016\\plate ^
        --tracks experiments\\blender_track\\out\\SH016__dense__blender.txt --sheet
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blio import Plate, OUT_DIR, read_3de  # noqa: E402  (sets OPENCV_IO_ENABLE_OPENEXR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.pattern_refine import _extract, _ncc_match  # noqa: E402


def gaps_of(pts: dict) -> list[tuple[int, int]]:
    """Internal holes only: (last frame before, first frame after)."""
    fs = sorted(pts)
    return [(a, b) for a, b in zip(fs, fs[1:]) if b - a > 1]


def _synthesise_gaps(tracks: dict, L: int) -> dict:
    """Punch an L-frame hole into the middle of every long CONTINUOUS run.

    The two points either side are the track's own, measured positions, so the feature
    really is at both -- the reappearance is correct by construction. Anything but ~0px
    offsets here means the measurement is broken.
    """
    out = {}
    for name, pts in tracks.items():
        fs = sorted(pts)
        if len(fs) < 2 * L + 6 or (fs[-1] - fs[0] + 1) != len(fs):
            continue                                # needs a run long enough, and no real gap
        mid = fs[len(fs) // 2]
        out[name] = {f: p for f, p in pts.items() if not (mid < f < mid + L)}
    return out


class Frames:
    """Grayscale frame cache. These plates are 4K and a gap check hits the same handful of
    frames repeatedly, so decoding each one once matters more than the memory."""

    def __init__(self, plate: Plate, limit: int = 64):
        self.plate, self.limit, self.cache = plate, limit, {}

    def gray(self, f1: int):
        g = self.cache.get(f1)
        if g is None:
            img = self.plate.frame(f1 - 1)          # exported frames are 1-based
            if img is None:
                return None
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(self.cache) >= self.limit:
                self.cache.pop(next(iter(self.cache)))
            self.cache[f1] = g
        return g


def ncc_pair(fr: Frames, fa: int, pa, fb: int, pb, half: int, search: int, h_img: int,
             ambiguity: float = 1.0):
    """(ncc at the given point, ncc of the best match nearby, px offset to it).

    `best`/`off` come back None when the search box holds a rival peak within `ambiguity`
    of the winner. That is not a failure to measure -- it is the honest answer on
    repetitive texture, where "the best match is 19px away" says nothing about the track
    because the two positions are indistinguishable to NCC. The first version of this
    checker lacked it and failed its own self-check at 45%.
    """
    ga, gb = fr.gray(fa), fr.gray(fb)
    if ga is None or gb is None:
        return None
    # Exports are y-up; the image is y-down.
    ax, ay = pa[0], (h_img - 1.0) - pa[1]
    bx, by = pb[0], (h_img - 1.0) - pb[1]
    pat = _extract(ga, ax, ay, half)
    at = _extract(gb, bx, by, half)
    if pat is None or at is None:
        return None
    a = pat.astype(np.float32)
    b = at.astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    here = float((a * b).mean())

    m = _ncc_match(gb, pat, bx, by, search, half, edge_clamp=True,
                   ambiguity_ratio=ambiguity)
    if m is None:
        return here, None, None
    mx, my, cc = m
    return here, float(cc), float(np.hypot(mx - bx, my - by))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--half", type=int, default=15, help="patch half-size in px")
    ap.add_argument("--search", type=int, default=24, help="recheck radius in px")
    ap.add_argument("--good-offset", type=float, default=1.5,
                    help="px; a reappearance this close to the best match is on the feature")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="NCC a rival must beat the resumed point by before the "
                         "reappearance is called wrong")
    ap.add_argument("--ambiguity", type=float, default=0.92,
                    help="refuse to judge when a rival peak scores this fraction of the "
                         "winner; 1.0 disables (and fails the self-check)")
    ap.add_argument("--sheet", action="store_true", help="write a before/after contact sheet")
    ap.add_argument("--sheet-n", type=int, default=12)
    ap.add_argument("--max-gaps", type=int, default=400)
    ap.add_argument("--selfcheck", type=int, default=0, metavar="L",
                    help="punch synthetic gaps of L frames into CONTINUOUS tracks and "
                         "measure those instead. The reappearance is then known-correct by "
                         "construction, so the offsets must come out ~0")
    args = ap.parse_args()

    plate = Plate(args.plate, ifl_dir=OUT_DIR)
    tracks = read_3de(args.tracks)
    if args.selfcheck:
        tracks = _synthesise_gaps(tracks, args.selfcheck)
        print(f"SELF-CHECK: {args.selfcheck}-frame gaps punched into continuous tracks.\n"
              f"The points either side are the track's own, so a working measurement must\n"
              f"report offsets ~0px and 'lands on the feature' ~100%. Anything else is the\n"
              f"CHECKER being wrong, not the tracker.")
    fr = Frames(plate)
    name = os.path.splitext(os.path.basename(args.tracks))[0]

    rows = []
    for tname, pts in tracks.items():
        for fa, fb in gaps_of(pts):
            if len(rows) >= args.max_gaps:
                break
            L = fb - fa
            r = ncc_pair(fr, fa, pts[fa], fb, pts[fb], args.half, args.search, plate.h,
                         args.ambiguity)
            if r is None:
                continue
            here, best, off = r
            # Control: same track, same separation, no gap in between. If the track does not
            # reach back that far, this gap simply has no control and is excluded from the
            # control column rather than being compared against something else.
            ctrl = None
            fc = fa - L
            if fc in pts and all((f in pts) for f in range(fc, fa + 1)):
                c = ncc_pair(fr, fc, pts[fc], fa, pts[fa], args.half, args.search, plate.h,
                             args.ambiguity)
                if c is not None:
                    ctrl = c[0]
            rows.append({"track": tname, "fa": fa, "fb": fb, "len": L, "ncc": here,
                         "best": best, "off": off, "ctrl": ctrl,
                         "pa": pts[fa], "pb": pts[fb]})

    if not rows:
        print(f"{name}: no gaps in {len(tracks)} tracks -- nothing disappeared and came back")
        return 0

    # A reappearance is judged WRONG only when a clearly better match sits somewhere else:
    # both far away AND better by a real margin. A point whose NCC ties the best match in
    # its neighbourhood is on its feature as far as image evidence goes, however far the
    # argmax happens to land -- that is the ambiguity the self-check exposed.
    ambig = [r for r in rows if r["off"] is None]
    onf = [r for r in rows if r["off"] is not None]
    for r in onf:
        r["margin"] = r["best"] - r["ncc"]
    good = [r for r in onf if r["off"] <= args.good_offset or r["margin"] <= args.margin]
    bad = [r for r in onf if r not in good]
    near = [r for r in bad if r["off"] <= 6.0]
    lost = [r for r in bad if r["off"] > 6.0]
    ctrls = [r["ctrl"] for r in rows if r["ctrl"] is not None]
    nccs = [r["ncc"] for r in rows]

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    def pct(n):
        return 100.0 * n / max(1, len(onf))

    print(f"\n{name}   {len(tracks)} tracks, {len(rows)} gaps "
          f"(median {med([r['len'] for r in rows]):.0f} frames long)\n")
    print(f"  judged: {len(onf)} of {len(rows)}   "
          f"({len(ambig)} on texture too repetitive to judge -- NCC cannot tell "
          f"two positions apart there)\n")
    print(f"  reappearance is on the feature      {len(good):4d}  ({pct(len(good)):.0f}%)")
    print(f"  off by {args.good_offset}-6px, and beatable       "
          f"{len(near):4d}  ({pct(len(near)):.0f}%)")
    print(f"  clearly on the wrong thing          {len(lost):4d}  ({pct(len(lost)):.0f}%)"
          f"   >6px away AND >{args.margin:g} NCC better there")
    print(f"\n  median offset to best match         {med([r['off'] for r in onf]):.2f}px")
    print(f"  median NCC across the gap           {med(nccs):.3f}")
    if ctrls:
        print(f"  median NCC on the CONTROL           {med(ctrls):.3f}   "
              f"(same tracks, same frame separation, no gap)")
        d = med(nccs) - med(ctrls)
        print(f"  difference                          {d:+.3f}   "
              + ("appearance across a gap matches an equal stretch without one"
                 if abs(d) < 0.10 else
                 "the gap costs real appearance -- read the offsets, not the NCC"))
    else:
        print("  no control available (tracks do not reach back far enough)")

    worst = sorted(bad, key=lambda r: -r["off"])[:8]
    if worst:
        print("\n  worst reappearances")
        for r in worst:
            print(f"    {r['track']:<12} f{r['fa']:>4}->{r['fb']:<4} gap {r['len']:>3}f   "
                  f"off {r['off']:6.2f}px   ncc {r['ncc']:.2f} -> best {r['best']:.2f}")

    if args.sheet:
        write_sheet(fr, plate, rows, args, os.path.join(OUT_DIR, name + "__replants.png"))
    plate.close()
    return 0


def write_sheet(fr: Frames, plate: Plate, rows, args, out_png: str) -> None:
    """before | after | best-match, one row per sampled gap. For eyeballing, not scoring."""
    onf = [r for r in rows if r["off"] is not None]
    onf.sort(key=lambda r: -r["off"])              # worst first: that is what to look at
    pick = onf[:args.sheet_n]
    if not pick:
        return
    P = 2 * args.half + 1
    Z = 3                                          # patches are small; upscale to see them
    cell = P * Z
    pad, top = 6, 18
    W = 3 * cell + 4 * pad + 190
    H = len(pick) * (cell + pad) + pad + top
    sheet = np.full((H, W, 3), 28, np.uint8)
    for i, lbl in enumerate(("before gap", "after gap", "best match")):
        cv2.putText(sheet, lbl, (190 + i * (cell + pad), 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)

    for i, r in enumerate(pick):
        y0 = top + pad + i * (cell + pad)
        ga, gb = fr.gray(r["fa"]), fr.gray(r["fb"])
        if ga is None or gb is None:
            continue
        ax, ay = r["pa"][0], (plate.h - 1.0) - r["pa"][1]
        bx, by = r["pb"][0], (plate.h - 1.0) - r["pb"][1]
        pa = _extract(ga, ax, ay, args.half)
        pb = _extract(gb, bx, by, args.half)
        m = _ncc_match(gb, pa, bx, by, args.search, args.half, edge_clamp=True) if pa is not None else None
        pc = _extract(gb, m[0], m[1], args.half) if m else None
        for j, p in enumerate((pa, pb, pc)):
            if p is None:
                continue
            tile = cv2.resize(p.astype(np.uint8), (cell, cell),
                              interpolation=cv2.INTER_NEAREST)
            x0 = 190 + j * (cell + pad)
            sheet[y0:y0 + cell, x0:x0 + cell] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        col = (90, 255, 120) if r["off"] <= args.good_offset else (
            (60, 200, 255) if r["off"] <= 6.0 else (80, 80, 255))
        cv2.putText(sheet, f"{r['track'][:12]}", (8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"f{r['fa']}->{r['fb']} ({r['len']}f)", (8, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"off {r['off']:.1f}px", (8, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
    cv2.imwrite(out_png, sheet)
    print(f"\n  contact sheet -> {out_png}  (worst {len(pick)} of {len(onf)}, worst first)")


if __name__ == "__main__":
    raise SystemExit(main())
