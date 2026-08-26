"""Judge the SET of tracks, not each track on its own.

Everything else in this addon asks "is this track still on its feature". That is the track
job. The matchmove job is a different question and no part of this answered it: will these
tracks SOLVE, and if not, what is missing and where.

An artist finds that out by exporting to 3DE, solving, and reading the error -- which is the
expensive way to learn that a shot was a nodal pan all along, or that eleven frames in the
middle have four tracks on them, or that the six tracks doing all the work are on a truck.

Four things are computed here, in the order they ruin a day:

  * **How many tracks are live on each frame.** A frame with too few is a frame the solve
    cannot constrain, and a run of them is where a solve breaks. Exact -- it is counting.
  * **Where they are.** Tracks bunched in one corner leave the camera free to rotate about
    the empty part of the frame. A grid occupancy per frame says which region is bare.
  * **Whether there is any usable parallax at all.** This is the one that costs a day. If a
    homography explains the motion as well as the epipolar geometry does, the pair carries no
    depth: a nodal pan, or a scene that is effectively flat. A 3D solve on that does not
    fail loudly, it produces a camera that looks nearly right and drifts.
  * **Which tracks disagree with everything else.** A track that is persistently an outlier
    to the dominant geometry is on a moving object or has slid off. Those are the ones a
    solver throws out, and the ones an artist wants to see before solving rather than after.
    It has one blind spot and it is geometric, not a threshold: a point moving along its own
    epipolar line looks exactly like a static point at another depth. See `disagreement`.

No solve is run and nothing is modified. This reports.

Coordinates in, throughout: image pixels, y-DOWN, original plate resolution.
"""

import math

import numpy as np

#: RANSAC reprojection threshold, in plate pixels. A hand track disagrees with itself by a
#: couple of pixels (measured: the artist's own click scatter against the correlator runs
#: 2.3-2.5 px on a 3840-wide plate), so anything under that would call correct tracks
#: outliers. 3.0 leaves room for that without admitting a track that has actually slid.
RANSAC_PX = 3.0

#: Below this many correspondences a frame pair cannot support a fundamental matrix at all
#: (seven is the algebraic minimum; fifteen is the point where RANSAC stops being a lottery).
MIN_PAIR = 15

#: A pair whose points barely moved says nothing about parallax either way -- the homography
#: and the epipolar geometry both fit a stationary field perfectly. In plate pixels, median.
MIN_MOTION_PX = 1.5

#: How much better the epipolar geometry has to explain a pair before it is called parallax.
#: `h_share` near 1.0 means a homography did just as well, which is the degenerate case.
#: 0.85 is provisional and is measured against synthetic shots with a known answer in
#: `tests/test_coverage.py` -- a pure rotation lands near 1.0, a translating camera over two
#: depth planes lands well below it.
PARALLAX_H_SHARE = 0.85


def frames_of(tracks):
    """Every frame any track has a sample on, in order."""
    got = set()
    for pts in tracks.values():
        got.update(pts)
    return sorted(got)


def per_frame_count(tracks, lo=None, hi=None):
    """{frame: how many tracks are live there}. Gaps count as absent, because they are."""
    fs = frames_of(tracks)
    if not fs:
        return {}
    lo = fs[0] if lo is None else int(lo)
    hi = fs[-1] if hi is None else int(hi)
    out = {f: 0 for f in range(lo, hi + 1)}
    for pts in tracks.values():
        for f in pts:
            if lo <= f <= hi:
                out[f] += 1
    return out


def thin_runs(counts, floor):
    """Runs of consecutive frames under `floor` tracks -- [(first, last, worst), ...].

    Runs and not a list of frames: an artist fixes a stretch, not a frame, and eleven
    separate warnings for eleven adjacent frames is the same fact printed eleven times.
    """
    out = []
    run = None
    for f in sorted(counts):
        if counts[f] < floor:
            if run and f == run[1] + 1:
                run[1] = f
                run[2] = min(run[2], counts[f])
            else:
                if run:
                    out.append(tuple(run))
                run = [f, f, counts[f]]
        elif run:
            out.append(tuple(run))
            run = None
    if run:
        out.append(tuple(run))
    return out


def cell_presence(tracks, w, h, frames, cols=3, rows=3):
    """For each grid cell, the fraction of frames that have at least one track in it.

    Per-frame and not once: a cell that is covered for the first half of the shot and bare
    for the second is a real hole, and looking at a single frame either misses it or reports
    it as total. Thirds, because that is how an artist frames coverage and because it maps
    one-to-one onto the words used to describe it -- a 4x3 grid had two different columns
    both called "left", so an occupied cell could be reported as bare.
    """
    fs = list(frames)
    if not fs:
        return [[0.0] * cols for _ in range(rows)]
    seen = [[0] * cols for _ in range(rows)]
    for f in fs:
        cell = grid_occupancy(tracks, w, h, f, cols, rows)
        for y in range(rows):
            for x in range(cols):
                if cell[y][x]:
                    seen[y][x] += 1
    return [[seen[y][x] / float(len(fs)) for x in range(cols)] for y in range(rows)]


def cell_name(x, y, cols, rows):
    """"top left", "middle centre" ... Exact only while the grid is 3x3."""
    v = ("top", "middle", "bottom")[min(rows - 1, y)] if rows == 3 else "row %d" % (y + 1)
    hpos = ("left", "centre", "right")[min(cols - 1, x)] if cols == 3 else "col %d" % (x + 1)
    return "%s %s" % (v, hpos)


def grid_occupancy(tracks, w, h, frame, cols=3, rows=3):
    """How many tracks sit in each cell of a cols x rows grid on one frame."""
    cell = [[0] * cols for _ in range(rows)]
    for pts in tracks.values():
        p = pts.get(frame)
        if p is None:
            continue
        cx = min(cols - 1, max(0, int(p[0] * cols / float(w))))
        cy = min(rows - 1, max(0, int(p[1] * rows / float(h))))
        cell[cy][cx] += 1
    return cell


def _pair_points(tracks, f1, f2):
    """Correspondences shared by two frames, as (ids, Nx2, Nx2)."""
    ids, a, b = [], [], []
    for tid, pts in tracks.items():
        p, q = pts.get(f1), pts.get(f2)
        if p is not None and q is not None:
            ids.append(tid)
            a.append(p)
            b.append(q)
    return ids, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def pair_geometry(tracks, f1, f2, ransac_px=RANSAC_PX):
    """Fit a homography and a fundamental matrix to one frame pair, and compare them.

    The comparison is the point. Both models are fitted to the SAME correspondences with the
    same threshold, and `h_share` is how much of the epipolar geometry's inlier set the
    homography could also explain. A homography cannot represent parallax; so when it
    explains everything the epipolar geometry does, there is none to represent -- the camera
    rotated about its own centre, or the scene is flat enough that it may as well have.

    Returns None when the pair cannot support the question: too few correspondences, or
    nothing moved.
    """
    import cv2                                                        # noqa: PLC0415

    ids, a, b = _pair_points(tracks, f1, f2)
    if len(ids) < MIN_PAIR:
        return None
    motion = float(np.median(np.linalg.norm(b - a, axis=1)))
    if motion < MIN_MOTION_PX:
        return None

    H, hm = cv2.findHomography(a, b, cv2.RANSAC, ransac_px)
    if H is None or hm is None:
        return None
    hm = hm.ravel().astype(bool)
    n_h = int(hm.sum())

    # A fundamental matrix is NOT DETERMINED when every point lies on one plane -- there is a
    # whole family of F consistent with the data, and RANSAC picks from it at random. OpenCV
    # does not report that politely: on a synthetic flat scene `findFundamentalMat` raised
    # from inside cv::Mat rather than returning None, which would have taken the whole report
    # down on one of the commonest shots there is (a wall, a road, a table top).
    #
    # So the failure is caught and READ, because it is evidence rather than an error. If a
    # homography fits and the epipolar geometry cannot even be estimated, that is the
    # degenerate case stated as loudly as it can be stated.
    F, fm = None, None
    try:
        F, fm = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, ransac_px, 0.99)
    except cv2.error:
        F, fm = None, None
    # It can also hand back several stacked solutions, or none, without raising.
    if F is None or fm is None or getattr(F, "shape", (0, 0))[0] != 3:
        if n_h < MIN_PAIR:
            return None
        return {"frames": (int(f1), int(f2)), "n": len(ids), "motion_px": motion,
                "inliers_h": n_h, "inliers_f": n_h, "h_share": 1.0, "f_ok": False,
                "ids": ids, "f_inlier": hm, "h_inlier": hm}

    fm = fm.ravel().astype(bool)
    n_f = int(fm.sum())
    if n_f < MIN_PAIR:
        return None
    return {"frames": (int(f1), int(f2)), "n": len(ids), "motion_px": motion,
            "inliers_h": n_h, "inliers_f": n_f,
            # Capped at 1.0: RANSAC fits the two models independently, so on a degenerate
            # pair the homography can take one or two more points than the epipolar geometry
            # did and push the ratio just over. That is noise, not more-than-total agreement.
            "h_share": min(1.0, n_h / float(n_f)), "f_ok": True,
            "ids": ids, "f_inlier": fm, "h_inlier": hm}


def parallax(tracks, frames, step=None, samples=12, ransac_px=RANSAC_PX):
    """Does this shot carry usable parallax, sampled across its length?

    Sampled and not exhaustive: consecutive frames of a slow move carry almost no baseline,
    so a pair one frame apart reports "no parallax" on a shot that has plenty. The step is a
    fraction of the shot, which is what actually separates viewpoints.
    """
    fs = sorted(frames)
    if len(fs) < 4:
        return {"verdict": "unknown", "reason": "too few frames", "pairs": []}
    step = step or max(1, len(fs) // 8)
    idx = np.linspace(0, len(fs) - 1 - step, min(samples, max(1, len(fs) - step)))
    got = []
    for i in sorted(set(int(round(v)) for v in idx)):
        g = pair_geometry(tracks, fs[i], fs[i + step], ransac_px)
        if g is not None:
            got.append(g)
    if not got:
        return {"verdict": "unknown",
                "reason": "no frame pair had enough shared tracks that moved", "pairs": []}
    shares = sorted(g["h_share"] for g in got)
    med = shares[len(shares) // 2]
    if med >= PARALLAX_H_SHARE:
        verdict, reason = "degenerate", (
            "a homography explains %d%% of the motion -- this pair carries no depth. Either "
            "the camera turned on the spot or everything tracked is on one plane. A 3D solve "
            "will not fail loudly here; it will produce a camera that looks nearly right and "
            "drifts" % round(med * 100))
    else:
        verdict, reason = "ok", (
            "a homography only explains %d%% of the motion, so the rest is depth -- there is "
            "parallax to solve with" % round(med * 100))
    return {"verdict": verdict, "reason": reason, "h_share_median": med,
            "pairs": [{k: g[k] for k in ("frames", "n", "motion_px", "inliers_h",
                                         "inliers_f", "h_share")} for g in got]}


def disagreement(tracks, frames, step=None, samples=12, ransac_px=RANSAC_PX):
    """How often each track is an outlier to the dominant geometry, and over how many tests.

    A track on a moving object, or one that has slid onto something else, does not fit the
    camera motion the rest agree on. This is what a solver discards -- reported before the
    solve rather than after it.

    Judged only where a track was actually TESTED. A track that appears in two of twelve
    pairs and fails both is not the same claim as one that fails twelve of twelve, and
    averaging them would say it is.

    **The blind spot, stated rather than tuned away.** A point moving ALONG its own epipolar
    line is indistinguishable from a static point at a different depth -- two views cannot
    tell those apart, and no threshold here changes that. Measured on six planted movers
    under a camera translating in X, where the epipolar lines run horizontal:

        travelling  0.6 deg off horizontal   ->  0 of 12 pairs disagreed   MISSED
        travelling  2.4 deg off horizontal   -> 12 of 12 pairs disagreed   found
        travelling 13.5 deg off horizontal   -> 12 of 12 pairs disagreed   found

    So the cliff is sharp and everything either side of it is decided cleanly. Real footage
    is kinder than the synthetic case because a hand-held or craned camera does not hold one
    baseline direction for a whole shot, and a mover invisible on one baseline shows up on
    the next -- but a locked-off dolly plus a car driving parallel to it is exactly the shot
    where this says nothing. It is a reason to read "no disagreement" as "nothing to report",
    never as "no movers".
    """
    fs = sorted(frames)
    if len(fs) < 4:
        return {}
    step = step or max(1, len(fs) // 8)
    idx = np.linspace(0, len(fs) - 1 - step, min(samples, max(1, len(fs) - step)))
    tested, failed = {}, {}
    for i in sorted(set(int(round(v)) for v in idx)):
        g = pair_geometry(tracks, fs[i], fs[i + step], ransac_px)
        if g is None:
            continue
        for tid, ok in zip(g["ids"], g["f_inlier"]):
            tested[tid] = tested.get(tid, 0) + 1
            if not ok:
                failed[tid] = failed.get(tid, 0) + 1
    return {tid: (failed.get(tid, 0), n) for tid, n in tested.items()}


def report(tracks, w, h, floor=8, cols=3, rows=3, min_tests=3, bad_share=0.6,
           hole_share=0.5):
    """Everything above, as one read-only answer.

    `floor` is how many simultaneous tracks a frame needs before it stops being called thin.
    Eight is a working number, not a measured one -- a solve wants more than the six degrees
    of freedom it is estimating, with margin for the ones that turn out to be wrong.
    """
    fs = frames_of(tracks)
    if not fs:
        return {"tracks": 0, "reason": "no tracks"}
    counts = per_frame_count(tracks)
    thin = thin_runs(counts, floor)
    par = parallax(tracks, fs)
    dis = disagreement(tracks, fs)
    suspect = sorted(
        ((tid, bad, n) for tid, (bad, n) in dis.items()
         if n >= min_tests and bad / float(n) >= bad_share),
        key=lambda r: (-r[1] / float(r[2]), -r[2]))
    worst_f = min(counts, key=lambda f: counts[f]) if counts else None
    pres = cell_presence(tracks, w, h, fs, cols, rows)
    # A hole is a region bare for a good part of the shot, not one bare on a single frame.
    bare = [(x, y, pres[y][x]) for y in range(rows) for x in range(cols)
            if pres[y][x] < hole_share]
    bare.sort(key=lambda c: c[2])
    spans = [len(p) for p in tracks.values()]

    # Does the plate size we were handed actually describe these coordinates?
    #
    # Every number here is a fraction of the frame, so the wrong resolution does not fail --
    # it produces a confident, detailed, wrong answer. Caught by this eval on a real file: a
    # 1920x1080 export read against a 3840x2160 plate put every track in one quadrant and
    # reported five of nine regions as having no coverage on any frame, which is exactly what
    # a badly-covered shot looks like. `ops_3de` already warns about the same mismatch on
    # import; the report has more to lose by staying quiet about it.
    xs = [p[0] for pts in tracks.values() for p in pts.values()]
    ys = [p[1] for pts in tracks.values() for p in pts.values()]
    extent = {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)]} if xs else None
    mismatch = None
    if xs and (max(xs) < w * 0.55 or max(ys) < h * 0.55):
        mismatch = ("every track falls inside %.0fx%.0f of a %dx%d plate. Either the shot is "
                    "tracked in one corner, or these tracks were made against a smaller "
                    "plate -- the coverage below is a fraction of the frame and means "
                    "nothing if the size is wrong" % (max(xs), max(ys), w, h))

    return {
        "extent": extent,
        "size_warning": mismatch,
        "tracks": len(tracks),
        "frames": [fs[0], fs[-1]],
        "per_frame": counts,
        "median_live": int(np.median(list(counts.values()))) if counts else 0,
        "min_live": min(counts.values()) if counts else 0,
        "thin_runs": thin,
        "floor": floor,
        "median_span": int(np.median(spans)) if spans else 0,
        "parallax": par,
        "suspect": [{"id": t, "failed": b, "tested": n} for t, b, n in suspect],
        "worst_frame": worst_f,
        "bare_cells": [{"cell": [x, y], "name": cell_name(x, y, cols, rows),
                        "present_share": round(sh, 3)} for x, y, sh in bare],
        "cell_presence": pres,
        "grid": [cols, rows],
    }
