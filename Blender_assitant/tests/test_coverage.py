"""The shot report, checked against shots whose answer is known by construction.

A coverage metric that looks plausible and measures the wrong thing is the defect this
project keeps rediscovering -- twice in 2026-08 a quality number turned out to be reading the
plate rather than the tracker, and each was caught by feeding it input whose answer was known
and seeing a figure that should have been zero. So none of these run on real footage: every
case here is generated from a camera model, and the right answer is a fact about how it was
built rather than a judgement about how it looks.

  * a NODAL PAN carries no depth however far it turns -- `parallax` must say degenerate;
  * a camera TRANSLATING past two depth planes carries plenty -- must say ok;
  * a FLAT scene shot by a translating camera is degenerate too, and this is the case that
    separates a real test from one that only detects rotation;
  * a track on a MOVING OBJECT must come out as the disagreement, and the static ones must
    not.

    runtime\\python311\\python.exe tests\\test_coverage.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSIST = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [os.path.join(ASSIST, "sidecar"), HERE]

import coverage                                                       # noqa: E402

W, H, FOCAL = 3840, 2160, 2400.0
FAILED = []


def check(name, got, want):
    ok = got == want
    print("  %-58s %s" % (name, "ok" if ok else "FAIL  got %r want %r" % (got, want)))
    if not ok:
        FAILED.append(name)


def note(name, value):
    print("  %-58s %s" % (name, value))


def _project(pts3, t):
    """Pinhole, camera translated by `t` = (tx, ty, tz). Returns Nx2 image points."""
    X = pts3[:, 0] - t[0]
    Y = pts3[:, 1] - t[1]
    Z = pts3[:, 2] - t[2]
    return np.stack([FOCAL * X / Z + W / 2.0, FOCAL * Y / Z + H / 2.0], axis=1)


def translating(n=120, frames=40, depths=(6.0, 60.0), rate=0.05, seed=0):
    """Camera sliding sideways past points at two depths -- real parallax by construction."""
    rng = np.random.default_rng(seed)
    z = rng.choice(depths, size=n)
    pts3 = np.stack([rng.uniform(-4, 4, n), rng.uniform(-2.2, 2.2, n), z], axis=1)
    tracks = {}
    for i in range(n):
        tracks["T%03d" % i] = {}
    for k in range(frames):
        uv = _project(pts3, (rate * k, 0.0, 0.0))
        for i in range(n):
            x, y = uv[i]
            if 0 <= x < W and 0 <= y < H:
                tracks["T%03d" % i][k + 1] = (float(x), float(y))
    return {k: v for k, v in tracks.items() if len(v) > frames * 0.6}


def flat(n=120, frames=40, depth=20.0, rate=0.05, seed=1):
    """The same translating camera, but everything on ONE plane.

    Included because a parallax test that only notices rotation would pass the nodal case and
    still miss this -- and a wall or a road fills the frame far more often than a nodal pan
    turns up. A homography maps one plane to one plane whatever the camera does, so there is
    no depth to recover here either.
    """
    rng = np.random.default_rng(seed)
    pts3 = np.stack([rng.uniform(-4, 4, n), rng.uniform(-2.2, 2.2, n),
                     np.full(n, depth)], axis=1)
    tracks = {"T%03d" % i: {} for i in range(n)}
    for k in range(frames):
        uv = _project(pts3, (rate * k, 0.0, 0.0))
        for i in range(n):
            x, y = uv[i]
            if 0 <= x < W and 0 <= y < H:
                tracks["T%03d" % i][k + 1] = (float(x), float(y))
    return {k: v for k, v in tracks.items() if len(v) > frames * 0.6}


def nodal(n=120, frames=40, per_frame_deg=0.25, seed=2):
    """Camera turning about its own centre. Depth is irrelevant to it, which is the point."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(5.0, 60.0, n)
    pts3 = np.stack([rng.uniform(-4, 4, n), rng.uniform(-2.2, 2.2, n), z], axis=1)
    K = np.array([[FOCAL, 0, W / 2.0], [0, FOCAL, H / 2.0], [0, 0, 1.0]])
    base = _project(pts3, (0.0, 0.0, 0.0))
    hom = np.concatenate([base, np.ones((n, 1))], axis=1)
    tracks = {"T%03d" % i: {} for i in range(n)}
    for k in range(frames):
        a = np.radians(per_frame_deg * k)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        M = K @ R @ np.linalg.inv(K)
        p = (M @ hom.T).T
        uv = p[:, :2] / p[:, 2:3]
        for i in range(n):
            x, y = uv[i]
            if 0 <= x < W and 0 <= y < H:
                tracks["T%03d" % i][k + 1] = (float(x), float(y))
    return {k: v for k, v in tracks.items() if len(v) > frames * 0.6}


def add_movers(tracks, k=6, seed=3):
    """Tracks that move on their own -- a truck crossing while the camera does its own thing."""
    rng = np.random.default_rng(seed)
    fs = coverage.frames_of(tracks)
    names = []
    for j in range(k):
        x0, y0 = rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.2, H * 0.8)
        vx, vy = rng.uniform(14, 26), rng.uniform(-6, 6)
        nm = "MOVER%d" % j
        names.append(nm)
        tracks[nm] = {f: (float(x0 + vx * i), float(y0 + vy * i))
                      for i, f in enumerate(fs)
                      if 0 <= x0 + vx * i < W and 0 <= y0 + vy * i < H}
    return names


def main():
    print("counting, which cannot be wrong about anything but itself")
    t = {"a": {1: (0, 0), 2: (0, 0), 5: (0, 0)}, "b": {1: (0, 0), 2: (0, 0)}}
    check("live per frame counts gaps as absent",
          coverage.per_frame_count(t), {1: 2, 2: 2, 3: 0, 4: 0, 5: 1})
    check("thin stretches are reported as runs, not as frames",
          coverage.thin_runs({1: 9, 2: 3, 3: 2, 4: 9, 5: 1}, 8), [(2, 3, 2), (5, 5, 1)])
    check("a shot that is never thin reports nothing",
          coverage.thin_runs({1: 9, 2: 10}, 8), [])
    g = coverage.grid_occupancy({"a": {1: (10, 10)}, "b": {1: (W - 10, H - 10)}}, W, H, 1,
                                cols=2, rows=2)
    check("grid puts a point in the corner it is in", (g[0][0], g[1][1]), (1, 1))
    check("  and leaves the other corners empty", (g[0][1], g[1][0]), (0, 0))

    print("\nparallax, on cameras whose answer is a fact about how they were built")
    tr = translating()
    par = coverage.parallax(tr, coverage.frames_of(tr))
    note("translating past two depths: h_share %.2f" % par.get("h_share_median", float("nan")),
         par["verdict"])
    check("  a camera that really moves past depth reads ok", par["verdict"], "ok")

    nd = nodal()
    par_n = coverage.parallax(nd, coverage.frames_of(nd))
    note("nodal pan: h_share %.2f" % par_n.get("h_share_median", float("nan")),
         par_n["verdict"])
    check("  a nodal pan is called degenerate", par_n["verdict"], "degenerate")

    fl = flat()
    par_f = coverage.parallax(fl, coverage.frames_of(fl))
    note("flat scene, moving camera: h_share %.2f" % par_f.get("h_share_median", float("nan")),
         par_f["verdict"])
    check("  a flat scene is degenerate even though the camera moved",
          par_f["verdict"], "degenerate")

    check("the two degenerate cases are separated from the real one by a clear margin",
          min(par_n.get("h_share_median", 0), par_f.get("h_share_median", 0))
          - par.get("h_share_median", 1) > 0.2, True)

    print("\ndisagreement, with the movers known by name")
    tr2 = translating(seed=7)
    movers = set(add_movers(tr2))
    rep = coverage.report(tr2, W, H)
    flagged = set(s["id"] for s in rep["suspect"])

    # The camera translates in X, so the epipolar lines run horizontal, and a mover
    # travelling horizontally is geometrically invisible -- it looks exactly like a static
    # point at another depth. No threshold reaches it. Measured on these six: 0.6 deg off
    # horizontal is missed on all 12 pairs, 2.4 deg is caught on all 12. So this asserts what
    # is TRUE -- every mover with a component across the epipolar direction is found -- rather
    # than asserting six and quietly settling for five.
    across = set()
    for nm in movers:
        pts = sorted(tr2[nm].items())
        (f0, p0), (f1, p1) = pts[0], pts[-1]
        vx = (p1[0] - p0[0]) / float(f1 - f0)
        vy = (p1[1] - p0[1]) / float(f1 - f0)
        if abs(math.degrees(math.atan2(vy, vx))) >= 2.0:
            across.add(nm)
    check("this set contains an epipolar-aligned mover to be blind to",
          len(movers) - len(across), 1)
    check("every mover that crosses the epipolar direction is flagged",
          sorted(flagged & movers), sorted(across))
    check("  and nothing static is", sorted(flagged - movers), [])

    print("\nthe report holds together")
    check("counts the tracks it was given", rep["tracks"], len(tr2))
    check("a healthy synthetic shot has no thin stretch", rep["thin_runs"], [])
    note("median live %d, min live %d, median span %d"
         % (rep["median_live"], rep["min_live"], rep["median_span"]), "")

    print("")
    if FAILED:
        print("COVERAGE: FAIL -- %s" % "; ".join(FAILED))
        return 1
    print("COVERAGE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
