# -*- coding: utf-8 -*-
"""Self-check of the epipolar residual column (`track_filter.geometric_residuals`).

Pure Python + numpy/cv2, no GPU, seconds to run. Both quality defects found in 2026-08 were
metrics that looked plausible and were measuring the PLATE instead of the tracker, each caught
by feeding ground truth and seeing a number that should have been zero. So:

  1. perfect tracks from a real 3D scene must read ~0px
  2. one track riding an independently-moving object must stand out from them
  3. adding pixel noise must raise it smoothly -- not flip it
  4. a PLANAR scene and a PURE ROTATION must report NOT MEASURABLE, not a small number.
     F is undetermined in both, and a confident answer there is the exact failure this
     column would otherwise introduce into a matchmove review.

    runtime\\python311\\python.exe tools\\check_geo_residual.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import track_filter as tf     # noqa: E402

W, H, NF = 1920, 1080, 60
K = np.array([[1400.0, 0, W / 2.0], [0, 1400.0, H / 2.0], [0, 0, 1]])


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def scene(n=160, planar=False, rotation_only=False, seed=3):
    """Project a rigid 3D scene through a moving camera -> {track: [(f,x,y)...]}."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-3, 3, n),
                         np.full(n, 12.0) if planar else rng.uniform(8, 20, n)])
    tracks = {f"T{i:03d}": [] for i in range(n)}
    for f in range(NF):
        u = f / (NF - 1.0)
        R = _rot(0.05 * u, 0.35 * u, 0.02 * u)
        t = np.zeros(3) if rotation_only else np.array([2.5 * u, 0.4 * u, 0.6 * u])
        P = K @ np.hstack([R, (-R @ t).reshape(3, 1)])
        h = np.hstack([X, np.ones((n, 1))]) @ P.T
        xy = h[:, :2] / h[:, 2:3]
        for i in range(n):
            x, y = float(xy[i, 0]), float(xy[i, 1])
            if 0 <= x < W and 0 <= y < H:
                tracks[f"T{i:03d}"].append((f + 1, x, y))
    return {k: v for k, v in tracks.items() if len(v) >= NF // 2}


def med(d, keys=None):
    v = sorted(x for k, x in d.items() if (keys is None or k in keys))
    return v[len(v) // 2] if v else float("nan")


def main() -> int:
    bad = 0

    def check(ok, label, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'}  {label}" + (f"   {detail}" if detail else ""))

    print("epipolar residual self-check\n")

    # 1. exact tracks on a real 3D scene
    tr = scene()
    g = tf.geometric_residuals(tr, W, H)
    m = med(g)
    check(len(g) > 50 and m < 0.05, "perfect rigid tracks read ~0px",
          f"median {m:.4f}px over {len(g)} tracks")

    # 2. one independently-moving point among them
    tr2 = dict(tr)
    victim = sorted(tr2)[0]
    tr2[victim] = [(f, x + 0.9 * (f - 1), y - 0.35 * (f - 1)) for f, x, y in tr2[victim]]
    g2 = tf.geometric_residuals(tr2, W, H)
    others = med(g2, keys=set(g2) - {victim})
    check(g2.get(victim, 0.0) > 20 * max(others, 1e-3),
          "an independently-moving track stands out",
          f"mover {g2.get(victim, float('nan')):.2f}px vs median {others:.4f}px")

    # 3. noise raises it smoothly
    rng = np.random.default_rng(5)
    prev = None
    mono = True
    line = []
    for s in (0.0, 0.25, 0.75, 1.5):
        trn = {k: [(f, x + rng.normal(0, s), y + rng.normal(0, s)) for f, x, y in v]
               for k, v in tr.items()}
        mv = med(tf.geometric_residuals(trn, W, H))
        line.append(f"{s:g}px->{mv:.3f}")
        if prev is not None and not (mv > prev):
            mono = False
        prev = mv
    check(mono, "pixel noise raises the residual monotonically", "  ".join(line))

    # 4. degenerate geometry must say so
    gp = tf.geometric_residuals(scene(planar=True), W, H)
    check(not gp, "a PLANAR scene reports not-measurable",
          f"returned {len(gp)} value(s)")
    gr = tf.geometric_residuals(scene(rotation_only=True), W, H)
    check(not gr, "a PURE ROTATION reports not-measurable",
          f"returned {len(gr)} value(s)")

    print("\n" + ("all checks passed" if not bad else f"{bad} check(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
