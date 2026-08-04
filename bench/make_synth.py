# -*- coding: utf-8 -*-
"""Build a synthetic shot with EXACT ground truth for every pixel.

Why this exists: `refs/` holds one hand-tracked corner, and `eval_refs` says so every time
it runs -- "per-class conclusions need one per class". Worse, a hand-tracked reference can
only ever cover the handful of features an artist had time to track, and the artist's own
track carries its own error. Neither problem is fixable by tracking more references.

A real plate frame warped by a KNOWN homography has neither problem. The scene is a rigid
plane, so the true position of any point on any frame is exact arithmetic, for every seed
the tracker chooses to make -- corner, blob, edge, dense, all of them, with no upper limit
and no human error term. The texture, grain and lens character are the real plate's.

What it deliberately does NOT model, and what must therefore never be concluded from it:
  * parallax  -- one plane, so nothing moves relative to anything else.
  * occlusion -- nothing passes in front of anything.
  * defocus / motion blur changes -- the warp resamples, it does not re-render optics.
So this measures LOCALISATION: sub-pixel accuracy, drift, and the wobble that
`refs/gt4k/baseline.json` reads as "position pulled toward where the point WAS". It does
not measure occlusion continuity or parallax handling; `refs/` still owns those.

The motion profile alternates fast and slow phases on purpose. The locked baseline's
fast=0.94 / slow=1.22 pair is only visible when the shot contains both.

    python bench/make_synth.py --out bench/synth/lab01
    python bench/make_synth.py --out bench/synth/lab01 --frames 100 --seed-frame 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default source: the 4K plate frames already on this box. Any single image works.
DEFAULT_SRC = "D:/Jefrin/liv1/shows/Test/AP_01/in/plates/v001/AP_plate_v01/3840x2160_jpg"


def _load_source(path: str, seed_frame: int) -> np.ndarray:
    """One real frame, BGR uint8. `path` may be an image or a directory of them."""
    if os.path.isdir(path):
        files = sorted(
            f for f in glob.glob(os.path.join(path, "*"))
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr")
        )
        if not files:
            raise SystemExit(f"no images in {path}")
        idx = min(max(0, seed_frame - 1), len(files) - 1)
        path = files[idx]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {path}")
    return img


def motion(t: int, n: int, cx: float, cy: float) -> np.ndarray:
    """Camera-like homography for frame t, mapping SOURCE pixels -> OUTPUT pixels.

    Four phases so one shot exercises both ends of the fast/slow metric:
      0-25%   slow drift      (~0.3 px/frame)  -- where over-shoot shows up as wobble
      25-55%  fast pan        (~7 px/frame)    -- where under-shoot shows up as lag
      55-75%  slow again      (~0.25 px/frame)
      75-100% medium + rotation and a slight zoom
    Handheld jitter rides on top at sub-pixel amplitude with irrational periods, so no two
    frames share a sub-pixel phase and nothing can accidentally land on whole pixels.
    """
    f = t / max(1, n - 1)

    # Piecewise speed integrated into a position (kept continuous across phase joins).
    def travel(u: float) -> float:
        a, b, c = 0.25, 0.55, 0.75
        s_slow, s_fast, s_med = 0.3, 7.0, 2.2
        d = 0.0
        d += s_slow * n * min(u, a)
        if u > a:
            d += s_fast * n * (min(u, b) - a)
        if u > b:
            d += s_slow * 0.8 * n * (min(u, c) - b)
        if u > c:
            d += s_med * n * (u - c)
        return d

    tx = travel(f)
    ty = 26.0 * math.sin(2.0 * math.pi * f * 1.3)
    # Sub-pixel handheld, deliberately non-commensurate periods.
    tx += 0.62 * math.sin(t / 3.7) + 0.31 * math.sin(t / 9.13)
    ty += 0.58 * math.cos(t / 4.3) + 0.27 * math.sin(t / 11.7)

    theta = math.radians(0.30) * math.sin(2.0 * math.pi * f * 1.1)
    scale = 1.0 + 0.012 * math.sin(2.0 * math.pi * f * 0.9)

    ct, st = math.cos(theta) * scale, math.sin(theta) * scale
    # Rotate+zoom about the output centre, then translate.
    rot = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], np.float64)
    to_c = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], np.float64)
    fro_c = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]], np.float64)
    return fro_c @ rot @ to_c


def _degrade(img: np.ndarray, t: int, step_px: float, noise: float, blur: float,
             exposure: float, rng: np.random.Generator, angle: float) -> np.ndarray:
    """Add the things that actually cause sub-pixel tracking error.

    A clean warp is tracked to ~0.05px by the current pipeline, which makes it useless for
    ranking a change -- everything scores perfect. Real plates are not clean, and each of
    these degradations attacks a different part of the tracker:

      noise     independent per frame, so it moves the NCC peak a little every frame. This
                is the direct cause of wobble on a static feature.
      blur      scaled by how far the frame moved, so fast phases are smeared and slow ones
                are sharp -- exactly the asymmetry the fast/slow ratio pair measures.
      exposure  slow gain/lift drift, which is what makes a template go stale and forces the
                re-reference decision (refine_ncc_reref) to be made at all.
    """
    out = img.astype(np.float32)
    if blur > 0.0 and step_px > 0.35:
        k = int(round(min(21.0, blur * step_px)))
        if k >= 3:
            k = k + 1 if k % 2 == 0 else k
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1.0 / k
            rot = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5),
                                          -math.degrees(angle), 1.0)
            ker = cv2.warpAffine(ker, rot, (k, k))
            s = ker.sum()
            if s > 1e-6:
                out = cv2.filter2D(out, -1, ker / s)
    if exposure > 0.0:
        gain = 1.0 + exposure * math.sin(2.0 * math.pi * t / 47.0)
        lift = 3.0 * exposure * math.cos(2.0 * math.pi * t / 61.0)
        out = out * gain + lift
    if noise > 0.0:
        out += rng.normal(0.0, noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def build(src_path: str, out_dir: str, frames: int, width: int, height: int,
          seed_frame: int, noise: float = 0.0, blur: float = 0.0,
          exposure: float = 0.0) -> str:
    src = _load_source(src_path, seed_frame)
    hs, ws = src.shape[:2]
    if ws < width + 200 or hs < height + 200:
        raise SystemExit(
            f"source {ws}x{hs} too small for a {width}x{height} output with motion margin")

    # Centre the output window in the source so the move has room on every side; every
    # output pixel then comes from real texture and no frame carries a black border.
    ox, oy = (ws - width) / 2.0, (hs - height) / 2.0
    base = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]], np.float64)
    cx, cy = width / 2.0, height / 2.0

    seq_dir = os.path.join(out_dir, "plate")
    os.makedirs(seq_dir, exist_ok=True)
    # Fixed seed: a bench whose noise changes between runs cannot attribute a delta to a
    # code change, which is the only thing it exists to do.
    rng = np.random.default_rng(20260804)
    mats = []
    prev_c = None
    for t in range(frames):
        m = motion(t, frames, cx, cy) @ base
        mats.append(m.tolist())
        warped = cv2.warpPerspective(src, m, (width, height), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_REFLECT)
        # How far the frame centre moved since the last frame -> blur length and direction.
        c_now = (m @ np.array([ox + cx, oy + cy, 1.0]))
        c_now = c_now[:2] / c_now[2]
        step, angle = 0.0, 0.0
        if prev_c is not None:
            d = c_now - prev_c
            step = float(np.hypot(d[0], d[1]))
            angle = float(math.atan2(d[1], d[0]))
        prev_c = c_now
        warped = _degrade(warped, t, step, noise, blur, exposure, rng, angle)
        # PNG, not JPEG: a codec would add its own sub-pixel error to what is being measured.
        cv2.imwrite(os.path.join(seq_dir, f"{t + 1:06d}.png"), warped)
        if (t + 1) % 20 == 0 or t == frames - 1:
            print(f"  rendered {t + 1}/{frames}")

    gt = {
        "source": src_path,
        "source_frame": seed_frame,
        "width": width,
        "height": height,
        "frames": frames,
        "degrade": {"noise": noise, "blur": blur, "exposure": exposure},
        "note": "H[t] maps SOURCE pixel coords -> OUTPUT pixel coords for frame t (0-based). "
                "A point seen at output q on frame s sits at H[t] @ inv(H[s]) @ q on frame t. "
                "Coords are IMAGE space (y down); the 3DE export flips y, the scorer un-flips.",
        "H": mats,
    }
    gt_path = os.path.join(out_dir, "gt.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f)
    print(f"wrote {frames} frames -> {seq_dir}")
    print(f"wrote ground truth -> {gt_path}")
    return gt_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="source image or folder of frames")
    ap.add_argument("--out", required=True, help="output shot folder")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--seed-frame", type=int, default=1,
                    help="which frame of a source folder to warp (1-based)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="gaussian sensor noise sigma in 0-255 levels (2-4 is plate-like)")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="motion-blur kernel px per px of inter-frame motion (~0.6 = 180deg shutter)")
    ap.add_argument("--exposure", type=float, default=0.0,
                    help="slow gain/lift drift, 0.02 = +-2%%")
    ap.add_argument("--hard", action="store_true",
                    help="plate-like preset: --noise 3 --blur 0.6 --exposure 0.02")
    a = ap.parse_args()
    if a.hard:
        a.noise, a.blur, a.exposure = 3.0, 0.6, 0.02
    os.makedirs(a.out, exist_ok=True)
    build(a.src, a.out, a.frames, a.width, a.height, a.seed_frame,
          noise=a.noise, blur=a.blur, exposure=a.exposure)


if __name__ == "__main__":
    main()
