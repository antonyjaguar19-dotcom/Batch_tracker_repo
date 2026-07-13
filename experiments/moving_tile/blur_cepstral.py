# -*- coding: utf-8 -*-
"""Cepstral motion-blur kernel estimation (ISOLATED experiment).

A linear motion blur of length L at angle theta convolves the image with a line PSF.
In the frequency domain that PSF is a sinc whose zeros form parallel dark stripes
perpendicular to the blur direction; equivalently, the CEPSTRUM (inverse FFT of the
log-magnitude spectrum) shows two symmetric negative peaks at distance ~L from the
origin, along the blur direction. So we can recover (length, angle) from the image
ITSELF -- no velocity guess, which is what failed on real footage.

estimate_blur_kernel(region_gray) -> (length_px, angle_rad, confidence).
Angle is mod pi (the line PSF is symmetric, so direction sign is irrelevant for blurring).
"""
from __future__ import annotations
import math
import numpy as np


def _hann2d(h, w):
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def estimate_blur_kernel(region, r_min=3, r_max_frac=0.48, conf_min=1.8):
    """Estimate linear-motion-blur (length, angle_rad, confidence) from a gray patch.

    length in px, angle in radians (mod pi). confidence = peak / annulus-mean; below
    conf_min or length < r_min -> return (0, 0, conf) meaning 'no clear blur'.
    """
    g = region.astype(np.float32)
    h, w = g.shape
    if min(h, w) < 2 * r_min + 4:
        return 0.0, 0.0, 0.0
    g = g - float(g.mean())
    win = _hann2d(h, w)
    F = np.fft.fft2(g * win)
    logmag = np.log(np.abs(F) ** 2 + 1e-6)
    ceps = np.abs(np.fft.ifft2(logmag))
    ceps = np.fft.fftshift(ceps)
    cy, cx = h // 2, w // 2

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r_max_frac * min(h, w)

    # Radially normalize: divide each pixel by the mean of its radius ring. The DC hump
    # near the centre is radially symmetric so it cancels, leaving the DIRECTIONAL blur
    # peak (localized at one angle, distance ~L) standing out above its ring.
    rint = np.round(r).astype(int)
    ring_sum = np.bincount(rint.ravel(), ceps.ravel())
    ring_cnt = np.bincount(rint.ravel())
    ring_mean = ring_sum / np.maximum(ring_cnt, 1)
    norm = ceps / (ring_mean[rint] + 1e-9)

    search = (r >= r_min) & (r <= r_max)
    if not np.any(search):
        return 0.0, 0.0, 0.0
    vals = np.where(search, norm, -np.inf)
    idx = int(np.argmax(vals))
    py, px = divmod(idx, w)
    conf = float(norm[py, px])       # how many x above its ring average

    dy, dx = (py - cy), (px - cx)
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) % math.pi
    if length < r_min or conf < conf_min:
        return 0.0, angle, conf
    return float(length), float(angle), float(conf)


# ---- self-test: recover a known kernel --------------------------------------------
if __name__ == "__main__":
    import cv2
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (160, 160)).astype(np.float32)
    base = cv2.GaussianBlur(base, (3, 3), 0)  # mild texture
    for (L, ang) in [(9, 0), (15, 30), (21, 90), (13, 135)]:
        k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((L / 2, L / 2), ang, 1.0)
        k = cv2.warpAffine(k, M, (L, L)); k /= k.sum()
        blurred = cv2.filter2D(base, -1, k)
        est_L, est_a, conf = estimate_blur_kernel(blurred)
        ea = math.degrees(est_a)
        # blur direction is perpendicular-invariant mod 180; compare to injected angle mod 180
        want = ang % 180
        derr = min(abs(ea - want), 180 - abs(ea - want))
        print(f"true L={L:2d} ang={ang:3d}  ->  est L={est_L:5.1f} ang={ea:5.1f} "
              f"conf={conf:.2f}  (Lerr={abs(est_L-L):.1f} angerr={derr:.1f})")
