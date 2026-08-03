# -*- coding: utf-8 -*-
"""Measure a shot and derive tracking parameters from it.

This is a BATCH tool, so per-shot tuning by hand is not an option -- yet feature_quality,
refine_patch_px, refine_ncc_lost/hold and min_feature_dist were fixed constants for every
shot, tuned on nothing in particular. A handheld plate with a defocused background and a
locked-off macro shot got identical settings, and at most one of those can be right.

Everything here is measured from the plate itself, because that works whether or not Analyze
has been run and because it is per-REGION rather than per-shot. Qwen's existing quality_flags
("out of focus", "low texture", "heavy grain", "motion blur") are folded in as a cross-check
where they exist -- they are a language model's opinion, useful for agreement, not authority.

No torch, no GPU: this must be importable from any path that needs it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import cv2  # type: ignore


# Defaults = today's constants, so an un-tuned run is byte-identical to the old behaviour.
_DEFAULTS = {
    "feature_quality": 0.02,
    "min_feature_dist": 12,
    "refine_patch_px": 31,
    "refine_ncc_lost": 0.60,
    "refine_ncc_hold": 0.45,
    "min_track_certainty": 0.0,
}

# Every tuned value is clamped, so a strange measurement can shift behaviour but never
# produce a nonsensical configuration.
_CLAMPS = {
    "feature_quality": (0.005, 0.20),
    "min_feature_dist": (6, 48),
    "refine_patch_px": (21, 61),
    "refine_ncc_lost": (0.35, 0.75),
    "refine_ncc_hold": (0.25, 0.65),
    "min_track_certainty": (0.0, 0.60),
}


@dataclass
class ShotProfile:
    """What the plate actually looks like. All values are scale-free where possible."""
    sharpness: float = 0.0        # median tile Laplacian variance / image variance (raw)
    sharp_frac: float = 1.0       # fraction of tiles that are acceptably sharp
    noise: float = 0.0            # per-pixel temporal noise, 0-255 scale
    texture: float = 0.0          # median absolute corner response
    motion: float = 0.0           # median inter-frame displacement, px
    flags: List[str] = field(default_factory=list)   # Qwen quality_flags, lowercased
    notes: List[str] = field(default_factory=list)   # agreement/disagreement remarks

    def describe(self) -> str:
        return (f"sharp={self.sharpness:.3f} sharp_area={self.sharp_frac * 100:.0f}% "
                f"grain={self.noise:.1f} texture={self.texture:.1f} motion={self.motion:.1f}px")


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def profile_shot(frames: Sequence[np.ndarray], flags: Optional[Sequence[str]] = None,
                 tile: int = 96) -> ShotProfile:
    """Measure a handful of frames. Degenerate input returns a neutral profile, never raises."""
    p = ShotProfile(flags=[str(f).strip().lower() for f in (flags or [])])
    try:
        if frames is None or len(frames) == 0:
            return p
        grays = [_gray(np.asarray(f)) for f in frames[:8]]
        grays = [g for g in grays if g is not None and g.ndim == 2 and min(g.shape) >= 16]
        if not grays:
            return p
        g0 = grays[0].astype(np.float32)
        h, w = g0.shape

        # --- focus: variance of the Laplacian per tile. High-frequency energy is exactly
        # what a defocused region lacks, and per-tile keeps it spatial: a sharp subject on a
        # soft background reads as "sharp enough, but only over part of the frame".
        lap = cv2.Laplacian(g0, cv2.CV_32F, ksize=3)
        ts = max(32, int(tile))
        vals = []
        for y in range(0, h - ts + 1, ts):
            for x in range(0, w - ts + 1, ts):
                vals.append(float(lap[y:y + ts, x:x + ts].var()))
        if vals:
            v = np.array(vals, dtype=np.float64)
            # Absolute focus, normalised by the frame's own contrast. Laplacian energy alone
            # tracks exposure as much as focus; dividing by the image variance cancels that
            # and leaves the high-frequency content, which is what blur destroys.
            #
            # It must NOT be a within-frame relative measure (e.g. median/P90 of the tiles):
            # that answers "is sharpness UNIFORM", so a uniformly blurred plate scores as
            # sharp -- which is exactly the shot we are trying to detect.
            # Left as a RAW ratio rather than squashed into 0-1: any normaliser would be a
            # constant guessed from nothing, and a guessed constant that saturates destroys
            # exactly the discrimination this measure exists for. Thresholds live in tune().
            img_var = float(g0.var()) or 1.0
            p.sharpness = float(np.median(v)) / img_var
            # Spatially: how much of the frame is in focus, judged against its own best
            # tiles. This one IS relative on purpose -- a sharp subject on a soft background
            # should read as "sharp, but only over part of the frame".
            hi = float(np.percentile(v, 90)) or 1.0
            p.sharp_frac = float(np.mean(v >= 0.25 * hi))

        # --- grain: temporal difference is dominated by noise where nothing moves, so the
        # low percentile of |frame difference| is a decent noise floor estimate.
        if len(grays) >= 2:
            d = np.abs(grays[1].astype(np.float32) - g0)
            p.noise = float(np.percentile(d, 25))

        # --- texture: absolute corner response, which sets whether a RELATIVE feature
        # threshold (goodFeaturesToTrack's qualityLevel) is meaningful on this plate.
        eig = cv2.cornerMinEigenVal(g0, blockSize=5, ksize=3)
        p.texture = float(np.percentile(eig, 99))

        # --- motion: coarse global displacement between two sampled frames.
        if len(grays) >= 2:
            a = cv2.resize(g0, (min(320, w), min(180, h)), interpolation=cv2.INTER_AREA)
            b = cv2.resize(grays[min(2, len(grays) - 1)].astype(np.float32),
                           (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            try:
                flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 15, 2, 5, 1.1, 0)
                mag = np.hypot(flow[..., 0], flow[..., 1])
                # scale back to full res, and per-frame rather than per-sample-gap
                p.motion = float(np.median(mag)) * (w / float(a.shape[1])) / max(1, min(2, len(grays) - 1))
            except cv2.error:
                p.motion = 0.0
    except Exception:
        return p
    return p


def tune(profile: ShotProfile, overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Map a profile to tracking parameters. `overrides` win, so a moved slider is honoured.

    Each rule below states WHY, because an auto-tuner nobody can audit is worse than a
    constant nobody can justify.
    """
    out = dict(_DEFAULTS)
    p = profile
    f = set(p.flags or [])

    # Thresholds sit on the raw measures. Texture (absolute corner response) is the most
    # reliable focus signal available -- blur destroys corner response directly, and it
    # separated a crisp plate from a blurred copy of the same plate by ~26x in testing.
    soft = p.texture < 400.0 or p.sharpness < 0.02 or "out of focus" in f
    low_tex = p.texture < 6.0 or "low texture" in f
    grainy = p.noise > 2.5 or "heavy grain" in f
    fast = p.motion > 6.0 or "motion blur" in f

    # Grain lowers every NCC score, so a fixed 0.60 rejects perfectly good matches on a
    # grainy plate and keeps junk on a clean one.
    if grainy:
        out["refine_ncc_lost"] = 0.48
        out["refine_ncc_hold"] = 0.34
    elif p.noise < 0.8:
        out["refine_ncc_lost"] = 0.68
        out["refine_ncc_hold"] = 0.52

    # A soft plate has little high-frequency detail, so a bigger box averages over more of
    # what little there is; a crisp plate is better served by a smaller, more local box.
    if soft:
        out["refine_patch_px"] = 41
    elif p.texture > 1500.0:
        out["refine_patch_px"] = 25

    # goodFeaturesToTrack's qualityLevel is RELATIVE to the frame's best corner. On a
    # low-texture plate 2% of the best is noise, so the bar has to come up.
    if low_tex:
        out["feature_quality"] = 0.06
        out["min_feature_dist"] = 18
    elif p.texture > 40.0:
        out["feature_quality"] = 0.015

    # Fast/handheld motion spreads features out; crowding them wastes the budget on points
    # that will chase each other.
    if fast:
        out["min_feature_dist"] = max(int(out["min_feature_dist"]), 16)

    # The certainty floor is the one that removes defocused tracks. Demand more of them
    # exactly when the plate has a lot of soft area to seed into.
    if soft or p.sharp_frac < 0.5:
        out["min_track_certainty"] = 0.25
    else:
        out["min_track_certainty"] = 0.12

    # Cross-check the measurement against Qwen, and SAY when they disagree rather than
    # letting one silently win.
    if f:
        if "out of focus" in f and p.texture >= 1500.0:
            p.notes.append("Qwen says out of focus but the plate measures sharp")
        if "heavy grain" in f and p.noise < 1.0:
            p.notes.append("Qwen says heavy grain but the plate measures clean")
        if soft and "out of focus" not in f:
            p.notes.append("plate measures soft though Qwen did not flag it")

    for k, (lo, hi) in _CLAMPS.items():
        v = out.get(k, _DEFAULTS[k])
        out[k] = type(_DEFAULTS[k])(max(lo, min(hi, v)))

    for k, v in (overrides or {}).items():        # a moved slider always wins
        if v is not None:
            out[k] = v
    return out
