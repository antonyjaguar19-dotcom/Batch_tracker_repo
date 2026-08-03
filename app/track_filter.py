# -*- coding: utf-8 -*-
"""Backend-agnostic track selection: score -> quality gate -> even spread -> cap.

This used to live inside tracker_core, which meant it only ever ran on the TAPNext path --
so SynthEyes (the DEFAULT backend) exported everything it tracked, up to 800 trackers on the
'Normal' preset, and the cleanup landed on the artist. The logic is pure Python (no torch, no
cv2) precisely so the SynthEyes path can share it without dragging the GPU stack in.

tracker_core delegates to these functions rather than keeping a second copy, so the two
backends cannot drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

Track = List[Tuple[int, float, float]]


@dataclass
class FilterConfig:
    """Everything the selection needs. Mirrors the same-named RunnerConfig fields."""
    # quality score weights (self-referential only -- must never judge vs global motion,
    # or a fast foreground point gets punished for not moving like the background)
    w_coverage: float = 0.5
    w_smoothness: float = 0.3
    w_stability: float = 0.2
    # final quality gate
    min_track_frames: int = 24
    min_track_span_frac: float = 0.0
    min_track_score: float = 0.35
    quality_gate_floor: int = 8
    # even spread + cap
    spread_min_dist_px: int = 40
    spread_scale_with_res: bool = True
    spread_ref_width: int = 1920
    spread_ref_frames: int = 5
    spread_max_starts_per_window: int = 0
    spread_start_window: int = 8
    max_output_tracks: int = 120

    @classmethod
    def from_state(cls, state) -> "FilterConfig":
        """Build from an AppState, so the UI drives both backends identically."""
        g = lambda k, d: getattr(state, k, d)
        return cls(
            min_track_frames=int(g("min_track_frames", 24) or 0),
            min_track_score=float(g("min_track_score", 0.35) or 0.0),
            spread_min_dist_px=int(g("track_spacing_px", 60) or 40),
            spread_ref_frames=int(g("spread_ref_frames", 5) or 1),
            spread_scale_with_res=bool(g("spread_scale_with_res", True)),
            spread_max_starts_per_window=int(g("spread_max_starts_per_window", 0) or 0),
            max_output_tracks=int(g("track_max_output", 120) or 0),
        )


def score_track(pts: Track, T: int, diag: float, cfg) -> float:
    """Per-track quality in ~[0,1] from the track's OWN trajectory only.

    PARALLAX-SAFE: no term compares the track against global/median motion, so a fast
    foreground (high-parallax) point is not penalised for moving unlike the background.
    Mistracks are caught by self-consistency (jitter/jump on the track's own path).
    Terms: coverage (long, gapless), smoothness (low own jitter), stability (no jumps).
    """
    n = len(pts)
    if n < 2:
        return 0.0
    frames = [p[0] for p in pts]
    span = max(1, frames[-1] - frames[0] + 1)
    cov = float(n) / float(max(1, T))       # fraction of shot the track is alive
    fill = float(n) / float(span)           # 1.0 = no internal gaps
    cov_score = 0.5 * min(1.0, cov) + 0.5 * fill

    vels = []
    for i in range(n - 1):
        dt = max(1, pts[i + 1][0] - pts[i][0])
        d = math.hypot(pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2])
        vels.append(d / dt)
    max_jump = max(vels) if vels else 0.0
    if len(vels) > 1:
        jitter = sum(abs(vels[k + 1] - vels[k]) for k in range(len(vels) - 1)) / (len(vels) - 1)
    else:
        jitter = 0.0
    d = float(diag) if diag > 0 else 1.0
    smooth_score = 1.0 / (1.0 + jitter / (0.005 * d))   # scale: 0.5% of frame diagonal
    stab_score = 1.0 / (1.0 + max_jump / (0.020 * d))   # scale: 2% of frame diagonal

    w_c = float(getattr(cfg, "w_coverage", 0.5))
    w_s = float(getattr(cfg, "w_smoothness", 0.3))
    w_j = float(getattr(cfg, "w_stability", 0.2))
    wsum = (w_c + w_s + w_j) or 1.0
    return (w_c * cov_score + w_s * smooth_score + w_j * stab_score) / wsum


def quality_gate(candidates: List[dict], T: int, cfg, log: Optional[Callable] = None) -> List[dict]:
    """Drop tracks that are not worth an artist's time, before spread selection.

    Scoring alone only ORDERED the candidates -- a short, scrappy track still shipped if
    there was room, which is how an export reaches four figures and the cleanup lands on
    the artist. These are absolute floors: too short to constrain a solve, or too poor to
    trust.

    Two safeguards, because an over-eager gate is worse than a permissive one:
      * the length requirement scales down on short shots (a 30-frame plate cannot
        produce 24-frame tracks in quantity);
      * if the gate would leave fewer than `quality_gate_floor` tracks it is relaxed and
        the best survivors are kept instead, so a hard shot still exports something.
    """
    def _log(m):
        if log:
            log(m)

    if not candidates:
        return candidates
    need_f = int(getattr(cfg, "min_track_frames", 0) or 0)
    frac = float(getattr(cfg, "min_track_span_frac", 0.0) or 0.0)
    if frac > 0.0:
        need_f = max(need_f, int(round(frac * max(1, T))))
    # Never demand more than a quarter of the shot: on a short plate that would gate
    # everything away for a reason the artist cannot act on.
    if need_f > 0:
        need_f = max(4, min(need_f, int(0.25 * max(1, T)) or 4))
    need_s = float(getattr(cfg, "min_track_score", 0.0) or 0.0)
    if need_f <= 0 and need_s <= 0.0:
        return candidates

    kept, short, weak = [], 0, 0
    for c in candidates:
        if need_f > 0 and len(c["pts"]) < need_f:
            short += 1
            continue
        if need_s > 0.0 and float(c["score"]) < need_s:
            weak += 1
            continue
        kept.append(c)

    floor = max(1, int(getattr(cfg, "quality_gate_floor", 8) or 1))
    if len(kept) < floor and len(candidates) > len(kept):
        kept = sorted(candidates, key=lambda c: c["score"], reverse=True)[:floor]
        _log(f"  quality gate: only {len(kept)} track(s) cleared it -> relaxed, "
             f"keeping the best {len(kept)} instead (short/poor shot)")
        return kept
    if short or weak:
        _log(f"  quality gate: dropped {short} too-short (<{need_f}f) and "
             f"{weak} low-quality (<{need_s:.2f}) of {len(candidates)} -> "
             f"{len(kept)} worth exporting")
    return kept


def spacing_px(cfg, out_width: int = 0) -> int:
    """Effective min spacing in PLATE pixels.

    The slider is quoted against spread_ref_width (1920). Candidate coords are in full
    plate space, so on a 4K plate a raw 40px gap is only ~1% of the width -- dense enough
    to still read as clumped -- and worse on 6K/8K. Scaling by plate width keeps one
    slider value looking the same at any resolution.
    """
    d = max(1, int(getattr(cfg, "spread_min_dist_px", 40)))
    if not bool(getattr(cfg, "spread_scale_with_res", True)):
        return d
    ref = max(1, int(getattr(cfg, "spread_ref_width", 1920) or 1920))
    w = int(out_width or 0)
    if w <= 0:
        return d
    return max(1, int(round(d * (w / float(ref)))))


def select_spread(candidates: List[dict], cfg, out_width: int = 0,
                  log: Optional[Callable] = None) -> Dict[str, Track]:
    """Greedy min-spacing selection over score-sorted candidates.

    Walk best-first; accept a track only if it stays >= spread_min_dist_px from every
    already-accepted track AT EVERY SAMPLED REFERENCE FRAME where both are visible.

    Spacing used to be measured between the two tracks' LIFETIME MEAN positions, which is
    not what the artist sees: with a moving camera two tracks can sit a few px apart for
    most of the shot while their means are 40px+ apart, so both were accepted and they
    clumped on screen. Sampling real positions fixes that; the dial keeps its meaning.

    Because spacing (default 40px) >> any pass duplicate offset, near-identical duplicates
    are still rejected implicitly -- this doubles as the pass dedup. One coarse cell grid
    PER reference frame keeps each spacing test O(neighbours).
    """
    def _log(m):
        if log:
            log(m)

    out: Dict[str, Track] = {}
    if not candidates:
        return out
    d = spacing_px(cfg, out_width)
    d2 = float(d * d)
    cap = int(getattr(cfg, "max_output_tracks", 0) or 0)
    if d != int(getattr(cfg, "spread_min_dist_px", 40)):
        _log(f"  spread: {getattr(cfg, 'spread_min_dist_px', 40)}px "
             f"@{getattr(cfg, 'spread_ref_width', 1920)} -> {d}px at this plate width ({out_width})")

    # Reference frames: evenly spaced over the frame range the candidates actually span.
    k = max(1, int(getattr(cfg, "spread_ref_frames", 5) or 1))
    f_lo = min(c["pts"][0][0] for c in candidates)
    f_hi = max(c["pts"][-1][0] for c in candidates)
    if k == 1 or f_hi <= f_lo:
        ref_frames = [f_lo]
    else:
        step = (f_hi - f_lo) / float(k - 1)
        ref_frames = sorted({int(round(f_lo + i * step)) for i in range(k)})

    # Per-candidate position at each reference frame. A track need not be visible at one:
    # it simply doesn't compete there. Nearest-sample within half a step keeps a track
    # that is alive around the reference frame but has a gap exactly on it.
    tol = max(1, int((f_hi - f_lo) / (2.0 * max(1, len(ref_frames) - 1)))) if f_hi > f_lo else 1

    def positions_at_refs(pts):
        by_frame = {f: (x, y) for f, x, y in pts}
        got = {}
        for rf in ref_frames:
            if rf in by_frame:
                got[rf] = by_frame[rf]
                continue
            near = min(by_frame, key=lambda f: abs(f - rf))
            if abs(near - rf) <= tol:
                got[rf] = by_frame[near]
        return got

    grids: Dict[int, Dict[Tuple[int, int], List[Tuple[float, float]]]] = {rf: {} for rf in ref_frames}
    mean_grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    # Temporal spacing: how many accepted tracks already START in each short window.
    start_cap = int(getattr(cfg, "spread_max_starts_per_window", 0) or 0)
    start_win = max(1, int(getattr(cfg, "spread_start_window", 8) or 8))
    starts: Dict[int, int] = {}
    n_start_pruned = 0

    def clashes(grid, px, py):
        cx, cy = int(px // d), int(py // d)
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for (ax, ay) in grid.get((gx, gy), ()):
                    if (ax - px) ** 2 + (ay - py) ** 2 < d2:
                        return True
        return False

    for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
        # Best-first, so a full start-window keeps the strongest tracks that began there
        # and turns the rest away -- thinning a burst rather than banning one outright.
        sbin = int(c["pts"][0][0]) // start_win if start_cap > 0 else None
        if sbin is not None and starts.get(sbin, 0) >= start_cap:
            n_start_pruned += 1
            continue
        at = positions_at_refs(c["pts"])
        mx, my = c["mean"]
        if not at:
            # Short track falling between every sample: fall back to the mean test rather
            # than waving it through unchecked.
            if clashes(mean_grid, mx, my):
                continue
        elif any(clashes(grids[rf], px, py) for rf, (px, py) in at.items()):
            continue

        out[c["id"]] = c["pts"]
        for rf, (px, py) in at.items():
            grids[rf].setdefault((int(px // d), int(py // d)), []).append((px, py))
        mean_grid.setdefault((int(mx // d), int(my // d)), []).append((mx, my))
        if sbin is not None:
            starts[sbin] = starts.get(sbin, 0) + 1
        if cap > 0 and len(out) >= cap:
            break
    if n_start_pruned:
        _log(f"  spread: thinned {n_start_pruned} track(s) that started in an "
             f"already-crowded {start_win}-frame window")
    return out


def filter_tracks(tracks: Dict[str, Track], T: int, width: int, height: int, cfg,
                  log: Optional[Callable] = None) -> Dict[str, Track]:
    """Score -> gate -> spread -> cap, on an already-exported set of tracks.

    Entry point for backends that produce finished tracks rather than candidate arrays
    (SynthEyes), so both paths end up with the same solve-ready selection.
    """
    if not tracks:
        return tracks
    diag = math.hypot(float(width or 0), float(height or 0)) or 1000.0
    candidates = []
    for tid, pts in tracks.items():
        pts = sorted(pts, key=lambda p: p[0])
        if len(pts) < 2:
            continue
        n = len(pts)
        candidates.append({
            "id": tid,
            "pts": pts,
            "mean": (sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n),
            "score": score_track(pts, T, diag, cfg),
        })
    if not candidates:
        return {}
    candidates = quality_gate(candidates, T, cfg, log=log)
    return select_spread(candidates, cfg, out_width=int(width or 0), log=log)
