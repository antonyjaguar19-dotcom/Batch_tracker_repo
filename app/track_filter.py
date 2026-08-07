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
import os
import statistics as st
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

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
    # Localisation certainty floor. The three score terms above are all self-referential
    # MOTION statistics, and a defocused blob tracks smoothly precisely because it has no
    # detail -- long, low-jitter, no jumps -- so it scored WELL. Certainty (how sharply the
    # correlation peak falls away) is the one axis a soft track cannot fake. 0 = off.
    min_track_certainty: float = 0.0
    # Relative certainty bar: keep tracks scoring at least this fraction of the shot's own
    # best decile. Self-calibrating, which an absolute number cannot be -- certainty depends
    # on the plate's contrast, grain and lens. 0 = off.
    certainty_rel: float = 0.80
    # Most of the track set a certainty bar may remove. If it wants more than this, the
    # measure is more likely wrong than the footage is, so it is capped and reported.
    certainty_max_cut: float = 0.6
    # even spread + cap
    spread_min_dist_px: int = 40
    spread_scale_with_res: bool = True
    spread_ref_width: int = 1920
    spread_ref_frames: int = 5
    spread_max_starts_per_window: int = 0
    spread_start_window: int = 8
    # The quality bar decides how many are exported -- if 400 clear it, export 400. This is
    # only a safety ceiling against a pathological shot, not a target.
    max_output_tracks: int = 600
    # ...but never hand back a handful with no explanation. Below this, fill with the best
    # remaining and mark them weak, so a thin shot is still workable and visibly so.
    min_export_tracks: int = 40
    # Holes a track may carry before it is cut into continuous runs instead. One or two long
    # gaps is an occluded track worth keeping whole; a dozen short ones is a marginal track
    # that kept losing lock, and it blinks on and off in the 3DE viewport. -1 = off.
    max_track_gaps: int = 2
    min_occlusion_run: int = 3      # a hole shorter than this was never a real occlusion
    refine_min_len: int = 8         # a run shorter than this is not worth exporting

    @classmethod
    def from_state(cls, state) -> "FilterConfig":
        """Build from an AppState, so the UI drives both backends identically."""
        g = lambda k, d: getattr(state, k, d)
        return cls(
            min_track_frames=int(g("min_track_frames", 24) or 0),
            min_track_score=float(g("min_track_score", 0.35) or 0.0),
            spread_min_dist_px=int(g("track_spacing_px", 60) or 40),
            max_track_gaps=int(g("max_track_gaps", 2)),
            spread_ref_frames=int(g("spread_ref_frames", 5) or 1),
            spread_scale_with_res=bool(g("spread_scale_with_res", True)),
            spread_max_starts_per_window=int(g("spread_max_starts_per_window", 0) or 0),
            max_output_tracks=int(g("track_max_output", 120) or 0),
            # These three were missing, and their absence was invisible because every one of
            # them defaults to "off": on the SynthEyes backend -- the default one -- the
            # certainty gate never fired and a thin export was never backed up to the floor,
            # whatever the artist set in the UI.
            min_export_tracks=int(g("min_export_tracks", 40) or 0),
            min_track_certainty=float(g("min_track_certainty", 0.0) or 0.0),
            certainty_rel=float(g("certainty_rel", 0.80) or 0.0),
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

    need_c = float(getattr(cfg, "min_track_certainty", 0.0) or 0.0)
    kept, short, weak, soft = [], 0, 0, 0
    for c in candidates:
        if need_f > 0 and len(c["pts"]) < need_f:
            short += 1
            continue
        if need_s > 0.0 and float(c["score"]) < need_s:
            weak += 1
            continue
        # Only judge certainty when it was actually measured (the refine stage supplies it);
        # a track with no measurement is not penalised for it.
        if need_c > 0.0 and c.get("certainty") is not None and float(c["certainty"]) < need_c:
            soft += 1
            continue
        kept.append(c)

    floor = max(1, int(getattr(cfg, "quality_gate_floor", 8) or 1))
    if len(kept) < floor and len(candidates) > len(kept):
        kept = sorted(candidates, key=lambda c: c["score"], reverse=True)[:floor]
        _log(f"  quality gate: only {len(kept)} track(s) cleared it -> relaxed, "
             f"keeping the best {len(kept)} instead (short/poor shot)")
        return kept
    if short or weak or soft:
        _log(f"  quality gate: dropped {short} too-short (<{need_f}f), "
             f"{weak} low-quality (<{need_s:.2f}) and {soft} poorly-localised "
             f"(certainty <{need_c:.2f}, e.g. defocused) of {len(candidates)} -> "
             f"{len(kept)} worth exporting")
    return kept


def stitch_passes(candidates: List[dict], T: int, diag: float, cfg,
                  log: Optional[Callable] = None) -> List[dict]:
    """Join tracks from different passes that sit on the SAME feature into one track.

    TAPNext is causal: a query entering at frame k only tracks forward from k. Seeds are
    deliberately staggered across the block (tracker_core._staggered_queries) so content
    appearing mid-shot gets tracked, which means a seed entering three-quarters of the way in
    covers only the final quarter. The backward pass tracks that same physical feature over
    the earlier frames -- but as a SEPARATE id, and nothing ever joined them. select_spread
    then sees two tracks closer than the spacing threshold, calls the weaker one a duplicate
    and throws it away. The artist gets a track that stops part-way through a shot whose
    pattern is plainly visible throughout, and the half that was successfully tracked is
    discarded.

    So join them here, before the quality gate, where the coverage term of score_track can
    see the full length. Nothing is re-tracked: these frames were already solved, just under
    two names.

    A wrong join is worse than a short track -- it teleports a point onto a different feature
    -- so a merge must clear all of:
      * agreement where they overlap: >= `min_overlap` shared frames, mean separation under
        `max_sep` px. Two distinct features do not sit half a pixel apart for four frames.
      * or bare adjacency (<= `max_gap` frames apart) with the join predicted by the primary's
        own velocity, so a blind append cannot invent a jump.
      * consistent velocity across the join.

    The donor is shifted by the constant offset measured over the overlap before being
    appended. Two independent estimates of the same feature legitimately differ by a constant
    sub-pixel amount (a point 0.3px from the corner is still that corner, rigidly tracked);
    leaving the offset in would put a visible step at the junction, which is the one artefact
    this must not introduce. The offset is constant, so the track's SHAPE -- the only thing a
    solve reads -- is untouched.

    FUSION (`fuse_passes`, off by default). The above is a JOIN: it uses the donor only where
    the primary has no point, and drops the donor's opinion everywhere they overlap. That
    throws away real information. The four passes are four INDEPENDENT estimates of the same
    physical feature, so where two of them agree, averaging them cuts the random component --
    and it cuts the systematic one too, because a causal tracker's error grows with distance
    from its own query frame and FWD and BWD accumulate that error from opposite ends.

    Averaged how: each estimate is weighted 1/(1 + age/tau), age being frames since the pass
    seeded THAT track (its last frame for the backward passes, its first for the forward
    ones). This is the accumulation model from ProTracker's probabilistic integration --
    variance grows along the chain, so an estimate 200 frames from its seed is worth less than
    one 5 frames from its seed. Weights accumulate across donors, so a 4-way agreement is a
    true 4-way weighted mean rather than three pairwise blends.

    (ProTracker also carries a correlation coefficient to stop N correlated estimates being
    counted as N independent ones. That term corrects the fused VARIANCE, not the fused mean,
    and nothing downstream of here reads a variance -- so it is deliberately not implemented
    rather than added as a knob that does nothing.)

    Fusion is also what makes a same-length partner worth merging at all. Without it, a donor
    covering exactly the frames the primary already covers "adds nothing" and is skipped here,
    then quietly deleted by select_spread as a duplicate -- which is the most common shape of
    a FWD/BWD pair on a feature visible all shot, i.e. precisely the pair worth averaging.
    """
    def _log(m):
        if log:
            log(m)

    if not candidates or len(candidates) < 2:
        return candidates
    if not bool(getattr(cfg, "stitch_passes", True)):
        return candidates

    max_sep = float(getattr(cfg, "stitch_max_sep_px", 1.5) or 1.5)
    min_overlap = int(getattr(cfg, "stitch_min_overlap", 4) or 4)
    max_gap = int(getattr(cfg, "stitch_max_gap", 2) or 2)
    fuse = bool(getattr(cfg, "fuse_passes", False))
    tau = float(getattr(cfg, "fuse_age_tau", 30.0) or 30.0)
    # A fused frame needs more agreement than a joined one. A join only ever writes where the
    # primary is silent, so a marginal partner costs coverage at worst; a fuse MOVES points
    # the primary already had, so a partner sitting 1.4px away on a neighbouring feature would
    # drag every shared frame toward it.
    fuse_max_sep = float(getattr(cfg, "fuse_max_sep_px", 0.75) or 0.75)

    def _seed_frame(idx: int, pmap: Dict[int, Tuple[float, float]]) -> int:
        """Frame this track was QUERIED on -- the origin its error grows from.

        The backward passes are seeded at the end of the block and tracked in reverse, so in
        absolute frame numbers their query is their LAST frame. Getting this backwards would
        weight every estimate by exactly the wrong end of its own drift.
        """
        p = str(candidates[idx].get("pass") or candidates[idx].get("id") or "").upper()
        fs = sorted(pmap)
        if not fs:
            return 0
        return fs[-1] if (p.startswith("BWD") or p.startswith("MID_B")) else fs[0]

    def _w(f: int, seed: int) -> float:
        return 1.0 / (1.0 + abs(int(f) - int(seed)) / max(1e-6, tau))

    # Index every candidate by frame so partners are found by POSITION at a shared frame.
    # Bucketing by lifetime-mean position would not work here: the whole point is that these
    # tracks cover different parts of the shot, so on a moving camera their means are far
    # apart even when they are the same feature.
    by_frame: Dict[int, List[Tuple[int, float, float]]] = {}
    for i, c in enumerate(candidates):
        for (f, x, y) in c["pts"]:
            by_frame.setdefault(int(f), []).append((i, float(x), float(y)))

    order = sorted(range(len(candidates)), key=lambda i: candidates[i]["score"], reverse=True)
    merged_into: Dict[int, int] = {}          # donor idx -> primary idx
    pts_by_idx: Dict[int, Dict[int, Tuple[float, float]]] = {
        i: {int(f): (float(x), float(y)) for (f, x, y) in candidates[i]["pts"]}
        for i in range(len(candidates))
    }
    n_merged = 0
    n_fused = 0
    wsum_by_idx: Dict[int, Dict[int, float]] = {}   # primary idx -> per-frame accumulated weight

    def _velocity(pmap: Dict[int, Tuple[float, float]], f0: int, back: bool) -> Optional[Tuple[float, float]]:
        fs = sorted(pmap)
        if len(fs) < 2:
            return None
        seq = [f for f in fs if (f <= f0 if back else f >= f0)]
        if len(seq) < 2:
            return None
        a, b = (seq[-2], seq[-1]) if back else (seq[0], seq[1])
        dt = max(1, b - a)
        return ((pmap[b][0] - pmap[a][0]) / dt, (pmap[b][1] - pmap[a][1]) / dt)

    for pi in order:
        if pi in merged_into:
            continue
        grew = True
        while grew:                       # a primary may absorb several partials in turn
            grew = False
            prim = pts_by_idx[pi]
            pf = sorted(prim)
            if not pf:
                break
            # Candidates sharing any frame with the primary, found via the frame index.
            seen: Dict[int, int] = {}
            probe = pf if len(pf) <= 8 else [pf[int(round(k * (len(pf) - 1) / 7.0))] for k in range(8)]
            for f in probe:
                for (j, x, y) in by_frame.get(f, ()):
                    if j == pi or j in merged_into:
                        continue
                    if abs(x - prim[f][0]) <= max_sep and abs(y - prim[f][1]) <= max_sep:
                        seen[j] = seen.get(j, 0) + 1
            # Adjacency: a track starting just after the primary ends, or ending just before
            # it starts, never shares a frame -- so the index cannot find it. Look for those
            # explicitly and judge them by the primary's own predicted position.
            for j in range(len(candidates)):
                if j == pi or j in merged_into or j in seen:
                    continue
                jf = sorted(pts_by_idx[j])
                if not jf:
                    continue
                if 0 < jf[0] - pf[-1] <= max_gap or 0 < pf[0] - jf[-1] <= max_gap:
                    seen[j] = 0

            best_j, best_shift, best_fuse = None, None, False
            for j in sorted(seen, key=lambda k: -seen[k]):
                don = pts_by_idx[j]
                shared = sorted(set(prim) & set(don))
                new_frames = [f for f in don if f not in prim]
                # Without fusion a same-coverage donor is dead weight. With it, that donor IS
                # the payload: a second independent estimate of every frame the primary has.
                can_fuse = fuse and len(shared) >= min_overlap
                if not new_frames and not can_fuse:
                    continue                      # adds nothing, so nothing to gain
                if len(shared) >= min_overlap:
                    dx = [prim[f][0] - don[f][0] for f in shared]
                    dy = [prim[f][1] - don[f][1] for f in shared]
                    mx, my = st.mean(dx), st.mean(dy)
                    sep = st.mean([math.hypot(prim[f][0] - don[f][0], prim[f][1] - don[f][1])
                                   for f in shared])
                    if sep > max_sep:
                        continue
                    # Same feature but disagreeing on how it MOVES is two features that
                    # happen to be crossing; the residual after removing the constant offset
                    # is what catches that.
                    resid = st.mean([math.hypot(prim[f][0] - don[f][0] - mx,
                                                prim[f][1] - don[f][1] - my) for f in shared])
                    if resid > max_sep * 0.5:
                        continue
                    shift = (mx, my)
                    # Fuse only where the two really are the same point. `sep` is the raw
                    # separation, before the offset is removed -- a partner further away than
                    # this is joinable (its shape agrees) but not averageable.
                    can_fuse = can_fuse and sep <= fuse_max_sep
                    if not new_frames and not can_fuse:
                        continue
                elif shared:
                    continue                      # overlapping but too briefly to judge
                else:
                    # Adjacent: predict where the primary would have been on the donor's first
                    # frame and require the donor to be there.
                    jf = sorted(don)
                    if jf[0] > pf[-1]:
                        v = _velocity(prim, pf[-1], back=True)
                        if v is None:
                            continue
                        dt = jf[0] - pf[-1]
                        px = prim[pf[-1]][0] + v[0] * dt
                        py = prim[pf[-1]][1] + v[1] * dt
                        tgt = don[jf[0]]
                    else:
                        v = _velocity(prim, pf[0], back=False)
                        if v is None:
                            continue
                        dt = pf[0] - jf[-1]
                        px = prim[pf[0]][0] - v[0] * dt
                        py = prim[pf[0]][1] - v[1] * dt
                        tgt = don[jf[-1]]
                    if math.hypot(px - tgt[0], py - tgt[1]) > max_sep:
                        continue
                    shift = (px - tgt[0], py - tgt[1])
                    can_fuse = False              # no shared frames: nothing to average
                best_j, best_shift, best_fuse = j, shift, can_fuse
                break

            if best_j is None:
                continue
            don = pts_by_idx[best_j]
            sx, sy = best_shift
            if best_fuse:
                # Accumulated weighted mean, so N donors give one N-way average rather than
                # N pairwise blends -- absorbing a second donor must not halve the first.
                if pi not in wsum_by_idx:
                    sp = _seed_frame(pi, prim)
                    wsum_by_idx[pi] = {f: _w(f, sp) for f in prim}
                wsum = wsum_by_idx[pi]
                sd = _seed_frame(best_j, don)
                for f, (x, y) in don.items():
                    if f not in prim:
                        prim[f] = (x + sx, y + sy)
                        wsum[f] = _w(f, sd)
                        continue
                    wd = _w(f, sd)
                    w0 = wsum.get(f, 1.0)
                    tot = w0 + wd
                    prim[f] = ((prim[f][0] * w0 + (x + sx) * wd) / tot,
                               (prim[f][1] * w0 + (y + sy) * wd) / tot)
                    wsum[f] = tot
                n_fused += 1
            else:
                for f, (x, y) in don.items():
                    if f not in prim:             # primary's own points always win
                        prim[f] = (x + sx, y + sy)
            merged_into[best_j] = pi
            n_merged += 1
            grew = True

    if not n_merged:
        return candidates

    out: List[dict] = []
    for i, c in enumerate(candidates):
        if i in merged_into:
            continue
        pmap = pts_by_idx[i]
        pts = [(f, pmap[f][0], pmap[f][1]) for f in sorted(pmap)]
        # Re-score at the new length: the coverage term is exactly what makes a stitched
        # track outrank the partials it replaced, and select_spread reads that score.
        out.append({**c, "pts": pts,
                    "mean": (st.mean([p[1] for p in pts]), st.mean([p[2] for p in pts])),
                    "score": score_track(pts, T, diag, cfg)})
    _log(f"  stitch: joined {n_merged} partial track(s) onto {len(out)} feature(s) "
         f"({len(candidates)} -> {len(out)} candidates)"
         + (f"; {n_fused} averaged across passes (tau={tau:g}f)" if n_fused else ""))
    return out


def identity_gate(tracks: Dict[str, Track], identity: Dict[str, float], cfg,
                  log: Optional[Callable] = None) -> Dict[str, Track]:
    """Drop tracks that no longer sit on the feature they started on.

    `identity` is the correlation between a track's patch at its FIRST frame and at its
    LAST. A point that slid off its feature ends up on unrelated pixels and scores near
    zero; a point that held on scores high even when lighting and perspective have moved.

    This is the only measure here that catches a SMOOTH drifter, and everything else was
    tried first. Certainty reads the sharpness of a correlation peak, and a drifting point
    correlates with its new surroundings perfectly well (-0.236 against real error). Score
    rewards long unbroken tracks, which is exactly what a drifter is (+0.398 -- it ranks
    them as the BEST tracks). Wobble measures deviation from the track's own smooth path,
    and a drift is smooth by definition, so gating on it cut 14 of 29 tracks and left the
    drifter untouched.

    An ABSOLUTE floor, not a relative one: "does this patch still resemble that patch" means
    the same thing on every plate, unlike certainty, which is why the certainty bar has to
    be self-calibrating and this one does not. The default is set far below the good
    population (0.50-0.99 on the measured shot) so it only ever removes a track that has
    genuinely lost its feature, not one whose appearance merely changed.
    """
    def _log(m):
        if log:
            log(m)

    need = float(getattr(cfg, "min_seed_identity", 0.0) or 0.0)
    if need <= 0.0 or not tracks or not identity:
        return tracks
    # Unmeasured is not evidence of failure -- same rule as the certainty gate.
    drop = [k for k, v in identity.items()
            if k in tracks and v == v and float(v) < need]
    if not drop:
        return tracks
    # Never empty a shot on this test alone. If most of the export fails, the plate changed
    # appearance wholesale (a lighting cut, a whip) rather than every track drifting.
    if len(drop) > 0.5 * len(tracks):
        _log(f"  identity gate: {len(drop)}/{len(tracks)} tracks fail the seed-identity check "
             f"(<{need:.2f}) -- too many to be drift, so the plate changed; not applied")
        return tracks
    keep = {k: v for k, v in tracks.items() if k not in drop}
    worst = ", ".join(f"{k} ({identity[k]:+.2f})" for k in sorted(drop, key=lambda k: identity[k])[:3])
    _log(f"  identity gate: dropped {len(drop)} track(s) no longer on their seeded feature "
         f"(seed-vs-end correlation <{need:.2f}) -> {len(keep)}   worst: {worst}")
    return keep


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


def defragment(tracks: Dict[str, Track], cfg,
               log: Optional[Callable] = None, registry=None) -> Dict[str, Track]:
    """Split heavily-broken tracks into their continuous runs.

    Occlusion continuity deliberately lets a track survive a crossing by carrying a HOLE, so
    a background point is not deleted because someone walked past it. One long gap is exactly
    that and is worth keeping under a single id.

    Many SHORT gaps are a different animal: not an occluded track but a marginal one that
    kept losing and regaining lock. Nothing limited that, so a point could lose lock fifteen
    times, re-acquire fifteen times, and export as one id -- which is why tracks blink on and
    off in the 3DE viewport like an LED.

    Rule: a track may carry up to `max_track_gaps` holes, and every hole must be at least
    `min_occlusion_run` frames (i.e. plausibly a real occlusion). Anything else is cut into
    its continuous runs, and runs too short to be useful are dropped. Every exported track is
    then either solid or holed only where something genuinely crossed it.
    """
    def _log(m):
        if log:
            log(m)

    max_gaps = int(getattr(cfg, "max_track_gaps", -1))
    if max_gaps < 0 or not tracks:
        return tracks
    min_hole = max(1, int(getattr(cfg, "min_occlusion_run", 1) or 1))
    min_run = max(2, int(getattr(cfg, "refine_min_len", 8) or 2))

    out: Dict[str, Track] = {}
    n_split = n_dropped = 0
    for tid, pts in tracks.items():
        p = sorted(pts, key=lambda q: int(q[0]))
        runs: List[Track] = []
        cur: Track = [p[0]] if p else []
        holes: List[int] = []
        for a, b in zip(p, p[1:]):
            step = int(b[0]) - int(a[0])
            if step == 1:
                cur.append(b)
            else:
                holes.append(step - 1)
                runs.append(cur)
                cur = [b]
        if cur:
            runs.append(cur)

        # A SOLID track is never this function's business, whatever its length -- length is
        # the quality gate's job, and silently applying one here would be a surprise from
        # something called "defragment".
        if not holes:
            out[tid] = p
            continue

        # Keep a holed track whole only if it looks genuinely occluded: few holes, each long
        # enough to be something crossing, AND every visible stretch substantial enough to
        # see. Without that last test, three 3-frame fragments separated by long gaps read as
        # "two occlusions" and still blink.
        if (len(holes) <= max_gaps and all(h >= min_hole for h in holes)
                and all(len(r) >= min_run for r in runs)):
            out[tid] = p
            continue

        # Too broken to be one track: emit the usable runs separately.
        keep_runs = [r for r in runs if len(r) >= min_run]
        if not keep_runs:
            n_dropped += 1
            continue
        n_split += 1
        for k, r in enumerate(keep_runs):
            new_id = tid if k == 0 else f"{tid}_f{k}"
            if registry is not None and new_id != tid:
                registry.derive(tid, new_id)   # the runs are all the same feature
            out[new_id] = r

    if n_split or n_dropped:
        _log(f"  defragment: {n_split} track(s) were breaking up repeatedly -> split into "
             f"continuous runs, {n_dropped} dropped entirely ({len(tracks)} -> {len(out)})")
    return out


def _rank(v) -> float:
    """Sort key for certainty that survives NaN.

    NaN means "not measured" (see certainty_gate). Python's sort is undefined with NaN in
    the key -- comparisons are all False, so the result depends on input order. Rank unknown
    below every real score instead: these tracks are never dropped for being unknown, so this
    only decides where they sit when a fallback is picking "the best N".
    """
    if v is None:
        return -1.0
    f = float(v)
    return -1.0 if math.isnan(f) else f


def wobble_gate(tracks: Dict[str, Track], wobble: Dict[str, float], cfg,
                log: Optional[Callable] = None) -> Dict[str, Track]:
    """Drop tracks that deviate from their own smooth path far more than the shot's norm.

    This is the gate that catches what the others miss. Measured on SH004 against an
    independent Lucas-Kanade re-track of every exported track (tools/verify_against_lk.py),
    the worst track in the delivery sat 20.51px from a reference closing to 0.71px -- and it
    had PASSED the certainty gate. Certainty predicts true error at -0.236, so it cannot
    reject it; `score` is worse than useless at +0.398, ranking bad tracks as good because
    its coverage term rewards length and length is where drift accumulates. Wobble is the
    one signal that ranks real error (+0.573 spearman), and only since it stopped detrending
    with a moving average, which reported 1.585px on ground truth correct by construction.

    RELATIVE to the shot's own median, for the same reason the certainty bar is: wobble
    scales with how much the camera actually moves, so an absolute px threshold would empty a
    handheld plate and pass everything on a locked-off one.

    Simulated on the real per-track numbers before it was written, then run: at 1.5x the
    median, SH004 goes from 2.34px median / 20.51px worst / 7 tracks over 3px, to 1.08px
    median / 2.58px worst / none over 3px.

    Applied to the FINAL set, after the top-up -- a filter that sees only part of the export
    cannot remove a track that entered through another door, which is exactly how an earlier
    backfill-only ceiling failed.

    A track with no wobble measurement is kept and not judged, as elsewhere.
    """
    def _log(m):
        if log:
            log(m)

    mult = float(getattr(cfg, "wobble_rel", 0.0) or 0.0)
    if mult <= 0.0 or not tracks or not wobble:
        return tracks
    vals = [float(wobble[k]) for k in tracks
            if k in wobble and not math.isnan(float(wobble[k]))]
    if len(vals) < 8:
        return tracks                      # too few to establish what this shot's norm is
    thr = mult * st.median(vals)
    if thr <= 0.0:
        return tracks
    keep = {k: v for k, v in tracks.items()
            if k not in wobble or math.isnan(float(wobble[k])) or float(wobble[k]) <= thr}
    floor = max(1, int(getattr(cfg, "quality_gate_floor", 8) or 1))
    if len(keep) < min(floor, len(tracks)):
        best = sorted(tracks, key=lambda k: (float(wobble[k])
                                             if k in wobble and not math.isnan(float(wobble[k]))
                                             else float("inf")))[:floor]
        _log(f"  wobble gate: only {len(keep)} track(s) under {thr:.2f}px -> relaxed, "
             f"keeping the steadiest {len(best)}")
        return {k: tracks[k] for k in best}
    if len(keep) < len(tracks):
        _log(f"  wobble gate: dropped {len(tracks) - len(keep)} track(s) deviating more than "
             f"{thr:.2f}px ({mult:g}x this shot's median) -> {len(keep)}")
    return keep


def backfill_to_floor(kept: Dict[str, Track], all_tracks: Dict[str, Track],
                      certainty: Dict[str, float], cfg,
                      log: Optional[Callable] = None,
                      wobble: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, Track], set]:
    """Top a thin export back up to min_export_tracks with the best of what was rejected.

    Seven filters stacked multiplicatively could take 200 candidates down to 10, with no
    single stage obviously at fault. A handful of tracks and no explanation is not a usable
    delivery, so the shortfall is made up from the best rejects -- and every one of them is
    FLAGGED, in the log and in the per-track CSV, so it is clear which to check rather than
    quietly padding the count.

    WHICH rejects get picked matters as much as how many, and it was being decided by the
    wrong number. Measured on SH004 against an independent Lucas-Kanade re-track of every
    exported track (tools/verify_against_lk.py), on real footage:

        predictor of true error     pearson   spearman
        score                        +0.398     +0.291   <- ranks BAD tracks as good
        certainty                    -0.236     -0.133
        wobble (after the S-G fix)   +0.127     +0.573   <- the usable one

    `score` is positively correlated with error because its coverage term is weighted 0.5, so
    a long track outranks a still one and long tracks are where drift accumulates. Picking
    the top 10 by score shipped a 20.51px track and four over 3px; picking the top 10 by
    wobble gave a median of 1.08px, a worst of 2.58px, and nothing over 3px.

    So rank by wobble when it has been measured, and fall back to certainty when it has not.
    Note this only became usable after measure_wobble stopped detrending with a moving
    average -- before that it reported 1.585px on ground truth that is correct by
    construction, i.e. it was ranking the plate's motion.

    Returns (tracks, weak_ids).
    """
    def _log(m):
        if log:
            log(m)

    floor = int(getattr(cfg, "min_export_tracks", 0) or 0)
    if floor <= 0 or len(kept) >= floor or len(all_tracks) <= len(kept):
        return kept, set()
    spare = [k for k in all_tracks if k not in kept]
    def _w(k):
        v = (wobble or {}).get(k)
        return float("inf") if v is None or math.isnan(float(v)) else float(v)

    if wobble:
        # Ascending: least wobble first. Unmeasured sorts last rather than first -- an
        # unknown is not evidence of steadiness.
        spare.sort(key=lambda k: (_w(k), -_rank(certainty.get(k, None))))
        why = "steadiest first"
    else:
        spare.sort(key=lambda k: _rank(certainty.get(k, None)), reverse=True)
        why = "highest certainty first"
    add = spare[:max(0, floor - len(kept))]

    if not add:
        return kept, set()
    out = dict(kept)
    for k in add:
        out[k] = all_tracks[k]
    _log(f"  only {len(kept)} track(s) cleared the quality bar -> topped up to {len(out)} "
         f"with the best {len(add)} rejected ({why}), marked weak in the report")
    return out, set(add)


def dump_track_report(path: str, tracks: Dict[str, Track], certainty: Dict[str, float],
                      T: int, width: int, height: int, cfg,
                      wobble_fn: Optional[Callable] = None,
                      weak: Optional[set] = None, registry=None) -> str:
    """Write a per-track CSV: length, score, certainty, wobble amplitude and period.

    Several rounds of tracking fixes have been judged by eye on real footage and by
    synthetic tests in the harness, and the two kept disagreeing. This puts the actual
    numbers for the actual shot on disk, so the next question is answered by reading a
    column rather than guessing. Best-effort -- a reporting failure must never cost a run.
    """
    try:
        diag = math.hypot(float(width or 0), float(height or 0)) or 1000.0
        weak = weak or set()
        # The per-track columns are appended, never inserted: anything already reading this
        # file by column index keeps working.
        extra = ["kind", "patch_px", "motion", "sam_obj", "depth_class", "risk",
                 "escalation", "geo_residual"] if registry is not None else []
        rows = ["name,frames,span,score,certainty,wobble_px,wobble_period,mean_x,mean_y,weak"
                + ("," + ",".join(extra) if extra else "")]
        for tid, pts in sorted(tracks.items()):
            p = sorted(pts, key=lambda q: q[0])
            if not p:
                continue
            n = len(p)
            span = int(p[-1][0]) - int(p[0][0]) + 1
            sc = score_track(p, T, diag, cfg)
            ct = float(certainty.get(tid, float("nan")))
            amp, per = wobble_fn(p) if wobble_fn else (float("nan"), 0)
            mx = sum(q[1] for q in p) / n
            my = sum(q[2] for q in p) / n
            row = (f"{tid},{n},{span},{sc:.4f},{ct:.4f},{amp:.4f},{per},"
                   f"{mx:.1f},{my:.1f},{1 if tid in weak else 0}")
            if extra:
                m = registry.row(tid)
                row += "," + ",".join(str(m.get(c, "")) for c in extra)
            rows.append(row)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
        return path
    except Exception:
        return ""


def certainty_gate(tracks: Dict[str, Track], certainty: Dict[str, float], cfg,
                   log: Optional[Callable] = None) -> Dict[str, Track]:
    """Drop poorly-localised tracks AFTER refine, when certainty is finally known.

    The main quality gate runs before refinement, so it cannot see this: certainty comes from
    the correlation peaks the refine stage measures. It is the only signal that catches a
    defocused point, because such a point's motion statistics -- long, smooth, no jumps --
    are excellent. Same safeguard as the main gate: if the floor would empty the shot it is
    relaxed and the best are kept.
    """
    def _log(m):
        if log:
            log(m)

    need = float(getattr(cfg, "min_track_certainty", 0.0) or 0.0)
    rel = float(getattr(cfg, "certainty_rel", 0.0) or 0.0)
    if (need <= 0.0 and rel <= 0.0) or not tracks or not certainty:
        return tracks

    # RELATIVE threshold, judged against this shot's own best tracks. An absolute number
    # cannot work across plates: certainty depends on contrast, grain and lens, so 0.5 might
    # be excellent on one shot and poor on another. Comparing each track to the best decile
    # of ITS OWN shot is self-calibrating -- on a uniformly sharp plate nothing is dropped,
    # while on a sharp-subject/soft-background plate the soft cluster falls away. That is
    # exactly the handheld-with-defocused-background case.
    # A track with no certainty measurement is not a track with bad certainty. Refine reports
    # NaN when it could not run at all (every segment "no-anchor"), and those points are kept
    # deliberately -- raw rather than deleted -- so they are ordinary tracks that this gate has
    # no evidence about. They are excluded from the distribution AND never dropped by it:
    # counting them as 0.0 both condemned them and manufactured a gap wide enough to look like
    # two populations, which is how 23 accurate tracks were dropped on bench/lab03.
    def _known(k) -> bool:
        v = certainty.get(k, None)
        return v is not None and not math.isnan(float(v))

    vals = [float(certainty[k]) for k in tracks if _known(k)]
    n_unknown = sum(1 for k in tracks if not _known(k))
    thr = need

    # Describe the distribution once: whether it holds two populations, and where they part.
    # This is a property of the numbers, not of which bar is switched on, so both the
    # threshold choice below and the safety rail further down read the same analysis.
    #
    # "Largest gap" alone does NOT establish two populations. Any sample of a continuum has a
    # largest gap, and in the sparsely-sampled tail it is routinely several times the typical
    # spacing -- so the test fired on ordinary spread and, because a clean split is allowed to
    # override the max_cut rail, deleted most of the shot. Measured on a synthetic plate whose
    # every track is accurate to 0.06px (bench/): certainty ran 0.35-0.87 with a 0.075 gap at
    # 0.79, which is 3.7x the median gap, and the gate dropped 20 of 23 tracks with nothing
    # wrong with any of them.
    #
    # The test is SEPARATION, not population size. Counting tracks either side looks sensible
    # and gets the headline case backwards: on the plate this gate exists for -- a sharp
    # subject against a defocused background -- the GOOD cluster is the minority, so any
    # "both sides must be big" rule throws the real split away and keeps the soft tracks.
    # What actually distinguishes two populations from one is the gap measured against how
    # spread out each side is on its own. A continuum has a gap smaller than its own scatter
    # (lab02: 0.075 gap against ~0.12 within-cluster spread -> one population). A genuine
    # focus split has a gap far larger than either side's scatter (soft 0.30 +/-0.03 versus
    # sharp 0.85 +/-0.03 -> gap 0.5, unmistakable) and stays unmistakable when the sharp
    # cluster is a tenth of the tracks.
    split_gap, gap_mid = 0.0, 0.0
    if vals:
        a = np.array(vals, dtype=float)
        sv = np.sort(a)
        gaps = np.diff(sv)
        # Below ~8 samples "two populations" is not a claim the numbers can support.
        if len(gaps) and len(sv) >= 8:
            gi = int(np.argmax(gaps))
            gap = float(gaps[gi])
            lo_side, hi_side = sv[:gi + 1], sv[gi + 1:]
            # Scatter WITHIN each side. A single-point side has no scatter of its own, so it
            # cannot vouch for itself -- fall back to the whole set's spacing scale instead of
            # reading 0.0 and declaring every isolated point a population.
            def _spread(side: np.ndarray) -> float:
                return float(side.std()) if side.size >= 2 else float("nan")
            s_lo, s_hi = _spread(lo_side), _spread(hi_side)
            known = [s for s in (s_lo, s_hi) if not math.isnan(s)]
            scatter = max(known) if known else float(np.median(gaps))
            # Two populations only when the gap dominates the scatter inside them.
            if gap >= 0.05 and gap >= 2.0 * max(scatter, 1e-6) and min(len(lo_side), len(hi_side)) >= 2:
                split_gap = gap
                gap_mid = float((sv[gi] + sv[gi + 1]) * 0.5)

    if rel > 0.0 and vals:
        a = np.array(vals, dtype=float)
        best = float(np.percentile(a, 90))
        lo = float(np.percentile(a, 10))
        # The relative bar applies ONLY when the numbers show a soft cluster to remove.
        #
        # Two ways they can fail to: no spread at all (a uniformly sharp or uniformly soft
        # plate, where every track scores alike), or a spread with no clean split (a single
        # population strung along a continuum). In both cases rel * P90 is slicing an
        # arbitrary point off a distribution rather than separating two groups. On
        # bench/lab02 that cut 13 of 23 tracks whose true error was 0.06px -- every one as
        # accurate as the tracks it kept -- and certainty there correlates with true error at
        # only -0.22, so the cut was made on noise.
        #
        # A narrow spread used to `return tracks` outright, which skipped the ABSOLUTE floor
        # too. That let a uniformly defocused plate -- every track alike and every one too
        # soft to trust -- pass the gate untouched, which is the exact case the floor exists
        # for. Both paths now fall through to the floor, which encodes "too soft to trust"
        # rather than "softer than its neighbours".
        narrow = float(a.max()) - float(a.min()) < 0.05
        if split_gap > 0.0:
            # Prefer the natural gap over a percentile bar. A percentile assumes the good
            # tracks are the majority, and on a defocused background they are the MINORITY --
            # most of the frame is soft, so P90 lands inside the bad cluster and the bar
            # collapses. A gap makes no such assumption.
            thr = max(thr, gap_mid)
        else:
            why = ("spread too narrow" if narrow else "no clean split") + \
                  f" ({lo:.2f}-{best:.2f})"
            if need <= 0.0:
                _log(f"  certainty gate: {why} -- one population and no absolute floor set, "
                     f"so nothing to judge against; skipped")
                return tracks
            _log(f"  certainty gate: {why} -- one population; absolute floor {need:.2f} only, "
                 f"no relative bar")
    if thr <= 0.0:
        return tracks
    keep = {k: v for k, v in tracks.items()
            if (not _known(k)) or float(certainty[k]) >= thr}
    if n_unknown:
        _log(f"  certainty gate: {n_unknown} track(s) carry no certainty measurement "
             f"(refine could not anchor) -- kept, not judged")

    # A large cut is fine when it is DISCRIMINATING and suspect when it is arbitrary -- and
    # size alone cannot tell those apart. A sharp subject against a mostly-defocused
    # background genuinely needs most tracks gone, while a noisy certainty measure would cut
    # just as deeply for no reason. So judge the split instead: if the threshold falls inside
    # a real gap in the distribution, there are two populations and the cut is well founded
    # at any size. If it lands mid-continuum, it is slicing an arbitrary point and gets capped.
    max_cut = float(getattr(cfg, "certainty_max_cut", 0.6) or 0.0)
    if max_cut > 0.0 and len(keep) < (1.0 - max_cut) * len(tracks) and len(vals) >= 4:
        if split_gap > 0.0:
            _log(f"  certainty gate: {len(tracks) - len(keep)}/{len(tracks)} dropped, but the "
                 f"split is clean (gap {split_gap:.2f} at {thr:.2f}) -- two populations, "
                 f"so the cut stands")
        else:
            n_keep = max(1, int(round((1.0 - max_cut) * len(tracks))))
            best_ids = sorted(tracks, key=lambda k: _rank(certainty.get(k, None)),
                              reverse=True)[:n_keep]
            _log(f"  certainty gate: would have cut {len(tracks) - len(keep)}/{len(tracks)} "
                 f"with no clear split in the numbers -- capped at {len(tracks) - n_keep}")
            return {k: tracks[k] for k in best_ids}
    need = thr
    floor = max(1, int(getattr(cfg, "quality_gate_floor", 8) or 1))
    if len(keep) < min(floor, len(tracks)):
        best = sorted(tracks, key=lambda k: _rank(certainty.get(k, None)), reverse=True)[:floor]
        _log(f"  certainty gate: only {len(keep)} track(s) cleared {need:.2f} -> relaxed, "
             f"keeping the best {len(best)} (soft plate)")
        return {k: tracks[k] for k in best}
    if len(keep) < len(tracks):
        _log(f"  certainty gate: dropped {len(tracks) - len(keep)} poorly-localised track(s) "
             f"(certainty <{need:.2f}) -> {len(keep)}")
    return keep


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
    sel = select_spread(candidates, cfg, out_width=int(width or 0), log=log)
    # Cut repeatedly-broken tracks into continuous runs before export, so nothing blinks on
    # and off in the viewport.
    return defragment(sel, cfg, log=log)
