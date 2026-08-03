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
               log: Optional[Callable] = None) -> Dict[str, Track]:
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
            out[tid if k == 0 else f"{tid}_f{k}"] = r

    if n_split or n_dropped:
        _log(f"  defragment: {n_split} track(s) were breaking up repeatedly -> split into "
             f"continuous runs, {n_dropped} dropped entirely ({len(tracks)} -> {len(out)})")
    return out


def backfill_to_floor(kept: Dict[str, Track], all_tracks: Dict[str, Track],
                      certainty: Dict[str, float], cfg,
                      log: Optional[Callable] = None) -> Tuple[Dict[str, Track], set]:
    """Top a thin export back up to min_export_tracks with the best of what was rejected.

    Seven filters stacked multiplicatively could take 200 candidates down to 10, with no
    single stage obviously at fault. A handful of tracks and no explanation is not a usable
    delivery, so the shortfall is made up from the best rejects -- and every one of them is
    FLAGGED, in the log and in the per-track CSV, so it is clear which to check rather than
    quietly padding the count.

    Returns (tracks, weak_ids).
    """
    def _log(m):
        if log:
            log(m)

    floor = int(getattr(cfg, "min_export_tracks", 0) or 0)
    if floor <= 0 or len(kept) >= floor or len(all_tracks) <= len(kept):
        return kept, set()
    spare = [k for k in all_tracks if k not in kept]
    spare.sort(key=lambda k: float(certainty.get(k, 0.0)), reverse=True)
    add = spare[:max(0, floor - len(kept))]
    if not add:
        return kept, set()
    out = dict(kept)
    for k in add:
        out[k] = all_tracks[k]
    _log(f"  only {len(kept)} track(s) cleared the quality bar -> topped up to {len(out)} "
         f"with the best {len(add)} rejected, marked weak in the report")
    return out, set(add)


def dump_track_report(path: str, tracks: Dict[str, Track], certainty: Dict[str, float],
                      T: int, width: int, height: int, cfg,
                      wobble_fn: Optional[Callable] = None,
                      weak: Optional[set] = None) -> str:
    """Write a per-track CSV: length, score, certainty, wobble amplitude and period.

    Several rounds of tracking fixes have been judged by eye on real footage and by
    synthetic tests in the harness, and the two kept disagreeing. This puts the actual
    numbers for the actual shot on disk, so the next question is answered by reading a
    column rather than guessing. Best-effort -- a reporting failure must never cost a run.
    """
    try:
        diag = math.hypot(float(width or 0), float(height or 0)) or 1000.0
        weak = weak or set()
        rows = ["name,frames,span,score,certainty,wobble_px,wobble_period,mean_x,mean_y,weak"]
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
            rows.append(f"{tid},{n},{span},{sc:.4f},{ct:.4f},{amp:.4f},{per},"
                        f"{mx:.1f},{my:.1f},{1 if tid in weak else 0}")
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
    vals = [float(certainty.get(k, 0.0)) for k in tracks if k in certainty]
    thr = need

    # Describe the distribution once: whether it holds two populations, and where they part.
    # This is a property of the numbers, not of which bar is switched on, so both the
    # threshold choice below and the safety rail further down read the same analysis.
    split_gap, gap_mid = 0.0, 0.0
    if vals:
        a = np.array(vals, dtype=float)
        sv = np.sort(a)
        gaps = np.diff(sv)
        if len(gaps):
            gi = int(np.argmax(gaps))
            if float(gaps[gi]) >= 0.05:
                split_gap = float(gaps[gi])
                gap_mid = float((sv[gi] + sv[gi + 1]) * 0.5)

    if rel > 0.0 and vals:
        a = np.array(vals, dtype=float)
        best = float(np.percentile(a, 90))
        lo = float(np.percentile(a, 10))
        # Only worth applying when there IS a spread to discriminate. On a uniformly sharp
        # (or uniformly soft) plate every track scores alike, and a relative bar would then
        # be cutting on noise rather than on a real soft cluster.
        if float(a.max()) - float(a.min()) < 0.05:
            _log(f"  certainty gate: spread too narrow ({lo:.2f}-{best:.2f}) to separate "
                 f"anything -- skipped")
            return tracks
        # Prefer the natural gap over a percentile bar. A percentile assumes the good tracks
        # are the majority, and on a defocused background they are the MINORITY -- most of
        # the frame is soft, so P90 lands inside the bad cluster and the bar collapses. A gap
        # makes no such assumption.
        thr = max(thr, gap_mid) if split_gap > 0.0 else max(thr, rel * best)
    if thr <= 0.0:
        return tracks
    keep = {k: v for k, v in tracks.items() if float(certainty.get(k, 1.0)) >= thr}

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
            best_ids = sorted(tracks, key=lambda k: float(certainty.get(k, 0.0)),
                              reverse=True)[:n_keep]
            _log(f"  certainty gate: would have cut {len(tracks) - len(keep)}/{len(tracks)} "
                 f"with no clear split in the numbers -- capped at {len(tracks) - n_keep}")
            return {k: tracks[k] for k in best_ids}
    need = thr
    floor = max(1, int(getattr(cfg, "quality_gate_floor", 8) or 1))
    if len(keep) < min(floor, len(tracks)):
        best = sorted(tracks, key=lambda k: float(certainty.get(k, 0.0)), reverse=True)[:floor]
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
