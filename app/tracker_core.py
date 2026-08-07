# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import gc
import time
import math
import threading
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Dict, List, Tuple

import numpy as np
import cv2  # type: ignore
import torch # Need torch for empty_cache

from app.video_io import read_video_frames_bgr_scaled, FrameSource, estimate_clip_bytes
from app.tapnext_engine import TapNextEngine
from app.export_3de import write_tracks_txt
from app import track_filter as _tf
from app.track_meta import (SeedFeat, SeedStats, TrackMeta, TrackRegistry,
                            classify_seed, policy_for)

StatusCB = Optional[Callable[[str], None]]


@dataclass
class RunnerConfig:
    input_dir: str
    output_dir: str
    
    seeding_mode: str = "features"
    bidirectional: bool = True

    # --- Per-shot auto-tune -------------------------------------------------------------
    # This is a batch tool, so the settings below cannot be hand-tuned per shot -- yet a
    # handheld plate with a defocused background and a locked-off macro shot were given
    # identical constants, and at most one of those can be right. When on, each shot is
    # measured (sharpness, grain, texture, motion) and the relevant values derived from it;
    # anything the user has explicitly set still wins. See app/shot_profile.py.
    auto_tune: bool = True
    auto_tune_overrides: Dict[str, object] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)   # Qwen's, when Analyze has run
    # Localisation certainty floor, normally set by the auto-tune (0 = off), plus a RELATIVE
    # bar judged against the shot's own best tracks -- an absolute number cannot travel
    # between plates, since certainty depends on contrast, grain and lens.
    min_track_certainty: float = 0.0
    certainty_rel: float = 0.80
    certainty_max_cut: float = 0.6   # a gate wanting more than this is measuring badly
    track_report: bool = True        # write a per-track CSV beside the exported tracks
    
    max_tracks: int = 1200      
    feature_quality: float = 0.02 
    min_feature_dist: int = 12    
    
    grid_size: int = 10
    flip_y_for_3de: bool = True
    selected_files: List[str] | None = None
    selected_scales: Dict[str, float] | None = None
    # True plate resolution to export coords in, when the tracked mp4 is a downscaled
    # proxy of a larger plate. 0 = coords stay in the tracked (mp4) resolution.
    out_w: int = 0
    out_h: int = 0
    # Track a full-res image SEQUENCE directly (no mp4). When set, this dir is the input
    # for one shot; W0/H0 become the plate res so refine runs native and coords are in
    # plate space (out_w/out_h not needed). sequence_name = the shot name for output.
    sequence_path: str = ""
    sequence_name: str = ""

    # Frame range (1-based inclusive). 0 = unset => full clip.
    frame_start: int = 0
    frame_end: int = 0

    mask_root_dir: str = ""
    mask_mode: str = "outside"
    mask_polarity: str = "auto"
    # Safety margin pulled IN from the mask edge, in px. SAM3 mattes often stop a few px
    # short of the real silhouette (hair, motion-blur fringe), and a track seeded in that
    # halo is stuck to the character and slides. Shrinks the seed/keep region on both the
    # seeding mask and the per-frame gate. 0 = the old, exact-mask behaviour.
    mask_margin_px: int = 8

    # Aperture-problem guard. A point on a 1-D feature (rope, plate line, TV-screen edge) can
    # only be localized ACROSS the edge, never ALONG it, so NCC finds a response ridge and the
    # point slides down it. Reject seeds whose structure-tensor anisotropy (lambda_min /
    # lambda_max) is below this: a corner tends to 1, a pure edge to 0. Contrast-invariant, so
    # unlike feature_quality it does not also throw away faint-but-real corners. 0 = off.
    min_corner_anisotropy: float = 0.08

    mask_subdir: str = "masks"
    output_tag: str = ""

    enable_filtering: bool = True
    min_visibility_ratio: float = 0.05 
    min_motion_inlier_ratio: float = 0.10
    motion_residual_diag_frac: float = 0.010
    max_jump_diag_frac: float = 0.050
    smooth_window: int = 0

    enable_mask_gating: bool = True
    inside_ratio: float = 0.80

    # --- Occlusion continuity ---------------------------------------------------------
    # 'outside' gating used to be a per-TRACK kill ("dropped if ever inside"), so one person
    # walking past for a few frames deleted an otherwise perfect background track for the
    # WHOLE shot. The mover mask is actually the best occlusion signal available (SAM3 knows
    # exactly where the character is on every frame), so it now marks those frames occluded
    # and the track survives with a gap. 3DE takes gaps natively (export writes a frame
    # number per point) and pattern_refine already refines per visible segment.
    occlusion_continuity: bool = True
    min_track_points: int = 8      # drop a track only if fewer than this many frames survive
    # A point sitting right on the (dilated) mask edge flips in and out from sub-pixel
    # jitter alone, which reads as the track flickering on and off while its pattern is
    # plainly visible. An occlusion has to LAST to be real: runs shorter than this are
    # treated as boundary chatter and the frames are kept. 1 = no hysteresis.
    min_occlusion_run: int = 3

    # --- Rejoin partial tracks of the same feature (pre-gate) ---------------------------
    # TAPNext is causal, and seeds are staggered across the block on purpose, so a seed
    # entering three-quarters of the way in covers only the final quarter. Another pass
    # tracked that same feature over the earlier frames, but under a different id -- and
    # spread selection then threw one of them away as a duplicate, which is why a track could
    # stop part-way through a shot whose pattern is visible throughout. Stitching runs before
    # the quality gate so the rejoined track is judged at its true length.
    stitch_passes: bool = True
    stitch_max_sep_px: float = 1.5   # two features do not sit this close for this long
    stitch_min_overlap: int = 4      # shared frames needed to trust the pairing
    stitch_max_gap: int = 2          # or join across at most this many missing frames
    # Average agreeing passes instead of picking one. Stitching JOINS -- it uses a donor only
    # where the primary is silent -- so where two passes both saw the feature, one opinion is
    # discarded and select_spread later deletes the donor as a duplicate. Those are four
    # independent estimates of the same point, and FWD/BWD accumulate their drift from
    # opposite ends of the shot, so averaging cancels part of it rather than picking a side.
    # Weighted 1/(1 + age/fuse_age_tau) by each estimate's distance from its OWN query frame
    # (ProTracker's variance-along-the-chain model).
    #
    # MEASURED 2026-08 and REJECTED. The arithmetic is right -- tools/check_fusion.py shows
    # 0.2729 -> 0.1559px against truth on constructed passes, beating an unweighted mean --
    # and it still does not help this pipeline:
    #
    #     shot     median      p90          worst track      fused
    #     lab02    0.10 =    0.53 -> 0.45   1.03 -> 2.66px    309 tracks
    #     lab03    0.04 =    0.06 =         0.09 -> 0.41px    215 tracks
    #
    # No median gain on either, and the WORST track -- the number a batch is judged by, see
    # bench/score_synth.py -- got worse on both. Two independent shots agreeing is why this
    # is a verdict and not noise. (lab02 also showed a 10x peak-locking spike that lab03 did
    # not reproduce; that was the single 2.66px mistrack, not a systematic bias.)
    #
    # The likely cause, for anyone revisiting: this runs BEFORE refine, so it averages two
    # COARSE 256px-guide estimates and hands pattern_refine a start point that can sit
    # between two features rather than on either. Fusing AFTER refine -- where the estimates
    # are already native-res and certainty is known per track -- is a different design and
    # the one worth trying. Do not simply re-enable this flag.
    fuse_passes: bool = False
    fuse_age_tau: float = 30.0       # frames at which an estimate is worth half a fresh one
    fuse_max_sep_px: float = 0.75    # tighter than stitch_max_sep_px: averaging needs SAME point

    # --- Quality-ranked, evenly-spread track selection (post-pass) ---
    # Collapses the 4x pass duplication, scores each surviving track on its OWN
    # trajectory (parallax-safe: no agreement-with-global-motion term), then greedily
    # accepts the strongest tracks that stay >= spread_min_dist_px apart -> an even
    # blanket instead of clumps. Count floats with footage: dense texture/parallax
    # clears the spacing test in more places. spread_min_dist_px is the only density dial.
    enable_spread_select: bool = True
    spread_min_dist_px: int = 40   # min px spacing between kept tracks (density dial)
    # Export a solve-ready set, not everything that survived. A 3DE camera solve wants on the
    # order of 100 long, clean, well-spread points -- handing over 1000+ just moves the work
    # to the artist. Best-scoring first, so the cap keeps the good ones.
    # The quality bar decides the count -- if 400 tracks clear it, export 400. This is only a
    # safety ceiling, not a target. Quality is earned by tracking better (a clean multi-frame
    # template, an upsampled peak, an iterated refine), not by discarding more.
    max_output_tracks: int = 600
    min_export_tracks: int = 40    # below this, top up from the best rejects and flag them
    # Final gate on deviation from a track's own smooth path, RELATIVE to the shot's median
    # (wobble scales with how much the camera moves, so an absolute px bar cannot travel).
    # This is the only signal measured to rank true error on real footage; certainty manages
    # -0.236 and `score` is +0.398, i.e. backwards. 0 = off. See track_filter.wobble_gate.
    wobble_rel: float = 0.0
    # Seed identity: correlation between a track's patch at its FIRST frame and its LAST.
    # The only test here that catches a point which slid SMOOTHLY off its feature -- such a
    # track has a sharp correlation peak every frame (certainty passes it), runs the whole
    # shot unbroken (score ranks it top) and never jitters (wobble sees nothing). Measured on
    # a real plate the drifter scored -0.055 while every other exported track scored 0.50 to
    # 0.99, so this is an outlier catch, not a ranking. Absolute rather than relative: "is
    # this the same patch" travels between plates in a way certainty does not. 0 = off.
    min_seed_identity: float = 0.25
    # Band-pass sigma applied to the frame before pattern matching. TM_CCOEFF_NORMED removes
    # a patch's mean but not its low-frequency shape, so on a defocused feature the smooth
    # ramp dominates the correlation and the peak goes broad and flat. Subtracting a blurred
    # copy leaves the mid-frequency detail that actually localises. 0 = off (match on the
    # plate as-is). Identity is still measured on unfiltered frames.
    refine_bandpass: float = 0.0
    # Holes a track may carry before it is cut into continuous runs. One or two long gaps is
    # an occluded track worth keeping whole; a dozen short ones is a marginal track that kept
    # losing lock, and it blinks on and off in the 3DE viewport. -1 = off.
    max_track_gaps: int = 2

    # --- Final quality gate ------------------------------------------------------------
    # _track_quality_score only RANKED tracks; nothing ever dropped a weak one, so a short or
    # scrappy track still shipped if there was room. These are absolute floors applied before
    # spread selection. All are scaled down on short shots and relaxed if they would empty
    # the export -- returning nothing is worse than returning too much.
    min_track_frames: int = 24        # visible frames a track must have; 0 = off
    min_track_span_frac: float = 0.0  # or a fraction of the shot length; 0 = off
    min_track_score: float = 0.35     # quality floor in [0,1]; 0 = off
    quality_gate_floor: int = 8       # never gate below this many surviving tracks
    # Spacing is measured ON SCREEN at this many evenly-spaced reference frames, not on each
    # track's lifetime MEAN position. With a moving camera two tracks can sit a few px apart
    # for most of the shot while their means are 40px+ apart -- the mean test passed both and
    # they clumped. 1 = effectively the old mean-only behaviour.
    spread_ref_frames: int = 5
    # Track coords are scaled to the FULL PLATE before spacing is measured, so a raw pixel
    # dial means something different per resolution: 40px is 2% of an HD width but ~1% of 4K
    # (~96 columns of tracks -> looks clumped) and ~0.7% of 6K. Scale the spacing by
    # plate_width / spread_ref_width so one slider value looks the same at any resolution.
    spread_scale_with_res: bool = True
    spread_ref_width: int = 1920   # the width spread_min_dist_px is quoted against
    # Spacing in TIME as well as space. Seeding is staggered now, but the 4 passes still all
    # begin at their own first frame, so start frames can still pile up. Cap how many tracks
    # may START within the same short window; the best-scoring ones win. 0 = off.
    spread_max_starts_per_window: int = 0   # 0 = unlimited (set with spread_start_window)
    spread_start_window: int = 8            # frames that counts as "the same start"
    # score weights (self-referential terms only; must not judge vs global motion)
    w_coverage: float = 0.5        # visible span / low internal gaps -> long, gapless
    w_smoothness: float = 0.3      # low own-path jitter -> steady, sub-pixel
    w_stability: float = 0.2       # small max single-frame jump -> no teleports

    # --- Bad-track filter (drop jittery/jumpy mistracks before export; plate px) ---
    filter_max_jump_px: float = 0.0     # drop track if any single-frame jump > this (0 = off)
    filter_max_jitter_px: float = 0.0   # drop track if mean |dvelocity| jitter > this (0 = off)

    # --- Moving-tile native re-track (post-selection, BEFORE pattern_refine) ---
    # TAPNext runs at 256px, so on a 4K plate the whole frame is squashed ~15x and the
    # coarse position lands several px off the real feature. NCC alone can't fix that
    # (its search box centres on the coarse point -> locks the wrong patch). Moving tiling
    # cuts a NATIVE 256px crop that follows the coarse path and re-runs TAPNext on it, so
    # the point lands on the real feature. Measured vs 17 manual 4K tracks: baseline 4.88px,
    # baseline+NCC 4.03px, moving-tile+NCC 1.30px. Non-destructive (see moving_tile_refine.py).
    enable_moving_tile: bool = True
    mt_window: int = 16             # frames per tile window before it re-centres on the guide
    # Windows used to butt-joint, each a fresh model call seeded from the previous window's
    # drifted end, so every seam put a small STEP into the path -- once per window, which is
    # precisely the regular ~16-frame beat seen in centre-2D. Overlapping and cross-fading
    # the seam removes the step, and with it the beat. 0 = the old butt-joint.
    mt_overlap: int = 4
    mt_edge_margin: int = 40        # keep the coarse guide this many px inside the tile edge
    # Edge tracking: keep refining a point right up to the frame border instead of trimming it
    # when the search box / tile clamps against the edge (edge tracks anchor lens/solve corners).
    mt_edge_track: bool = True

    # --- 3DE-style NCC + affine pattern refinement (full-res, post-selection) ---
    # TAPNext (256px) only centres the search box; a contrast patch tracked at native
    # resolution decides the sub-pixel position -> tracks stick to the pattern like 3DE.
    # Hybrid re-reference + trim-on-lost + affine motion (see app/pattern_refine.py).
    enable_pattern_refine: bool = True
    refine_patch_px: int = 31       # pattern box (odd); larger = more stable, less local
    refine_search_px: int = 24      # search radius around TAPNext coarse position
    # Adaptive search: a fixed radius is wrong at both ends. Too small and a fast-moving
    # feature falls OUTSIDE the box, so NCC returns the best match inside it -- the wrong
    # place -- and the point slides. Too large and rival peaks creep in. Grow the radius only
    # as fast as the point actually moves; cost is quadratic in radius, so it is capped.
    refine_search_max: int = 64      # ceiling; == refine_search_px disables adaptation
    refine_search_speed_k: float = 1.5   # px of extra radius per px/frame of local speed
    # Distinctiveness test. Repetitive detail -- bolts, rivets, window grids, tiles -- gives
    # NCC several near-equal answers, and taking the single best silently snapped the point
    # to the identical feature next door at high correlation, so nothing downstream noticed.
    # Reject the match when a rival peak scores >= this fraction of the winner. 1.0 = off.
    match_ambiguity_ratio: float = 0.90
    # Re-acquisition after an occlusion is where that matters most: search this fraction of
    # the normal radius around the neighbour-predicted position, so a rival further away
    # cannot win. 1.0 = search the full box (old behaviour).
    reacquire_search_frac: float = 0.5
    refine_ncc_lost: float = 0.60   # corr below this = lock lost -> trim the track here
    # Hysteresis (Schmitt trigger). refine_ncc_lost alone is a single hard edge, so on grainy
    # footage the correlation hovers around it and the point drops out for a frame here and
    # there while its pattern is plainly visible -- the track appears to flicker. Once locked,
    # hold on down to this lower bar; only below it is the lock genuinely lost. A frame held
    # this way is never used to re-grab the pattern. Set == refine_ncc_lost to disable.
    refine_ncc_hold: float = 0.45
    # Re-referencing is now the EXCEPTION, not the routine. At 0.85 against lost=0.60 the
    # pattern was re-grabbed on most frames of real footage (grain, motion blur, a slight
    # lighting shift all land in that band), which quietly turned a pattern tracker into an
    # incremental frame-to-frame one -- and those random-walk, so the point wandered smoothly
    # around the feature it seeded on. Keep the seeded pattern unless correlation is close to
    # collapsing; the anchor is matched every frame regardless (see pattern_refine).
    refine_ncc_reref: float = 0.68  # corr below this (but >= lost) = re-grab the pattern
    # translation default: with moving-tile placing the point accurately, affine's extra
    # rot/scale DoF only adds wobble on pan/translation-dominant plates (measured regression).
    # Set to "affine"/"euclidean" for shots with real camera roll or zoom.
    refine_motion: str = "translation"   # translation | euclidean | affine
    # Sub-pixel polish. _ecc_refine is proper iterative estimation against the real pixels,
    # but it only ever ran when refine_motion was NOT translation -- and translation is the
    # default, so the exported position was always a 3-sample curve fit. Running it in
    # translation mode too is what stops a track wobbling on a feature that never moved.
    refine_ecc_polish: bool = True
    # Quality bought with time, per the "despite the time" brief. All three sharpen the
    # correlation peak, which is what certainty measures -- so they raise the surviving track
    # COUNT as well as the accuracy, rather than trading one for the other.
    template_frames: int = 5      # frames averaged into the reference pattern (1 = old)
    refine_iterations: int = 3    # match -> polish -> re-match passes per frame (1 = old)
    refine_min_len: int = 8         # drop a refined track shorter than this many frames
    # Per-track policy. Everything above is ONE setting applied to every point in the shot,
    # so the crisp corner on a bolt head and the soft blob on a wall get the same 31px box,
    # the same translation-only motion and the same NCC bars -- and those two features want
    # opposite settings. With this on, each seed is measured (structure tensor, its own
    # scale, rival density) and tracked with parameters chosen for what it actually is.
    # Off by default so one binary produces baseline and treatment on the same footage in
    # the same run. See app/track_meta.py and tools/eval_refs.py.
    per_track_policy: bool = False

    # --- Re-acquisition after an occlusion --------------------------------------------
    # A lost lock used to END that side of the track, which is how points were lost to
    # occluders SAM3 never masked (poles, props, a hand). Instead, treat it as a candidate
    # occlusion: keep stepping and try to re-find the ORIGINAL anchor patch. The search
    # centre is predicted from nearby surviving tracks -- the one thing CoTracker does well
    # -- but the sub-pixel position still comes from native-res NCC, which is what actually
    # carries the accuracy here (measured 4.88 -> 4.03 -> 1.30px, see enable_moving_tile).
    reacquire_max_gap: int = 24        # frames to keep trying before giving up; 0 = off
    refine_ncc_reacquire: float = 0.75  # correlation vs the ANCHOR needed to call it the same point
    reacquire_neighbours: int = 8       # nearest tracks used to predict where it reappears
    # Post-gap segments that FAIL the anchor check are emitted as a separate track rather
    # than welded into one id: welding two different features is invisible until the solve
    # fails, so on doubt it splits.
    split_unverified_segments: bool = True

    # --- Carry the track past its ends -------------------------------------------------
    # A track starts where a seed entered and stops where its pass ran out; neither is a
    # statement about the feature, which is usually still visible either side. Extension
    # walks outward from the OUTER ends only (never into an internal gap -- that gap IS an
    # occlusion, and re-acquisition above is what crosses it, on evidence). It matches the
    # ORIGINAL anchor at the position the track's own velocity predicts, demands
    # refine_ncc_reacquire rather than the looser mid-track bar, verifies each frame
    # backwards, and STOPS at the first failure: skipping a bad frame is exactly how a point
    # walks through an occluder and reattaches on the far side.
    refine_extend: bool = True
    refine_extend_max: int = 48        # frames per end; 0 = off

    # --- Accuracy passes ---------------------------------------------------------------
    # Drift guard: re-referencing used to re-grab the patch at the CURRENT position with
    # nothing tying it to the original, so over a long shot the pattern random-walks off the
    # feature. A re-reference is now accepted only if it still correlates with the ANCHOR.
    refine_drift_floor: float = 0.55     # 0 = off (old unbounded re-referencing)
    refine_drift_check_every: int = 24   # also re-validate against the anchor this often
    # Forward-backward consistency: refine the segment, then re-run it backwards; a good
    # point returns to where it started. Self-referential, so parallax-safe (it never judges
    # a track against global motion). Roughly doubles refine time. 0 = off.
    refine_fb_max_px: float = 1.5
    # Gap-aware refine: a track that disappears (occlusion) and reappears is refined per
    # contiguous VISIBLE segment (each re-acquires its own reference patch) and reassembled
    # under ONE id -> the reappeared segment is kept, not trimmed by the pre-occlusion patch.
    refine_gap_aware: bool = True

    # --- Track replenishment (re-seeding) ---
    # A single frame-0 seed leaves the frame uncovered once those points exit (fast/low-angle
    # shots: the close FG sweeps out in a few frames -> FG goes untracked). The chunked path
    # already seeds FRESH features per window; this bounds the window to `reseed_every` frames
    # so new features are seeded at least that often (re-seeding), reusing the existing
    # seed->carry->merge->filter->gate->spread machinery (so re-seeded tracks are mover-gated
    # and quality-filtered like any other). 0 = off (keep VRAM-only chunking).
    enable_reseed: bool = True
    reseed_every: int = 30     # max frames between re-seeds (window cap); 0 = disable
    reseed_max_windows: int = 40  # safety cap on forced window count
    # ORGANIC per-window seed budget (prevents the fixed max_tracks * n_windows blow-up that
    # saturated the machine). Fresh seeds PER WINDOW are capped to a resolution-scaled density,
    # so the TOTAL across the clip = density * n_windows, and n_windows = frames/reseed_every ->
    # total scales linearly with the FRAME RANGE and with resolution, not a fixed multiplied N.
    # goodFeatures still returns fewer on low-texture frames (content-organic); max_tracks is
    # only the upper bound. Applies to the chunked/re-seed path (single-block seeds once).
    reseed_density_per_mp: float = 60.0  # fresh seeds per window per megapixel
    reseed_seed_floor: int = 64          # min fresh seeds per window (small frames)
    # How many entry times a window's fresh seeds are split across. TAPNext queries carry a
    # time index that was always 0, so every new track began on its window's first frame and
    # they arrived in visible bulk every reseed_every frames. 1 = that old behaviour.
    seed_stagger: int = 4

    # --- VRAM/RAM safety: temporal chunking + OOM downscale-retry + streamed decode ---
    chunks: int = 0            # 0 = auto (VRAM-estimated); >=1 forces that many chunks
    chunk_overlap: int = 24    # frames of overlap between consecutive chunks (chaining band)
    max_chunks: int = 6        # upper bound for auto chunk count
    oom_retry: bool = True     # on CUDA OOM, downscale the block and retry
    oom_scale_step: float = 0.7
    oom_scale_floor: float = 0.25
    stream_decode: str = "auto"  # auto | always | never  (per-window decode for huge clips)
    host_ram_frac: float = 0.5   # fraction of available RAM a full-clip decode may use


class BatchTrackerRunner:
    def __init__(self, cfg: RunnerConfig, on_status: StatusCB = None):
        self.cfg = cfg
        self.on_status = on_status
        self._stop = threading.Event()
        # Frame-range bookkeeping (set per shot in _run_impl).
        self._frame_offset = 0   # 0-based index of first tracked frame within the full clip
        self._orig_total = 0     # full clip length, for mask<->frame proportional mapping
        # VRAM auto-sizing: bytes of CUDA peak per (frame*Hs*Ws); measured after first chunk.
        self._cal = None
        self._MEM_PER_FPX_DEFAULT = 50.0  # conservative until calibrated
        # Per-track metadata (app/track_meta.py). _pass_seeds holds one (SeedFeat, kind) per
        # query row, keyed by pass name, until _merge_filter_export turns the surviving
        # columns into registry entries. Both are reset per shot in _run_impl.
        self._pass_seeds: Dict[str, List[Tuple[SeedFeat, str]]] = {}
        self.registry = TrackRegistry(enabled=bool(getattr(cfg, "per_track_policy", False)))

    def _resolve_frame_range(self, total: int) -> tuple[int, int]:
        """Return (fs, fe) 0-based half-open slice bounds from cfg.frame_start/end (1-based incl)."""
        fs = self.cfg.frame_start
        fe = self.cfg.frame_end
        fs0 = max(0, (int(fs) - 1)) if fs and int(fs) > 0 else 0
        fe0 = int(fe) if fe and int(fe) > 0 else total
        fe0 = min(fe0, total)
        if fs0 >= fe0:  # invalid/empty => full clip
            return 0, total
        return fs0, fe0

    def request_stop(self):
        self._stop.set()

    def _status(self, msg: str):
        if self.on_status:
            self.on_status(msg)

    def _resolve_videos(self) -> List[str]:
        all_mp4 = sorted([f for f in os.listdir(self.cfg.input_dir) if f.lower().endswith(".mp4")])
        if not all_mp4:
            return []
        if self.cfg.selected_files:
            sel = set(self.cfg.selected_files)
            return [f for f in all_mp4 if f in sel]
        return all_mp4

    def _scale_for(self, filename: str) -> float:
        if not self.cfg.selected_scales:
            return 1.0
        s = float(self.cfg.selected_scales.get(filename, 1.0))
        if s <= 0.0 or s > 1.0:
            return 1.0
        return s

    def _log_paths(self) -> tuple[str, str]:
        return (
            os.path.join(self.cfg.output_dir, "track_log.txt"),
            os.path.join(self.cfg.output_dir, "track_log.csv"),
        )

    def _append_log(self, txt_path: str, line: str):
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line.rstrip("\n") + "\n")

    def _append_csv(self, csv_path: str, row: List[str]):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        exists = os.path.isfile(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="\n") as f:
            if not exists:
                f.write(
                    "timestamp,shot,file,mode,scale,orig_res,scaled_res,frames,tracks,output_txt,seconds,status,message\n"
                )

            def esc(s: str) -> str:
                s = "" if s is None else str(s)
                if any(c in s for c in [",", '"', "\n", "\r"]):
                    s = '"' + s.replace('"', '""') + '"'
                return s

            f.write(",".join(esc(x) for x in row) + "\n")

    @staticmethod
    def _moving_average_1d(a: np.ndarray, win: int) -> np.ndarray:
        win = int(win)
        if win <= 1:
            return a
        if win % 2 == 0:
            win += 1
        pad = win // 2
        ap = np.pad(a, (pad, pad), mode="edge")
        ker = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(ap, ker, mode="valid")

    def _post_filter_tracks(self, x_all: np.ndarray, y_all: np.ndarray, vis: np.ndarray, diag: float) -> np.ndarray:
        T, N = x_all.shape
        if N == 0 or T < 2:
            return np.ones((N,), dtype=bool)

        vis_pair = vis[:-1, :] & vis[1:, :]
        dx = np.diff(x_all, axis=0)
        dy = np.diff(y_all, axis=0)

        dx_med = np.zeros((T - 1,), dtype=np.float32)
        dy_med = np.zeros((T - 1,), dtype=np.float32)
        for t in range(T - 1):
            m = vis_pair[t]
            if np.any(m):
                dx_med[t] = np.median(dx[t, m])
                dy_med[t] = np.median(dy[t, m])
            else:
                dx_med[t] = 0.0
                dy_med[t] = 0.0

        res = np.sqrt((dx - dx_med[:, None]) ** 2 + (dy - dy_med[:, None]) ** 2)

        vis_ratio = np.mean(vis, axis=0)
        inlier_thr = float(self.cfg.motion_residual_diag_frac) * float(diag)
        jump_thr = float(self.cfg.max_jump_diag_frac) * float(diag)

        inlier_ratio = np.zeros((N,), dtype=np.float32)
        max_jump = np.zeros((N,), dtype=np.float32)

        for j in range(N):
            m = vis_pair[:, j]
            if np.any(m):
                rj = res[m, j]
                inlier_ratio[j] = float(np.mean(rj < inlier_thr)) if rj.size else 0.0
                max_jump[j] = float(np.max(np.sqrt(dx[m, j] ** 2 + dy[m, j] ** 2))) if rj.size else 0.0
            else:
                inlier_ratio[j] = 0.0
                max_jump[j] = float("inf")

        keep = (
            (vis_ratio >= float(self.cfg.min_visibility_ratio))
            & (inlier_ratio >= float(self.cfg.min_motion_inlier_ratio))
            & (max_jump <= float(jump_thr))
        )
        return keep

    @staticmethod
    def _find_child_dir_case_insensitive(parent: str, child_name: str) -> str | None:
        if not parent or not os.path.isdir(parent) or not child_name:
            return None
        direct = os.path.join(parent, child_name)
        if os.path.isdir(direct):
            return direct
        want = child_name.lower()
        try:
            for d in os.listdir(parent):
                p = os.path.join(parent, d)
                if os.path.isdir(p) and d.lower() == want:
                    return p
        except Exception:
            return None
        return None

    def _resolve_mask_dir_for_shot(self, shot_name: str) -> tuple[str | None, str]:
        root = (self.cfg.mask_root_dir or "").strip()
        if not root:
            return None, "no mask_root"

        shot_dir = self._find_child_dir_case_insensitive(root, shot_name)
        if not shot_dir:
            return None, f"no shot folder in mask_root ({root}\\{shot_name})"

        masks_dir = self._find_child_dir_case_insensitive(shot_dir, (self.cfg.mask_subdir or 'masks'))
        if masks_dir:
            return masks_dir, f"mask_dir={masks_dir}"
        return shot_dir, f"mask_dir={shot_dir} (no 'masks' subfolder)"

    def _mask_region_from_gray(self, gray: np.ndarray, pol: str | None = None) -> np.ndarray:
        pol = (pol if pol is not None else (self.cfg.mask_polarity or "auto")).strip().lower()
        white_region = gray >= 128
        if pol == "white":
            return white_region
        if pol == "black":
            return ~white_region
        pct_white = float(np.mean(white_region))
        if pct_white > 0.5:
            return ~white_region
        return white_region

    def _resolve_auto_polarity(self, sample_paths: list[str], target_w: int, target_h: int) -> str:
        """Decide 'white'/'black' ONCE for the whole clip when polarity=='auto'.

        Per-mask auto flips near 50% white and, unioned across frames, blows the exclude
        region up to 100% (-> zero seeds). Deciding once from the clip-wide mean keeps the
        polarity consistent so the union stays meaningful.
        """
        whites = []
        for p in sample_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.ndim == 3:
                img = img.squeeze()
            whites.append(float(np.mean(img >= 128)))
        if not whites:
            return "white"
        # Mirror the original per-frame rule, but on the clip-wide mean (majority = background).
        return "black" if float(np.mean(whites)) > 0.5 else "white"

    def _list_mask_files(self, shot_name: str) -> tuple[list[str], str]:
        mask_dir, where = self._resolve_mask_dir_for_shot(shot_name)
        if not mask_dir:
            return [], where
        files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
        files.sort()
        if not files:
            return [], f"{where} | no png masks found"
        return [os.path.join(mask_dir, f) for f in files], where

    def _load_mask_union(self, shot_name: str, target_w: int, target_h: int) -> tuple[np.ndarray | None, str, int]:
        mask_paths, where = self._list_mask_files(shot_name)
        if not mask_paths:
            return None, where, 0

        max_samples = 300
        step = max(1, int(np.ceil(len(mask_paths) / float(max_samples))))
        sample_paths = mask_paths[::step]

        # Resolve 'auto' to a single polarity for the whole clip (never flip per-frame).
        pol = (self.cfg.mask_polarity or "auto").strip().lower()
        eff_pol = self._resolve_auto_polarity(sample_paths, target_w, target_h) if pol == "auto" else pol

        union_region = np.zeros((target_h, target_w), dtype=bool)
        used = 0
        for p in sample_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            if img.ndim == 3:
                img = img.squeeze()

            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            if img.ndim == 3:
                img = img.squeeze()

            union_region |= self._mask_region_from_gray(img, pol=eff_pol)
            used += 1

        pct = float(np.mean(union_region)) * 100.0
        info = f"{where} | masks={len(mask_paths)} sampled={used} union_mask={pct:.1f}% polarity={self.cfg.mask_polarity}->{eff_pol}"
        return union_region, info, len(mask_paths)

    @staticmethod
    def _drop_short_runs(occ: np.ndarray, min_run: int) -> np.ndarray:
        """Clear occluded runs shorter than min_run, per track (column) of a (T,N) array.

        Sub-pixel jitter around a mask boundary marks a point occluded for one frame here
        and there, which shows up as a track flickering on and off even though its pattern
        never went anywhere. A genuine crossing lasts several frames.
        """
        if min_run <= 1 or occ.size == 0:
            return occ
        out = occ.copy()
        T = out.shape[0]
        for j in range(out.shape[1]):
            col = out[:, j]
            if not col.any():
                continue
            t = 0
            while t < T:
                if not col[t]:
                    t += 1
                    continue
                s = t
                while t < T and col[t]:
                    t += 1
                if (t - s) < min_run:
                    col[s:t] = False      # too brief to be a real occlusion
        return out

    def _shrink_region(self, incl: np.ndarray, margin: int) -> np.ndarray:
        """Pull the keep-region IN by `margin` px (erode). SAM3 mattes routinely stop a few px
        short of the real silhouette, and a point seeded in that halo is stuck to the character
        and slides with it. Mirrors sam3_runner's mask_dilation_px, but applied at track time
        so it works on masks that already exist."""
        if margin <= 0 or incl is None or not incl.any():
            return incl
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        return cv2.erode(incl.astype(np.uint8), k).astype(bool)

    def _make_seed_inclusion_mask(self, shot_name: str, Ws: int, Hs: int) -> tuple[np.ndarray | None, str, int]:
        region, info, n_masks = self._load_mask_union(shot_name, Ws, Hs)
        mode = (self.cfg.mask_mode or "outside").strip().lower()

        if region is None:
            return None, info, n_masks

        margin = max(0, int(getattr(self.cfg, "mask_margin_px", 0) or 0))
        incl = region if mode == "inside" else ~region
        before = float(np.mean(incl)) * 100.0
        incl = self._shrink_region(incl, margin)
        incl_pct = float(np.mean(incl)) * 100.0
        extra = f" | margin={margin}px ({before:.1f}%->{incl_pct:.1f}%)" if margin > 0 else ""
        return incl, f"{info} | mode={mode} | seed_region={incl_pct:.1f}%{extra}", n_masks

    # Block sizes probed to find a feature's own size. The one whose corner response peaks is
    # the scale the feature actually lives at, which is what should be setting its pattern box
    # -- today one refine_patch_px is applied to a hairline corner and a soft blob alike.
    _SCALE_BLOCKS = (3, 5, 9, 15)

    def _seed_measurements(self, gray: np.ndarray, pts: np.ndarray,
                           lmin: np.ndarray, lmax: np.ndarray,
                           eig: np.ndarray) -> Tuple[List[SeedFeat], SeedStats]:
        """Measure what each surviving seed is sitting on.

        The eigenvalues are already in hand from the aperture test; this adds the three
        things that test does not need but a per-track policy does: the feature's own scale,
        how many near-identical rivals surround it, and its local contrast. Every threshold
        derived here is a percentile of THIS frame, never an absolute -- an absolute
        cornerness bar does not survive a change of exposure, codec or plate.
        """
        h, w = gray.shape[:2]
        xi = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, h - 1)

        # Eigenvector angle: which direction an edge-like feature is UNCONSTRAINED in.
        theta = np.arctan2(eig[yi, xi, 3], eig[yi, xi, 2])

        # Scale: response at each probe block size, take the argmax per seed.
        resp = np.stack([cv2.cornerMinEigenVal(gray, blockSize=b, ksize=3)[yi, xi]
                         for b in self._SCALE_BLOCKS], axis=1)
        scale_px = np.asarray(self._SCALE_BLOCKS, dtype=np.float32)[np.argmax(resp, axis=1)]

        # Density of rivals: repetitive detail (bolts, rivets, window grids, tiles) is where
        # NCC quietly snaps to the identical feature next door, so it needs measuring, not
        # assuming. One box filter over a strong-corner mask, sampled at the seeds.
        #
        # The bar has to be high. Most of a plate is flat, so the cornerness map is mostly
        # zeros and every percentile up to ~P95 IS zero -- a bar below that marks the whole
        # frame "strong" and every seed comes back saturated at 1.0, which is no measurement
        # at all. At P99 a rivet grid reads ~5x an isolated corner, which is the signal.
        lmin_map = np.minimum(np.abs(eig[:, :, 0]), np.abs(eig[:, :, 1]))
        thr = float(np.percentile(lmin_map, 99.0))
        if thr <= 0.0:
            thr = float(lmin_map.max()) * 0.05
        strong = (lmin_map >= max(thr, 1e-12)).astype(np.float32)
        density = cv2.boxFilter(strong, -1, (65, 65), normalize=True)[yi, xi]

        # Local contrast, to guard the near-zero-variance patch NCC cannot use at all.
        g32 = gray.astype(np.float32)
        m = cv2.boxFilter(g32, -1, (33, 33), normalize=True)
        m2 = cv2.boxFilter(g32 * g32, -1, (33, 33), normalize=True)
        contrast = np.sqrt(np.maximum(0.0, m2 - m * m))[yi, xi]

        stats = SeedStats(
            lmin_p25=float(np.percentile(lmin, 25.0)) if lmin.size else 0.0,
            lmin_p75=float(np.percentile(lmin, 75.0)) if lmin.size else 0.0,
            density_p90=float(np.percentile(density, 90.0)) if density.size else 1.0,
            scale_med=float(np.median(scale_px)) if scale_px.size else 0.0,
        )
        ratio = np.divide(lmin, lmax, out=np.zeros_like(lmax), where=lmax > 1e-12)
        feats = [
            SeedFeat(lmin=float(lmin[i]), lmax=float(lmax[i]), aniso=float(ratio[i]),
                     theta=float(theta[i]), scale_px=float(scale_px[i]),
                     local_density=float(density[i]), local_contrast=float(contrast[i]))
            for i in range(pts.shape[0])
        ]
        return feats, stats

    def _drop_edge_points(self, gray: np.ndarray, pts: np.ndarray
                          ) -> Tuple[np.ndarray, List[SeedFeat], SeedStats]:
        """Drop seeds sitting on 1-D features (aperture problem), and measure the survivors.

        A point on an edge -- a rope, a plate line, the border of a TV screen -- can only be
        localized ACROSS the edge, never ALONG it, so the NCC refine finds a response ridge
        instead of a peak and the point slides up and down the edge. The structure tensor's
        eigenvalue RATIO (lambda_min / lambda_max) measures exactly that: ~1 for a corner, ~0
        for a pure edge. It is contrast-invariant, so unlike goodFeaturesToTrack's relative
        qualityLevel it does not also discard faint-but-real corners.

        The eigenvalues used to be reduced to that keep/drop boolean and thrown away. They
        are the beginning of knowing what each track is sitting on, so they now come back
        with the points (see _seed_measurements and app/track_meta.py).
        """
        thr = float(getattr(self.cfg, "min_corner_anisotropy", 0.0) or 0.0)
        want_feats = bool(getattr(self.cfg, "per_track_policy", False))
        if (thr <= 0.0 and not want_feats) or pts.shape[0] == 0:
            return pts, [], SeedStats()
        # 6 channels: l1, l2, and the two eigenvectors. blockSize matches goodFeaturesToTrack.
        eig = cv2.cornerEigenValsAndVecs(gray, blockSize=5, ksize=3)
        h, w = gray.shape[:2]
        xi = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, h - 1)
        l1 = eig[yi, xi, 0]
        l2 = eig[yi, xi, 1]
        lmax = np.maximum(np.abs(l1), np.abs(l2))
        lmin = np.minimum(np.abs(l1), np.abs(l2))
        if thr > 0.0:
            ratio = np.divide(lmin, lmax, out=np.zeros_like(lmax), where=lmax > 1e-12)
            keep = ratio >= thr
            n_drop = int(pts.shape[0] - np.sum(keep))
            if n_drop:
                self._status(f"  seeds: dropped {n_drop}/{pts.shape[0]} edge-like point(s) "
                             f"(anisotropy < {thr:.2f}) - these slide along edges")
            pts, lmin, lmax = pts[keep], lmin[keep], lmax[keep]
        if not want_feats or pts.shape[0] == 0:
            return pts, [], SeedStats()
        feats, stats = self._seed_measurements(gray, pts, lmin, lmax, eig)
        return pts, feats, stats

    def _stagger_offsets(self, n_frames: int) -> List[int]:
        """Frame offsets within a block at which fresh seeds should enter.

        Every seed used to enter at offset 0 -- the TAPNext query format is (t, x, y) and the
        t was hardcoded to zero -- so with a re-seed window of N frames, a whole batch of new
        tracks began at frames 0, N, 2N... and none in between. In 3DE that reads as tracks
        arriving in bulk every N frames. Spreading the entry times uses a capability the model
        already has; 1 = the old behaviour.
        """
        k = max(1, int(getattr(self.cfg, "seed_stagger", 1) or 1))
        if k <= 1 or n_frames < 4:
            return [0]
        k = min(k, max(1, n_frames // 2))          # never denser than every other frame
        step = n_frames / float(k)
        # Offsets stay inside the block and always include 0 so frame-0 content is seeded.
        return sorted({min(n_frames - 2, int(round(i * step))) for i in range(k)})

    def _staggered_queries(self, frames: np.ndarray, seed_mask, total: int,
                           n_frames: int) -> Tuple[Optional[np.ndarray], List[Tuple[SeedFeat, str]]]:
        """Build a (1,N,3) TAPNext query array whose seeds ENTER at staggered times.

        The per-offset budget is `total` split across the offsets, so staggering redistributes
        the same seed budget over time instead of multiplying it. Each batch detects features
        on the frame it actually enters on, so it seeds what is visible THERE -- content that
        only appears mid-window now gets tracked, which seeding frame 0 alone could never do.

        Also returns one (SeedFeat, kind) per query row, in the same order. Classification
        happens HERE because it is relative to the detection frame's own distribution, which
        is only in hand at this point; the policy those classes imply is decided later, where
        the proxy-to-plate scale factor is known (see _merge_filter_export).
        """
        offsets = self._stagger_offsets(int(n_frames))
        # The SHOT's seed budget is authoritative, not the per-round one. Each round runs its
        # own goodFeaturesToTrack with a fresh min-distance allowance, so a naive
        # total/len(offsets) split would return more points in four rounds than in one --
        # staggering must redistribute the budget over time, never inflate it.
        remaining = int(total)
        rows: List[np.ndarray] = []
        # Ground already claimed by an earlier offset. Without this every offset re-detects
        # the SAME features -- the frames barely differ -- and we would emit one duplicate
        # track per offset for every point. Masking claimed ground means a later batch seeds
        # only what is genuinely new (content that has just entered or been revealed), which
        # is the whole reason to seed mid-window.
        taken = None
        rad = max(2, int(self.cfg.min_feature_dist) or 2)
        min_aniso = float(getattr(self.cfg, "min_corner_anisotropy", 0.0) or 0.0)
        seeds: List[Tuple[SeedFeat, str]] = []
        for oi, off in enumerate(offsets):
            if off >= frames.shape[0] or remaining <= 0:
                continue
            left = len(offsets) - oi
            per = max(1, int(round(remaining / float(left))))
            m = seed_mask
            if taken is not None:
                m = (~taken) if seed_mask is None else (seed_mask & ~taken)
            pts, feats, stats = self._detect_features(
                frames[off], mask=m, count=per,
                quality=self.cfg.feature_quality, min_dist=self.cfg.min_feature_dist,
            )
            if pts.shape[0] > remaining:
                pts, feats = pts[:remaining], feats[:remaining]
            if pts.shape[0] == 0:
                continue
            remaining -= int(pts.shape[0])
            # Pad rather than skip when measurement is off: the seeds list must stay index-
            # aligned with the query rows, or every track downstream inherits its neighbour's
            # policy -- a silent, plausible-looking wrong answer.
            if len(feats) == pts.shape[0]:
                seeds.extend((f, classify_seed(f, stats, min_aniso)) for f in feats)
            else:
                seeds.extend((SeedFeat(), "") for _ in range(pts.shape[0]))
            if taken is None:
                taken = np.zeros(frames.shape[1:3], dtype=bool)
            claim = taken.astype(np.uint8)
            for (px, py) in pts:
                cv2.circle(claim, (int(round(px)), int(round(py))), rad, 1, -1)
            taken = claim.astype(bool)
            q = np.zeros((pts.shape[0], 3), dtype=np.float32)
            q[:, 0] = float(off)
            q[:, 1] = pts[:, 0]
            q[:, 2] = pts[:, 1]
            rows.append(q)
        if not rows:
            return None, []
        return np.concatenate(rows, axis=0)[None], seeds

    def _detect_features(self, first_frame_bgr: np.ndarray, mask: np.ndarray | None,
                         count: int, quality: float, min_dist: int
                         ) -> Tuple[np.ndarray, List[SeedFeat], SeedStats]:
        gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
        m_uint8 = None
        if mask is not None:
            m_uint8 = (mask.astype(np.uint8) * 255)

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=count,
            qualityLevel=quality,
            minDistance=min_dist,
            mask=m_uint8,
            blockSize=5
        )
        if pts is None or len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float32), [], SeedStats()
        return self._drop_edge_points(gray, pts.reshape(-1, 2).astype(np.float32))

    def _apply_per_frame_mask_gating(
        self,
        shot_name: str,
        tracks_xy: np.ndarray,
        vis: np.ndarray,
        Ws: int,
        Hs: int,
        T_seg: int,
        global_T: int,
        start_frame: int,
        is_reverse: bool,
        want_occlusion: bool = False,
    ) -> tuple[np.ndarray, str]:
        """Returns (keep, info). With want_occlusion, `keep` is instead a per-FRAME (T,N)
        boolean of which samples are usable -- False where the mover mask covers the point --
        so a crossed track survives with a gap instead of being deleted outright."""
        def _all_ok(msg):
            shape = (T_seg, tracks_xy.shape[1]) if want_occlusion else (tracks_xy.shape[1],)
            return np.ones(shape, dtype=bool), msg

        if not self.cfg.enable_mask_gating:
            return _all_ok("mask gating disabled")

        mask_paths, where = self._list_mask_files(shot_name)
        if not mask_paths:
            return _all_ok("no masks for gating")

        N = int(tracks_xy.shape[1])
        inside_count = np.zeros((N,), dtype=np.int32)
        total_count = np.zeros((N,), dtype=np.int32)
        # Per-frame occlusion: True where the mover mask covers this point on this frame.
        occluded = np.zeros((T_seg, N), dtype=bool) if want_occlusion else None
        M = len(mask_paths)

        # Masks cover the FULL clip; tracked frames may be a sub-range. Map the local
        # frame -> full-clip frame (add offset) -> mask index (proportional to orig total).
        orig_total = int(self._orig_total) if int(self._orig_total) > 0 else int(global_T)
        # If SAM3 masked only the tracked sub-range, M ~= tracked length (< full clip):
        # masks then align 1:1 with the tracked (sliced) frames. Otherwise masks cover
        # the whole clip and we map proportionally over the full frame index.
        masks_are_subrange = M < orig_total

        def get_global_mask_idx(t: int) -> int:
            global_t = (start_frame - t) if is_reverse else (start_frame + t)
            global_t = max(0, min(global_t, global_T - 1))
            if masks_are_subrange:
                return max(0, min(global_t, M - 1))
            full_frame = int(self._frame_offset) + global_t
            full_frame = max(0, min(full_frame, orig_total - 1))
            if orig_total <= 1 or M <= 1: return 0
            return int(round(full_frame * (M - 1) / float(orig_total - 1)))

        mode = (self.cfg.mask_mode or "outside").strip().lower()

        # Resolve 'auto' polarity ONCE for the clip. Deciding per frame flips near 50% white
        # and makes the gate mean different things on different frames -- the exact failure
        # _resolve_auto_polarity exists to prevent (the seeding path already does this).
        pol = (self.cfg.mask_polarity or "auto").strip().lower()
        if pol == "auto":
            step_s = max(1, int(np.ceil(M / 300.0)))
            pol = self._resolve_auto_polarity(mask_paths[::step_s], Ws, Hs)

        # Same edge margin as the seeding mask. 'outside' gates on the mover region, so the
        # margin GROWS it; 'inside' gates on the keep region, so the margin SHRINKS it. Either
        # way the boundary band stops counting as safe, and a track that DRIFTS into the halo
        # is dropped -- which seeding-only filtering would miss.
        margin = max(0, int(getattr(self.cfg, "mask_margin_px", 0) or 0))
        mk = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
              if margin > 0 else None)

        for t in range(T_seg):
            if self._stop.is_set(): break
            p = mask_paths[get_global_mask_idx(t)]
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue

            if gray.ndim == 3: gray = gray.squeeze()
            if gray.shape[1] != Ws or gray.shape[0] != Hs:
                gray = cv2.resize(gray, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
            if gray.ndim == 3: gray = gray.squeeze()

            region = self._mask_region_from_gray(gray, pol=pol)
            if mk is not None:
                r8 = region.astype(np.uint8)
                region = (cv2.dilate(r8, mk) if mode == "outside"
                          else cv2.erode(r8, mk)).astype(bool)
            vt = vis[t]
            if not np.any(vt): continue

            xs = np.rint(tracks_xy[t, :, 0]).astype(np.int32)
            ys = np.rint(tracks_xy[t, :, 1]).astype(np.int32)
            xs = np.clip(xs, 0, Ws - 1)
            ys = np.clip(ys, 0, Hs - 1)

            in_region = region[ys, xs]
            total_count[vt] += 1
            inside_count[vt & in_region] += 1
            if occluded is not None:
                # Mark the sample occluded whether or not the tracker thought it was visible:
                # the mask is the authority on where the character is, and TAPNext frequently
                # keeps reporting a covered point as visible (it locks onto the character).
                occluded[t] = in_region

        if int(np.max(total_count)) <= 0:
            return _all_ok(f"{where} | gating: no usable mask frames")

        marg = f" margin={margin}px" if margin > 0 else ""
        if mode == "outside":
            if occluded is not None:
                min_run = max(1, int(getattr(self.cfg, "min_occlusion_run", 1) or 1))
                raw_frames = int(np.sum(occluded))
                occluded = self._drop_short_runs(occluded, min_run)
                de_flickered = raw_frames - int(np.sum(occluded))
                usable = ~occluded
                n_gapped = int(np.sum(np.any(occluded, axis=0)))
                n_dead = int(np.sum(np.sum(usable, axis=0) == 0))
                extra = (f", ignored {de_flickered} 1-2 frame boundary flicker(s)"
                         if de_flickered else "")
                return usable, (f"{where} | gating(outside): {n_gapped}/{N} track(s) occluded "
                                f"for part of the shot -> kept with a gap, {n_dead} fully "
                                f"covered{extra}{marg}")
            keep = inside_count == 0
            kept = int(np.sum(keep))
            return keep, f"{where} | gating(outside): kept={kept}/{N} (dropped if ever inside){marg}"

        ratio = np.zeros((N,), dtype=np.float32)
        nz = total_count > 0
        ratio[nz] = inside_count[nz].astype(np.float32) / total_count[nz].astype(np.float32)
        thr = float(self.cfg.inside_ratio)
        keep = ratio >= thr
        kept = int(np.sum(keep))
        info = f"{where} | gating(inside): kept={kept}/{N} (inside_ratio>={thr:.2f}){marg}"
        if want_occlusion:
            # 'inside' semantics are unchanged -- an object track SHOULD stay on its object,
            # so this stays a whole-track decision. Broadcast it so the caller has one shape.
            return np.broadcast_to(keep, (T_seg, N)).copy(), info
        return keep, info

    def run(self):
        try:
            self._run_impl()
            self._status("Stopped." if self._stop.is_set() else "Done.")
        except Exception as e:
            self._status(f"Error: {e}")

    def _process_single_pass(
        self,
        engine: TapNextEngine,
        frames: np.ndarray,
        shot_name: str,
        Ws: int,
        Hs: int,
        mask_info: str,
        seed_mask: np.ndarray | None,
        n_masks: int,
        is_reverse: bool,
        global_T: int,
        start_frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        
        torch.cuda.empty_cache()
        pass_name = "BWD" if is_reverse else "FWD"
        T_seg = frames.shape[0]

        if T_seg < 2:
            empty_xy = np.zeros((T_seg, 0, 2), dtype=np.float32)
            empty_vis = np.zeros((T_seg, 0), dtype=bool)
            empty_gate = np.ones((0,), dtype=bool)
            return empty_xy, empty_vis, empty_gate, f"[{pass_name}] Skipped (Segment < 2 frames)"
        
        # 1. SEEDING
        seeds: List[Tuple[SeedFeat, str]] = []
        if self.cfg.seeding_mode == "features":
            q, seeds = self._staggered_queries(frames, seed_mask, self.cfg.max_tracks, T_seg)
            if q is None or q.shape[1] < 5:
                # Grid fallback seeds on its own rules, so the measured features describe
                # points that are not being tracked. Drop them rather than mis-attribute.
                seeds = []
                tracks_xy, vis = engine.track_grid(frames, grid_size=int(self.cfg.grid_size), segm_mask=seed_mask)
            else:
                tracks_xy, vis = engine.track_queries(frames, q)
        else:
            tracks_xy, vis = engine.track_grid(frames, grid_size=int(self.cfg.grid_size), segm_mask=seed_mask)
        # Query row order is the track column order all the way to _merge_filter_export, so
        # the seed a column came from is its index -- no matching needed. Keyed by pass name
        # because the (xy, vis, gate) tuple those columns travel in is deliberately unchanged.
        self._pass_seeds[pass_name] = seeds
            
        N = int(tracks_xy.shape[1])
        
        # 2. MASK GATING
        occl = bool(getattr(self.cfg, "occlusion_continuity", True))
        gate_keep = np.ones((T_seg, N) if occl else (N,), dtype=bool)
        gate_msg = "no gating"
        if n_masks > 0 and self.cfg.enable_mask_gating and N > 0:
            gate_keep, gate_msg = self._apply_per_frame_mask_gating(
                shot_name, tracks_xy, vis.astype(bool), Ws, Hs,
                T_seg=T_seg, global_T=global_T, start_frame=start_frame, is_reverse=is_reverse,
                want_occlusion=occl,
            )

        return tracks_xy, vis, gate_keep, f"[{pass_name}] N={N} {gate_msg}"

    def _merge_filter_export(self, passes, T: int, W0: int, H0: int, diag: float, inv: float):
        """Convert passes -> filtered/gated/smoothed final tracks at original coords.

        `passes`: list of (name, xy(T,N,2) processed-coords, vis(T,N) bool, gate_keep(N) bool).
        `inv` scales processed coords -> original (single-block: 1/scale; chunked: already
        original, pass 1.0). Returns (final_tracks_out, kept, candidates, after_filter,
        after_gate, short) so the caller can write + explain 0-track shots uniformly.
        """
        final_tracks_out: Dict[str, List[Tuple[int, float, float]]] = {}
        total_kept = total_candidates = 0
        diag_after_filter = diag_after_gate = diag_short = 0

        # Pool every filtered+gated survivor from all passes, THEN dedup/score/spread once
        # (below). Passes seed independently on the same frame, so the raw union carries
        # up to ~4x duplicates of the same physical point -- the spread pass collapses them.
        candidates: List[dict] = []

        for p_name, xy_raw, vis_raw, gate_mask in passes:
            if xy_raw is None or xy_raw.shape[1] == 0: continue
            N = xy_raw.shape[1]
            total_candidates += N

            x_all = (xy_raw[:, :, 0].astype(np.float32) * float(inv))
            y_all = (xy_raw[:, :, 1].astype(np.float32) * float(inv))
            # If the tracked mp4 was a downscaled proxy, map coords up to the true plate
            # resolution so they overlay the full plate (else they'd sit in a corner).
            ow = int(getattr(self.cfg, "out_w", 0) or 0)
            oh = int(getattr(self.cfg, "out_h", 0) or 0)
            if ow > 0 and oh > 0 and W0 > 0 and H0 > 0:
                x_all = x_all * (ow / float(W0))
                y_all = y_all * (oh / float(H0))
                flip_h = float(oh)
            else:
                flip_h = float(H0)
            if self.cfg.flip_y_for_3de and flip_h > 0: y_all = (flip_h - 1.0) - y_all

            # Pre-mask any immediate NaNs/Infs that the tracker leaked
            vis_bool = vis_raw.astype(bool) & ~np.isnan(x_all) & ~np.isnan(y_all) & ~np.isinf(x_all) & ~np.isinf(y_all)

            # Occlusion continuity: a (T,N) gate marks the frames the mover mask covers, so
            # the point simply goes INVISIBLE there and the track survives with a gap. The
            # old (N,) gate deleted the whole track for one crossing. Applied BEFORE the
            # motion filter so the occluded samples can't be read as a jump either.
            if gate_mask is not None and getattr(gate_mask, "ndim", 1) == 2:
                vis_bool = vis_bool & gate_mask.astype(bool)
                gate_mask = None

            keep = np.ones((N,), dtype=bool)
            if self.cfg.enable_filtering and N > 0:
                keep = self._post_filter_tracks(x_all, y_all, vis_bool, diag=diag)
            diag_after_filter += int(np.sum(keep))

            if gate_mask is not None:
                keep = keep & gate_mask
            # A fully-covered track has nothing left; drop anything too short to be useful.
            min_pts = max(2, int(getattr(self.cfg, "min_track_points", 8) or 2))
            keep = keep & (np.sum(vis_bool, axis=0) >= min_pts)
            diag_after_gate += int(np.sum(keep))
            kept_idx = np.where(keep)[0]
            win = max(1, int(self.cfg.smooth_window or 1))
            if win % 2 == 0: win += 1

            # Seeds were measured on the tracked proxy; the pattern box they imply is applied
            # by pattern_refine at NATIVE resolution. Carry the same factor the coordinates
            # just took, or a 4K plate tracked at half res gets boxes half the size it needs.
            pass_seeds = self._pass_seeds.get(p_name, [])
            px_scale = float(inv)
            if ow > 0 and W0 > 0:
                px_scale *= ow / float(W0)

            for j in kept_idx.tolist():
                xs, ys = x_all[:, j].copy(), y_all[:, j].copy()
                if win > 1:
                    xs, ys = self._moving_average_1d(xs, win), self._moving_average_1d(ys, win)

                out_id = f"{p_name}_{j+1:04d}"
                valid_pts = []
                sx = sy = 0.0
                for t in range(T):
                    if vis_bool[t, j]:
                        x_val, y_val = float(xs[t]), float(ys[t])
                        if not (np.isnan(x_val) or np.isnan(y_val) or np.isinf(x_val) or np.isinf(y_val)):
                            valid_pts.append((t + 1 + self._frame_offset, x_val, y_val))
                            sx += x_val; sy += y_val

                if len(valid_pts) > 1:
                    if self._bad_track(valid_pts):
                        diag_short += 1   # dropped: too jittery/jumpy
                        continue
                    n = len(valid_pts)
                    candidates.append({
                        "id": out_id,
                        "pts": valid_pts,
                        "mean": (sx / n, sy / n),
                        "score": self._track_quality_score(valid_pts, T, diag),
                        # Which pass produced this. stitch_passes needs it to know which END
                        # of the track was the query frame: the backward passes are seeded at
                        # the block's last frame, so their error grows toward frame 0.
                        "pass": p_name,
                    })
                    if self.registry.enabled and j < len(pass_seeds):
                        feat, kind = pass_seeds[j]
                        if kind and kind != "flat":
                            feat = replace(feat, scale_px=feat.scale_px * px_scale)
                            self.registry.register(out_id, TrackMeta(
                                seed_frame=valid_pts[0][0],
                                seed_xy=(valid_pts[0][1], valid_pts[0][2]),
                                pass_name=p_name, feat=feat, kind=kind,
                                policy=policy_for(kind, feat, self.cfg),
                            ))
                else:
                    diag_short += 1

        n_scored = len(candidates)
        # Join partial tracks of the SAME feature from different passes before anything
        # judges them on length. A seed entering late covers only the tail of the shot while
        # another pass holds the head; un-joined, the quality gate sees two short tracks and
        # spread selection then discards one of them as a duplicate.
        n_before_stitch = len(candidates)
        candidates = _tf.stitch_passes(candidates, T, diag, self.cfg, log=self._status)
        n_stitched = n_before_stitch - len(candidates)
        candidates = self._apply_quality_gate(candidates, T)
        n_after_quality = len(candidates)

        if self.cfg.enable_spread_select and candidates:
            # Spacing is measured in the same space the candidate coords are in: the full
            # plate (out_w when the tracked file was a downscaled proxy, else W0).
            ow = int(getattr(self.cfg, "out_w", 0) or 0)
            final_tracks_out = self._select_spread(candidates, out_width=(ow if ow > 0 else int(W0)))
        else:
            final_tracks_out = {c["id"]: c["pts"] for c in candidates}
        total_kept = len(final_tracks_out)

        # Per-stage accounting. Seven filters stack multiplicatively here, and a thin export
        # used to leave all of them as suspects. One line naming each stage's toll makes the
        # responsible one obvious instead of inferred.
        self._status(
            f"  tracks: {total_candidates} seeded -> {diag_after_filter} past motion filter "
            f"-> {diag_after_gate} past mask gate -> {n_scored} scored "
            f"({diag_short} too short/jumpy) -> {n_after_quality} past quality bar "
            f"-> {total_kept} after spacing"
            + (f", capped at {int(self.cfg.max_output_tracks)}"
               if self.cfg.max_output_tracks and total_kept >= int(self.cfg.max_output_tracks)
               else ""))
        return final_tracks_out, total_kept, total_candidates, diag_after_filter, diag_after_gate, diag_short

    def _apply_quality_gate(self, candidates: List[dict], T: int) -> List[dict]:
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
        if not candidates:
            return candidates
        need_f = int(getattr(self.cfg, "min_track_frames", 0) or 0)
        frac = float(getattr(self.cfg, "min_track_span_frac", 0.0) or 0.0)
        if frac > 0.0:
            need_f = max(need_f, int(round(frac * max(1, T))))
        # Never demand more than a quarter of the shot: on a short plate that would gate
        # everything away for a reason the artist cannot act on.
        if need_f > 0:
            need_f = max(4, min(need_f, int(0.25 * max(1, T)) or 4))
        need_s = float(getattr(self.cfg, "min_track_score", 0.0) or 0.0)
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

        floor = max(1, int(getattr(self.cfg, "quality_gate_floor", 8) or 1))
        if len(kept) < floor and len(candidates) > len(kept):
            kept = sorted(candidates, key=lambda c: c["score"], reverse=True)[:floor]
            self._status(f"  quality gate: only {len(kept)} track(s) cleared it -> relaxed, "
                         f"keeping the best {len(kept)} instead (short/poor shot)")
            return kept
        if short or weak:
            self._status(f"  quality gate: dropped {short} too-short (<{need_f}f) and "
                         f"{weak} low-quality (<{need_s:.2f}) of {len(candidates)} -> "
                         f"{len(kept)} worth exporting")
        return kept

    def _bad_track(self, pts: List[Tuple[int, float, float]]) -> bool:
        """True if a track is too jittery or jumpy (plate px), per the user thresholds.
        Self-consistency signals only (a fast but SMOOTH high-parallax point passes).
        Returns False when both thresholds are 0 (filter off) or the track is too short."""
        jmax = float(getattr(self.cfg, "filter_max_jump_px", 0.0) or 0.0)
        jit_thr = float(getattr(self.cfg, "filter_max_jitter_px", 0.0) or 0.0)
        if jmax <= 0.0 and jit_thr <= 0.0:
            return False
        n = len(pts)
        if n < 3:
            return False
        vels = []
        for i in range(n - 1):
            dt = max(1, pts[i + 1][0] - pts[i][0])
            d = math.hypot(pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2])
            vels.append(d / dt)
        max_jump = max(vels) if vels else 0.0
        jitter = (sum(abs(vels[k + 1] - vels[k]) for k in range(len(vels) - 1)) / (len(vels) - 1)
                  ) if len(vels) > 1 else 0.0
        if jmax > 0.0 and max_jump > jmax:
            return True
        if jit_thr > 0.0 and jitter > jit_thr:
            return True
        return False

    # --- Track selection ---------------------------------------------------------
    # The implementation lives in app/track_filter.py (pure Python, no torch) so the
    # SynthEyes backend can share it. These stay as thin delegates: one implementation
    # means the two backends cannot drift apart.

    def _track_quality_score(self, pts: List[Tuple[int, float, float]], T: int, diag: float) -> float:
        return _tf.score_track(pts, T, diag, self.cfg)

    def _apply_quality_gate(self, candidates: List[dict], T: int) -> List[dict]:
        return _tf.quality_gate(candidates, T, self.cfg, log=self._status)

    def _spacing_px(self, out_width: int = 0) -> int:
        return _tf.spacing_px(self.cfg, out_width)

    def _select_spread(self, candidates: List[dict],
                       out_width: int = 0) -> Dict[str, List[Tuple[int, float, float]]]:
        return _tf.select_spread(candidates, self.cfg, out_width=out_width, log=self._status)

    # -------------------------------------------------------------------------
    # VRAM/RAM-safe chunked tracking
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_oom(e: Exception) -> bool:
        try:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        return "out of memory" in str(e).lower()

    def _free_vram_bytes(self) -> int:
        try:
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return int(free)
        except Exception:
            pass
        return 0

    def _calibrate(self, frames: int, Hs: int, Ws: int):
        """Record measured CUDA peak per (frame*px) to refine later chunk sizing."""
        try:
            peak = int(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            unit = int(frames) * int(Hs) * int(Ws)
            if unit > 0 and peak > 0:
                self._cal = peak / float(unit)
        except Exception:
            pass

    def _auto_tune(self, source, fs0: int, T: int) -> None:
        """Measure this shot and write the derived parameters onto self.cfg.

        Best-effort by design: if anything here fails the shot simply tracks with the
        existing values. An auto-tuner that can break a batch is worse than one that
        occasionally declines to help.
        """
        try:
            from app import shot_profile as _sp
            n = int(min(6, max(2, T)))
            frames = source.get(fs0, n)
            prof = _sp.profile_shot(frames, flags=list(getattr(self.cfg, "quality_flags", []) or []))
            tuned = _sp.tune(prof, overrides=dict(getattr(self.cfg, "auto_tune_overrides", {}) or {}))
            changed = []
            for k, v in tuned.items():
                if hasattr(self.cfg, k) and getattr(self.cfg, k) != v:
                    changed.append(f"{k}={v}")
                setattr(self.cfg, k, v)
            self._status(f"  auto-tune: {prof.describe()}"
                         + (f" | flags: {', '.join(prof.flags)}" if prof.flags else ""))
            if changed:
                self._status(f"  auto-tune -> {', '.join(changed)}")
            for note in prof.notes:
                self._status(f"  auto-tune note: {note}")
        except Exception as e:
            self._status(f"  auto-tune skipped ({e}); using the configured values")

    def _probe_vram(self, engine, frames: np.ndarray, Ws: int, Hs: int) -> None:
        """Measure this card's real bytes-per-(frame*pixel) BEFORE chunking is decided.

        _MEM_PER_FPX_DEFAULT is documented "conservative until calibrated", but _calibrate
        only runs after the first real track_queries call while _decide_chunks runs before
        it -- so the first shot of every run was sized against the pessimistic guess and cut
        into more chunks than the card needs. More chunks means more seams and less temporal
        context, i.e. worse tracking. A few frames and a handful of points cost almost
        nothing and replace the guess with a measurement. Best-effort: any failure just
        leaves the old constant in place.
        """
        if self._cal and self._cal > 0:
            return                                  # already measured this run
        try:
            n = int(min(8, frames.shape[0]))
            if n < 2:
                return
            blk = frames[:n]
            q = np.zeros((1, 16, 3), dtype=np.float32)
            q[0, :, 1] = np.linspace(Ws * 0.2, Ws * 0.8, 16)
            q[0, :, 2] = np.linspace(Hs * 0.2, Hs * 0.8, 16)
            torch.cuda.reset_peak_memory_stats()
            engine.track_queries(blk, q)
            self._calibrate(n, Hs, Ws)
            torch.cuda.empty_cache()
            if self._cal:
                self._status(f"  VRAM probe: {self._cal:.1f} B per frame-pixel measured "
                             f"(was assuming {self._MEM_PER_FPX_DEFAULT:.0f})")
        except Exception as e:
            self._status(f"  VRAM probe skipped ({e}); using the conservative default")

    def _decide_chunks(self, T: int, Ws: int, Hs: int) -> int:
        # Manual override wins.
        if int(self.cfg.chunks) >= 1:
            return max(1, min(int(self.cfg.chunks), 64))
        measured = bool(self._cal and self._cal > 0)
        factor = self._cal if measured else self._MEM_PER_FPX_DEFAULT
        free = self._free_vram_bytes()
        if free <= 0 or Ws <= 0 or Hs <= 0:
            return 1
        # Lean harder on a MEASURED figure than on the guess; the OOM ladder
        # (oom_retry -> oom_scale_step -> oom_scale_floor) is still the safety net either way.
        budget = free * (0.85 if measured else 0.8)
        per_frame = float(Hs) * float(Ws) * float(factor)
        if per_frame <= 0:
            return 1
        max_frames = max(1, int(budget / per_frame))
        n = int(math.ceil(T / float(max_frames)))
        return max(1, min(n, int(self.cfg.max_chunks)))

    def _gate_assembled(self, shot: str, xy: np.ndarray, vis: np.ndarray, W0: int, H0: int, T: int) -> np.ndarray:
        """Mask-gate an assembled full-length sweep (coords in ORIGINAL, top-left orientation).
        Returns a per-frame (T,N) usable mask when occlusion continuity is on, else per-track."""
        N = int(xy.shape[1])
        occl = bool(getattr(self.cfg, "occlusion_continuity", True))
        if N == 0 or not self.cfg.enable_mask_gating:
            return np.ones((T, N) if occl else (N,), dtype=bool)
        keep, _msg = self._apply_per_frame_mask_gating(
            shot, xy, vis.astype(bool), W0, H0,
            T_seg=T, global_T=T, start_frame=0, is_reverse=False,
            want_occlusion=occl,
        )
        return keep

    def _chain_core(self, engine, fetch_block, shot: str, T: int, windows, W0: int, H0: int,
                    base_Ws: int, base_Hs: int):
        """Chain queries across overlapping windows (processing order).

        `fetch_block(proc_start, proc_count)` returns BGR frames already in processing order.
        Tracks that survive a window's overlap region are carried as queries into the next
        window, KEEPING their global id -> continuous tracks across seams. Returns
        (store, feats_by_gid) where store is {gid: {order_idx: (x_orig, y_orig)}}
        (order_idx = index in processing order).

        feats_by_gid matters more than it looks: re-seeding is on by default, so this is the
        path most real shots take. Here columns are keyed by global id rather than by query
        row, so without carrying the seed measurements alongside them the per-track policy
        would quietly apply to nothing on production footage.
        """
        store: Dict[int, Dict[int, Tuple[float, float]]] = {}
        feats_by_gid: Dict[int, Tuple[SeedFeat, str]] = {}
        next_gid = 0
        carry: Dict[int, Tuple[float, float]] = {}

        for wi, (ps, pc) in enumerate(windows):
            if self._stop.is_set():
                break
            block = fetch_block(ps, pc)
            if block is None or block.shape[0] < 2:
                carry = {}
                continue

            cur_scale = 1.0
            result = None
            while True:
                cur_Ws = max(1, int(round(base_Ws * cur_scale)))
                cur_Hs = max(1, int(round(base_Hs * cur_scale)))
                if cur_scale != 1.0:
                    blk = np.stack(
                        [cv2.resize(f, (cur_Ws, cur_Hs), interpolation=cv2.INTER_AREA) for f in block],
                        axis=0,
                    )
                else:
                    blk = block

                sx, sy = cur_Ws / float(W0), cur_Hs / float(H0)
                q_list: List[Tuple[float, float, float]] = []
                gid_list: List[int] = []
                # carried points (persist ids) at this window's first frame
                for gid, (xo, yo) in carry.items():
                    q_list.append((0.0, xo * sx, yo * sy))
                    gid_list.append(gid)
                # fresh features (new ids) for content that appears in this window.
                # ORGANIC per-window cap: resolution-scaled density, not a fixed count -> total
                # seeds = cap * n_windows scales with frame range (n_windows = T/reseed_every),
                # so re-seeding REDISTRIBUTES a shot-sized budget over time instead of stacking
                # max_tracks per window. max_tracks stays only as the upper bound.
                area_mp = max(0.1, (cur_Ws * cur_Hs) / 1_000_000.0)
                fresh_cap = int(round(float(getattr(self.cfg, "reseed_density_per_mp", 60.0)) * area_mp))
                fresh_cap = max(int(getattr(self.cfg, "reseed_seed_floor", 64)),
                                min(int(self.cfg.max_tracks), fresh_cap))
                seed_mask, _info, _nm = self._make_seed_inclusion_mask(shot, cur_Ws, cur_Hs)
                # Stagger the entry times across the window instead of dumping every fresh
                # seed on its first frame -- that is what made tracks arrive in bulk every
                # reseed_every frames. The budget is split across the offsets, so this
                # redistributes seeds over time rather than adding more of them.
                fresh_q, fresh_seeds = self._staggered_queries(
                    blk, seed_mask, fresh_cap, int(blk.shape[0]))
                if fresh_q is not None:
                    # Seeds were measured at this window's working resolution, which an OOM
                    # retry may have shrunk. Put the scale into ORIGINAL pixels here, while
                    # the factor for THIS window is still in hand.
                    seed_scale = W0 / float(cur_Ws) if cur_Ws > 0 else 1.0
                    for qi, (qt, px, py) in enumerate(fresh_q[0]):
                        q_list.append((float(qt), float(px), float(py)))
                        if qi < len(fresh_seeds):
                            f, kind = fresh_seeds[qi]
                            feats_by_gid[next_gid] = (
                                replace(f, scale_px=f.scale_px * seed_scale), kind)
                        gid_list.append(next_gid); next_gid += 1

                if not q_list:
                    result = None
                    break
                queries = np.array(q_list, dtype=np.float32)[None]
                try:
                    txy, tvis = engine.track_queries(blk, queries)
                    self._calibrate(blk.shape[0], cur_Hs, cur_Ws)
                    result = (txy, tvis, gid_list, cur_Ws, cur_Hs)
                    break
                except Exception as e:
                    if (self._is_oom(e) and self.cfg.oom_retry
                            and cur_scale * self.cfg.oom_scale_step >= self.cfg.oom_scale_floor):
                        torch.cuda.empty_cache()
                        cur_scale *= self.cfg.oom_scale_step
                        self._status(f"OOM on {shot} chunk {wi+1}/{len(windows)} -> retry at scale {cur_scale:.2f}")
                        continue
                    raise

            if result is None:
                carry = {}
                continue
            txy, tvis, gid_list, cur_Ws, cur_Hs = result
            Tb = int(txy.shape[0])
            ox, oy = W0 / float(cur_Ws), H0 / float(cur_Hs)

            for qi, gid in enumerate(gid_list):
                d = store.setdefault(gid, {})
                for tb in range(Tb):
                    if tvis[tb, qi]:
                        oidx = ps + tb
                        if 0 <= oidx < T:
                            d[oidx] = (float(txy[tb, qi, 0]) * ox, float(txy[tb, qi, 1]) * oy)

            carry = {}
            if wi + 1 < len(windows):
                ns = windows[wi + 1][0]
                tb_at = ns - ps
                if 0 <= tb_at < Tb:
                    for qi, gid in enumerate(gid_list):
                        if tvis[tb_at, qi]:
                            carry[gid] = (float(txy[tb_at, qi, 0]) * ox, float(txy[tb_at, qi, 1]) * oy)

        return store, feats_by_gid

    @staticmethod
    def _assemble(store: Dict[int, Dict[int, Tuple[float, float]]], T: int, to_local
                  ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Returns (xy, vis, cols); cols[i] is the global id of column i, which is what lets
        a caller line the per-seed measurements back up with the assembled columns."""
        cols = [gid for gid, d in store.items() if len(d) >= 1]
        N = len(cols)
        xy = np.zeros((T, N, 2), dtype=np.float32)
        vis = np.zeros((T, N), dtype=bool)
        for ci, gid in enumerate(cols):
            for oidx, (x, y) in store[gid].items():
                t = to_local(oidx)
                if 0 <= t < T:
                    xy[t, ci, 0] = x; xy[t, ci, 1] = y; vis[t, ci] = True
        return xy, vis, cols

    def _track_chunked(self, engine, source: FrameSource, shot: str, fs0: int, T: int,
                       n_chunks: int, W0: int, H0: int, base_Ws: int, base_Hs: int):
        """FWD-chain + BWD-chain over overlapping windows; returns merge-ready passes
        with coords already in ORIGINAL resolution (caller uses inv=1.0)."""
        ov = max(0, int(self.cfg.chunk_overlap))
        win = max(1, int(math.ceil(T / float(n_chunks))))
        windows = []
        k = 0
        while k * win < T:
            ls = k * win
            le = min(T, ls + win + ov)
            windows.append((ls, le - ls))
            k += 1

        passes = []

        def _seeds_for(cols: List[int], feats: Dict[int, Tuple[SeedFeat, str]]
                       ) -> List[Tuple[SeedFeat, str]]:
            # One entry per assembled column, in column order -- an unmeasured carried point
            # gets a blank rather than being skipped, so the list can never slip out of step
            # and hand a track its neighbour's policy.
            return [feats.get(g, (SeedFeat(), "")) for g in cols]

        # Forward
        fwd_fetch = lambda ps, pc: source.get(fs0 + ps, pc)
        store_f, feats_f = self._chain_core(engine, fwd_fetch, shot, T, windows, W0, H0, base_Ws, base_Hs)
        fxy, fvis, cols_f = self._assemble(store_f, T, lambda o: o)
        self._pass_seeds["FWD"] = _seeds_for(cols_f, feats_f)
        passes.append(("FWD", fxy, fvis, self._gate_assembled(shot, fxy, fvis, W0, H0, T)))

        # Backward (reverse processing order: proc frame o -> actual local (T-1-o))
        if self.cfg.bidirectional and not self._stop.is_set():
            rev_fetch = lambda ps, pc: source.get(fs0 + (T - (ps + pc)), pc)[::-1].copy()
            store_b, feats_b = self._chain_core(engine, rev_fetch, shot, T, windows, W0, H0, base_Ws, base_Hs)
            bxy, bvis, cols_b = self._assemble(store_b, T, lambda o: (T - 1 - o))
            self._pass_seeds["BWD"] = _seeds_for(cols_b, feats_b)
            passes.append(("BWD", bxy, bvis, self._gate_assembled(shot, bxy, bvis, W0, H0, T)))

        return passes

    def _run_impl(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        seq_mode = bool(self.cfg.sequence_path)
        if seq_mode:
            vids = [self.cfg.sequence_name or os.path.basename(self.cfg.sequence_path.rstrip("/\\"))]
        else:
            vids = self._resolve_videos()
            if not vids:
                raise RuntimeError("No .mp4 files found (or none selected).")

        txt_log, csv_log = self._log_paths()
        ts0 = time.strftime("%Y-%m-%d %H:%M:%S")
        self._append_log(
            txt_log,
            f"===== Batch start {ts0} | mode={self.cfg.seeding_mode} | bidir={self.cfg.bidirectional} | flip_y={self.cfg.flip_y_for_3de} =====",
        )

        tool_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._status("Loading TAPNext++ (256px, streaming)...")
        engine = TapNextEngine(tool_root=tool_root, device="cuda")

        for i, fn in enumerate(vids, start=1):
            if self._stop.is_set(): break
            shot_start = time.time()
            in_path = self.cfg.sequence_path if seq_mode else os.path.join(self.cfg.input_dir, fn)
            shot = fn if seq_mode else os.path.splitext(fn)[0]
            scale = self._scale_for(fn)
            # Per-track metadata is per shot: ids repeat across shots (FWD_0001 every time),
            # so carrying a registry over would hand shot 2 shot 1's measurements.
            self._pass_seeds = {}
            self.registry = TrackRegistry(
                enabled=bool(getattr(self.cfg, "per_track_policy", False)))

            try:
                # Decide host-RAM streaming (auto: only when a full decode would be too big).
                sd = (self.cfg.stream_decode or "auto").strip().lower()
                if sd == "always":
                    stream = True
                elif sd == "never":
                    stream = False
                else:
                    try:
                        import psutil  # type: ignore
                        vm = psutil.virtual_memory()
                        avail = int(vm.available)
                        need = int(estimate_clip_bytes(in_path, scale))
                        # A fixed 0.5 is over-cautious on a big machine: holding the whole
                        # clip lets moving-tile and refine share ONE decode, which is both
                        # faster and more accurate than streaming. Scale the allowance with
                        # total RAM, keeping the conservative fraction on small machines.
                        frac = float(self.cfg.host_ram_frac)
                        total_gb = vm.total / (1024.0 ** 3)
                        if total_gb >= 96:
                            frac = max(frac, 0.75)
                        elif total_gb >= 48:
                            frac = max(frac, 0.6)
                        stream = need > avail * frac
                    except Exception:
                        stream = False

                self._status(f"[{i}/{len(vids)}] Reading: {fn} (scale={scale}, stream={stream})")
                source = FrameSource(in_path, scale=scale, stream=stream)
                meta = source.meta

                orig_total = int(source.total)
                fs0, fe0 = self._resolve_frame_range(orig_total)
                self._frame_offset = fs0
                self._orig_total = orig_total
                T = fe0 - fs0
                if T <= 1: continue
                if (fs0, fe0) != (0, orig_total):
                    self._status(f"[{i}/{len(vids)}] Frame range: {fs0 + 1}-{fe0} of {orig_total}")

                W0, H0 = int(source.w0), int(source.h0)
                Ws, Hs = int(source.scaled_w), int(source.scaled_h)
                diag = float(np.sqrt(float(W0 * W0 + H0 * H0))) if (W0 > 0 and H0 > 0) else 1000.0

                # Read THIS shot and set the tracking parameters from it. A batch tool cannot
                # be hand-tuned per shot, so the alternative was one set of constants for
                # every plate -- which is why a handheld shot with a defocused background
                # tracked badly with settings that suit a sharp locked-off one.
                if bool(getattr(self.cfg, "auto_tune", False)):
                    self._auto_tune(source, fs0, T)

                # Measure the card BEFORE sizing chunks, so the first shot of a run isn't
                # cut up against the conservative constant (see _probe_vram).
                if not (self._cal and self._cal > 0) and int(self.cfg.chunks) < 1:
                    try:
                        self._probe_vram(engine, source.get(fs0, min(8, T)), Ws, Hs)
                    except Exception:
                        pass
                n_chunks = self._decide_chunks(T, Ws, Hs)
                # Re-seeding: cap the window to reseed_every frames so fresh features are
                # seeded at least that often (the chunked path seeds per window). Only ever
                # INCREASES the window count vs the VRAM decision, so headroom is preserved.
                reseed_msg = ""
                if getattr(self.cfg, "enable_reseed", True) and int(getattr(self.cfg, "reseed_every", 0) or 0) > 0:
                    every = int(self.cfg.reseed_every)
                    n_reseed = int(math.ceil(T / float(max(1, every))))
                    n_reseed = min(n_reseed, int(getattr(self.cfg, "reseed_max_windows", 40) or 40))
                    if n_reseed > n_chunks:
                        n_chunks = n_reseed
                        reseed_msg = f" (re-seed every ~{every}f)"
                free_gb = self._free_vram_bytes() / (1024.0 ** 3)
                cal_msg = (f" cal={self._cal:.1f}B/frame-px" if (self._cal and self._cal > 0)
                           else " cal=default(uncalibrated)")
                self._status(f"[{i}/{len(vids)}] frames={T} res={Ws}x{Hs} freeVRAM={free_gb:.1f}GB"
                             f"{cal_msg} -> chunks={n_chunks}{reseed_msg}")

                log_f = log_b = log_mid_f = log_mid_b = ""

                if n_chunks <= 1:
                    # --- Single-block path (unchanged 4-pass behavior) ---
                    frames_fwd = source.get(fs0, T)
                    if int(frames_fwd.shape[0]) <= 1: continue
                    inv = 1.0 / float(scale) if float(scale) != 0.0 else 1.0

                    seed_mask, mask_info, n_masks = self._make_seed_inclusion_mask(shot, Ws, Hs)
                    self._status(f"[{i}/{len(vids)}] Mask: {mask_info}")

                    self._status(f"[{i}/{len(vids)}] Tracking Forward...")
                    tracks_xy_f, vis_f, keep_gate_f, log_f = self._process_single_pass(
                        engine, frames_fwd, shot, Ws, Hs, mask_info, seed_mask, n_masks,
                        is_reverse=False, global_T=T, start_frame=0
                    )

                    tracks_xy_b, vis_b, keep_gate_b, log_b = (None, None, None, "")
                    if self.cfg.bidirectional:
                        self._status(f"[{i}/{len(vids)}] Tracking Backward...")
                        frames_bwd = frames_fwd[::-1].copy()
                        tracks_xy_rev, vis_rev, keep_gate_b, log_b = self._process_single_pass(
                            engine, frames_bwd, shot, Ws, Hs, mask_info, seed_mask, n_masks,
                            is_reverse=True, global_T=T, start_frame=T-1
                        )
                        tracks_xy_b = tracks_xy_rev[::-1, :, :].copy()
                        vis_b = vis_rev[::-1, :].copy()

                    self._status(f"[{i}/{len(vids)}] Tracking Mid-Forward...")
                    mid_idx = T // 2
                    frames_mid_f = frames_fwd[mid_idx:].copy()
                    tr_mid_f, vis_mid_f, keep_gate_mid_f, log_mid_f = self._process_single_pass(
                        engine, frames_mid_f, shot, Ws, Hs, mask_info, seed_mask, n_masks,
                        is_reverse=False, global_T=T, start_frame=mid_idx
                    )

                    self._status(f"[{i}/{len(vids)}] Tracking Mid-Backward...")
                    frames_mid_b = frames_fwd[:mid_idx+1][::-1].copy()
                    tr_mid_b, vis_mid_b, keep_gate_mid_b, log_mid_b = self._process_single_pass(
                        engine, frames_mid_b, shot, Ws, Hs, mask_info, seed_mask, n_masks,
                        is_reverse=True, global_T=T, start_frame=mid_idx
                    )

                    self._status(f"[{i}/{len(vids)}] Merging & Filtering 4-Pass Results...")
                    N_mid_f = tr_mid_f.shape[1]
                    xy_mid_f_full = np.zeros((T, N_mid_f, 2), dtype=np.float32)
                    vis_mid_f_full = np.zeros((T, N_mid_f), dtype=bool)
                    if N_mid_f > 0:
                        xy_mid_f_full[mid_idx:, :, :] = tr_mid_f
                        vis_mid_f_full[mid_idx:, :] = vis_mid_f.astype(bool)

                    N_mid_b = tr_mid_b.shape[1]
                    xy_mid_b_full = np.zeros((T, N_mid_b, 2), dtype=np.float32)
                    vis_mid_b_full = np.zeros((T, N_mid_b), dtype=bool)
                    if N_mid_b > 0:
                        xy_mid_b_full[:mid_idx+1, :, :] = tr_mid_b[::-1, :, :]
                        vis_mid_b_full[:mid_idx+1, :] = vis_mid_b[::-1, :].astype(bool)

                    passes = [("FWD", tracks_xy_f, vis_f, keep_gate_f)]
                    if tracks_xy_b is not None: passes.append(("BWD", tracks_xy_b, vis_b, keep_gate_b))
                    passes.append(("MID_F", xy_mid_f_full, vis_mid_f_full, keep_gate_mid_f))
                    passes.append(("MID_B", xy_mid_b_full, vis_mid_b_full, keep_gate_mid_b))

                    (final_tracks_out, total_kept, total_candidates,
                     diag_after_filter, diag_after_gate, diag_short) = self._merge_filter_export(
                        passes, T=T, W0=W0, H0=H0, diag=diag, inv=inv)
                else:
                    # --- Chunked path: chained FWD/BWD sweeps over overlapping windows ---
                    self._status(f"[{i}/{len(vids)}] Tracking in {n_chunks} chained chunks (overlap {self.cfg.chunk_overlap})...")
                    passes = self._track_chunked(engine, source, shot, fs0, T, n_chunks, W0, H0, Ws, Hs)
                    # coords already at ORIGINAL resolution -> inv=1.0
                    (final_tracks_out, total_kept, total_candidates,
                     diag_after_filter, diag_after_gate, diag_short) = self._merge_filter_export(
                        passes, T=T, W0=W0, H0=H0, diag=diag, inv=1.0)
                    log_f = f"chunked x{n_chunks}"

                # Explain a 0-track result instead of silently writing an empty file.
                if total_kept == 0:
                    if total_candidates == 0:
                        why = ("no points seeded - goodFeaturesToTrack found <5 features AND grid "
                               "seeding produced nothing (mask may cover the whole frame, or footage "
                               "is flat/low-contrast)")
                    elif diag_after_filter == 0:
                        why = (f"all {total_candidates} seeded tracks rejected by motion/jitter post-filter "
                               f"(jump/residual outliers vs frame diagonal); try disabling filtering or "
                               f"raising thresholds")
                    elif diag_after_gate == 0:
                        mode = (self.cfg.mask_mode or 'outside').strip().lower()
                        reason = ("every track entered the mask region" if mode == "outside"
                                  else f"no track stayed inside the mask >= inside_ratio ({self.cfg.inside_ratio:.2f})")
                        why = (f"{diag_after_filter} tracks survived filtering but ALL dropped by SAM3 mask "
                               f"gating (mode={mode}: {reason}); check mask polarity/coverage for this shot")
                    else:
                        why = (f"{diag_after_gate} tracks survived filters+gating but had <2 visible points "
                               f"(occluded/left frame almost immediately)")
                    zmsg = f"{shot}: 0 tracks - {why}"
                    self._status(f"[{i}/{len(vids)}] {zmsg}")
                    self._append_log(txt_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ZERO {zmsg} "
                                              f"(seeded={total_candidates} after_filter={diag_after_filter} "
                                              f"after_gate={diag_after_gate} short={diag_short})")

                # --- Host-RAM: free the (scaled) tracking frames and build ONE native decode
                # shared by moving-tile + pattern-refine. Three concurrent full decodes of a
                # 4K/1440p clip (tracking + moving-tile BGR + refine gray) exhaust host RAM and
                # freeze the machine. When tracking already holds the NATIVE frames (scale 1.0,
                # full-decode), reuse them directly -> a single decode for the whole shot.
                refine_src = None
                if final_tracks_out and (getattr(self.cfg, "enable_moving_tile", True)
                                         or self.cfg.enable_pattern_refine):
                    if float(scale) == 1.0 and source is not None:
                        refine_src = source                       # already native (full or streamed) -> reuse
                    else:
                        # free the scaled tracking array + any single-block frames, then decode
                        # (or stream) the native clip ONCE for both refine stages.
                        try:
                            if source is not None and getattr(source, "_arr", None) is not None:
                                source._arr = None
                        except Exception:
                            pass
                        try:
                            frames_fwd = None  # single-block path holds this at scale
                        except Exception:
                            pass
                        gc.collect(); torch.cuda.empty_cache()
                        stream_native = False
                        try:
                            import psutil  # type: ignore
                            need = int(estimate_clip_bytes(in_path, 1.0))
                            budget = int(psutil.virtual_memory().available * float(self.cfg.host_ram_frac))
                            stream_native = need > budget
                        except Exception:
                            stream_native = False
                        sdn = (self.cfg.stream_decode or "auto").strip().lower()
                        if sdn == "always":
                            stream_native = True
                        elif sdn == "never":
                            stream_native = False
                        try:
                            refine_src = FrameSource(in_path, scale=1.0, stream=stream_native)
                        except Exception:
                            refine_src = None

                # Moving-tile native re-track: fix the coarse 256px position on the full-res
                # plate BEFORE NCC (NCC can't recover a position that started several px off).
                # Runs only on the handful of selected tracks; non-destructive (count/len kept).
                if getattr(self.cfg, "enable_moving_tile", True) and final_tracks_out:
                    try:
                        from app.moving_tile_refine import moving_tile_refine
                        self._status(f"[{i}/{len(vids)}] Moving-tile native re-track at {W0}x{H0}...")
                        final_tracks_out, minfo = moving_tile_refine(
                            final_tracks_out, in_path, W0, H0, orig_total, engine, self.cfg,
                            status=lambda m: self._status(f"TRACK: {m}"), src=refine_src,
                            registry=(self.registry if self.registry.enabled else None))
                        self._append_log(txt_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MOVINGTILE {shot}: {minfo}")
                    except Exception as e:
                        self._status(f"[{i}/{len(vids)}] Moving-tile skipped: {e}")

                # 3DE-style NCC/affine pattern lock at native resolution (post-selection,
                # so it runs only on the handful of spread-selected tracks -> cheap).
                if self.cfg.enable_pattern_refine and final_tracks_out:
                    try:
                        from app.pattern_refine import refine_tracks
                        self._status(f"[{i}/{len(vids)}] Pattern-refine (NCC/{self.cfg.refine_motion}) at {W0}x{H0}...")
                        _reg = self.registry if self.registry.enabled else None
                        final_tracks_out, rinfo = refine_tracks(
                            final_tracks_out, in_path, W0, H0, orig_total, self.cfg,
                            status=lambda m: self._status(f"TRACK: {m}"), bgr_source=refine_src,
                            registry=_reg)
                        # Certainty is only known once refine has measured the correlation
                        # peaks, so this gate cannot live with the others before it.
                        _certs = getattr(refine_tracks, "last_certainty", {}) or {}
                        # Identity FIRST, and on the full set -- a track that no longer sits
                        # on the feature it started on is not a track, so it must leave the
                        # backfill pool as well. Gating after the top-up would have achieved
                        # nothing: the pool is snapshotted before the certainty gate, so
                        # anything dropped here is simply re-added there. That is exactly how
                        # the first attempt at this failed, and how the backfill ceiling
                        # failed before it -- filtering one stage while the drifter comes
                        # back through another.
                        final_tracks_out = _tf.identity_gate(
                            final_tracks_out, getattr(refine_tracks, "last_identity", {}) or {},
                            self.cfg, log=self._status)
                        _before_gate = dict(final_tracks_out)
                        final_tracks_out = _tf.certainty_gate(
                            final_tracks_out, _certs, self.cfg, log=self._status)
                        # A handful of tracks and no explanation is not a usable delivery.
                        # Top a thin export back up from the best rejects, flagged.
                        #
                        # Measure wobble FIRST so the top-up can rank on it. On real footage
                        # it is the only signal here that predicts true error (spearman
                        # +0.573, against +0.291 for score, which is actively misleading --
                        # see backfill_to_floor). It costs one pass over tracks already in
                        # memory, and it decides which rejects an artist has to look at.
                        _wob = {}
                        try:
                            from app.pattern_refine import measure_wobble as _mw
                            _wob = {k: _mw(v)[0] for k, v in _before_gate.items()}
                        except Exception:
                            _wob = {}
                        final_tracks_out, _weak_ids = _tf.backfill_to_floor(
                            final_tracks_out, _before_gate, _certs, self.cfg,
                            log=self._status, wobble=_wob)
                        # Last, on the FINAL set: the one gate that catches a track which
                        # deviates far more than this shot's norm. It has to see everything,
                        # including what the top-up just added and what certainty let through
                        # -- a 20px track on SH004 passed the certainty gate and only this
                        # removes it. See track_filter.wobble_gate.
                        final_tracks_out = _tf.wobble_gate(
                            final_tracks_out, _wob, self.cfg, log=self._status)
                        _weak_ids = {k for k in _weak_ids if k in final_tracks_out}
                        # A track that kept losing and regaining lock exports as one id full
                        # of holes, which blinks on and off in the 3DE viewport. Cut those
                        # into continuous runs; genuine occlusion gaps stay.
                        final_tracks_out = _tf.defragment(final_tracks_out, self.cfg,
                                                          log=self._status, registry=_reg)
                        total_kept = len(final_tracks_out)
                        if _reg is not None:
                            self._status(f"  {_reg.summary()}")
                        self._append_log(txt_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REFINE {shot}: {rinfo}")
                    except Exception as e:
                        self._status(f"[{i}/{len(vids)}] Pattern-refine skipped: {e}")

                # Per-track numbers for THIS shot, beside the tracks. Eyeballing real footage
                # and passing synthetic tests kept disagreeing; this makes the next question
                # answerable from a column instead of another guess.
                if final_tracks_out and bool(getattr(self.cfg, "track_report", True)):
                    try:
                        from app.pattern_refine import measure_wobble, refine_tracks as _rt
                        rep = os.path.join(
                            self.cfg.output_dir,
                            f"{shot}{('__' + self.cfg.output_tag.strip()) if (self.cfg.output_tag or '').strip() else ''}"
                            f"__trackreport.csv")
                        # Epipolar agreement, measured once for the whole shot. This is the
                        # only number here that looks ACROSS tracks; everything else is
                        # deliberately self-referential so parallax is not punished, which is
                        # also why this one is reported and never gated -- a point on a real
                        # mover is off-epipolar and correct. See geometric_residuals.
                        _geo = _tf.geometric_residuals(final_tracks_out, W0, H0,
                                                       log=self._status)
                        if _geo and self.registry.enabled:
                            for _tid, _v in _geo.items():
                                _m = self.registry.get(_tid)
                                if _m is not None:
                                    _m.geo_residual = float(_v)
                        got = _tf.dump_track_report(
                            rep, final_tracks_out, getattr(_rt, "last_certainty", {}) or {},
                            T, W0, H0, self.cfg, wobble_fn=measure_wobble,
                            weak=locals().get("_weak_ids") or set(),
                            registry=(self.registry if self.registry.enabled else None),
                            geo=(_geo or None))
                        if got:
                            self._status(f"[{i}/{len(vids)}] Track report -> {os.path.basename(got)}")
                        # Full records beside the CSV: the CSV is for reading, this is for
                        # analysing a whole batch offline once the columns raise a question.
                        if self.registry.enabled:
                            self.registry.dump(rep[:-len("__trackreport.csv")] + "__trackmeta.json")
                    except Exception as e:
                        self._status(f"[{i}/{len(vids)}] Track report skipped: {e}")

                tag = (self.cfg.output_tag or '').strip()
                base = f"{shot}__tapnext.txt" if not tag else f"{shot}__{tag}__tapnext.txt"
                out_txt = os.path.join(self.cfg.output_dir, base)
                write_tracks_txt(out_txt, final_tracks_out, end_frame=T)

                refine_src = None            # release the shared native decode before next shot
                gc.collect(); torch.cuda.empty_cache()
                secs = time.time() - shot_start
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                msg = f"kept {total_kept}/{total_candidates} | F:{log_f} | B:{log_b} | MF:{log_mid_f} | MB:{log_mid_b}"
                self._append_log(txt_log, f"[{ts}] OK  {shot}: scale={scale} {msg} -> {os.path.basename(out_txt)} ({secs:.2f}s)")
                self._append_csv(csv_log, [ts, shot, fn, self.cfg.seeding_mode + ("+BIDIR+MID" if self.cfg.bidirectional else ""), str(scale), f"{W0}x{H0}", f"{Ws}x{Hs}", str(T), str(total_kept), out_txt, f"{secs:.3f}", "OK", msg])
                self._status(f"[{i}/{len(vids)}] Exported: {os.path.basename(out_txt)} ({total_kept} tracks)")

            except Exception as e:
                torch.cuda.empty_cache()
                secs = time.time() - shot_start
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                self._append_log(txt_log, f"[{ts}] FAIL {shot}: {e} ({secs:.2f}s)")
                self._append_csv(csv_log, [ts, shot, fn, str(self.cfg.grid_size), str(scale), "", "", "", "", "", f"{secs:.3f}", "FAIL", str(e)])
                self._status(f"[{i}/{len(vids)}] Error on {fn}: {e}")