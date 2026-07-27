# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import gc
import time
import math
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple

import numpy as np
import cv2  # type: ignore
import torch # Need torch for empty_cache

from app.video_io import read_video_frames_bgr_scaled, FrameSource, estimate_clip_bytes
from app.tapnext_engine import TapNextEngine
from app.export_3de import write_tracks_txt

StatusCB = Optional[Callable[[str], None]]


@dataclass
class RunnerConfig:
    input_dir: str
    output_dir: str
    
    seeding_mode: str = "features" 
    bidirectional: bool = True
    
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

    # Frame range (1-based inclusive). 0 = unset => full clip.
    frame_start: int = 0
    frame_end: int = 0

    mask_root_dir: str = ""
    mask_mode: str = "outside"
    mask_polarity: str = "auto"

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

    # --- Quality-ranked, evenly-spread track selection (post-pass) ---
    # Collapses the 4x pass duplication, scores each surviving track on its OWN
    # trajectory (parallax-safe: no agreement-with-global-motion term), then greedily
    # accepts the strongest tracks that stay >= spread_min_dist_px apart -> an even
    # blanket instead of clumps. Count floats with footage: dense texture/parallax
    # clears the spacing test in more places. spread_min_dist_px is the only density dial.
    enable_spread_select: bool = True
    spread_min_dist_px: int = 40   # min px spacing between kept tracks (density dial)
    max_output_tracks: int = 0     # soft cap; 0 = unlimited
    # score weights (self-referential terms only; must not judge vs global motion)
    w_coverage: float = 0.5        # visible span / low internal gaps -> long, gapless
    w_smoothness: float = 0.3      # low own-path jitter -> steady, sub-pixel
    w_stability: float = 0.2       # small max single-frame jump -> no teleports

    # --- Moving-tile native re-track (post-selection, BEFORE pattern_refine) ---
    # TAPNext runs at 256px, so on a 4K plate the whole frame is squashed ~15x and the
    # coarse position lands several px off the real feature. NCC alone can't fix that
    # (its search box centres on the coarse point -> locks the wrong patch). Moving tiling
    # cuts a NATIVE 256px crop that follows the coarse path and re-runs TAPNext on it, so
    # the point lands on the real feature. Measured vs 17 manual 4K tracks: baseline 4.88px,
    # baseline+NCC 4.03px, moving-tile+NCC 1.30px. Non-destructive (see moving_tile_refine.py).
    enable_moving_tile: bool = True
    mt_window: int = 16             # frames per tile window before it re-centres on the guide
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
    refine_ncc_lost: float = 0.60   # corr below this = lock lost -> trim the track here
    refine_ncc_reref: float = 0.85  # corr below this (but >= lost) = re-grab the pattern
    # translation default: with moving-tile placing the point accurately, affine's extra
    # rot/scale DoF only adds wobble on pan/translation-dominant plates (measured regression).
    # Set to "affine"/"euclidean" for shots with real camera roll or zoom.
    refine_motion: str = "translation"   # translation | euclidean | affine
    refine_min_len: int = 8         # drop a refined track shorter than this many frames
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

    def _make_seed_inclusion_mask(self, shot_name: str, Ws: int, Hs: int) -> tuple[np.ndarray | None, str, int]:
        region, info, n_masks = self._load_mask_union(shot_name, Ws, Hs)
        mode = (self.cfg.mask_mode or "outside").strip().lower()

        if region is None:
            return None, info, n_masks

        if mode == "inside":
            incl = region
            incl_pct = float(np.mean(incl)) * 100.0
            return incl, f"{info} | mode=inside | seed_region={incl_pct:.1f}%", n_masks
        else:
            incl = ~region
            incl_pct = float(np.mean(incl)) * 100.0
            return incl, f"{info} | mode=outside | seed_region={incl_pct:.1f}%", n_masks

    def _detect_features(self, first_frame_bgr: np.ndarray, mask: np.ndarray | None, 
                         count: int, quality: float, min_dist: int) -> np.ndarray:
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
            return np.zeros((0, 2), dtype=np.float32)
        return pts.reshape(-1, 2).astype(np.float32)

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
        is_reverse: bool
    ) -> tuple[np.ndarray, str]:
        if not self.cfg.enable_mask_gating:
            return np.ones((tracks_xy.shape[1],), dtype=bool), "mask gating disabled"

        mask_paths, where = self._list_mask_files(shot_name)
        if not mask_paths:
            return np.ones((tracks_xy.shape[1],), dtype=bool), "no masks for gating"

        N = int(tracks_xy.shape[1])
        inside_count = np.zeros((N,), dtype=np.int32)
        total_count = np.zeros((N,), dtype=np.int32)
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

        for t in range(T_seg):
            if self._stop.is_set(): break
            p = mask_paths[get_global_mask_idx(t)]
            gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            
            if gray.ndim == 3: gray = gray.squeeze()
            if gray.shape[1] != Ws or gray.shape[0] != Hs:
                gray = cv2.resize(gray, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
            if gray.ndim == 3: gray = gray.squeeze()

            region = self._mask_region_from_gray(gray)
            vt = vis[t]
            if not np.any(vt): continue

            xs = np.rint(tracks_xy[t, :, 0]).astype(np.int32)
            ys = np.rint(tracks_xy[t, :, 1]).astype(np.int32)
            xs = np.clip(xs, 0, Ws - 1)
            ys = np.clip(ys, 0, Hs - 1)

            in_region = region[ys, xs]
            total_count[vt] += 1
            inside_count[vt & in_region] += 1

        if int(np.max(total_count)) <= 0:
            return np.ones((N,), dtype=bool), f"{where} | gating: no usable mask frames"

        if mode == "outside":
            keep = inside_count == 0
            kept = int(np.sum(keep))
            return keep, f"{where} | gating(outside): kept={kept}/{N} (dropped if ever inside)"

        ratio = np.zeros((N,), dtype=np.float32)
        nz = total_count > 0
        ratio[nz] = inside_count[nz].astype(np.float32) / total_count[nz].astype(np.float32)
        thr = float(self.cfg.inside_ratio)
        keep = ratio >= thr
        kept = int(np.sum(keep))
        return keep, f"{where} | gating(inside): kept={kept}/{N} (inside_ratio>={thr:.2f})"

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
        if self.cfg.seeding_mode == "features":
            pts = self._detect_features(
                frames[0], 
                mask=seed_mask,
                count=self.cfg.max_tracks,
                quality=self.cfg.feature_quality,
                min_dist=self.cfg.min_feature_dist
            )
            if pts.shape[0] < 5:
                tracks_xy, vis = engine.track_grid(frames, grid_size=int(self.cfg.grid_size), segm_mask=seed_mask)
            else:
                N_pts = pts.shape[0]
                queries = np.zeros((1, N_pts, 3), dtype=np.float32)
                queries[0, :, 0] = 0.0
                queries[0, :, 1] = pts[:, 0]
                queries[0, :, 2] = pts[:, 1]
                tracks_xy, vis = engine.track_queries(frames, queries)
        else:
            tracks_xy, vis = engine.track_grid(frames, grid_size=int(self.cfg.grid_size), segm_mask=seed_mask)
            
        N = int(tracks_xy.shape[1])
        
        # 2. MASK GATING
        gate_keep = np.ones((N,), dtype=bool)
        gate_msg = "no gating"
        if n_masks > 0 and self.cfg.enable_mask_gating and N > 0:
            gate_keep, gate_msg = self._apply_per_frame_mask_gating(
                shot_name, tracks_xy, vis.astype(bool), Ws, Hs, 
                T_seg=T_seg, global_T=global_T, start_frame=start_frame, is_reverse=is_reverse
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

            keep = np.ones((N,), dtype=bool)
            if self.cfg.enable_filtering and N > 0:
                keep = self._post_filter_tracks(x_all, y_all, vis_bool, diag=diag)
            diag_after_filter += int(np.sum(keep))

            if gate_mask is not None:
                keep = keep & gate_mask
            diag_after_gate += int(np.sum(keep))
            kept_idx = np.where(keep)[0]
            win = max(1, int(self.cfg.smooth_window or 1))
            if win % 2 == 0: win += 1

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
                    n = len(valid_pts)
                    candidates.append({
                        "id": out_id,
                        "pts": valid_pts,
                        "mean": (sx / n, sy / n),
                        "score": self._track_quality_score(valid_pts, T, diag),
                    })
                else:
                    diag_short += 1

        if self.cfg.enable_spread_select and candidates:
            final_tracks_out = self._select_spread(candidates)
        else:
            final_tracks_out = {c["id"]: c["pts"] for c in candidates}
        total_kept = len(final_tracks_out)

        return final_tracks_out, total_kept, total_candidates, diag_after_filter, diag_after_gate, diag_short

    def _track_quality_score(self, pts: List[Tuple[int, float, float]], T: int, diag: float) -> float:
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

        w_c, w_s, w_j = self.cfg.w_coverage, self.cfg.w_smoothness, self.cfg.w_stability
        wsum = (w_c + w_s + w_j) or 1.0
        return (w_c * cov_score + w_s * smooth_score + w_j * stab_score) / wsum

    def _select_spread(self, candidates: List[dict]) -> Dict[str, List[Tuple[int, float, float]]]:
        """Greedy min-spacing selection over score-sorted candidates.

        Walk best-first; accept a track only if its mean position is >= spread_min_dist_px
        from every already-accepted track. Because spacing (default 40px) >> any pass
        duplicate offset, near-identical duplicates are rejected implicitly -- this doubles
        as the pass dedup. A coarse cell grid keeps each spacing test O(neighbours).
        """
        d = max(1, int(self.cfg.spread_min_dist_px))
        d2 = float(d * d)
        cap = int(self.cfg.max_output_tracks or 0)
        grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        out: Dict[str, List[Tuple[int, float, float]]] = {}

        for c in sorted(candidates, key=lambda c: c["score"], reverse=True):
            mx, my = c["mean"]
            cx, cy = int(mx // d), int(my // d)
            ok = True
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for (ax, ay) in grid.get((gx, gy), ()):  # type: ignore[arg-type]
                        if (ax - mx) ** 2 + (ay - my) ** 2 < d2:
                            ok = False
                            break
                    if not ok: break
                if not ok: break
            if not ok:
                continue
            out[c["id"]] = c["pts"]
            grid.setdefault((cx, cy), []).append((mx, my))
            if cap > 0 and len(out) >= cap:
                break
        return out

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

    def _decide_chunks(self, T: int, Ws: int, Hs: int) -> int:
        # Manual override wins.
        if int(self.cfg.chunks) >= 1:
            return max(1, min(int(self.cfg.chunks), 64))
        factor = self._cal if (self._cal and self._cal > 0) else self._MEM_PER_FPX_DEFAULT
        free = self._free_vram_bytes()
        if free <= 0 or Ws <= 0 or Hs <= 0:
            return 1
        budget = free * 0.8
        per_frame = float(Hs) * float(Ws) * float(factor)
        if per_frame <= 0:
            return 1
        max_frames = max(1, int(budget / per_frame))
        n = int(math.ceil(T / float(max_frames)))
        return max(1, min(n, int(self.cfg.max_chunks)))

    def _gate_assembled(self, shot: str, xy: np.ndarray, vis: np.ndarray, W0: int, H0: int, T: int) -> np.ndarray:
        """Mask-gate an assembled full-length sweep (coords in ORIGINAL, top-left orientation)."""
        N = int(xy.shape[1])
        if N == 0 or not self.cfg.enable_mask_gating:
            return np.ones((N,), dtype=bool)
        keep, _msg = self._apply_per_frame_mask_gating(
            shot, xy, vis.astype(bool), W0, H0,
            T_seg=T, global_T=T, start_frame=0, is_reverse=False,
        )
        return keep

    def _chain_core(self, engine, fetch_block, shot: str, T: int, windows, W0: int, H0: int,
                    base_Ws: int, base_Hs: int):
        """Chain queries across overlapping windows (processing order).

        `fetch_block(proc_start, proc_count)` returns BGR frames already in processing order.
        Tracks that survive a window's overlap region are carried as queries into the next
        window, KEEPING their global id -> continuous tracks across seams. Returns
        store: {gid: {order_idx: (x_orig, y_orig)}} (order_idx = index in processing order).
        """
        store: Dict[int, Dict[int, Tuple[float, float]]] = {}
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
                pts = self._detect_features(
                    blk[0], mask=seed_mask, count=fresh_cap,
                    quality=self.cfg.feature_quality, min_dist=self.cfg.min_feature_dist,
                )
                for (px, py) in pts:
                    q_list.append((0.0, float(px), float(py)))
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

        return store

    @staticmethod
    def _assemble(store: Dict[int, Dict[int, Tuple[float, float]]], T: int, to_local) -> Tuple[np.ndarray, np.ndarray]:
        cols = [gid for gid, d in store.items() if len(d) >= 1]
        N = len(cols)
        xy = np.zeros((T, N, 2), dtype=np.float32)
        vis = np.zeros((T, N), dtype=bool)
        for ci, gid in enumerate(cols):
            for oidx, (x, y) in store[gid].items():
                t = to_local(oidx)
                if 0 <= t < T:
                    xy[t, ci, 0] = x; xy[t, ci, 1] = y; vis[t, ci] = True
        return xy, vis

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
        # Forward
        fwd_fetch = lambda ps, pc: source.get(fs0 + ps, pc)
        store_f = self._chain_core(engine, fwd_fetch, shot, T, windows, W0, H0, base_Ws, base_Hs)
        fxy, fvis = self._assemble(store_f, T, lambda o: o)
        passes.append(("FWD", fxy, fvis, self._gate_assembled(shot, fxy, fvis, W0, H0, T)))

        # Backward (reverse processing order: proc frame o -> actual local (T-1-o))
        if self.cfg.bidirectional and not self._stop.is_set():
            rev_fetch = lambda ps, pc: source.get(fs0 + (T - (ps + pc)), pc)[::-1].copy()
            store_b = self._chain_core(engine, rev_fetch, shot, T, windows, W0, H0, base_Ws, base_Hs)
            bxy, bvis = self._assemble(store_b, T, lambda o: (T - 1 - o))
            passes.append(("BWD", bxy, bvis, self._gate_assembled(shot, bxy, bvis, W0, H0, T)))

        return passes

    def _run_impl(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
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
            in_path = os.path.join(self.cfg.input_dir, fn)
            shot = os.path.splitext(fn)[0]
            scale = self._scale_for(fn)

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
                        avail = int(psutil.virtual_memory().available)
                        need = int(estimate_clip_bytes(in_path, scale))
                        stream = need > avail * float(self.cfg.host_ram_frac)
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
                self._status(f"[{i}/{len(vids)}] frames={T} res={Ws}x{Hs} freeVRAM={free_gb:.1f}GB -> chunks={n_chunks}{reseed_msg}")

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
                            status=lambda m: self._status(f"TRACK: {m}"), src=refine_src)
                        self._append_log(txt_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MOVINGTILE {shot}: {minfo}")
                    except Exception as e:
                        self._status(f"[{i}/{len(vids)}] Moving-tile skipped: {e}")

                # 3DE-style NCC/affine pattern lock at native resolution (post-selection,
                # so it runs only on the handful of spread-selected tracks -> cheap).
                if self.cfg.enable_pattern_refine and final_tracks_out:
                    try:
                        from app.pattern_refine import refine_tracks
                        self._status(f"[{i}/{len(vids)}] Pattern-refine (NCC/{self.cfg.refine_motion}) at {W0}x{H0}...")
                        final_tracks_out, rinfo = refine_tracks(
                            final_tracks_out, in_path, W0, H0, orig_total, self.cfg,
                            status=lambda m: self._status(f"TRACK: {m}"), bgr_source=refine_src)
                        total_kept = len(final_tracks_out)
                        self._append_log(txt_log, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REFINE {shot}: {rinfo}")
                    except Exception as e:
                        self._status(f"[{i}/{len(vids)}] Pattern-refine skipped: {e}")

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