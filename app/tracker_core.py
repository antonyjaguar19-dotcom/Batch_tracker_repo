# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple

import numpy as np
import cv2  # type: ignore
import torch # Need torch for empty_cache

from app.video_io import read_video_frames_bgr_scaled
from app.cotracker_engine import CoTracker3Engine
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


class BatchTrackerRunner:
    def __init__(self, cfg: RunnerConfig, on_status: StatusCB = None):
        self.cfg = cfg
        self.on_status = on_status
        self._stop = threading.Event()

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

    def _mask_region_from_gray(self, gray: np.ndarray) -> np.ndarray:
        pol = (self.cfg.mask_polarity or "auto").strip().lower()
        white_region = gray >= 128
        if pol == "white":
            return white_region
        if pol == "black":
            return ~white_region
        pct_white = float(np.mean(white_region))
        if pct_white > 0.5:
            return ~white_region
        return white_region

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

            union_region |= self._mask_region_from_gray(img)
            used += 1

        pct = float(np.mean(union_region)) * 100.0
        info = f"{where} | masks={len(mask_paths)} sampled={used} union_mask={pct:.1f}% polarity={self.cfg.mask_polarity}"
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

        def get_global_mask_idx(t: int) -> int:
            global_t = (start_frame - t) if is_reverse else (start_frame + t)
            global_t = max(0, min(global_t, global_T - 1))
            if global_T <= 1 or M <= 1: return 0
            return int(round(global_t * (M - 1) / float(global_T - 1)))

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
        engine: CoTracker3Engine,
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
        self._status("Loading CoTracker3 (offline) [FP16 mode]...")
        engine = CoTracker3Engine(tool_root=tool_root, device="cuda")

        for i, fn in enumerate(vids, start=1):
            if self._stop.is_set(): break
            shot_start = time.time()
            in_path = os.path.join(self.cfg.input_dir, fn)
            shot = os.path.splitext(fn)[0]
            scale = self._scale_for(fn)

            try:
                self._status(f"[{i}/{len(vids)}] Reading: {fn} (scale={scale})")
                frames_fwd, meta = read_video_frames_bgr_scaled(in_path, scale=scale)

                T = int(meta.get("total_frames", frames_fwd.shape[0]))
                if T <= 1: continue

                W0, H0 = int(meta.get("width", 0) or 0), int(meta.get("height", 0) or 0)
                Ws, Hs = int(meta.get("scaled_width", frames_fwd.shape[2]) or frames_fwd.shape[2]), int(meta.get("scaled_height", frames_fwd.shape[1]) or frames_fwd.shape[1])
                diag = float(np.sqrt(float(W0 * W0 + H0 * H0))) if (W0 > 0 and H0 > 0) else 1000.0
                inv = 1.0 / float(scale) if float(scale) != 0.0 else 1.0

                seed_mask, mask_info, n_masks = self._make_seed_inclusion_mask(shot, Ws, Hs)
                self._status(f"[{i}/{len(vids)}] Mask: {mask_info}")

                # --- PASS 1: FORWARD ---
                self._status(f"[{i}/{len(vids)}] Tracking Forward...")
                tracks_xy_f, vis_f, keep_gate_f, log_f = self._process_single_pass(
                    engine, frames_fwd, shot, Ws, Hs, mask_info, seed_mask, n_masks, 
                    is_reverse=False, global_T=T, start_frame=0
                )

                # --- PASS 2: BACKWARD ---
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

                # --- PASS 3: MID-FORWARD ---
                self._status(f"[{i}/{len(vids)}] Tracking Mid-Forward...")
                mid_idx = T // 2
                frames_mid_f = frames_fwd[mid_idx:].copy()
                tr_mid_f, vis_mid_f, keep_gate_mid_f, log_mid_f = self._process_single_pass(
                    engine, frames_mid_f, shot, Ws, Hs, mask_info, seed_mask, n_masks, 
                    is_reverse=False, global_T=T, start_frame=mid_idx
                )

                # --- PASS 4: MID-BACKWARD ---
                self._status(f"[{i}/{len(vids)}] Tracking Mid-Backward...")
                frames_mid_b = frames_fwd[:mid_idx+1][::-1].copy()
                tr_mid_b, vis_mid_b, keep_gate_mid_b, log_mid_b = self._process_single_pass(
                    engine, frames_mid_b, shot, Ws, Hs, mask_info, seed_mask, n_masks, 
                    is_reverse=True, global_T=T, start_frame=mid_idx
                )
                
                # --- MERGE & EXPORT ---
                self._status(f"[{i}/{len(vids)}] Merging & Filtering 3-Pass Results...")
                
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

                final_tracks_out: Dict[str, List[Tuple[int, float, float]]] = {}
                total_kept, total_candidates = 0, 0

                passes = [("FWD", tracks_xy_f, vis_f, keep_gate_f)]
                if tracks_xy_b is not None: passes.append(("BWD", tracks_xy_b, vis_b, keep_gate_b))
                passes.append(("MID_F", xy_mid_f_full, vis_mid_f_full, keep_gate_mid_f))
                passes.append(("MID_B", xy_mid_b_full, vis_mid_b_full, keep_gate_mid_b))

                for p_name, xy_raw, vis_raw, gate_mask in passes:
                    if xy_raw is None or xy_raw.shape[1] == 0: continue
                    N = xy_raw.shape[1]
                    total_candidates += N
                    
                    x_all = (xy_raw[:, :, 0].astype(np.float32) * float(inv))
                    y_all = (xy_raw[:, :, 1].astype(np.float32) * float(inv))
                    if self.cfg.flip_y_for_3de and H0 > 0: y_all = (float(H0 - 1) - y_all)
                    
                    # Pre-mask any immediate NaNs/Infs that CoTracker leaked
                    vis_bool = vis_raw.astype(bool) & ~np.isnan(x_all) & ~np.isnan(y_all) & ~np.isinf(x_all) & ~np.isinf(y_all)
                    
                    keep = np.ones((N,), dtype=bool)
                    if self.cfg.enable_filtering and N > 0:
                        keep = self._post_filter_tracks(x_all, y_all, vis_bool, diag=diag)
                    
                    keep = keep & gate_mask
                    kept_idx = np.where(keep)[0]
                    win = max(1, int(self.cfg.smooth_window or 1))
                    if win % 2 == 0: win += 1

                    for j in kept_idx.tolist():
                        xs, ys = x_all[:, j].copy(), y_all[:, j].copy()
                        if win > 1:
                            xs, ys = self._moving_average_1d(xs, win), self._moving_average_1d(ys, win)
                            
                        # Format organized track name e.g., "MID_F_0042"
                        out_id = f"{p_name}_{j+1:04d}"
                        
                        valid_pts = []
                        for t in range(T):
                            if vis_bool[t, j]:
                                x_val, y_val = float(xs[t]), float(ys[t])
                                # Final safety check before pushing to text file
                                if not (np.isnan(x_val) or np.isnan(y_val) or np.isinf(x_val) or np.isinf(y_val)):
                                    valid_pts.append((t + 1, x_val, y_val))
                                    
                        if len(valid_pts) > 1:
                            final_tracks_out[out_id] = valid_pts
                            total_kept += 1

                tag = (self.cfg.output_tag or '').strip()
                base = f"{shot}__cotracker3_bidir.txt" if not tag else f"{shot}__{tag}__cotracker3_bidir.txt"
                out_txt = os.path.join(self.cfg.output_dir, base)
                write_tracks_txt(out_txt, final_tracks_out, end_frame=T)

                torch.cuda.empty_cache()
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