# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import numpy as np

# Help PyTorch manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _add_cotracker_to_path(tool_root: str) -> str:
    thirdparty = os.path.join(tool_root, "thirdparty")
    candidates = [
        os.path.join(thirdparty, "co-tracker-main"),
        os.path.join(thirdparty, "co-tracker"),
    ]
    repo = None
    for c in candidates:
        if os.path.isdir(c) and os.path.isdir(os.path.join(c, "cotracker")):
            repo = c
            break
    if repo is None:
        raise RuntimeError(
            "CoTracker repo not found. Expected in tool_root\\thirdparty.\n"
            f"Tried: {candidates}"
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo

class CoTracker3Engine:
    def __init__(self, tool_root: str, device: str = "cuda"):
        self.tool_root = tool_root
        self.repo_root = _add_cotracker_to_path(tool_root)

        import torch  # type: ignore
        from cotracker.predictor import CoTrackerPredictor  # type: ignore

        self.torch = torch
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        ckpt = os.path.join(self.repo_root, "checkpoints", "scaled_offline.pth")
        if not os.path.isfile(ckpt):
            alt = os.path.join(tool_root, "checkpoints", "scaled_offline.pth")
            if os.path.isfile(alt):
                ckpt = alt
            else:
                raise RuntimeError(
                    "Missing checkpoint scaled_offline.pth. Put it in either:\n"
                    f"  {os.path.join(self.repo_root,'checkpoints')}\n"
                    f"  {os.path.join(tool_root,'checkpoints')}\n"
                )
        self.checkpoint = ckpt
        
        # FIX 1: Load model in standard FP32. 
        # We rely on autocast to handle mixed precision, rather than forcing .half() on the weights.
        self.model = CoTrackerPredictor(checkpoint=self.checkpoint, offline=True).to(self.device)

    def _video_tensor(self, frames_bgr: np.ndarray):
        """
        Helper to convert (T,H,W,3) BGR numpy array -> (1,T,3,H,W) float16 tensor (RGB).
        """
        torch = self.torch
        frames_rgb = frames_bgr[..., ::-1].copy()
        
        # FIX 2: We STILL convert the video to Half (FP16).
        # The video is the biggest memory hog (4GB+ for 4K). 
        # Autocast can handle FP16 input fed into an FP32 model.
        video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].half().to(self.device)
        return video

    def track_queries(self, frames_bgr: np.ndarray, queries: np.ndarray):
        """
        Explicitly tracks specific points (queries).
        """
        torch = self.torch
        
        # 1. Convert BGR frames to RGB Tensor (FP16)
        video = self._video_tensor(frames_bgr)
        
        # 2. Convert queries to tensor (Float32)
        # We keep points in Float32 for coordinate precision. Autocast handles the mix.
        q_tensor = torch.from_numpy(queries).to(self.device).float()
        
        # 3. Run model with Autocast
        # This Context Manager automatically casts operations to FP16 where safe,
        # preventing "Half vs Float" mismatch errors.
        with torch.autocast("cuda"):
            pred_tracks, pred_visibility = self.model(video, queries=q_tensor)
        
        # 4. Return as float32 numpy
        return pred_tracks[0].float().detach().cpu().numpy(), pred_visibility[0].float().detach().cpu().numpy()

    def track_grid(self, frames_bgr: np.ndarray, grid_size: int, grid_query_frame: int = 0, segm_mask: np.ndarray | None = None):
        torch = self.torch
        video = self._video_tensor(frames_bgr)

        mask_tensor = None
        if segm_mask is not None:
            m = segm_mask
            if m.dtype != np.bool_:
                m = (m > 0)
            m = m.astype(np.float32)[None, None, :, :]
            # Mask is Float32, Model is Float32, Video is Half. Autocast manages this.
            mask_tensor = torch.from_numpy(m).to(self.device).float()

        with torch.autocast("cuda"):
            tracks, vis = self.model(
                video,
                grid_size=int(grid_size),
                grid_query_frame=int(grid_query_frame),
                segm_mask=mask_tensor,
            )
        return tracks[0].float().detach().cpu().numpy(), vis[0].float().detach().cpu().numpy().astype(bool)