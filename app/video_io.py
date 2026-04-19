# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np
from app.reformat_plate_core import reload_and_rescale_video

def read_video_frames_bgr_scaled(path: str, scale: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    scale = float(scale or 1.0)
    if scale <= 0.0 or scale > 1.0:
        scale = 1.0

    frames_rgb, fps, (h0, w0) = reload_and_rescale_video(path, scale)
    frames_bgr = frames_rgb[..., ::-1].copy()

    meta = {
        "fps": float(fps),
        "width": int(w0),
        "height": int(h0),
        "total_frames": int(frames_bgr.shape[0]),
        "scaled_width": int(frames_bgr.shape[2]),
        "scaled_height": int(frames_bgr.shape[1]),
        "scale": float(scale),
    }
    return frames_bgr, meta

def read_video_frames_bgr(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    return read_video_frames_bgr_scaled(path, scale=1.0)
