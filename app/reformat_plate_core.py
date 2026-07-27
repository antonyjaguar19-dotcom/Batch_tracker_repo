# -*- coding: utf-8 -*-
"""reformat_plate_core.py

Downscaling/reformat helpers adapted from your T-rex script reformat_plate.py,
with the PySide6 UI removed so this tool stays dependency-light.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2

def _to_rgb_u8(img):
    if img is None:
        return None
    im = img
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    if im.dtype == np.float32 or im.dtype == np.float64:
        im = np.clip(im, 0.0, 1.0)
        im = (im * 255.0 + 0.5).astype(np.uint8)
    elif im.dtype == np.uint16:
        im = (im / 257.0 + 0.5).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def _resize_rgb(img_rgb, scale: float):
    if float(scale) == 1.0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    nw = max(1, int(round(w * float(scale))))
    nh = max(1, int(round(h * float(scale))))
    interp = cv2.INTER_AREA if float(scale) < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img_rgb, (nw, nh), interpolation=interp)

def _probe_video_hw(path: str) -> Tuple[Tuple[int,int], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if w <= 0 or h <= 0:
        ok, frm = cap.read()
        if not ok or frm is None:
            cap.release()
            raise RuntimeError("Failed to read first frame for probing size.")
        h, w = frm.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.release()
    return (int(h), int(w)), fps

def _load_video_scaled(path: str, scale: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = _to_rgb_u8(bgr)
        if rgb is None:
            continue
        rgb = _resize_rgb(rgb, scale)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded from video.")
    return np.stack(frames, axis=0).astype(np.uint8), fps

def reload_and_rescale_video(path: str, scale: float):
    (h0, w0), _fps0 = _probe_video_hw(path)
    frames_rgb, fps = _load_video_scaled(path, float(scale))
    return frames_rgb, fps, (h0, w0)
