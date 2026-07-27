# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from typing import Tuple, Dict, Any, List
import numpy as np
import cv2  # type: ignore
from app.reformat_plate_core import reload_and_rescale_video, _to_rgb_u8, _resize_rgb, _probe_video_hw

# Feeding TAPNext a full-res image SEQUENCE (instead of a downscaled proxy mp4) keeps the
# tracker + native-res refine at true plate resolution and avoids the mp4 codec entirely.
_SEQ_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr"}

def _seq_frames(path: str) -> List[str]:
    """Sorted image frames under `path`: the deepest subfolder holding the most images,
    ordered by trailing frame number (falls back to lexical)."""
    best, best_n = None, 0
    for dp, _dn, fns in os.walk(path):
        imgs = [f for f in fns if os.path.splitext(f)[1].lower() in _SEQ_EXTS]
        if len(imgs) > best_n:
            best_n, best = len(imgs), (dp, imgs)
    if not best:
        return []
    dp, imgs = best
    def _num(f):
        m = re.search(r"(\d+)$", os.path.splitext(f)[0])
        return int(m.group(1)) if m else 0
    return [os.path.join(dp, f) for f in sorted(imgs, key=_num)]

def _read_img_bgr(fp: str, scale: float):
    """Read one frame -> BGR uint8 at `scale` (UNCHANGED handles 8/16-bit + exr float)."""
    im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    rgb = _to_rgb_u8(im)
    if rgb is None:
        return None
    rgb = _resize_rgb(rgb, float(scale or 1.0))
    return rgb[..., ::-1]  # RGB -> BGR

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


def _probe_total_frames(path: str) -> int:
    """Best-effort total frame count. Falls back to a decode-count if metadata lies."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n > 0:
        cap.release()
        return n
    # Metadata missing -> count by decoding (slow, only when needed).
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def read_window_bgr_scaled(path: str, start: int, count: int, scale: float = 1.0) -> np.ndarray:
    """Decode frames [start, start+count) at `scale`, return (n,H,W,3) BGR uint8.

    Sequential decode (codec-safe): seeks via CAP_PROP_POS_FRAMES then verifies by
    discarding forward if the seek under-shot. Used only for clips too big to hold whole.
    """
    scale = float(scale or 1.0)
    if scale <= 0.0 or scale > 1.0:
        scale = 1.0
    start = max(0, int(start))
    count = max(0, int(count))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        # Verify/repair seek: if the backend landed short, grab forward to `start`.
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        while pos < start:
            if not cap.grab():
                break
            pos += 1
    out = []
    for _ in range(count):
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = _to_rgb_u8(bgr)
        if rgb is None:
            continue
        rgb = _resize_rgb(rgb, scale)
        out.append(rgb[..., ::-1])  # RGB -> BGR
    cap.release()
    if not out:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(out, axis=0).astype(np.uint8)


class FrameSource:
    """Uniform frame access for the tracker. Holds the whole clip in RAM when it fits,
    otherwise decodes each requested window from disk. `get(start, count)` indexes the
    FULL clip (0-based) and returns a BGR uint8 block at `scale`."""

    def __init__(self, path: str, scale: float = 1.0, stream: bool = False):
        scale = float(scale or 1.0)
        if scale <= 0.0 or scale > 1.0:
            scale = 1.0
        self.path = path
        self.scale = scale
        self.stream = bool(stream)
        self._arr = None
        self._seq = None

        # --- Image-sequence input (a directory of frames) ---
        if os.path.isdir(path):
            self._seq = _seq_frames(path)
            if not self._seq:
                raise RuntimeError(f"No image frames in sequence dir: {path}")
            first = _read_img_bgr(self._seq[0], 1.0)
            if first is None:
                raise RuntimeError(f"Could not read first frame in: {path}")
            self.h0, self.w0, self.fps = int(first.shape[0]), int(first.shape[1]), 24.0
            self.total = len(self._seq)
            self.scaled_w = max(1, int(round(self.w0 * scale)))
            self.scaled_h = max(1, int(round(self.h0 * scale)))
            if not self.stream:
                frames = [a for a in (_read_img_bgr(fp, scale) for fp in self._seq) if a is not None]
                if not frames:
                    raise RuntimeError(f"No readable frames in: {path}")
                self._arr = np.stack(frames, axis=0).astype(np.uint8)
                self.total = int(self._arr.shape[0])
                self.scaled_h, self.scaled_w = int(self._arr.shape[1]), int(self._arr.shape[2])
            return

        # --- Video input (mp4/mov via OpenCV) ---
        (h0, w0), fps = _probe_video_hw(path)
        self.w0, self.h0, self.fps = int(w0), int(h0), float(fps)
        if not self.stream:
            arr, meta = read_video_frames_bgr_scaled(path, scale)
            self._arr = arr
            self.total = int(arr.shape[0])
            self.scaled_h = int(meta.get("scaled_height", arr.shape[1]))
            self.scaled_w = int(meta.get("scaled_width", arr.shape[2]))
        else:
            self.total = _probe_total_frames(path)
            self.scaled_w = max(1, int(round(self.w0 * scale)))
            self.scaled_h = max(1, int(round(self.h0 * scale)))

    @property
    def meta(self) -> Dict[str, Any]:
        return {
            "fps": self.fps, "width": self.w0, "height": self.h0,
            "total_frames": self.total, "scaled_width": self.scaled_w,
            "scaled_height": self.scaled_h, "scale": self.scale,
        }

    def get(self, start: int, count: int) -> np.ndarray:
        if self._arr is not None:
            return self._arr[int(start):int(start) + int(count)]
        if self._seq is not None:
            sel = self._seq[int(start):int(start) + int(count)]
            out = [a for a in (_read_img_bgr(fp, self.scale) for fp in sel) if a is not None]
            return np.stack(out, axis=0).astype(np.uint8) if out else np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return read_window_bgr_scaled(self.path, start, count, self.scale)


def estimate_clip_bytes(path: str, scale: float = 1.0) -> int:
    """Approx host-RAM bytes to hold the whole clip decoded at `scale` (uint8 BGR)."""
    try:
        s = float(scale or 1.0)
        if os.path.isdir(path):
            seq = _seq_frames(path)
            if not seq:
                return 0
            im = cv2.imread(seq[0], cv2.IMREAD_UNCHANGED)
            if im is None:
                return 0
            h0, w0 = im.shape[:2]
            return int(len(seq) * round(h0 * s) * round(w0 * s) * 3)
        (h0, w0), _ = _probe_video_hw(path)
        n = _probe_total_frames(path)
        return int(n * round(h0 * s) * round(w0 * s) * 3)
    except Exception:
        return 0
