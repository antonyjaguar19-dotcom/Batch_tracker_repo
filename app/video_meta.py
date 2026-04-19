# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any
import cv2

def probe_video_meta(path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False, "error": "failed_to_open", "fps": 0.0, "width": 0, "height": 0, "total_frames": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"ok": True, "fps": fps, "width": width, "height": height, "total_frames": total_frames}
