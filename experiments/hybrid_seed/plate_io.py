"""One way to open a plate, whether it is a movie or an image sequence.

The dev-box experiment only ever used an mp4. Prod plates are usually EXR sequences, and
SynthEyes takes those through an IFL (a text file listing one frame path per line) exactly
as `SynthEyesEngine.process_shot` builds one. This wraps both so the probes and the hybrid
runner do not each grow their own half of it.

`OPENCV_IO_ENABLE_OPENEXR` must be set before cv2 is imported anywhere, so it is set at the
top of this module and every entry point imports this before touching cv2.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.syntheyes_engine import find_shot_frames, read_image_size  # noqa: E402

MOVIE_EXT = (".mp4", ".mov", ".avi", ".mkv")


class Plate:
    """load_path -> hand to NewSceneAndShot; frame(i) -> BGR uint8, i is 0-based."""

    def __init__(self, path: str, ifl_dir: str):
        self.path = os.path.normpath(path)
        self.is_movie = os.path.isfile(self.path) and self.path.lower().endswith(MOVIE_EXT)
        self._cap = None
        self._files: list[str] = []

        if self.is_movie:
            self.load_path = self.path
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise RuntimeError(f"could not open movie: {self.path}")
            self.count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            if not os.path.isdir(self.path):
                raise RuntimeError(f"not a movie file or a folder of frames: {self.path}")
            seq = find_shot_frames(self.path)
            if not seq:
                raise RuntimeError(f"no image sequence found under: {self.path}")
            self._files = list(seq["all_frames"])
            self.count = int(seq["frame_count"])
            self.w, self.h = read_image_size(seq["first_frame"])
            os.makedirs(ifl_dir, exist_ok=True)
            ifl = os.path.join(ifl_dir, os.path.basename(self.path.rstrip("\\/")) + ".ifl")
            with open(ifl, "w", encoding="utf-8") as f:
                for p in self._files:
                    f.write(os.path.abspath(p).replace("\\", "/") + "\n")
            self.load_path = ifl

    @property
    def name(self) -> str:
        base = os.path.basename(self.path.rstrip("\\/"))
        return os.path.splitext(base)[0] if self.is_movie else base

    def frame(self, i: int):
        """0-based frame as BGR uint8, or None."""
        if self.is_movie:
            # Sequential reads dominate here, so keep one capture open and only seek when
            # the caller jumps -- seeking every frame on a long 4K plate is far slower.
            if self._cap is None:
                self._cap = cv2.VideoCapture(self.path)
                self._pos = 0
            if i != getattr(self, "_pos", -1):
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                self._pos = i
            ok, img = self._cap.read()
            self._pos = i + 1
            return img if ok else None

        if not (0 <= i < len(self._files)):
            return None
        img = cv2.imread(self._files[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return _to_bgr8(img)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _to_bgr8(img: np.ndarray) -> np.ndarray:
    """EXR/DPX come back float or 16-bit, sometimes with alpha. Seeding and NCC both want
    an 8-bit BGR image, and a naive cast on linear float EXR gives a near-black frame with
    no features at all -- so float input is gamma-encoded first."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 257.0).astype(np.uint8)
    f = np.clip(img.astype(np.float32), 0.0, None)
    f = np.power(f / (1.0 + f), 1.0 / 2.2) if f.max() > 1.0 else np.power(f, 1.0 / 2.2)
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def to_uv(px: float, py: float, w: int, h: int):
    """Plate pixels -> SynthEyes u/v. Inverse of the engine's export mapping
    (app/syntheyes_engine.py:2062): px = 0.5*(u+1)*w, py = 0.5*(1-v)*h."""
    return (2.0 * px / w) - 1.0, 1.0 - (2.0 * py / h)


def pick_feature(img, n: int = 1, quality: float = 0.05, min_dist: int = 60):
    """Strongest corner(s) on a frame -- used by the probes so a tracker is planted on
    something real. A probe that plants on flat sky cannot tell 'the mechanism failed'
    apart from 'there was nothing to lock onto'."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=quality,
                                  minDistance=min_dist, blockSize=5)
    if pts is None or len(pts) == 0:
        raise RuntimeError("no trackable feature found on that frame")
    return pts.reshape(-1, 2).astype(float)
