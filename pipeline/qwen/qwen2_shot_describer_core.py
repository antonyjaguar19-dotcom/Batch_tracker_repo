from __future__ import annotations

"""
Qwen2 Shot Describer Core (Batch Tracker integration)

Offline-by-design when `model_id_or_path` is a LOCAL folder.
We call `from_pretrained(..., local_files_only=True)` so missing files error fast and never hit the network.

STRUCT OUTPUT:
{
  "scene_elements": "...",
  "camera_movement": "...",
  "things": ["...", "..."]
}

Dependencies:
- torch, transformers, pillow, opencv-python
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import re
import time

import numpy as np

try:
    import cv2
except Exception as e:
    raise RuntimeError(
        "OpenCV (cv2) is required. Install with: pip install opencv-python\n"
        f"Original error: {e}"
    )

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Pillow is required. Install with: pip install pillow\n"
        f"Original error: {e}"
    )

import torch

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except Exception as e:
    raise RuntimeError(
        "Failed to import Qwen2.5-VL from transformers.\n"
        "If you see 'Could not import Qwen2_5_VLForConditionalGeneration', "
        "upgrade transformers (often from source) and accelerate.\n"
        f"Original error: {e}"
    )


@dataclass
class ModelConfig:
    # kept for compatibility with earlier runners; not required
    use_torchao_int4: bool = False

    # IMPORTANT: should be a LOCAL PATH for true offline mode
    model_id_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    fps: int = 1

    # VL inputs
    max_new_tokens: int = 220
    # VLM frames are spread across the WHOLE clip (not the first N). Coverage comes
    # from spreading, NOT from sheer count: too many images dilute the VLM's attention
    # and DEGRADE answers. Keep the count in the 6-8 sweet spot; raise resolution
    # instead so small/brief subjects (e.g. a dog) stay visible.
    num_frames_min: int = 6
    num_frames_cap: int = 8
    max_side: int = 768

    # CV-only sampling for camera classification
    cv_fps_min: int = 3
    cv_frames_cap: int = 20
    cv_scale: float = 0.5

    # Camera thresholds (downscaled px)
    cam_static_p95_px: float = 0.12
    cam_static_median_px: float = 0.06

    # Small-motion handheld bias
    cam_small_p95_px: float = 1.2
    cam_small_med_px: float = 0.55

    # Direction variance gating
    cam_direction_circvar_freemove: float = 0.33
    cam_direction_circvar_mounted: float = 0.46

    # Zoom (conservative)
    zoom_min_abs_log_scale_med: float = 0.020
    zoom_consistency: float = 0.75

    def __post_init__(self):
        self.fps = max(1, int(self.fps))
        self.max_new_tokens = int(self.max_new_tokens)
        self.num_frames_min = max(2, int(self.num_frames_min))
        self.num_frames_cap = max(self.num_frames_min, int(self.num_frames_cap))
        self.max_side = int(self.max_side)

        self.cv_fps_min = max(1, int(self.cv_fps_min))
        self.cv_frames_cap = max(4, int(self.cv_frames_cap))
        self.cv_scale = float(self.cv_scale)


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr", ".dpx"}


def list_shots_in_folder(in_dir: str) -> List[Tuple[str, Path]]:
    p = Path(in_dir)
    if not p.exists():
        return []
    items: List[Tuple[str, Path]] = []
    for child in sorted(p.iterdir()):
        if child.is_file() and child.suffix.lower() in _VIDEO_EXTS:
            items.append((child.stem, child))
        elif child.is_dir():
            items.append((child.name, child))
    return items


def _find_images_recursive(seq_dir: Path) -> List[Path]:
    all_imgs = [p for p in seq_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    if not all_imgs:
        return []
    by_parent: Dict[Path, List[Path]] = {}
    for p in all_imgs:
        by_parent.setdefault(p.parent, []).append(p)
    best_parent = max(by_parent.items(), key=lambda kv: len(kv[1]))[0]
    return sorted(by_parent[best_parent])


def sequence_folder_to_temp_mp4(seq_dir: str | Path, tmp_dir: str | Path,
                                max_side: int = 1280) -> str:
    seq_dir = Path(seq_dir)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    imgs = _find_images_recursive(seq_dir)
    if not imgs:
        raise RuntimeError(f"No images found in sequence folder: {seq_dir}")

    first = None
    for p in imgs:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            first = im
            break
    if first is None:
        raise RuntimeError(
            f"Found {len(imgs)} images in {seq_dir} but none could be read by OpenCV. "
            f"If these are EXR/DPX, generate JPG/PNG proxies or add OpenImageIO."
        )

    # Downscale to a safe max side BEFORE encoding: the mp4v (MPEG-4 Part 2) codec
    # stalls/hangs on frames wider than ~4096 (a 6K/8K plate would freeze here), and
    # Qwen downsamples the frames for inference anyway, so native res is wasted work.
    first = _resize_max_side_bgr(first, max_side)
    h, w = first.shape[:2]
    out_path = tmp_dir / f"{seq_dir.name}_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 24.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    written = 0
    for p in imgs:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = _resize_max_side_bgr(frame, max_side)
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    writer.release()

    if written < 2:
        raise RuntimeError(f"Sequence conversion wrote only {written} frames from: {seq_dir}")

    return str(out_path)


# ----------------------------
# Camera classification
# ----------------------------

def _resize_max_side_bgr(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame_bgr
    scale = max_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _sample_frames(video_path: str, fps: int, cap_frames: int) -> List[np.ndarray]:
    """Sample up to `cap_frames` frames spread EVENLY across the whole clip.

    Spreading across the full duration (not the first N) is what lets the VLM see
    subjects that appear anywhere in the shot. Falls back to a sequential stride
    when the frame count is unknown/unreliable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n = max(2, int(cap_frames))

    frames: List[np.ndarray] = []

    if total and total > 1:
        if total <= n:
            targets = list(range(total))
        else:
            targets = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
        targets_sorted = sorted(targets)
        idx = 0
        ti = 0
        while ti < len(targets_sorted):
            ok, frame = cap.read()
            if not ok:
                break
            while ti < len(targets_sorted) and targets_sorted[ti] == idx:
                frames.append(frame)
                ti += 1
            idx += 1

    if len(frames) < 2:
        # Unknown/odd frame count: sequential stride fallback (covers as much as it can).
        cap.release()
        cap = cv2.VideoCapture(video_path)
        step = max(1, int(round(src_fps / float(max(1, fps)))))
        frames = []
        idx = 0
        grabbed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
                grabbed += 1
                if grabbed >= n:
                    break
            idx += 1

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"Not enough frames decoded from: {video_path}")
    return frames


def _sample_frames_cv(video_path: str, fps_min: int, cap_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 24.0
    use_fps = max(fps_min, 1)
    step = max(1, int(round(src_fps / float(use_fps))))

    frames: List[np.ndarray] = []
    idx = 0
    grabbed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
            grabbed += 1
            if grabbed >= cap_frames:
                break
        idx += 1
    cap.release()
    return frames if len(frames) >= 2 else []


def _sample_frames_from_folder(seq_dir, cap_frames: int) -> List[np.ndarray]:
    """Evenly-spread sample straight from an image-sequence folder — no mp4 encode.
    Mirrors _sample_frames' spread so the VLM sees subjects anywhere in the shot."""
    imgs = _find_images_recursive(Path(seq_dir))
    if not imgs:
        raise RuntimeError(f"No images found in sequence folder: {seq_dir}")
    total = len(imgs)
    n = max(2, int(cap_frames))
    if total <= n:
        targets = list(range(total))
    else:
        targets = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
    frames: List[np.ndarray] = []
    for ti in targets:
        im = cv2.imread(str(imgs[ti]), cv2.IMREAD_COLOR)
        if im is not None:
            frames.append(im)
    if len(frames) < 2:
        raise RuntimeError(f"Not enough readable frames in: {seq_dir}")
    return frames


def _sample_frames_cv_from_folder(seq_dir, fps_min: int, cap_frames: int,
                                  src_fps: float = 24.0) -> List[np.ndarray]:
    """Stride sample from a folder for the CV camera-motion classifier (no mp4)."""
    imgs = _find_images_recursive(Path(seq_dir))
    if not imgs:
        return []
    step = max(1, int(round(src_fps / float(max(fps_min, 1)))))
    frames: List[np.ndarray] = []
    grabbed = 0
    for i, p in enumerate(imgs):
        if i % step == 0:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                frames.append(im)
                grabbed += 1
                if grabbed >= cap_frames:
                    break
    return frames if len(frames) >= 2 else []


def _estimate_affine(prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray:
    try:
        orb = cv2.ORB_create(1200)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(cur_gray, None)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            raise ValueError("ORB insufficient")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        if len(matches) < 12:
            raise ValueError("ORB matches insufficient")
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        M, _inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            raise ValueError("Affine fit failed")
        return M.astype(np.float32)
    except Exception:
        a = prev_gray.astype(np.float32)
        b = cur_gray.astype(np.float32)
        hann = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        shift, _resp = cv2.phaseCorrelate(a, b, hann)
        dx, dy = float(shift[0]), float(shift[1])
        return np.array([[1.0, 0.0, dx],
                         [0.0, 1.0, dy]], dtype=np.float32)


def _estimate_global_motion(frames_bgr: List[np.ndarray], scale: float):
    mags, vecs, scales = [], [], []
    prev = frames_bgr[0]
    for cur in frames_bgr[1:]:
        p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            p = cv2.resize(p, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            c = cv2.resize(c, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        A = _estimate_affine(p, c)
        dx, dy = float(A[0, 2]), float(A[1, 2])
        a, b = float(A[0, 0]), float(A[0, 1])
        sc = float(np.sqrt(a * a + b * b))
        if not np.isfinite(sc) or sc <= 0:
            sc = 1.0

        mags.append(float((dx * dx + dy * dy) ** 0.5))
        vecs.append((dx, dy))
        scales.append(sc)
        prev = cur

    return np.asarray(mags, np.float32), np.asarray(vecs, np.float32), np.asarray(scales, np.float32)


def _circular_variance(dx: np.ndarray, dy: np.ndarray) -> float:
    ang = np.arctan2(dy, dx)
    s = float(np.mean(np.sin(ang)))
    c = float(np.mean(np.cos(ang)))
    R = float(np.sqrt(s * s + c * c))
    return float(1.0 - R)


def _classify_camera(mags: np.ndarray, vecs: np.ndarray, scales: np.ndarray, cfg: ModelConfig) -> str:
    if mags.size == 0 or vecs.size == 0:
        return "Static"

    med = float(np.median(mags))
    p95 = float(np.percentile(mags, 95))

    zoom_label: Optional[str] = None
    if scales.size > 0:
        log_sc = np.log(np.clip(scales, 1e-4, 1e4))
        abs_med = float(np.median(np.abs(log_sc)))
        med_sign = float(np.median(log_sc))
        sign_cons = float(np.mean(np.sign(log_sc) == np.sign(med_sign))) if med_sign != 0 else 0.0
        if abs_med >= cfg.zoom_min_abs_log_scale_med and sign_cons >= cfg.zoom_consistency:
            zoom_label = "Static zoom" if p95 <= max(cfg.cam_static_p95_px * 3.0, 0.35) else "Freemove zoom"

    if p95 <= cfg.cam_static_p95_px and med <= cfg.cam_static_median_px:
        return zoom_label or "Static"

    dx = vecs[:, 0]
    dy = vecs[:, 1]
    net_dx = float(np.sum(dx))
    net_dy = float(np.sum(dy))
    circ_var = _circular_variance(dx, dy)

    if p95 <= cfg.cam_small_p95_px and med <= cfg.cam_small_med_px:
        return zoom_label or ("Mounted" if circ_var >= cfg.cam_direction_circvar_mounted else "Freemove")

    if circ_var >= cfg.cam_direction_circvar_freemove:
        return zoom_label or ("Mounted" if circ_var >= cfg.cam_direction_circvar_mounted else "Freemove")

    if abs(net_dx) > abs(net_dy) * 1.5:
        return zoom_label or "Truck"
    if abs(net_dy) > abs(net_dx) * 1.5:
        return zoom_label or "Pedestal"

    return zoom_label or "Freemove"


# ----------------------------
# VL helpers
# ----------------------------

def _pil_from_bgr(frame_bgr: np.ndarray, max_side: int) -> Image.Image:
    frame_bgr = _resize_max_side_bgr(frame_bgr, max_side=max_side)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _normalize_item(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("s") and len(s) > 3:
        s = s[:-1]
    return s


def _parse_csvish_list(text: str) -> List[str]:
    text = re.sub(r"[\r\n]+", ", ", text)
    text = re.sub(r"[\u2022\-•]+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip(" .;:-").strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    seen = set()
    out = []
    for p in parts:
        k = _normalize_item(p)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _short_paragraph(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    para = " ".join(sents[:2]).strip()
    return para if para else (text[:280].strip() if text else "None")


import json as _json

_QUALITY_VOCAB = {
    "motion blur", "out of focus", "low texture", "overexposed",
    "underexposed", "low light", "flicker", "heavy grain", "compression artifacts",
}


def _extract_json_obj(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ``` fences / extra prose)."""
    if not text:
        return {}
    t = text.strip()
    # strip ```json ... ``` fences
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = _json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _as_str_list(v, cap: int = 12) -> List[str]:
    """Coerce a model field into a clean de-duplicated string list."""
    items: List[str] = []
    if isinstance(v, list):
        items = [str(x).strip() for x in v]
    elif isinstance(v, str):
        items = [x.strip() for x in re.split(r"[,;\n]", v)]
    seen = set()
    out: List[str] = []
    for s in items:
        if not s:
            continue
        k = s.lower()
        if k in ("none", "n/a", "na", "unknown"):
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _is_prequantized(model_dir: str) -> bool:
    """True if the checkpoint already carries a quantization_config (AWQ/GPTQ/etc).

    Pre-quantized checkpoints are loaded as-is; transformers reads the quant
    config from the model itself, so we must NOT pass our own quant config or
    a conflicting torch_dtype.
    """
    try:
        import json as _json
        cfgp = Path(model_dir) / "config.json"
        if cfgp.is_file():
            data = _json.loads(cfgp.read_text(encoding="utf-8"))
            return isinstance(data, dict) and bool(data.get("quantization_config"))
    except Exception:
        pass
    return False


def _make_bnb_4bit_config():
    """Return a BitsAndBytesConfig for NF4 4-bit, or None if bitsandbytes missing."""
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    except Exception:
        return None


class QwenShotDescriber:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # Optional progress sink (set by the runner). Without it the long VLM
        # stages are silent and a slow box looks frozen — so log every step.
        self.log = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Never hit the network
        self.processor = AutoProcessor.from_pretrained(cfg.model_id_or_path, local_files_only=True)

        load_kwargs: Dict = {"local_files_only": True}
        if self.device.type == "cuda":
            load_kwargs["device_map"] = "auto"

        prequant = _is_prequantized(cfg.model_id_or_path)
        want_4bit = bool(getattr(cfg, "use_torchao_int4", False)) and self.device.type == "cuda"

        if prequant:
            # AWQ/GPTQ checkpoint: load as-is, do not override dtype/quant.
            self.quant_mode = "prequantized"
        elif want_4bit:
            bnb = _make_bnb_4bit_config()
            if bnb is not None:
                load_kwargs["quantization_config"] = bnb
                load_kwargs["torch_dtype"] = torch.float16
                self.quant_mode = "bnb-nf4"
            else:
                # bitsandbytes not available -> degrade to fp16, do not crash.
                load_kwargs["torch_dtype"] = torch.float16
                self.quant_mode = "fp16 (int4 requested but bitsandbytes missing)"
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self.device.type == "cuda" else torch.float32
            self.quant_mode = "fp16" if self.device.type == "cuda" else "fp32"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id_or_path,
            **load_kwargs,
        )
        self.model.eval()

    def _emit(self, msg: str) -> None:
        if self.log:
            try:
                self.log(msg)
            except Exception:
                pass

    @torch.inference_mode()
    def _generate(self, prompt: str, images: List[Image.Image], tag: str = "gen") -> str:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_text], images=[images], return_tensors="pt", padding=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)

        t0 = time.time()
        self._emit(f"    · {tag}: generating ({len(images)} img, {int(self.cfg.max_new_tokens)} tok max)…")
        out_ids = self.model.generate(**inputs, max_new_tokens=int(self.cfg.max_new_tokens))
        in_len = int(inputs["input_ids"].shape[-1])
        gen_ids = out_ids[0][in_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)
        self._emit(f"    · {tag}: done in {time.time() - t0:.1f}s")
        return text.strip()

    def _frame_budget(self) -> int:
        # Frame count nudges with the FPS slider but stays within the VLM sweet spot
        # [min, cap] (~6-8); more than ~8 images dilutes the VLM, so it plateaus by design.
        return max(self.cfg.num_frames_min, min(self.cfg.num_frames_cap, int(self.cfg.fps) + 4))

    @torch.inference_mode()
    def describe_video_struct(self, video_path: str, client_requirement: Optional[str] = None) -> Dict:
        frames_bgr = _sample_frames(video_path, fps=self.cfg.fps, cap_frames=self._frame_budget())
        cv_frames = _sample_frames_cv(
            video_path, fps_min=max(self.cfg.cv_fps_min, self.cfg.fps), cap_frames=self.cfg.cv_frames_cap)
        return self._describe_from_frames(frames_bgr, cv_frames, client_requirement)

    @torch.inference_mode()
    def describe_sequence_struct(self, seq_dir, client_requirement: Optional[str] = None) -> Dict:
        """Same analysis as describe_video_struct but samples directly from an image
        sequence folder — no mp4 encode, so it can't hit the OpenCV VideoWriter crash."""
        frames_bgr = _sample_frames_from_folder(seq_dir, cap_frames=self._frame_budget())
        cv_frames = _sample_frames_cv_from_folder(
            seq_dir, fps_min=max(self.cfg.cv_fps_min, self.cfg.fps), cap_frames=self.cfg.cv_frames_cap)
        return self._describe_from_frames(frames_bgr, cv_frames, client_requirement)

    def _describe_from_frames(self, frames_bgr, cv_frames, client_requirement: Optional[str] = None) -> Dict:
        self._emit(f"  - {len(frames_bgr)} VLM frame(s) sampled; classifying camera motion…")
        images = [_pil_from_bgr(f, max_side=self.cfg.max_side) for f in frames_bgr]

        cam_src = cv_frames if len(cv_frames) >= 2 else frames_bgr
        mags, vecs, scales = _estimate_global_motion(cam_src, scale=self.cfg.cv_scale)
        cam_label = _classify_camera(mags, vecs, scales, self.cfg)
        self._emit(f"  - camera: {cam_label}. Running VLM (3 passes)…")

        scene_prompt = (
            "Describe what is happening in this shot as a short paragraph (2-3 sentences). "
            "Mention key visible subjects and any obvious action."
        )
        scene = _short_paragraph(self._generate(scene_prompt, images, tag="scene [1/3]"))

        list_prompt = (
            "These frames are sampled across one shot; an object may appear in only some of them. "
            "List ALL distinct visible things/elements across ALL frames as a comma-separated list. "
            "Explicitly include any people, animals/pets (e.g., dog, cat, bird, horse), and vehicles, "
            "even if small, partially visible, or present in only one frame. "
            "Also include important environment/background elements (floor, wall, ground, table, rocks, trees) if visible. "
            "Avoid repetition and synonyms/duplicates. "
            "Return ONLY the comma-separated list. No extra text."
        )
        things = _parse_csvish_list(self._generate(list_prompt, images, tag="things [2/3]"))

        analysis = self._analyze_for_matchmove(images, things)

        return {
            "scene_elements": scene,
            "camera_movement": cam_label,
            "things": things,
            **analysis,
        }

    @torch.inference_mode()
    def _analyze_for_matchmove(self, images: List[Image.Image], things: List[str]) -> Dict:
        """One structured-JSON prompt extracting matchmove-relevant signals.

        Returns keys: moving_things, bad_track_regions, foreground_occluders,
        quality_flags, depth_layers (fg/mg/bg), parallax.
        Empty/safe defaults on any parse failure (never raises).
        """
        things_str = ", ".join(things) if things else "(none listed)"
        prompt = (
            "You are assisting a VFX matchmove (camera tracking) pipeline. "
            "Analyze this shot for tracking. Detected things: " + things_str + ".\n"
            "Return ONLY a JSON object with these exact keys:\n"
            '{"moving_things": [list of things that MOVE/deform relative to the scene (people, vehicles, animals, fluids)],\n'
            ' "bad_track_regions": [surfaces bad for point tracking: reflections, glass, water, sky, mirrors, screens, plain/textureless areas],\n'
            ' "foreground_occluders": [things passing in FRONT of and occluding the background],\n'
            ' "quality_flags": [any of: motion blur, out of focus, low texture, overexposed, underexposed, low light, flicker, heavy grain, compression artifacts],\n'
            ' "depth_layers": {"fg": "short foreground desc", "mg": "short midground desc", "bg": "short background desc"},\n'
            ' "parallax": "yes" or "weak" or "no"}\n'
            "Use only nouns visible in the shot. Empty list if none. No prose outside the JSON."
        )
        defaults = {
            "moving_things": [],
            "bad_track_regions": [],
            "foreground_occluders": [],
            "quality_flags": [],
            "depth_layers": {"fg": "", "mg": "", "bg": ""},
            "parallax": "",
        }
        try:
            raw = self._generate(prompt, images, tag="matchmove [3/3]")
        except Exception:
            return defaults
        obj = _extract_json_obj(raw)
        if not obj:
            return defaults

        out = dict(defaults)
        out["moving_things"] = _as_str_list(obj.get("moving_things"))
        out["bad_track_regions"] = _as_str_list(obj.get("bad_track_regions"))
        out["foreground_occluders"] = _as_str_list(obj.get("foreground_occluders"))
        qf = [q for q in _as_str_list(obj.get("quality_flags")) if q.lower() in _QUALITY_VOCAB]
        out["quality_flags"] = qf
        dl = obj.get("depth_layers")
        if isinstance(dl, dict):
            out["depth_layers"] = {
                "fg": str(dl.get("fg") or "").strip(),
                "mg": str(dl.get("mg") or "").strip(),
                "bg": str(dl.get("bg") or "").strip(),
            }
        pax = str(obj.get("parallax") or "").strip().lower()
        out["parallax"] = pax if pax in ("yes", "weak", "no") else ""
        return out
