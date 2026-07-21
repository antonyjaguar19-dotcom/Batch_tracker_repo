from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

LogCb = Callable[[str], None]
ProgressCb = Callable[[int, int], None]
StatusCb = Callable[[str], None]

FRAME_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")
VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")


@dataclass(frozen=True)
class RunnerConfig:
    guide_json_path: Path
    input_root: Optional[Path]
    output_root: Path
    weights_path: Path
    # CV motion backstop: for camera shots, also black out pixels that move
    # independently of the camera (optical-flow residual), catching movers that
    # the VLM/heuristics missed entirely. Disable via env BTR_MOTION_BACKSTOP=0.
    motion_backstop: bool = True
    motion_thresh: float = 1.5          # residual flow magnitude (px @ proc width)
    motion_proc_width: int = 480        # downscale width for flow compute
    motion_min_area_frac: float = 0.0008  # ignore blobs smaller than this frac of frame
    motion_dilate_px: int = 9           # grow moving region for safety margin
    # Mask ML-style edge dilation: grow the EXCLUDE (mover/black) region of the
    # final SAM mask by this many pixels so auto-trackers stay off soft edges
    # (hair, motion-blur fringes) -- mirrors SynthEyes Mask ML's "Mask Dilation".
    # 0 = off (tight SAM edges). Try ~8-15 to match SynthEyes tracking behavior.
    mask_dilation_px: int = 0


def load_masking_guide(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Masking guide JSON must be an object/dict at the top level.")
    data["shots"] = normalize_shots(data)
    return data


def normalize_shots(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize shot entries from various JSON schemas.

    Supports BOTH legacy single-task schema and new multi-task schema:
      shots[].tasks = [ {task_id, include_prompts/mask_includes, exclude_prompts/mask_excludes,
                         track_mode, mask_subdir}, ... ]

    Each task becomes an individual normalized entry, but keeps the same shot_name
    for frame discovery. Output directories are separated via mask_subdir.
    """
    shots = data.get("shots", data.get("Shots", []))
    if not isinstance(shots, list):
        raise ValueError("Expected 'shots' to be a list.")

    def split_csv(s: str) -> List[str]:
        return [p.strip() for p in s.split(",") if p.strip()]

    def list_from(d: Dict[str, Any], *keys: str) -> List[str]:
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        return []

    def normalize_prompts(include_prompts: List[str], exclude_prompts: List[str]) -> tuple[List[str], List[str]]:
        # Expand comma-separated single strings
        if len(include_prompts) == 1 and "," in include_prompts[0]:
            include_prompts = split_csv(include_prompts[0])
        if len(exclude_prompts) == 1 and "," in exclude_prompts[0]:
            exclude_prompts = split_csv(exclude_prompts[0])
        return include_prompts, exclude_prompts

    norm: List[Dict[str, Any]] = []
    for s in shots:
        if not isinstance(s, dict):
            continue

        masking = s.get("masking", {})
        combined = {**s, **masking} if isinstance(masking, dict) else dict(s)

        shot_name = combined.get("shot_name") or combined.get("name") or combined.get("shot")
        if not shot_name:
            continue

        input_dir = combined.get("input_dir") or combined.get("frames_dir") or combined.get("source_dir")

        # Multi-task schema
        tasks = combined.get("tasks")
        if isinstance(tasks, list) and tasks:
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                task_id = str(t.get("task_id") or t.get("id") or "task").strip() or "task"

                include_prompts = list_from(
                    t,
                    "include_prompts", "mask_include", "mask_includes", "mask_includes_list",
                    "include", "include_text", "text_include", "sam3_include_prompt"
                )
                exclude_prompts = list_from(
                    t,
                    "exclude_prompts", "mask_exclude", "mask_excludes", "mask_excludes_list",
                    "exclude", "exclude_text", "text_exclude", "sam3_exclude_prompt"
                )
                include_prompts, exclude_prompts = normalize_prompts(include_prompts, exclude_prompts)

                track_mode = str(t.get("track_mode") or combined.get("track_mode") or "").strip()

                # mask_mode is advisory only; keep legacy behavior
                mask_mode = str(t.get("mask_mode") or combined.get("mask_mode") or "include").strip().lower()
                if mask_mode not in ("include", "exclude"):
                    mask_mode = "include"

                mask_subdir = str(t.get("mask_subdir") or f"masks_{task_id}").strip()

                norm.append(
                    {
                        "shot_name": str(shot_name),
                        "task_id": task_id,
                        "track_mode": track_mode,
                        "mask_mode": mask_mode,
                        "include_prompts": include_prompts,
                        "exclude_prompts": exclude_prompts,
                        "input_dir": str(input_dir) if input_dir else None,
                        "mask_subdir": mask_subdir,
                        "frame_start": int(combined.get("frame_start") or 0),
                        "frame_end": int(combined.get("frame_end") or 0),
                    }
                )
            continue

        # Legacy single-task schema
        include_prompts = list_from(
            combined,
            "include_prompts", "mask_include", "mask_includes", "mask_includes_list",
            "include", "include_text", "text_include", "sam3_include_prompt"
        )
        exclude_prompts = list_from(
            combined,
            "exclude_prompts", "mask_exclude", "mask_excludes", "mask_excludes_list",
            "exclude", "exclude_text", "text_exclude", "sam3_exclude_prompt"
        )
        include_prompts, exclude_prompts = normalize_prompts(include_prompts, exclude_prompts)

        track_mode = str(combined.get("track_mode", "") or "").strip()
        mask_mode = str(combined.get("mask_mode", "") or "include").strip().lower()
        if mask_mode not in ("include", "exclude"):
            mask_mode = "include"

        norm.append(
            {
                "shot_name": str(shot_name),
                "task_id": "",
                "track_mode": track_mode,
                "mask_mode": mask_mode,
                "include_prompts": include_prompts,
                "exclude_prompts": exclude_prompts,
                "input_dir": str(input_dir) if input_dir else None,
                "mask_subdir": "masks",
                "frame_start": int(combined.get("frame_start") or 0),
                "frame_end": int(combined.get("frame_end") or 0),
            }
        )

    return norm


def is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def has_frames(dir_path: Path) -> bool:
    if not dir_path.exists() or not dir_path.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in FRAME_EXTS for p in dir_path.iterdir())


def find_frames_dir_recursive(root_dir: Path, max_depth: int = 8) -> Optional[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    root_depth = len(root_dir.parts)
    for current, dirs, files in os.walk(root_dir):
        cur_path = Path(current)
        depth = len(cur_path.parts) - root_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        for fn in files:
            if Path(fn).suffix.lower() in FRAME_EXTS:
                return cur_path
        dirs.sort()
    return None


def find_video_file_any(root_dir: Path, max_depth: int = 6) -> Optional[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    root_depth = len(root_dir.parts)
    for current, dirs, files in os.walk(root_dir):
        cur_path = Path(current)
        depth = len(cur_path.parts) - root_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        for fn in sorted(files):
            p = cur_path / fn
            if is_video_file(p):
                return p
        dirs.sort()
    return None


def find_video_file_matching_shot(root_dir: Path, shot_name: str, max_depth: int = 6) -> Optional[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    target = shot_name.lower()
    root_depth = len(root_dir.parts)
    for current, dirs, files in os.walk(root_dir):
        cur_path = Path(current)
        depth = len(cur_path.parts) - root_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        for fn in sorted(files):
            p = cur_path / fn
            if is_video_file(p) and target in p.stem.lower():
                return p
        dirs.sort()
    return None


def ensure_ffmpeg_available() -> str:
    """Return ffmpeg path. Prefer system PATH, else imageio-ffmpeg bundled binary."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg2 = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg2 and Path(ffmpeg2).exists():
            return ffmpeg2
    except Exception:
        pass

    raise RuntimeError(
        "ffmpeg was not found. This tool needs ffmpeg to extract JPG frames from MOV/MP4.\n\n"
        "Fix options:\n"
        "  A) Install ffmpeg system-wide and verify in a NEW PowerShell:  ffmpeg -version\n"
        "  B) Or install python package imageio-ffmpeg into the SAME venv (already included in requirements).\n"
    )


def extract_jpeg_sequence(video_path: Path, out_dir: Path, log: Optional[LogCb] = None) -> Path:
    """Extract JPEG sequence to out_dir, skipping if frames already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted([p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")])
    if len(existing) >= 2:
        if log:
            log(f"  Using existing extracted JPG sequence: {out_dir}")
        return out_dir

    ffmpeg = ensure_ffmpeg_available()
    out_pattern = str(out_dir / "%06d.jpg")

    cmd = [ffmpeg, "-y", "-i", str(video_path), "-qscale:v", "2", out_pattern]

    if log:
        log("  Extracting JPG sequence via ffmpeg:")
        log(f"    video: {video_path}")
        log(f"    out  : {out_dir}")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to extract frames.\n\n"
            f"Video: {video_path}\nOut: {out_dir}\n\n"
            f"STDERR (tail):\n{proc.stderr[-4000:]}"
        )

    extracted = sorted([p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")])
    if not extracted:
        raise RuntimeError(f"ffmpeg completed but no JPG frames were created in: {out_dir}")
    return out_dir


def find_video_for_shot(input_root: Optional[Path], output_root: Optional[Path], shot_name: str) -> Optional[Path]:
    """Locate a video for a shot across common layouts."""
    # A) <input_root>/<shot_name>/...
    if input_root:
        base = input_root / shot_name
        if base.exists() and base.is_dir():
            v = find_video_file_matching_shot(base, shot_name, max_depth=10) or find_video_file_any(base, max_depth=10)
            if v:
                return v

        # B) video directly under input_root with shot name in filename
        if input_root.exists() and input_root.is_dir():
            v = find_video_file_matching_shot(input_root, shot_name, max_depth=3)
            if v:
                return v

            # fallback: if there is exactly one video within depth, use it
            v_any = find_video_file_any(input_root, max_depth=3)
            return v_any

    # C) <output_root>/<shot_name>/... (rare, but allowed)
    if output_root:
        base = output_root / shot_name
        if base.exists() and base.is_dir():
            v = find_video_file_matching_shot(base, shot_name, max_depth=10) or find_video_file_any(base, max_depth=10)
            if v:
                return v

    return None


def resolve_shot_frames_dir(
    shot: Dict[str, Any],
    input_root: Optional[Path],
    output_root: Path,
    log: Optional[LogCb] = None,
) -> Path:
    """Resolve frames dir; if only a video exists, extract JPGs into OUT and return that dir."""
    shot_name = shot["shot_name"]
    common_subdirs = ["frames", "frame", "images", "image", "renders", "render", "plates", "plate", "img", "rgb", "input", "source"]

    tried: List[Path] = []

    # 1) JSON input_dir: may be frames dir, parent dir, or a video file path
    if shot.get("input_dir"):
        p = Path(shot["input_dir"])
        tried.append(p)
        if p.exists():
            if p.is_file() and is_video_file(p):
                return extract_jpeg_sequence(p, output_root / shot_name / "frames_jpg", log=log)
            if p.is_dir():
                if has_frames(p):
                    return p
                rec = find_frames_dir_recursive(p, max_depth=10)
                if rec:
                    return rec
                v = find_video_file_matching_shot(p, shot_name, max_depth=10) or find_video_file_any(p, max_depth=10)
                if v:
                    return extract_jpeg_sequence(v, output_root / shot_name / "frames_jpg", log=log)

    def candidates(root_dir: Optional[Path]) -> List[Path]:
        if not root_dir:
            return []
        base = root_dir / shot_name
        return [base] + [base / sub for sub in common_subdirs]

    # 2) input_root candidates
    for c in candidates(input_root):
        tried.append(c)
        if has_frames(c):
            return c

    # 3) output_root candidates (if user already has frames there)
    for c in candidates(output_root):
        tried.append(c)
        if has_frames(c):
            return c

    # 4) recursive scan in shot folders
    if input_root:
        base = input_root / shot_name
        tried.append(base)
        rec = find_frames_dir_recursive(base, max_depth=10)
        if rec:
            return rec

    base_out = output_root / shot_name
    tried.append(base_out)
    rec2 = find_frames_dir_recursive(base_out, max_depth=10)
    if rec2:
        return rec2

    # 5) auto video detect + extract
    v = find_video_for_shot(input_root, output_root, shot_name)
    if v:
        return extract_jpeg_sequence(v, output_root / shot_name / "frames_jpg", log=log)

    tried_str = "\n".join(f"  - {t}" for t in tried[:60])
    raise FileNotFoundError(
        f"Could not resolve input frames dir for shot '{shot_name}'.\n\n"
        "Supported frame formats: png/jpg/tif sequences.\n"
        "Video formats supported for auto-extract: mov/mp4/m4v/avi/mkv (requires ffmpeg).\n\n"
        "Fix options:\n"
        f"  1) Put frames in: <input_root>\\{shot_name}\\frames\\000001.jpg (or png/tif)\n"
        f"  2) Put a video in: <input_root>\\{shot_name}\\{shot_name}.mp4 (or mov)\n"
        f"  3) Or put video directly under input_root named like '{shot_name}.mp4'\n"
        "  4) Or set JSON input_dir to the full path of the frames folder or video file.\n\n"
        f"Paths tried (first 60):\n{tried_str}"
    )


def list_frames(frames_dir: Path) -> List[Path]:
    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in FRAME_EXTS]
    frames.sort()
    return frames


def frame_hw(frame_path: Path) -> Tuple[int, int]:
    img = Image.open(frame_path)
    w, h = img.size
    return h, w


def write_full_white_mask(frame_path: Path, out_path: Path) -> None:
    h, w = frame_hw(frame_path)
    alpha = Image.fromarray(np.full((h, w), 255, dtype=np.uint8), mode="L")
    alpha.save(out_path)


def ensure_out_dirs(out_root: Path, shot_name: str, masks_subdir: str = "masks") -> Path:
    """Create the masks output directory for a shot.

    Supports multi-task outputs by allowing per-task subdirectories, e.g.:
      OUT/Shot_01/masks_camera
      OUT/Shot_01/masks_object
    """
    masks_dir = out_root / shot_name / (masks_subdir or "masks")
    masks_dir.mkdir(parents=True, exist_ok=True)
    return masks_dir


def union_masks_from_results(results: Any, frame_path: Path, keep: bool) -> np.ndarray:
    h, w = frame_hw(frame_path)
    union = np.zeros((h, w), dtype=bool)

    if results is None:
        return np.full((h, w), 255 if not keep else 0, dtype=np.uint8)

    if not isinstance(results, (list, tuple)):
        results = [results]

    for r in results:
        masks_obj = getattr(r, "masks", None)
        if masks_obj is None:
            continue
        data = getattr(masks_obj, "data", None)
        if data is None:
            continue
        try:
            m = data.detach().float().cpu().numpy()
        except Exception:
            try:
                m = np.array(data)
            except Exception:
                continue

        if m.ndim == 2:
            union |= (m > 0.5)
        elif m.ndim == 3:
            union |= np.any(m > 0.5, axis=0)

    if keep:
        return np.where(union, 255, 0).astype(np.uint8)
    return np.where(union, 0, 255).astype(np.uint8)


def compute_motion_keep_mask(prev_path: Path, cur_path: Path, out_hw: Tuple[int, int],
                             cfg: "RunnerConfig") -> Optional[np.ndarray]:
    """Return a keep-mask (255=static/keep, 0=independently-moving/ignore) for `cur_path`.

    Optical-flow residual after subtracting the dominant (camera) motion. Pixels whose
    flow deviates from the global field = moving independently of the camera. This is the
    4th backstop: it masks movers the VLM never detected (no noun, no scene mention).
    Returns None if OpenCV is unavailable or frames can't be read (caller skips gracefully).
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    prev = cv2.imread(str(prev_path))
    cur = cv2.imread(str(cur_path))
    if prev is None or cur is None:
        return None

    H, W = int(out_hw[0]), int(out_hw[1])
    src_w = max(1, cur.shape[1])
    scale = min(1.0, float(cfg.motion_proc_width) / float(src_w))
    sw = max(32, int(round(cur.shape[1] * scale)))
    sh = max(32, int(round(cur.shape[0] * scale)))

    pg = cv2.cvtColor(cv2.resize(prev, (sw, sh)), cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(cv2.resize(cur, (sw, sh)), cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Dominant (camera) motion = median flow vector; residual reveals independent movers.
    med = np.median(flow.reshape(-1, 2), axis=0)
    res = flow - med
    mag = np.sqrt(res[..., 0] ** 2 + res[..., 1] ** 2)
    moving = (mag > float(cfg.motion_thresh)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, k)
    moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, k)

    min_area = int(float(cfg.motion_min_area_frac) * moving.shape[0] * moving.shape[1])
    if min_area > 0 and moving.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(moving, 8)
        clean = np.zeros_like(moving)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[lab == i] = 255
        moving = clean

    if cfg.motion_dilate_px and moving.any():
        d = max(1, int(round(cfg.motion_dilate_px * scale)))
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        moving = cv2.dilate(moving, dk)

    moving_full = cv2.resize(moving, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.where(moving_full > 0, 0, 255).astype(np.uint8)


def run_sam3_batch(
    cfg: RunnerConfig,
    log_cb: Optional[LogCb] = None,
    progress_cb: Optional[ProgressCb] = None,
    status_cb: Optional[StatusCb] = None,
) -> None:
    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def status(msg: str) -> None:
        if status_cb:
            status_cb(msg)
        log(msg)

    guide = load_masking_guide(cfg.guide_json_path)
    shots = guide.get("shots", [])
    if not shots:
        raise ValueError("No shots found in JSON under 'shots'.")

    status("Loading Ultralytics SAM3… (first time can take a while)")
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed or does not include SAM3. "
            "Install/upgrade ultralytics (>= 8.3.237) and try again."
        ) from e

    if not cfg.weights_path.exists():
        raise FileNotFoundError(f"sam3 weights not found: {cfg.weights_path}")

    overrides = dict(conf=0.25, task="segment", mode="predict", model=str(cfg.weights_path), half=True, save=False, verbose=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Pre-scan and (if needed) extract video frames
    total_frames = 0
    per_shot_frames: Dict[str, List[Path]] = {}
    per_shot_frames_dir: Dict[str, Path] = {}

    for shot in shots:
        frames_dir = resolve_shot_frames_dir(shot, cfg.input_root, cfg.output_root, log=log)
        frames = list_frames(frames_dir)

        if not frames:
            # force one more attempt: maybe a video exists and extraction failed to be triggered earlier
            v = find_video_for_shot(cfg.input_root, cfg.output_root, shot["shot_name"])
            if v:
                frames_dir = extract_jpeg_sequence(v, cfg.output_root / shot["shot_name"] / "frames_jpg", log=log)
                frames = list_frames(frames_dir)

        if not frames:
            raise FileNotFoundError(
                f"No frames found for shot '{shot['shot_name']}'. "
                "Provide an image sequence or a video (mov/mp4) accessible via input_root or JSON input_dir."
            )

        # Honor per-shot frame range (1-based inclusive); only mask those frames.
        fs = int(shot.get("frame_start") or 0)
        fe = int(shot.get("frame_end") or 0)
        if fs > 0 or fe > 0:
            n = len(frames)
            a = max(0, fs - 1) if fs > 0 else 0
            b = fe if (fe > 0 and fe <= n) else n
            if a < b:
                frames = frames[a:b]
                log(f"  frame range: {a + 1}-{b} of {n}")

        per_shot_frames_dir[shot["shot_name"]] = frames_dir
        per_shot_frames[shot["shot_name"]] = frames
        total_frames += len(frames)

    done = 0
    if progress_cb:
        progress_cb(done, max(total_frames, 1))

    for shot in shots:
        shot_name = shot["shot_name"]
        task_id = str(shot.get("task_id") or "").strip()
        status(f"Shot: {shot_name}" + (f" | task={task_id}" if task_id else ""))

        frames_dir = per_shot_frames_dir[shot_name]
        frames = per_shot_frames[shot_name]
        log(f"  frames_dir: {frames_dir}")

        masks_dir = ensure_out_dirs(cfg.output_root, shot_name, masks_subdir=str(shot.get("mask_subdir") or "masks"))

        track_mode = (shot.get("track_mode") or "").strip().lower()
        mask_mode = (shot.get("mask_mode") or "include").strip().lower()
        include_prompts: List[str] = shot.get("include_prompts") or []
        exclude_prompts: List[str] = shot.get("exclude_prompts") or []

        # If mask_mode == exclude and exclude empty, treat include as exclude list
        if mask_mode == "exclude" and (not exclude_prompts) and include_prompts:
            exclude_prompts = include_prompts
            include_prompts = []

        log(f"  track_mode: {track_mode or '(default)'}")
        log(f"  mask_mode : {mask_mode}")
        log(f"  include   : {include_prompts}")
        log(f"  exclude   : {exclude_prompts}")

        # CV motion backstop applies to CAMERA shots (track outside / exclude / task=camera),
        # including no_mask_needed (Qwen missed all movers => full-white mask otherwise).
        is_camera = (task_id.lower() == "camera") or (track_mode == "track_outside_mask") or (mask_mode == "exclude")
        motion_on = bool(cfg.motion_backstop) and is_camera and (os.environ.get("BTR_MOTION_BACKSTOP", "1") != "0")
        if motion_on:
            log("  motion backstop: ON (mask pixels moving independently of camera)")
        prev_frame_path: Optional[Path] = None

        for frame_path in frames:
            # Frame number must be the TRAILING token (name_####.png) so SynthEyes
            # detects it as a numbered image sequence for +Alpha matte loading.
            # (Old "####_alpha.png" put digits first -> SynthEyes couldn't read the
            # sequence.) frame_path.stem is the padded frame number for numbered
            # sequences, so mask_<stem>.png == mask_000001.png. Downstream readers
            # (tracker_core get_global_mask_idx, existing-mask glob) use sorted order,
            # not the filename, so this rename is safe; sort order is preserved.
            out_mask = masks_dir / f"mask_{frame_path.stem}.png"

            if track_mode == "no_mask_needed" and not motion_on:
                write_full_white_mask(frame_path, out_mask)
                prev_frame_path = frame_path
                done += 1
                if progress_cb:
                    progress_cb(done, max(total_frames, 1))
                continue

            alpha_keep: Optional[np.ndarray] = None

            if track_mode != "no_mask_needed":
                predictor.set_image(str(frame_path))

                if include_prompts:
                    res = predictor(text=include_prompts)
                    alpha_keep = union_masks_from_results(res, frame_path, keep=True)

                if alpha_keep is None:
                    h, w = frame_hw(frame_path)
                    alpha_keep = np.full((h, w), 255, dtype=np.uint8)

                if exclude_prompts:
                    res = predictor(text=exclude_prompts)
                    ex_keep = union_masks_from_results(res, frame_path, keep=False)
                    alpha_keep = np.minimum(alpha_keep, ex_keep)
            else:
                h, w = frame_hw(frame_path)
                alpha_keep = np.full((h, w), 255, dtype=np.uint8)

            # 4th backstop: black out independently-moving pixels (camera shots only).
            if motion_on and prev_frame_path is not None:
                motion_keep = compute_motion_keep_mask(prev_frame_path, frame_path, alpha_keep.shape, cfg)
                if motion_keep is not None:
                    alpha_keep = np.minimum(alpha_keep, motion_keep)

            # Mask ML-style edge dilation: grow the EXCLUDE (black/mover) region by
            # mask_dilation_px so trackers avoid soft edges. Growing black = eroding
            # the white keep-mask. Applied last, after all mask sources are merged.
            if int(cfg.mask_dilation_px) > 0 and alpha_keep is not None and (alpha_keep == 0).any():
                import cv2
                d = int(cfg.mask_dilation_px)
                dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
                alpha_keep = cv2.erode(alpha_keep, dk)

            Image.fromarray(alpha_keep, mode="L").save(out_mask)

            prev_frame_path = frame_path
            done += 1
            if progress_cb:
                progress_cb(done, max(total_frames, 1))

    status("All shots complete.")
