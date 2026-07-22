# -*- coding: utf-8 -*-
"""
Unified Batch Tracker — Automated 5-Step Workflow
Merges Qwen2/LLaMA Analysis, SAM3 Masking, and SynthEyes/TAPNext++ Execution into one UI.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
import queue
import traceback
import importlib
import importlib.util
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr
import pandas as pd

# -----------------------------------------------------------------------------
# Fixed runtime endpoints / weights (kept out of the UI)
# You can override these via environment variables if needed.
# -----------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.environ.get("BTR_OLLAMA_URL", "http://localhost:11434")
# Repo root = parent of this file's dir (app.py lives in app/). Old "BTr" tree kept
# only as last-resort fallback.
_HERE = Path(__file__).resolve().parents[1]
_LEGACY_ROOT = Path(r"D:\Jefrin\BTr\batch_tracker_v001_starter")
DEFAULT_SAM3_WEIGHTS = os.environ.get(
    "BTR_SAM3_WEIGHTS",
    str(_HERE / "pipeline" / "sam3" / "weights" / "sam3.pt"),
)


# -----------------------------------------------------------------------------
# 1. SYSTEM PATH BOOTSTRAP
# -----------------------------------------------------------------------------
def _add_to_sys_path(path_str: str):
    """Safely add a path to sys.path if missing."""
    if path_str and path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"DEBUG: Added to sys.path: {path_str}")

def _bootstrap_paths():
    """Recursively find critical modules (core, SAM3, Qwen) and fix sys.path."""
    _add_to_sys_path(os.getcwd())

    candidates = [
        _HERE,
        Path.cwd(),
        _LEGACY_ROOT,  # last-resort fallback for the old BTr layout
    ]

    found_core = False

    print("DEBUG: Starting Path Bootstrap...")

    for root in candidates:
        if not root.exists(): continue

        # Core
        if not found_core:
            try:
                for p in root.rglob("io_parsers.py"):
                    if p.parent.name == "core":
                        package_root = str(p.parent.parent)
                        _add_to_sys_path(package_root)
                        found_core = True
                        break
            except Exception: pass

    if not found_core:
        print("WARNING: Could not find 'core' package automatically.")

    return _HERE

PROJECT_ROOT = _bootstrap_paths()

# -----------------------------------------------------------------------------
# 2. IMPORTS
# -----------------------------------------------------------------------------

# --- 2.1 Core Imports ---
load_requirements = None
load_qwen2_v1_scene_cam_things = None
OllamaConfig = None
OllamaReasoner = None
build_batch_tracker_json = None

# --- 2.2 Execution Imports (DIRECT FILE LOADING) ---
BatchTrackerRunner = None
RunnerConfig = None
probe_video_meta = None
TRACKER_IMPORT_ERROR = None

def _load_tracker_direct():
    """Load tracker + video_meta in a way that preserves `app.*` imports."""
    try:
        from app.video_meta import probe_video_meta as PVM  # type: ignore
        from app.tracker_core import BatchTrackerRunner as BTR, RunnerConfig as RC  # type: ignore
        return PVM, BTR, RC, None
    except Exception as e_pkg:
        pkg_err = str(e_pkg)

    roots = [
        _HERE,
        Path(os.getcwd()),
        _LEGACY_ROOT,
    ]

    tracker_path: Path | None = None
    meta_path: Path | None = None

    preferred_tracker = _HERE / "app" / "syntheyes_runner.py"
    preferred_meta = _HERE / "app" / "video_meta.py"
    if preferred_tracker.exists():
        tracker_path = preferred_tracker
    if preferred_meta.exists():
        meta_path = preferred_meta

    for r in roots:
        if not r.exists(): continue
        if tracker_path is None:
            for h in r.rglob("tracker_core.py"):
                if "venv" in str(h) or "site-packages" in str(h): continue
                tracker_path = h
                break
        if meta_path is None:
            for h in r.rglob("video_meta.py"):
                if "venv" in str(h) or "site-packages" in str(h): continue
                meta_path = h
                break
        if tracker_path is not None and meta_path is not None:
            break

    if tracker_path is None:
        return None, None, None, f"Could not find 'tracker_core.py'. Package import error: {pkg_err}"

    try:
        project_root = tracker_path.parent.parent if tracker_path.parent.name.lower() == "app" else tracker_path.parent
        _add_to_sys_path(str(project_root))

        spec = importlib.util.spec_from_file_location("tracker_core_module", str(tracker_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {tracker_path}")
        tc_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tc_mod)

        BTR = getattr(tc_mod, "BatchTrackerRunner", None)
        RC = getattr(tc_mod, "RunnerConfig", None)

        PVM = None
        if meta_path and meta_path.exists():
            project_root2 = meta_path.parent.parent if meta_path.parent.name.lower() == "app" else meta_path.parent
            _add_to_sys_path(str(project_root2))

            spec2 = importlib.util.spec_from_file_location("video_meta_module", str(meta_path))
            if spec2 is None or spec2.loader is None:
                raise RuntimeError(f"Could not create import spec for {meta_path}")
            vm_mod = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(vm_mod)
            PVM = getattr(vm_mod, "probe_video_meta", None)

        if BTR is None or RC is None:
            raise ImportError(f"Loaded {tracker_path} but could not find BatchTrackerRunner/RunnerConfig")

        return PVM, BTR, RC, None

    except Exception as e:
        return None, None, None, f"{e} | pkg_err={pkg_err}"


_TRACKER_LOADED = False

def _load_video_meta_only():
    """Load probe_video_meta (cv2-only, torch-free) — safe to run at boot."""
    try:
        from app.video_meta import probe_video_meta as PVM  # type: ignore
        return PVM
    except Exception:
        pass
    for r in [_HERE, Path(os.getcwd()), _LEGACY_ROOT]:
        try:
            if not r.exists(): continue
        except Exception:
            continue
        cand = r / "app" / "video_meta.py"
        if cand.exists():
            try:
                _add_to_sys_path(str(cand.parent.parent))
                spec = importlib.util.spec_from_file_location("video_meta_module", str(cand))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, "probe_video_meta", None)
            except Exception:
                pass
    return None

def _ensure_tracker_loaded():
    """Import the TAPNext++/torch stack lazily (only when the tapnext fallback runs).
    Torch/CUDA is kept OFF the boot path: initialising CUDA at UI startup can wedge the
    GPU display driver on some machines and hard-freeze Windows."""
    global probe_video_meta, BatchTrackerRunner, RunnerConfig, TRACKER_IMPORT_ERROR, _TRACKER_LOADED
    if _TRACKER_LOADED:
        return
    _TRACKER_LOADED = True
    PVM, BTR, RC, err = _load_tracker_direct()
    if PVM is not None and probe_video_meta is None:
        probe_video_meta = PVM
    BatchTrackerRunner, RunnerConfig, TRACKER_IMPORT_ERROR = BTR, RC, err
    if err:
        print(f"CRITICAL: Tracker Import Failed: {err}")

# Boot: only the lightweight cv2 probe. The torch tracker loads on demand.
probe_video_meta = _load_video_meta_only()

# --- 2.2b SynthEyes tracking backend (replaces CoTracker as the track stage) ---
SynthEyesEngine = None
se_find_shot_frames = None
se_read_image_size = None
se_preset_track_count = None
SE_PRESET_NAMES = []
SE_DEFAULT_PRESET = "Normal / Handheld"
SYNTHEYES_IMPORT_ERROR = None

def _load_syntheyes_direct():
    """Load the SynthEyes engine module (fail-silent like the other stages)."""
    try:
        from app.syntheyes_engine import (  # type: ignore
            SynthEyesEngine as SE, find_shot_frames as FSF, read_image_size as RIS,
            preset_track_count as PTC, PRESET_NAMES as PN, DEFAULT_PRESET as DP,
        )
        return SE, FSF, RIS, PTC, PN, DP, None
    except Exception as e_pkg:
        pkg_err = str(e_pkg)
    try:
        # Known location only — never rglob the whole repo at boot (runtime/ is huge
        # and a directory walk here could stall app startup).
        se_path = _HERE / "app" / "syntheyes_engine.py"
        if not se_path.exists():
            return (None, None, None, None, [], "Normal / Handheld",
                    f"Could not find 'syntheyes_engine.py' at {se_path}. Package import error: {pkg_err}")
        _add_to_sys_path(str(se_path.parent.parent))
        spec = importlib.util.spec_from_file_location("syntheyes_engine_module", str(se_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return (getattr(mod, "SynthEyesEngine", None), getattr(mod, "find_shot_frames", None),
                getattr(mod, "read_image_size", None), getattr(mod, "preset_track_count", None),
                getattr(mod, "PRESET_NAMES", []), getattr(mod, "DEFAULT_PRESET", "Normal / Handheld"),
                None)
    except Exception as e:
        return (None, None, None, None, [], "Normal / Handheld", f"{e} | pkg_err={pkg_err}")

(SynthEyesEngine, se_find_shot_frames, se_read_image_size, se_preset_track_count,
 SE_PRESET_NAMES, SE_DEFAULT_PRESET, SYNTHEYES_IMPORT_ERROR) = _load_syntheyes_direct()

if SYNTHEYES_IMPORT_ERROR:
    print(f"WARNING: SynthEyes engine import failed: {SYNTHEYES_IMPORT_ERROR}")

# --- 2.3 SAM3 Imports ---
SamConfig = None
load_masking_guide = None
run_sam3_batch = None
SAM3_IMPORT_ERROR = None

def _load_sam3_direct():
    roots = [_HERE, Path(os.getcwd()), _LEGACY_ROOT]
    sam3_path = None
    for r in roots:
        if not r.exists(): continue
        hits = list(r.rglob("sam3_runner.py"))
        for h in hits:
            if "venv" not in str(h) and "site-packages" not in str(h):
                sam3_path = h
                break
        if sam3_path: break
        
    if not sam3_path:
        return None, None, None, "Could not find 'sam3_runner.py'."
        
    try:
        parent_dir = str(sam3_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if "sam3_runner" in sys.modules:
            del sys.modules["sam3_runner"]
        import sam3_runner as mod
        import importlib
        importlib.reload(mod)
        
        SC = getattr(mod, "RunnerConfig", None)
        LMG = getattr(mod, "load_masking_guide", None)
        RSB = getattr(mod, "run_sam3_batch", None)
        if SC is None:
            return None, None, None, "Loaded sam3_runner.py but 'RunnerConfig' class was missing."
        return SC, LMG, RSB, None
    except Exception as e:
        return None, None, None, f"{str(e)} | Path: {sam3_path}"

SAM3_IMPORT_ERROR = None
_SAM3_LOADED = False

def _ensure_sam3_loaded():
    """Import the SAM3/torch stack lazily (only at the Mask step) — off the boot path."""
    global SamConfig, load_masking_guide, run_sam3_batch, SAM3_IMPORT_ERROR, _SAM3_LOADED
    if _SAM3_LOADED:
        return
    _SAM3_LOADED = True
    SamConfig, load_masking_guide, run_sam3_batch, SAM3_IMPORT_ERROR = _load_sam3_direct()
    if SAM3_IMPORT_ERROR:
        print(f"CRITICAL: SAM3 Import Failed: {SAM3_IMPORT_ERROR}")

# --- 2.4 Qwen2 Loader ---
run_qwen2_batch = None
def _load_qwen_robustly():
    roots = [_HERE, Path(os.getcwd()), _LEGACY_ROOT]
    found = None
    for r in roots:
        if not r.exists(): continue
        for p in r.rglob("run_qwen2_shot_describer.py"):
            if "venv" in str(p): continue
            found = p
            break
        if found: break
    if found:
        _add_to_sys_path(str(found.parent))
        try:
            spec = importlib.util.spec_from_file_location("run_qwen2_shot_describer", str(found))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "run_batch", None)
        except Exception as e:
            print(f"Qwen load error: {e}")
            return None
    return None

_QWEN_LOADED = False

def _ensure_qwen_loaded():
    """Import the Qwen2.5-VL/torch stack lazily (only at Analyze) — off the boot path."""
    global run_qwen2_batch, _QWEN_LOADED
    if _QWEN_LOADED:
        return
    _QWEN_LOADED = True
    run_qwen2_batch = _load_qwen_robustly()

# -----------------------------------------------------------------------------
# 3. UTILITIES
# -----------------------------------------------------------------------------
def _tk_pick_file(initial: str = "") -> str:
    """Opens a system dialog to pick a FILE."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(initialdir=os.path.dirname(initial) if initial else None)
        root.destroy()
        return path or ""
    except Exception: return ""

def _tk_pick_folder(initial: str = "") -> str:
    """Opens a system dialog to pick a FOLDER."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(initialdir=initial if initial else None)
        root.destroy()
        return path or ""
    except Exception: return ""

def _gb(bytes_count: float) -> float:
    return float(bytes_count) / (1024.0 ** 3)

def est_vram(w: int, h: int, frames: int, grid_size: int = 10) -> str:
    if w <= 0 or h <= 0 or frames <= 0: return "0.0 GB"
    grid = max(1, int(grid_size))
    video_bytes = float(w) * float(h) * float(frames) * 3.0 * 4.0
    tracks_bytes = float(frames) * (grid**2) * 2.0 * 4.0
    gb = _gb(video_bytes + tracks_bytes)
    return f"{gb:.2f} GB"

def list_shots(in_root: str) -> List[str]:
    if not in_root or not os.path.exists(in_root): return []
    root = Path(in_root)
    dirs = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if dirs: return sorted(dirs)
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    files = [f.stem for f in root.iterdir() if f.is_file() and f.suffix.lower() in exts]
    return sorted(list(set(files)))

# -----------------------------------------------------------------------------
# Studio network plate structure:
#   <shows_root>/<show>/<shot>/in/plates/<version>/<plate.exr | .jpg | .jpeg>
# The show dropdown scans <shows_root>; each shot exposes its own version list
# (folders vNNN under in/plates), defaulting to the highest = latest.
# -----------------------------------------------------------------------------
_VER_RE = re.compile(r"^[vV](\d+)$")

# Studio pipeline folders that live alongside real shots under <shows_root>/<show>
# but are NOT shots (some are admin-only and raise on stat). Compared case-insensitively.
_NON_SHOT_DIRS = {"assets", "bid", "bip", "common", "home", "ingest", "lib", "shots"}

def _iter_subdir_names(path: Path):
    """Yield immediate subdirectory names, skipping hidden dirs and any entry that
    raises (permission-denied / offline share). One unreadable folder must NOT abort
    the whole scan — protected pipeline dirs are simply skipped."""
    try:
        entries = list(path.iterdir())
    except OSError:
        return
    for d in entries:
        name = d.name
        if name.startswith("."):
            continue
        try:
            if d.is_dir():
                yield name
        except OSError:
            continue  # admin-only / unreadable folder — skip, keep scanning

def list_shows(shows_root: str) -> List[str]:
    """Immediate subfolders of the shows root (each one is a show)."""
    if not shows_root or not os.path.exists(shows_root):
        return []
    return sorted(_iter_subdir_names(Path(shows_root)))

def list_shots_for_show(shows_root: str, show: str) -> List[str]:
    """Shot folders under <shows_root>/<show>, excluding studio pipeline dirs and any
    folder we can't stat (admin-only). Resilient: skips bad entries, never returns []
    just because one sibling folder is locked."""
    if not shows_root or not show:
        return []
    base = Path(shows_root) / show
    if not base.exists():
        return []
    return sorted(n for n in _iter_subdir_names(base)
                  if n.lower() not in _NON_SHOT_DIRS)

def list_shot_versions(shows_root: str, show: str, shot: str) -> List[str]:
    """Version folders under <show>/<shot>/in/plates, sorted ascending by number.
    e.g. ['v001','v002','v010'] — the last (highest) is the latest."""
    if not (shows_root and show and shot):
        return []
    plates = Path(shows_root) / show / shot / "in" / "plates"
    if not plates.exists():
        return []
    vers = [n for n in _iter_subdir_names(plates) if _VER_RE.match(n)]
    return sorted(vers, key=lambda v: int(_VER_RE.match(v).group(1)))

_SEQ_EXTS = {".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dpx"}

def count_plate_frames(plate_dir: str) -> int:
    """Number of image frames in a plate/version dir. Cheap listing only (no decode),
    so it stays fast over the network. Tolerant of unreadable dirs."""
    if not plate_dir or not os.path.exists(plate_dir):
        return 0
    try:
        return sum(1 for f in Path(plate_dir).iterdir()
                   if f.is_file() and f.suffix.lower() in _SEQ_EXTS)
    except OSError:
        return 0

def resolve_plate_dir(shows_root: str, show: str, shot: str, version: str) -> str:
    """Absolute frames dir: <shows_root>/<show>/<shot>/in/plates/<version>."""
    if not (shows_root and show and shot and version):
        return ""
    return str(Path(shows_root) / show / shot / "in" / "plates" / version)


_EXR_EXTS = {".exr"}
_PROXY_OK_EXTS = {".jpg", ".jpeg", ".png"}

def ensure_plate_proxies(plate_dir: str, cache_root: str, log_cb=None) -> str:
    """SAM3 (PIL) and Qwen (cv2.imread) cannot decode EXR. If `plate_dir` holds .exr
    frames, tonemap each to an 8-bit JPG under <cache_root>/_proxies/<key>/ and return
    that dir. If frames are already jpg/jpeg/png, return `plate_dir` unchanged.
    Idempotent: skips regen when the proxy count already matches the source count.

    SynthEyes reads EXR natively, so it should keep using the raw plate_dir (not this)."""
    def _log(m):
        if log_cb:
            try: log_cb(m)
            except Exception: pass
    if not plate_dir or not os.path.exists(plate_dir):
        return plate_dir or ""
    src = Path(plate_dir)
    try:
        files = sorted(f for f in src.iterdir() if f.is_file())
    except OSError:
        return plate_dir
    exr = [f for f in files if f.suffix.lower() in _EXR_EXTS]
    if not exr:
        # Already viewable by PIL/cv2 — no proxy needed.
        return plate_dir

    import hashlib
    key = hashlib.md5(str(src.resolve()).encode("utf-8", "ignore")).hexdigest()[:16]
    pdir = Path(cache_root) / "_proxies" / key
    pdir.mkdir(parents=True, exist_ok=True)
    existing = [f for f in pdir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"] if pdir.exists() else []
    if len(existing) >= len(exr):
        return str(pdir)  # already generated

    # OpenCV needs this set BEFORE importing/first use to enable the OpenEXR codec.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import numpy as _np
    try:
        import cv2 as _cv2
    except Exception:
        _cv2 = None

    def _read_exr_linear(fp: str):
        """Return an HxWx3 float32 linear RGB image in [0,inf), or None."""
        if _cv2 is not None:
            img = _cv2.imread(fp, _cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 2:
                    img = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
                if img.shape[2] >= 3:
                    return img[:, :, :3].astype(_np.float32)  # BGR order kept; written by cv2
        # Fallback: imageio (+ freeimage plugin) reads EXR without the cv2 codec.
        try:
            import imageio.v3 as _iio
            arr = _iio.imread(fp)
            arr = _np.asarray(arr, dtype=_np.float32)
            if arr.ndim == 2:
                arr = _np.stack([arr] * 3, axis=-1)
            rgb = arr[:, :, :3]
            return rgb[:, :, ::-1].copy()  # RGB -> BGR for cv2.imwrite
        except Exception as ex:
            _log(f"EXR read failed for {os.path.basename(fp)}: {ex}")
            return None

    if _cv2 is None:
        _log("cv2 unavailable — cannot write JPG proxies; returning raw EXR dir.")
        return plate_dir

    _log(f"Generating {len(exr)} JPG proxy frame(s) from EXR → {pdir}")
    made = 0
    for f in exr:
        lin = _read_exr_linear(str(f))
        if lin is None:
            continue
        # Linear -> sRGB-ish (gamma 2.2) tonemap, clip, 8-bit.
        disp = _np.clip(lin, 0.0, None)
        disp = _np.power(disp / (disp.max() if disp.max() > 1.0 else 1.0), 1.0 / 2.2) if disp.max() > 1.0 \
            else _np.power(_np.clip(disp, 0.0, 1.0), 1.0 / 2.2)
        out8 = _np.clip(disp * 255.0, 0, 255).astype(_np.uint8)
        outfp = pdir / (f.stem + ".jpg")
        try:
            _cv2.imwrite(str(outfp), out8, [int(_cv2.IMWRITE_JPEG_QUALITY), 95])
            made += 1
        except Exception as ex:
            _log(f"EXR proxy write failed for {f.name}: {ex}")
    _log(f"Proxy generation done: {made}/{len(exr)} frame(s).")
    return str(pdir) if made else plate_dir

def _extract_prompt_list(shot_dict: dict, keys: list) -> str:
    found_list = []
    for k in keys:
        val = shot_dict.get(k)
        if val:
            if isinstance(val, list):
                found_list = val
            elif isinstance(val, str):
                found_list = [x.strip() for x in val.split(",") if x.strip()]
            break
    if not found_list:
        tasks = shot_dict.get("tasks")
        if isinstance(tasks, list) and tasks:
            want_include = any("include" in str(k).lower() for k in keys)
            want_exclude = any("exclude" in str(k).lower() for k in keys)
            def _pick_task(tid: str):
                for t in tasks:
                    if isinstance(t, dict) and str(t.get("task_id") or "").strip().lower() == tid:
                        return t
                return tasks[0] if isinstance(tasks[0], dict) else None
            t = _pick_task("object") if want_include else _pick_task("camera") if want_exclude else None
            if isinstance(t, dict):
                if want_include:
                    val = t.get("mask_includes") or t.get("include_prompts") or []
                elif want_exclude:
                    val = t.get("mask_excludes") or t.get("exclude_prompts") or []
                else:
                    val = []
                if isinstance(val, list):
                    found_list = [str(x).strip() for x in val if str(x).strip()]
                elif isinstance(val, str):
                    found_list = [x.strip() for x in val.split(",") if x.strip()]
    return ",".join(found_list)

def _derive_strategy(shot_dict: dict) -> str:
    tasks = shot_dict.get('tasks')
    if isinstance(tasks, list) and tasks:
        ids = []
        for t in tasks:
            if isinstance(t, dict):
                tid = str(t.get('task_id') or '').strip()
                if tid: ids.append(tid)
        if ids: return '+'.join(ids)
    intent = str(shot_dict.get('intent') or '').strip()
    if intent: return intent
    return str(shot_dict.get('track_scope') or 'unknown')

def _as_list(v) -> List[str]:
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str): return [x.strip() for x in v.split(",") if x.strip()]
    return []

def _apply_analysis_fields(shot: "ShotData", s_item: dict) -> None:
    """Copy Qwen2.5-VL matchmove analysis fields from a guide-shot dict onto a ShotData."""
    shot.moving_things = _as_list(s_item.get("qwen2_moving_things"))
    shot.bad_track_regions = _as_list(s_item.get("qwen2_bad_track_regions"))
    shot.foreground_occluders = _as_list(s_item.get("qwen2_foreground_occluders"))
    shot.quality_flags = _as_list(s_item.get("qwen2_quality_flags"))
    dl = s_item.get("qwen2_depth_layers")
    if isinstance(dl, dict):
        shot.depth_layers = {"fg": str(dl.get("fg") or ""), "mg": str(dl.get("mg") or ""), "bg": str(dl.get("bg") or "")}
    shot.parallax = str(s_item.get("qwen2_parallax") or "").strip()

def _quality_cell(shot: "ShotData") -> str:
    """Short table cell: warn icon + flags, else OK."""
    if shot.quality_flags:
        return "⚠ " + ", ".join(shot.quality_flags)
    return "OK"

def _analysis_markdown(shot: "ShotData") -> str:
    """Full per-shot analysis panel (browser display)."""
    if not shot or not shot.name:
        return "Select a shot to see its AI analysis."
    dl = shot.depth_layers or {}
    def _j(xs): return ", ".join(xs) if xs else "_(none)_"
    q = ("⚠ " + ", ".join(shot.quality_flags)) if shot.quality_flags else "OK"
    pax = shot.parallax or "_(unknown)_"
    return (
        f"### AI Analysis — {shot.name}\n"
        f"- **Quality:** {q}\n"
        f"- **Parallax (3D-solvable?):** {pax}\n"
        f"- **Moving things:** {_j(shot.moving_things)}\n"
        f"- **Foreground occluders:** {_j(shot.foreground_occluders)}\n"
        f"- **Bad-track regions:** {_j(shot.bad_track_regions)}\n"
        f"- **Depth — FG:** {dl.get('fg') or '_(n/a)_'}  |  **MG:** {dl.get('mg') or '_(n/a)_'}  |  **BG:** {dl.get('bg') or '_(n/a)_'}\n"
        f"- **Detected things:** {_j(shot.detected_things)}\n"
    )

# -----------------------------------------------------------------------------
# 4. STATE
# -----------------------------------------------------------------------------
@dataclass
class ShotData:
    name: str
    use: bool = False   # default OFF -> user ticks the shot(s) to work on (one or many)
    res: str = ""
    width: int = 0
    height: int = 0
    frames: int = 0
    strategy: str = "Pending"
    include_prompts: str = ""
    exclude_prompts: str = ""
    detected_things: List[str] = field(default_factory=list)
    mask_mode: str = "outside"
    scale: str = "100%"
    frame_start: int = 0
    frame_end: int = 0
    vram: str = "0.0 GB"
    notes: str = ""
    track_metrics_summary: str = ""
    track_metrics_full: str = ""
    # Qwen2.5-VL 7B matchmove analysis
    moving_things: List[str] = field(default_factory=list)
    bad_track_regions: List[str] = field(default_factory=list)
    foreground_occluders: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    depth_layers: Dict[str, str] = field(default_factory=lambda: {"fg": "", "mg": "", "bg": ""})
    parallax: str = ""
    # Studio plate-versioning (network fetch): <show>/<shot>/in/plates/<version>
    show: str = ""
    version: str = ""
    versions: List[str] = field(default_factory=list)
    plate_dir: str = ""

@dataclass
class AppState:
    shots_data: Dict[str, ShotData] = field(default_factory=dict)
    guide_path: str = ""
    current_shot_name: str = ""
    log_history: List[str] = field(default_factory=list)
    filter_query: str = ""
    motion_backstop: bool = True   # CV optical-flow masking of movers (4th backstop)
    chunk_long_shots: bool = True  # SynthEyes: blip/peel long shots per window (avoid OOM)
    chunk_threshold: int = 1000    # frames above which SynthEyes chunks blip/peel
    reuse_existing_masks: bool = True   # if a shot already has masks in OUT, skip re-running SAM3
    track_chunks: int = 0          # TAPNext temporal chunks: 0=auto (VRAM-sized), >=1 forces
    track_spacing_px: int = 40     # TAPNext: min px spacing between kept tracks (density dial)
    track_max_output: int = 0      # TAPNext: soft cap on exported tracks per task, 0=unlimited
    moving_tile: bool = True       # TAPNext: native moving-tile re-track before NCC (4K accuracy)
    reseed: bool = True            # TAPNext: periodic re-seeding (replenish tracks on fast shots)
    reseed_every: int = 30         # TAPNext: max frames between re-seeds (window cap)
    edge_track: bool = True        # TAPNext: keep refining points to the frame border (edge tracks)
    gap_aware_refine: bool = True  # TAPNext: keep disappear/reappear points as one track (per-segment refine)
    pattern_refine: bool = True    # TAPNext: 3DE-style NCC/affine pattern lock at native res
    refine_patch_px: int = 31      # pattern-box size (px) for the refine pass
    # --- Tracking backend selection + SynthEyes settings ---
    track_backend: str = "syntheyes"   # "syntheyes" (default) | "tapnext" (fallback)
    syntheyes_exe: str = field(default_factory=lambda: os.environ.get("BTR_SYNTHEYES_EXE", ""))
    se_port: int = field(default_factory=lambda: int(os.environ.get("BTR_SE_PORT", "2222") or 2222))
    se_pin: str = field(default_factory=lambda: os.environ.get("BTR_SE_PIN", "listen") or "listen")
    track_preset: str = "Normal / Handheld"  # maps to SynthEyes max-track count
    use_sam3_matte: bool = True        # feed SAM3 masks to SynthEyes as a tracker matte
    auto_3de: bool = False             # build a .3de project after export
    tde4_exe: str = field(default_factory=lambda: os.environ.get("BTR_TDE4_EXE", ""))  # 3DEqualizer4 exe
    sensor_width: float = 36.0
    sensor_height: float = 24.0
    focal_length: float = 35.0
    # Per-shot client requirements typed in the UI for shots missing from the
    # uploaded requirements file (or when no file uploaded). {shot_name: note}.
    # Persisted to <OUT>/manual_requirements.json for later reuse.
    manual_notes: Dict[str, str] = field(default_factory=dict)

# NOTE: single-session tool. JOB_QUEUE / CURRENT_JOB_THREAD / STOP_EVENT are
# module globals shared across browser tabs — do not open multiple tabs at once.
JOB_QUEUE = queue.Queue()
CURRENT_JOB_THREAD = None
CURRENT_JOB_NAME = ""
CURRENT_JOB_START = 0.0     # epoch secs; drives the UI's elapsed timer
LAST_PROGRESS = ""
LAST_PROGRESS_FRAC = None   # 0..1 for a determinate bar; None = indeterminate (unknown total)
STOP_EVENT = threading.Event()

def _set_progress(msg: str, done=None, total=None):
    """Progress text + (optionally) a real fraction for the UI bar. Callers that know a
    done/total pass both; those that don't leave the fraction None -> indeterminate bar."""
    global LAST_PROGRESS, LAST_PROGRESS_FRAC
    LAST_PROGRESS = str(msg or "")
    try:
        LAST_PROGRESS_FRAC = (max(0.0, min(1.0, float(done) / float(total)))
                              if done is not None and total else None)
    except Exception:
        LAST_PROGRESS_FRAC = None

def _job_running() -> bool:
    return CURRENT_JOB_THREAD is not None and CURRENT_JOB_THREAD.is_alive()

def logger(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    JOB_QUEUE.put(full_msg)   # UI log keeps full unicode - the browser renders it fine
    try:
        print(full_msg)
    except UnicodeEncodeError:
        # Windows consoles default to cp1252, which can't encode plenty of what flows
        # through here: Qwen-derived mask prompts (sam3_runner logs include/exclude), shot
        # names, and exception text. A bare print would raise INSIDE logger and kill the
        # worker mid-stage, so degrade the CONSOLE copy only. (SynthEyesEngine.log does the
        # same for its own output.)
        print(full_msg.encode("ascii", "replace").decode("ascii"))

# -----------------------------------------------------------------------------
# Track QC Metrics
# -----------------------------------------------------------------------------
def _parse_tracks_txt(path: str) -> Tuple[int, Dict[str, List[Tuple[int, float, float]]]]:
    tracks: Dict[str, List[Tuple[int, float, float]]] = {}
    end_frame = 0
    if not path or not os.path.exists(path): return end_frame, tracks
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]
    if not lines: return end_frame, tracks
    try: n = int(lines[0])
    except Exception: return end_frame, tracks
    i = 1
    for _ in range(max(0, n)):
        if i >= len(lines): break
        tid = lines[i].split()
        if len(tid) != 1: break
        track_id = tid[0]
        i += 2 # skip header "0"
        if i >= len(lines): break
        try: end_frame = max(end_frame, int(float(lines[i].split()[0])))
        except Exception: pass
        i += 1
        pts: List[Tuple[int, float, float]] = []
        while i < len(lines):
            toks = lines[i].split()
            if len(toks) == 1: break
            if len(toks) >= 3:
                try: pts.append((int(float(toks[0])), float(toks[1]), float(toks[2])))
                except Exception: pass
            i += 1
        tracks[track_id] = pts
    return end_frame, tracks

def _fmt_pct(x: float) -> str:
    try: return f"{max(0.0, min(1.0, x)) * 100.0:.1f}%"
    except Exception: return "0.0%"
def _safe_div(a: float, b: float) -> float: return a / b if b else 0.0

def compute_track_metrics(tracks_txt_path: str, width: int = 0, height: int = 0) -> Tuple[str, str]:
    end_frame, tracks = _parse_tracks_txt(tracks_txt_path)
    if not tracks: return "No tracks", "No tracks (file missing or empty)."
    n_tracks = len(tracks)
    lengths, gap_ratios, mean_speeds, max_jumps, jitters = [], [], [], [], []
    edge_hits, edge_total = 0, 0
    occ = set(); bins_x, bins_y = 4, 4
    margin = int(max(10, 0.03 * min(width, height))) if (width>0 and height>0) else 0

    for tid, pts in tracks.items():
        if not pts: continue
        pts = sorted(pts, key=lambda t: t[0])
        frames, xs, ys = zip(*pts)
        lengths.append(len(pts))
        span = max(1, frames[-1] - frames[0] + 1)
        missing = 0
        for a, b in zip(frames[:-1], frames[1:]):
            if (b - a) > 1: missing += (b - a - 1)
        gap_ratios.append(_safe_div(missing, span))
        
        vels = []
        loc_max_j = 0.0
        for i in range(len(pts)-1):
            dt = max(1, pts[i+1][0] - pts[i][0])
            dist = math.sqrt((pts[i+1][1]-pts[i][1])**2 + (pts[i+1][2]-pts[i][2])**2)
            v = dist/dt
            vels.append(v)
            loc_max_j = max(loc_max_j, v)
        if vels:
            mean_speeds.append(sum(vels)/len(vels))
            max_jumps.append(loc_max_j)
            acc = [abs(vels[j+1]-vels[j]) for j in range(len(vels)-1)]
            jitters.append(sum(acc)/len(acc) if acc else 0.0)
        else:
            mean_speeds.append(0.0); max_jumps.append(0.0); jitters.append(0.0)

        if margin > 0:
            for x,y in zip(xs, ys):
                edge_total += 1
                if x <= margin or y <= margin or x >= (width-margin) or y >= (height-margin): edge_hits += 1
        if width > 0 and height > 0:
            bx = int(min(bins_x - 1, max(0, (xs[0] / max(1, width)) * bins_x)))
            by = int(min(bins_y - 1, max(0, (ys[0] / max(1, height)) * bins_y)))
            occ.add((bx, by))

    if not lengths: return "No valid tracks", "No valid track points parsed."
    mean_len = sum(lengths) / len(lengths)
    mean_gap = sum(gap_ratios) / len(gap_ratios) if gap_ratios else 0.0
    mean_jit = sum(jitters) / len(jitters) if jitters else 0.0
    edge_r = _safe_div(edge_hits, edge_total) if edge_total else 0.0
    
    summary = f"N={n_tracks} | lenμ={mean_len:.1f} | gaps={_fmt_pct(mean_gap)} | jitter={mean_jit:.3f} | edge={_fmt_pct(edge_r)}"
    full = (f"Track File: {os.path.basename(tracks_txt_path)}\nTracks: {n_tracks}\n"
            f"Mean length: {mean_len:.2f}\nMean Speed: {sum(mean_speeds)/len(mean_speeds) if mean_speeds else 0:.4f}\n")
    return summary, full

# -----------------------------------------------------------------------------
# 5. WORKERS
# -----------------------------------------------------------------------------
def _free_vram(tag: str = ""):
    """Drop dereferenced models from the CUDA caching allocator.

    No-op if torch was never imported (e.g. a SynthEyes-only run) — must not trigger a
    fresh torch/CUDA import here, which is exactly what we keep off non-GPU code paths."""
    if "torch" not in sys.modules:
        return
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved() / 1024**3
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after = torch.cuda.memory_reserved() / 1024**3
            logger(f"VRAM freed{(' ('+tag+')') if tag else ''}: {before:.2f} -> {after:.2f} GB reserved")
    except Exception as e:
        print(f"_free_vram: {e}")

def _free_ollama(model: str = "llama3.1:8b", base_url: str = None):
    """Ask Ollama to unload the LLM from memory (keep_alive=0)."""
    import urllib.request
    url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/") + "/api/generate"
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        logger(f"Ollama: unloaded {model} from memory.")
    except Exception as e:
        logger(f"Ollama unload skipped ({e}).")

def _norm_shot_name(s: str) -> str:
    """Match core.io_parsers._norm_shot: strip whitespace/dash/underscore, lowercase."""
    return re.sub(r"[\s_\-]+", "", str(s or "").strip()).lower()


def requirement_shot_names(req_path: str) -> set:
    """Normalized shot names that ALREADY carry a non-empty client note in the file.
    Tolerant: returns empty set if no file / unreadable (treat all shots as missing)."""
    global load_requirements
    names = set()
    if not req_path or not os.path.exists(req_path):
        return names
    try:
        if load_requirements is None:
            try:
                from core.io_parsers import load_requirements as _lr
            except ImportError:
                sys.path.append(os.getcwd())
                from core.io_parsers import load_requirements as _lr
            load_requirements = _lr
        for it in load_requirements(req_path):
            if str(getattr(it, "client_note", "") or "").strip():
                names.add(_norm_shot_name(getattr(it, "shot", "")))
    except Exception as e:
        print(f"requirement_shot_names: {e}")
    return names


def _manual_notes_path(out_dir: str) -> str:
    return os.path.join(out_dir or ".", "manual_requirements.json")


def load_manual_notes(out_dir: str) -> Dict[str, str]:
    """Load previously-saved per-shot UI requirements ({shot: note})."""
    p = _manual_notes_path(out_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                return {str(k): str(v) for k, v in d.items()}
        except Exception as e:
            print(f"load_manual_notes: {e}")
    return {}


def save_manual_notes(out_dir: str, notes: Dict[str, str]) -> str:
    """Persist per-shot UI requirements for later reuse."""
    p = _manual_notes_path(out_dir)
    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        logger(f"Saved {len(notes)} manual requirement note(s) -> {p}")
    except Exception as e:
        logger(f"save_manual_notes failed: {e}")
    return p


def _ensure_ollama(base_url: str = None, model: str = "llama3.1:8b", timeout: float = 40.0) -> bool:
    """Ensure the Ollama SERVER is up (start it if needed) and warm+pin the model so it
    stays resident through Analyze. keep_alive=-1 pins until Mask calls _free_ollama.
    Returns True if reachable. ConnectionRefused == server not running (not just unloaded)."""
    import time, shutil, subprocess, urllib.request
    base = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")

    def _alive() -> bool:
        try:
            urllib.request.urlopen(base + "/api/tags", timeout=3).read()
            return True
        except Exception:
            return False

    if not _alive():
        exe = shutil.which("ollama")
        if not exe:
            # Windows installer default locations (not always on PATH).
            for cand in (
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
                os.path.expandvars(r"%PROGRAMFILES(X86)%\Ollama\ollama.exe"),
            ):
                if cand and os.path.isfile(cand):
                    exe = cand
                    break
        if not exe:
            logger("Ollama not reachable and 'ollama.exe' not found (PATH or default install). "
                   "Install/start Ollama, then retry.")
            return False
        logger("Ollama server not running — starting 'ollama serve'…")
        try:
            flags = 0
            if os.name == "nt":
                # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: detached, no console window.
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x08000000
            subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, creationflags=flags)
        except Exception as e:
            logger(f"Failed to launch 'ollama serve': {e}")
            return False
        t0 = time.time()
        while time.time() - t0 < timeout and not _alive():
            time.sleep(1.0)
        if not _alive():
            logger(f"Ollama did not come up within {int(timeout)}s.")
            return False
        logger("Ollama server is up.")

    # Warm + pin the model so it stays resident through Analyze (freed at Mask).
    try:
        body = json.dumps({"model": model, "keep_alive": -1}).encode("utf-8")
        req = urllib.request.Request(base + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
        logger(f"Ollama: {model} loaded + pinned (resident until Mask).")
    except Exception as e:
        logger(f"Ollama warm skipped ({e}).")
    return True


def _latest_guide_file(out_root, exclude_dir=None):
    """Newest mask_guidance.json / overdrive_guide.json across <out>/_batches (or None)."""
    broot = Path(out_root) / "_batches"
    if not broot.exists():
        return None
    cands = []
    for d in broot.iterdir():
        if not d.is_dir():
            continue
        if exclude_dir and Path(d).resolve() == Path(exclude_dir).resolve():
            continue
        for fn in ("mask_guidance.json", "overdrive_guide.json"):
            g = d / fn
            if g.exists():
                cands.append(g)
    return max(cands, key=os.path.getmtime) if cands else None


def _guide_shot_name(s: dict) -> str:
    return str(s.get("shot_name") or s.get("shot") or s.get("name") or "")


def _merge_prior_guide_shots(guide_data, out_root, exclude_batch=None):
    """Cumulative analysis memory: copy shots from the most recent PRIOR guide that are NOT
    in this run into guide_data, so analyzing a new subset never forgets earlier shots."""
    prior = _latest_guide_file(out_root, exclude_dir=exclude_batch)
    if not prior:
        return
    try:
        with open(prior, "r", encoding="utf-8") as f:
            pdata = json.load(f)
    except Exception as e:
        logger(f"  (cumulative memory: could not read prior guide: {e})")
        return
    have = {_norm_shot_name(_guide_shot_name(s)) for s in guide_data.get("shots", [])}
    carried = 0
    for s in pdata.get("shots", []):
        nm = _norm_shot_name(_guide_shot_name(s))
        if nm and nm not in have:
            guide_data.setdefault("shots", []).append(s)
            have.add(nm)
            carried += 1
    if carried:
        logger(f"Carried forward {carried} previously-analyzed shot(s) (cumulative memory).")


def clear_shot_memory(out_root, shot_name):
    """Forget one shot's analysis: drop it from the latest guide + its manual requirement
    note. Returns True if anything was removed."""
    norm = _norm_shot_name(shot_name)
    removed = False
    g = _latest_guide_file(out_root)
    if g:
        try:
            with open(g, "r", encoding="utf-8") as f:
                data = json.load(f)
            shots = data.get("shots", [])
            kept = [s for s in shots if _norm_shot_name(_guide_shot_name(s)) != norm]
            if len(kept) != len(shots):
                data["shots"] = kept
                with open(g, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                removed = True
        except Exception as e:
            logger(f"clear_shot_memory (guide): {e}")
    mr = Path(out_root) / "manual_requirements.json"
    if mr.exists():
        try:
            with open(mr, "r", encoding="utf-8") as f:
                notes = json.load(f)
            newnotes = {k: v for k, v in notes.items() if _norm_shot_name(k) != norm}
            if len(newnotes) != len(notes):
                with open(mr, "w", encoding="utf-8") as f:
                    json.dump(newnotes, f, indent=2)
                removed = True
        except Exception as e:
            logger(f"clear_shot_memory (notes): {e}")
    logger(f"Cleared analysis memory for '{shot_name}'." if removed
           else f"No stored analysis found for '{shot_name}'.")
    return removed


def worker_analyze(in_dir, out_dir, req_path, fps, ollama_url, state: AppState):
    try:
        logger("--- Starting Step 2: Analysis & Decision ---")
        global load_requirements, load_qwen2_v1_scene_cam_things, build_batch_tracker_json
        try:
            from core.io_parsers import load_requirements, load_qwen2_v1_scene_cam_things
            from core.bridge import build_batch_tracker_json
        except ImportError:
            sys.path.append(os.getcwd())
            from core.io_parsers import load_requirements, load_qwen2_v1_scene_cam_things
            from core.bridge import build_batch_tracker_json

        batch_dir = Path(out_dir) / "_batches" / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        _ensure_qwen_loaded()
        if not run_qwen2_batch: raise ImportError("Qwen2 runner failed to load.")
        logger("Running Qwen2 Visual Description...")
        selected = [n for n, d in state.shots_data.items() if getattr(d, "use", False)]
        if selected:
            logger(f"Analyzing {len(selected)} selected shot(s): {', '.join(sorted(selected))}")
        # Per-shot plate dirs (network fetch). EXR gets JPG proxies since Qwen reads via cv2.
        plate_map = {n: ensure_plate_proxies(d.plate_dir, str(batch_dir), logger)
                     for n, d in state.shots_data.items()
                     if getattr(d, "use", False) and getattr(d, "plate_dir", "")}
        qwen_json = run_qwen2_batch(in_dir=in_dir, out_dir=str(batch_dir), fps=int(fps),
                                    use_int4=True, log_cb=logger, only_shots=(selected or None),
                                    shot_dirs=(plate_map or None))
        
        logger("Building mask guidance (Qwen signals + deterministic heuristics)...")
        reqs = []
        if req_path and os.path.exists(req_path):
            try:
                reqs = load_requirements(req_path)
                logger(f"Loaded {len(reqs)} requirements.")
            except Exception as e:
                logger(f"Warning: Requirements load failed ({e}). using defaults.")

        # Merge per-shot notes typed in the UI for shots missing from the file
        # (or when no file uploaded). UI gathered + saved these before launch.
        manual = getattr(state, "manual_notes", {}) or {}
        if manual:
            try:
                from core.io_parsers import ShotItem
            except ImportError:
                sys.path.append(os.getcwd())
                from core.io_parsers import ShotItem
            have = {_norm_shot_name(getattr(it, "shot", "")) for it in reqs}
            added = 0
            for shot, note in manual.items():
                note = str(note or "").strip()
                if note and _norm_shot_name(shot) not in have:
                    reqs.append(ShotItem(shot=str(shot), client_note=note))
                    have.add(_norm_shot_name(shot))
                    added += 1
            if added:
                logger(f"Merged {added} manual requirement note(s) into analysis.")

        qmap = load_qwen2_v1_scene_cam_things(str(qwen_json))

        # Keep only requirements for shots actually analyzed this run (in qmap).
        # Per-shot Analyze restricts Qwen to the selected shot(s); merging notes for
        # OTHER shots would emit broken "Missing shot entry" placeholders that wipe
        # those shots' good data when the guide reloads in the UI.
        before = len(reqs)
        reqs = [it for it in reqs if _norm_shot_name(getattr(it, "shot", "")) in qmap]
        if before != len(reqs):
            logger(f"Skipped {before - len(reqs)} requirement(s) for shots not analyzed this run.")

        guide_data = build_batch_tracker_json(items=reqs, qwen2_map=qmap)

        # Cumulative memory: fold in previously-analyzed shots not in this run so the newest
        # guide always holds ALL analyzed shots (Scan restore + Mask both read the newest).
        _merge_prior_guide_shots(guide_data, out_dir, exclude_batch=batch_dir)

        guide_path = batch_dir / "mask_guidance.json"
        with open(guide_path, "w", encoding="utf-8") as f:
            json.dump(guide_data, f, indent=2)
            
        logger("Analysis Complete.")
        # Set guide_path DIRECTLY as well as queueing it: the queue is only drained by the
        # UI's poll(), so a chained run (worker_pipeline calls analyze->mask in one thread)
        # would otherwise reach worker_mask with guide_path still unset and silently mask
        # against a fresh overdrive_guide instead of this analysis.
        state.guide_path = str(guide_path)
        JOB_QUEUE.put(f"GUIDE_PATH_UPDATE:{str(guide_path)}")
        JOB_QUEUE.put("DONE_ANALYSIS")
        return True
    except Exception as e:
        logger(f"ERROR in Analysis: {e}")
        traceback.print_exc()
        JOB_QUEUE.put("DONE_ANALYSIS")   # signal end on failure too, so the UI settles
        return False

def _shot_mask_dirs(out_dir: str, shot_name: str) -> List[str]:
    """Return mask dirs (containing >=1 .png) already present for a shot in OUT.
    Shot folder matched case-insensitively; any subdir named 'masks*' counts."""
    found = []
    try:
        if not out_dir or not os.path.isdir(out_dir):
            return found
        target = shot_name.strip().lower()
        shot_dir = None
        for child in os.listdir(out_dir):
            p = os.path.join(out_dir, child)
            if os.path.isdir(p) and child.strip().lower() == target:
                shot_dir = p; break
        if not shot_dir:
            return found
        for sub in os.listdir(shot_dir):
            sp = os.path.join(shot_dir, sub)
            if os.path.isdir(sp) and sub.strip().lower().startswith("masks"):
                if any(f.lower().endswith(".png") for f in os.listdir(sp)):
                    found.append(sp)
    except Exception:
        pass
    return found


def shots_with_existing_masks(out_dir: str, state: AppState) -> List[str]:
    """Selected shots that already have masks on disk in OUT."""
    names = []
    for name, d in state.shots_data.items():
        if not getattr(d, "use", False):
            continue
        if _shot_mask_dirs(out_dir, name):
            names.append(name)
    return names


def shots_missing_masks(out_dir: str, state: AppState) -> List[str]:
    """Selected shots with NO masks on disk. These will track the FULL PLATE (both backends
    fall back to un-gated tracking when a shot has no mask dir), so the UI warns first."""
    names = []
    for name, d in state.shots_data.items():
        if not getattr(d, "use", False):
            continue
        if not _shot_mask_dirs(out_dir, name):
            names.append(name)
    return names


def shots_missing_analysis(out_dir: str, state: AppState) -> List[str]:
    """Selected shots absent from the newest guide (i.e. never analyzed). Context only."""
    g = _latest_guide_file(out_dir)
    analyzed = set()
    if g:
        try:
            with open(g, "r", encoding="utf-8") as f:
                data = json.load(f)
            analyzed = {_norm_shot_name(_guide_shot_name(s)) for s in data.get("shots", [])}
        except Exception:
            analyzed = set()
    return [name for name, d in state.shots_data.items()
            if getattr(d, "use", False) and _norm_shot_name(name) not in analyzed]


def worker_mask(in_dir, out_dir, weights, state: AppState):
    try:
        logger("--- Starting Step 3: Mask Generation ---")
        n_sel = sum(1 for d in state.shots_data.values() if getattr(d, "use", False))
        if n_sel == 0:
            logger("No shots selected. Tick the 'Use' box on at least one shot, then retry.")
            JOB_QUEUE.put("DONE_MASKING")
            return False
        # Free analysis models before loading SAM3: clear any leftover Qwen VRAM.
        # (No Ollama LLM to unload — decisioning is Qwen-only now.)
        _free_vram("before SAM3")
        _ensure_sam3_loaded()
        if SamConfig is None: raise ImportError(f"SAM3 module missing. {SAM3_IMPORT_ERROR}")
        if not state.guide_path:
            batch_dir = Path(out_dir) / "_batches" / f"manual_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            state.guide_path = str(batch_dir / "overdrive_guide.json")

        guide_obj = None
        try:
            if state.guide_path and os.path.isfile(state.guide_path):
                with open(state.guide_path, "r", encoding="utf-8") as rf: guide_obj = json.load(rf)
        except Exception as e:
            logger(f"Warning: could not read guide {state.guide_path}: {e}")
        if not isinstance(guide_obj, dict): guide_obj = {"shots": []}
        
        shots = guide_obj.get("shots", [])
        if not isinstance(shots, list): shots = []; guide_obj["shots"] = shots
        by_name = {str(sh.get("shot_name") or sh.get("shot")): sh for sh in shots if isinstance(sh, dict)}

        for name, data in state.shots_data.items():
            if not getattr(data, "use", False): continue
            inc = [x.strip() for x in (data.include_prompts or "").split(",") if x.strip()]
            exc = [x.strip() for x in (data.exclude_prompts or "").split(",") if x.strip()]
            sh = by_name.get(name)
            if sh is None:
                sh = {"shot_name": name, "shot": name, "mask_includes": [], "mask_excludes": [], "track_mode": "no_mask_needed"}
                shots.append(sh); by_name[name] = sh

            # Carry the per-shot frame range into the guide so SAM3 masks only that range.
            sh["frame_start"] = int(getattr(data, "frame_start", 0) or 0)
            sh["frame_end"] = int(getattr(data, "frame_end", 0) or 0)

            # Point SAM3 at this shot's chosen plate version (network fetch). SAM3's
            # resolver honors shot["input_dir"] first. EXR plates get JPG proxies since
            # SAM3 (PIL/cv2) can't decode EXR.
            pd = str(getattr(data, "plate_dir", "") or "")
            if pd:
                sh["input_dir"] = ensure_plate_proxies(pd, out_dir, logger)

            # User edits are AUTHORITATIVE: write inc/exc verbatim (empty list CLEARS the
            # old value) so removing a term in the UI actually removes it from masking.
            tasks = sh.get("tasks")
            if isinstance(tasks, list) and tasks:
                for t in tasks:
                    if not isinstance(t, dict): continue
                    tid = str(t.get("task_id") or "").strip().lower()
                    if tid == "object":
                        t["mask_includes"] = inc; t["include_prompts"] = inc
                        t["track_mode"] = "track_inside_mask" if inc else "no_mask_needed"
                    elif tid == "camera":
                        t["mask_excludes"] = exc; t["exclude_prompts"] = exc
                        t["track_mode"] = "track_outside_mask" if exc else "no_mask_needed"
                    else:  # "other" / single task: apply both
                        t["mask_includes"] = inc; t["include_prompts"] = inc
                        t["mask_excludes"] = exc; t["exclude_prompts"] = exc
                        t["track_mode"] = ("track_inside_mask" if inc else
                                           "track_outside_mask" if exc else "no_mask_needed")
            else:
                sh["mask_includes"] = inc
                sh["mask_excludes"] = exc
                sh["track_mode"] = ("track_inside_mask" if inc else
                                    "track_outside_mask" if exc else "no_mask_needed")
            
        with open(state.guide_path, "w", encoding="utf-8") as f:
            json.dump(guide_obj, f, indent=2)

        # SAM3 runs SELECTED shots only — never the whole guide (which may carry
        # non-selected shots from a prior Analyze).
        reuse = bool(getattr(state, "reuse_existing_masks", True))
        sel_names = {n for n, d in state.shots_data.items() if getattr(d, "use", False)}
        def _shot_nm(sh): return str(sh.get("shot_name") or sh.get("shot") or "")
        run_shots = [sh for sh in shots if _shot_nm(sh) in sel_names]

        # Reuse-existing-masks: also skip selected shots that already have masks on disk.
        existing = {n for n in sel_names if _shot_mask_dirs(out_dir, n)}
        if reuse and existing:
            run_shots = [sh for sh in run_shots if _shot_nm(sh) not in existing]
            logger(f"Reusing existing masks for {len(existing)} shot(s): {', '.join(sorted(existing))}")
            if not run_shots:
                logger("All selected shots already have masks. Nothing to generate.")
                _free_vram("after SAM3")
                JOB_QUEUE.put("DONE_MASKING")
                return True   # success: masks are present — a chained pipeline must continue
        elif existing and not reuse:
            logger(f"Regenerating (overwriting) masks for {len(existing)} shot(s) with existing masks.")

        filtered = dict(guide_obj); filtered["shots"] = run_shots
        run_guide_path = str(Path(state.guide_path).with_name(Path(state.guide_path).stem + "_run.json"))
        with open(run_guide_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2)
        logger(f"Generating masks for {len(run_shots)} selected shot(s).")

        if not os.path.isfile(weights): raise FileNotFoundError(f"SAM3 Weights file not found: {weights}")
        mb = bool(getattr(state, "motion_backstop", True))
        logger(f"CV motion backstop: {'ON' if mb else 'OFF'}")
        try:
            cfg = SamConfig(guide_json_path=Path(run_guide_path), input_root=Path(in_dir), output_root=Path(out_dir), weights_path=Path(weights), motion_backstop=mb)
        except TypeError:
            # Older SamConfig without the motion_backstop field
            cfg = SamConfig(guide_json_path=Path(run_guide_path), input_root=Path(in_dir), output_root=Path(out_dir), weights_path=Path(weights))
        run_sam3_batch(cfg, log_cb=logger, progress_cb=lambda d,t: _set_progress(f"{d}/{t} frames", d, t), status_cb=logger)
        _free_vram("after SAM3")
        logger("Masking Complete.")
        JOB_QUEUE.put("DONE_MASKING")
        return True
    except Exception as e:
        logger(f"ERROR in Masking: {e}")
        traceback.print_exc()
        JOB_QUEUE.put("DONE_MASKING")   # signal end on failure too (also refreshes the table)
        return False

def _find_mask_dir(out_root, shot_name, mask_subdir):
    """Locate a SAM3 mask folder for a shot (case-insensitive child), trying the
    task's mask_subdir first, then a plain 'masks' fallback. Returns path or None."""
    shot_dir = None
    try:
        for child in Path(out_root).iterdir():
            if child.is_dir() and child.name.lower() == str(shot_name).lower():
                shot_dir = child
                break
    except Exception:
        return None
    if shot_dir is None:
        return None
    candidates = [mask_subdir, "masks"]
    for cand in candidates:
        if not cand:
            continue
        for child in shot_dir.iterdir():
            if child.is_dir() and child.name.lower() == cand.lower():
                pngs = [f for f in child.iterdir() if f.suffix.lower() == ".png"]
                if pngs:
                    return str(child)
    return None


def _track_shots_tapnext(in_root, out_root, shot_tasks_map, state, grid, seed_count, seed_min_dist):
    """TAPNext++ tracking path (Apache-2.0 GPU tracker, fallback backend)."""
    any_ran = False
    stopped = False
    sel_total = sum(1 for d in state.shots_data.values() if getattr(d, "use", False))
    sel_done = 0
    for shot_name, data in state.shots_data.items():
        if STOP_EVENT.is_set():
            logger("Tracking stopped by user."); stopped = True; break
        if not getattr(data, "use", False): continue

        video_dir, filename = None, None
        shot_dir = in_root / shot_name
        if shot_dir.exists() and shot_dir.is_dir():
            mp4s = sorted([p for p in shot_dir.iterdir() if p.suffix.lower() == ".mp4"])
            if mp4s: video_dir, filename = shot_dir, mp4s[0].name
        if not video_dir:
            exact = in_root / f"{shot_name}.mp4"
            if exact.exists(): video_dir, filename = in_root, exact.name
        if not video_dir:
            logger(f"Skip {shot_name}: No video found."); continue

        tasks = shot_tasks_map.get(shot_name)
        if not tasks:
            tasks = [{"task_id": "camera", "track_mode": "track_inside_mask" if data.mask_mode=="inside" else "track_outside_mask", "mask_subdir": "masks"}]

        qc_parts = []
        for t in tasks:
            if STOP_EVENT.is_set():
                logger("Tracking stopped by user."); stopped = True; break
            task_id = str(t.get("task_id") or "task").strip()
            tm = (t.get("track_mode") or "").strip().lower()
            mode = "inside" if tm=="track_inside_mask" else "outside"
            mask_subdir = str(t.get("mask_subdir") or ("masks_" + task_id)).strip()
            output_tag = "" if task_id.lower() == "camera" else task_id

            _set_progress(f"{shot_name} · {task_id}", sel_done, sel_total)
            logger(f"Tracking: {shot_name} | {task_id} | {mode.upper()}")
            cfg = RunnerConfig(
                input_dir=str(video_dir), output_dir=str(out_root), mask_root_dir=str(out_root),
                # SAM3 contract: white=keep/track, black=ignore. Derive polarity from that + mode
                # (inside seeds the white keep-region; outside seeds background, excluding black
                # movers). Beats "auto" pixel-majority guessing, which flips per-frame and unions to
                # a 100% exclude region on ~50/50 masks (movers filling half the frame) -> 0 seeds.
                mask_mode=mode, mask_polarity=("white" if mode == "inside" else "black"),
                mask_subdir=mask_subdir, output_tag=output_tag,
                grid_size=int(grid), seeding_mode="features",
                max_tracks=int(seed_count), min_feature_dist=int(seed_min_dist),
                flip_y_for_3de=True, selected_files=[filename], selected_scales={filename: float(data.scale.strip('%'))/100.0 if '%' in data.scale else 1.0},
                frame_start=int(getattr(data, "frame_start", 0) or 0), frame_end=int(getattr(data, "frame_end", 0) or 0),
                chunks=int(getattr(state, "track_chunks", 0) or 0),
                spread_min_dist_px=int(getattr(state, "track_spacing_px", 40) or 40),
                max_output_tracks=int(getattr(state, "track_max_output", 0) or 0),
                enable_moving_tile=bool(getattr(state, "moving_tile", True)),
                enable_reseed=bool(getattr(state, "reseed", True)),
                reseed_every=int(getattr(state, "reseed_every", 30) or 30),
                mt_edge_track=bool(getattr(state, "edge_track", True)),
                refine_gap_aware=bool(getattr(state, "gap_aware_refine", True)),
                enable_pattern_refine=bool(getattr(state, "pattern_refine", True)),
                refine_patch_px=int(getattr(state, "refine_patch_px", 31) or 31),
            )
            runner = BatchTrackerRunner(cfg, on_status=lambda m: logger(f"TRACK: {m}"))
            runner.run()
            any_ran = True
            try:
                stem = Path(filename).stem
                out_base = f"{stem}__tapnext.txt" if not output_tag else f"{stem}__{output_tag}__tapnext.txt"
                summ, _ = compute_track_metrics(
                    os.path.join(str(out_root), out_base),
                    width=int(getattr(data, "width", 0) or 0), height=int(getattr(data, "height", 0) or 0),
                )
                qc_parts.append(f"{task_id}: {summ}")
            except Exception as e:
                logger(f"QC metrics failed for {shot_name}/{task_id}: {e}")
        if qc_parts: data.track_metrics_summary = " | ".join(qc_parts)
        sel_done += 1
        _set_progress(f"{shot_name} · done", sel_done, sel_total)
        if stopped: break
    return any_ran, stopped


def _track_shots_syntheyes(in_root, out_root, shot_tasks_map, state, seed_count):
    """SynthEyes tracking path: drive one SynthEyes instance over SyPy3 across shots."""
    any_ran = False
    stopped = False
    sel_total = sum(1 for d in state.shots_data.values() if getattr(d, "use", False))
    sel_done = 0

    settings = {
        "syntheyes_exe": str(getattr(state, "syntheyes_exe", "") or ""),
        "port":          int(getattr(state, "se_port", 2222) or 2222),
        "pin":           str(getattr(state, "se_pin", "listen") or "listen"),
        "auto_3de":      bool(getattr(state, "auto_3de", False)),
        "tde4_exe":      str(getattr(state, "tde4_exe", "") or ""),
        "sensor_width":  float(getattr(state, "sensor_width", 36.0) or 36.0),
        "sensor_height": float(getattr(state, "sensor_height", 24.0) or 24.0),
        "focal_length":  float(getattr(state, "focal_length", 35.0) or 35.0),
        "chunk_long_shots": bool(getattr(state, "chunk_long_shots", True)),
        "chunk_threshold":  int(getattr(state, "chunk_threshold", 1000) or 1000),
    }
    use_matte = bool(getattr(state, "use_sam3_matte", True))
    preset = str(getattr(state, "track_preset", "Normal / Handheld") or "Normal / Handheld")

    if not settings["syntheyes_exe"]:
        logger("ERROR: SynthEyes .exe not set. Open Settings and point to SynthEyes64.exe.")
        return any_ran, stopped
    engine = SynthEyesEngine(settings, on_log=lambda m: logger(f"SE: {m}"))
    if not engine.setup_sypy():
        logger("ERROR: SyPy3 not found — install SynthEyes / check its bundled Python.")
        return any_ran, stopped
    # Force a CLEAN instance for the batch: reusing a stale/hung SynthEyes left over from a
    # prior run desyncs the socket and makes process_shot fail (shots then get swallowed ->
    # a bland "Nothing to track"). launch() kills any existing instance first, so this
    # guarantees a fresh, in-sync SynthEyes. (connect_or_launch would silently reuse it.)
    logger("Starting a clean SynthEyes instance for this batch…")
    if not engine.launch() or not engine.connect():
        logger("ERROR: Could not start/connect to SynthEyes.")
        return any_ran, stopped

    prev_was_heavy = False
    try:
        for shot_name, data in state.shots_data.items():
            if STOP_EVENT.is_set():
                logger("Tracking stopped by user."); stopped = True; break
            if not getattr(data, "use", False): continue

            # Chosen plate version (network fetch) wins; SynthEyes reads EXR natively so
            # it uses the raw plate dir, no proxy. Falls back to the flat <in_root>/<shot>.
            pd = str(getattr(data, "plate_dir", "") or "")
            shot_dir = Path(pd) if pd else (in_root / shot_name)
            seq = se_find_shot_frames(str(shot_dir)) if shot_dir.exists() else None
            movie_path = None
            if seq:
                first_frame = seq["first_frame"]
                all_frames = seq["all_frames"]
                frame_count = seq["frame_count"]
                seq_start, seq_end = seq["start_frame"], seq["end_frame"]
            else:
                # Flat movie layout: <in_root>/<shot>.<ext> (or a movie inside the shot folder).
                # SynthEyes decodes movies natively, so feed the file directly.
                VID = (".mp4", ".mov", ".avi", ".mkv")
                mv = None
                for ext in VID:
                    cand = in_root / f"{shot_name}{ext}"
                    if cand.exists():
                        mv = cand
                        break
                if mv is None and shot_dir.exists() and shot_dir.is_dir():
                    vids = sorted([p for p in shot_dir.iterdir() if p.suffix.lower() in VID])
                    if vids:
                        mv = vids[0]
                if mv is None:
                    logger(f"Skip {shot_name}: no image sequence or movie found.")
                    continue
                movie_path = str(mv)
                first_frame = movie_path
                all_frames = [movie_path]
                frame_count = int(getattr(data, "frames", 0) or 0)
                seq_start, seq_end = 1, (frame_count or 0)

            fr_start = int(getattr(data, "frame_start", 0) or 0) or seq_start
            fr_end = int(getattr(data, "frame_end", 0) or 0) or seq_end

            img_w = int(getattr(data, "width", 0) or 0) or None
            img_h = int(getattr(data, "height", 0) or 0) or None
            if (not img_w or not img_h) and se_read_image_size:
                w2, h2 = se_read_image_size(first_frame)
                img_w, img_h = img_w or w2, img_h or h2

            # heavy-shot RAM hygiene + liveness
            if prev_was_heavy:
                logger("Previous shot was heavy (1000+ frames) — restarting SynthEyes...")
                if not engine.restart():
                    logger("ERROR: Could not restart SynthEyes! Aborting batch."); stopped = True; break
            if not engine.is_alive():
                logger("SynthEyes not responding — attempting restart...")
                if not engine.restart():
                    logger("ERROR: Could not restart SynthEyes! Aborting batch."); stopped = True; break

            tasks = shot_tasks_map.get(shot_name)
            if not tasks:
                tasks = [{"task_id": "camera", "mask_subdir": "masks"}]

            track_count = se_preset_track_count(preset, seed_count) if se_preset_track_count else int(seed_count)

            qc_parts = []
            for t in tasks:
                if STOP_EVENT.is_set():
                    engine.request_stop()
                    logger("Tracking stopped by user."); stopped = True; break
                task_id = str(t.get("task_id") or "task").strip()
                mask_subdir = str(t.get("mask_subdir") or ("masks_" + task_id)).strip()
                output_tag = "" if task_id.lower() == "camera" else task_id
                out_base = f"{shot_name}__syntheyes.txt" if not output_tag else f"{shot_name}__{output_tag}__syntheyes.txt"

                mask_dir = _find_mask_dir(out_root, shot_name, mask_subdir) if use_matte else None
                if use_matte and not mask_dir:
                    logger(f"  No SAM3 masks for {shot_name}/{task_id} — tracking full frame.")

                _set_progress(f"{shot_name} · {task_id}", sel_done, sel_total)
                logger(f"Tracking (SynthEyes): {shot_name} | {task_id} | tracks={track_count} | matte={'yes' if mask_dir else 'no'}")
                try:
                    n_trk, _ = engine.process_shot(
                        shot_name=shot_name, output_dir=str(out_root), out_txt_name=out_base,
                        first_frame=first_frame, all_frames=all_frames, frame_count=frame_count,
                        start_frame=fr_start, end_frame=fr_end,
                        track_count=track_count, mask_dir=mask_dir,
                        image_width=img_w, image_height=img_h, movie_path=movie_path,
                    )
                    any_ran = True
                    logger(f"  DONE: {shot_name}/{task_id} — {n_trk} trackers")
                except Exception as e:
                    logger(f"ERROR tracking '{shot_name}/{task_id}': {e}")
                    traceback.print_exc()
                    continue

                try:
                    summ, _ = compute_track_metrics(
                        os.path.join(str(out_root), out_base),
                        width=int(img_w or 0), height=int(img_h or 0),
                    )
                    qc_parts.append(f"{task_id}: {summ}")
                except Exception as e:
                    logger(f"QC metrics failed for {shot_name}/{task_id}: {e}")

            if qc_parts: data.track_metrics_summary = " | ".join(qc_parts)

            # post-shot RAM flush
            try:
                shots_list = engine.hlev.Shots()
                if shots_list: engine.flush_after_shot(shots_list[0])
            except Exception as e:
                logger(f"  Post-shot flush failed: {e}")

            sel_done += 1
            _set_progress(f"{shot_name} · done", sel_done, sel_total)
            prev_was_heavy = (frame_count >= 1000)
            if stopped: break
    finally:
        logger("Closing SynthEyes...")
        try:
            engine.kill_syntheyes()
        except Exception as e:
            logger(f"  Could not close SynthEyes: {e}")
    return any_ran, stopped


def worker_track(in_dir, out_dir, grid, seed_count, seed_min_dist, state: AppState):
    try:
        logger("--- Starting Step 5: Tracking ---")
        n_sel = sum(1 for d in state.shots_data.values() if getattr(d, "use", False))
        if n_sel == 0:
            logger("No shots selected. Tick the 'Use' box on at least one shot, then retry.")
            JOB_QUEUE.put("DONE_TRACKING")
            return False
        in_root = Path(in_dir) if in_dir else None
        out_root = Path(out_dir) if out_dir else None
        if not in_root or not in_root.exists(): raise RuntimeError("Input folder does not exist.")
        if not out_root: raise RuntimeError("Output folder is empty.")
        out_root.mkdir(parents=True, exist_ok=True)

        guide = {}
        shot_tasks_map = {}
        if state and getattr(state, "guide_path", "") and os.path.isfile(state.guide_path):
            try:
                with open(state.guide_path, "r", encoding="utf-8") as f: guide = json.load(f) or {}
                for sh in (guide.get("shots") or []):
                    nm = sh.get("shot_name") or sh.get("shot")
                    if nm and sh.get("tasks"): shot_tasks_map[str(nm)] = sh.get("tasks")
            except Exception as e:
                logger(f"Warning: could not parse guide tasks: {e}")

        backend = str(getattr(state, "track_backend", "syntheyes") or "syntheyes").lower()
        if backend in ("cotracker", "cotracker3"):
            backend = "tapnext"   # legacy value -> new GPU tracker
        if backend == "syntheyes" and SynthEyesEngine is None:
            logger(f"SynthEyes backend unavailable ({SYNTHEYES_IMPORT_ERROR}); falling back to TAPNext++.")
            backend = "tapnext"

        if backend == "syntheyes":
            logger("Tracking backend: SynthEyes")
            _free_vram("before SynthEyes")  # release torch cache so SynthEyes can use the GPU
            any_ran, stopped = _track_shots_syntheyes(in_root, out_root, shot_tasks_map, state, seed_count)
        else:
            logger("Tracking backend: TAPNext++")
            _ensure_tracker_loaded()
            _free_vram("before TAPNext")
            if BatchTrackerRunner is None: raise ImportError(f"Tracker module missing. {TRACKER_IMPORT_ERROR}")
            any_ran, stopped = _track_shots_tapnext(in_root, out_root, shot_tasks_map, state, grid, seed_count, seed_min_dist)
            _free_vram("after TAPNext")

        if stopped: logger("Tracking halted.")
        elif not any_ran:
            logger("Nothing to track — no selected shot produced tracks. Check the lines above: "
                   "a shot was skipped (no image sequence/movie under the Input Folder) or "
                   "SynthEyes errored on it. Confirm the Input Folder is set and the shot is ticked.")
        else: logger("Tracking Complete.")
        JOB_QUEUE.put("DONE_TRACKING")
        return not stopped
    except Exception as e:
        logger(f"ERROR in Tracking: {e}")
        traceback.print_exc()
        JOB_QUEUE.put("DONE_TRACKING")   # signal end on failure too (also refreshes the table)
        return False


def worker_pipeline(in_dir, out_dir, req_path, fps, ollama_url, weights,
                    grid, seed_count, seed_min_dist, state: AppState):
    """Run the whole chain for the ticked shots in ONE job: Analyze -> Mask -> Track.

    Each stage is the same worker the numbered buttons call, so behavior is identical; they
    return False on failure and queue their own DONE_* (which drives the UI refresh). We stop
    the chain on the first failed stage or a Stop press, rather than masking/tracking against
    a broken analysis. Existing masks are reused (regenerating is the manual Mask button's
    job) so re-running the pipeline doesn't burn SAM3 time redoing work.
    """
    stages = [("Analyze", lambda: worker_analyze(in_dir, out_dir, req_path, fps, ollama_url, state)),
              ("Mask",    lambda: worker_mask(in_dir, out_dir, weights, state)),
              ("Track",   lambda: worker_track(in_dir, out_dir, grid, seed_count, seed_min_dist, state))]
    try:
        n_sel = sum(1 for d in state.shots_data.values() if getattr(d, "use", False))
        logger(f"=== Run Pipeline: Analyze -> Mask -> Track on {n_sel} shot(s) ===")
        state.reuse_existing_masks = True
        for i, (label, fn) in enumerate(stages, 1):
            if STOP_EVENT.is_set():
                logger(f"Pipeline stopped by user before {label}."); return False
            logger(f"--- Pipeline {i}/{len(stages)}: {label} ---")
            if not fn():
                logger(f"Pipeline aborted: {label} did not complete. See the errors above.")
                return False
            if STOP_EVENT.is_set():
                logger(f"Pipeline stopped by user after {label}."); return False
        logger("=== Run Pipeline: complete ===")
        return True
    except Exception as e:
        logger(f"ERROR in Pipeline: {e}")
        traceback.print_exc()
        return False


# -----------------------------------------------------------------------------
# 6. HANDLERS (Editor & Polling Logic)
# -----------------------------------------------------------------------------
def on_browse_file(current):
    return _tk_pick_file(current) or current

def on_browse_folder(current):
    return _tk_pick_folder(current) or current

def on_scan(in_dir, out_dir, state_store):
    st = state_store or AppState()
    shots = list_shots(in_dir)
    st.shots_data = {}
    st.log_history = []
    
    for s in shots:
        w, h, frames = 0, 0, 0
        if probe_video_meta:
            p = Path(in_dir)
            fpath = next((f for f in p.glob(f"{s}.*") if f.suffix.lower() in {'.mp4','.mov','.avi','.mkv'}), None)
            if fpath:
                meta = probe_video_meta(str(fpath))
                w, h = int(meta.get("width",0)), int(meta.get("height",0))
                frames = int(meta.get("total_frames",0))
        st.shots_data[s] = ShotData(name=s, res=f"{w}x{h}", width=w, height=h, frames=frames, scale="100%", vram=est_vram(w,h,frames))
    
    msg_extra = ""
    if out_dir and os.path.exists(out_dir):
        batches_root = Path(out_dir) / "_batches"
        if batches_root.exists():
            try:
                batch_dirs = sorted([d for d in batches_root.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
                # Newest guide across all batches; prefer the most recently modified of
                # mask_guidance.json / overdrive_guide.json (frame range is saved at Mask).
                guide_candidates = []
                for d in batch_dirs:
                    for fn in ("mask_guidance.json", "overdrive_guide.json"):
                        g = d / fn
                        if g.exists(): guide_candidates.append(g)
                latest_guide = max(guide_candidates, key=os.path.getmtime) if guide_candidates else None
                if latest_guide and latest_guide.exists():
                        with open(latest_guide, "r", encoding="utf-8") as f: data = json.load(f)
                        count_loaded = 0
                        for s_item in data.get("shots", []):
                            name = s_item.get("shot_name") or s_item.get("shot") or s_item.get("name")
                            if name and name in st.shots_data:
                                shot = st.shots_data[name]
                                shot.strategy = (_derive_strategy(s_item))
                                shot.include_prompts = _extract_prompt_list(s_item, ["mask_includes", "include_prompts", "sam3_include_prompt"])
                                shot.exclude_prompts = _extract_prompt_list(s_item, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
                                # Restore previously saved frame range (0 = full), if present.
                                try:
                                    fs = int(s_item.get("frame_start", 0) or 0)
                                    fe = int(s_item.get("frame_end", 0) or 0)
                                    if fs or fe:
                                        shot.frame_start, shot.frame_end = fs, fe
                                except Exception: pass
                                raw_things = s_item.get("qwen2_things", [])
                                if isinstance(raw_things, list): shot.detected_things = raw_things
                                elif isinstance(raw_things, str): shot.detected_things = [x.strip() for x in raw_things.split(",")]
                                _apply_analysis_fields(shot, s_item)
                                count_loaded += 1
                        st.guide_path = str(latest_guide)
                        msg_extra = f" (Loaded {count_loaded} from previous analysis)"
                        st.log_history.append(f"System: Loaded previous analysis (prompts + frame range) from {latest_guide.name}")
            except Exception as e: print(f"Error loading existing JSON: {e}")

    return _refresh_table(st), st, f"Found {len(shots)} shots.{msg_extra}", gr.update(choices=_shot_choices(st))

TABLE_COLUMNS = ["Use", "Shot Name", "Strategy", "Quality", "Prompts (Preview)", "Scale", "Range / Frames", "Track Metrics"]
USE_COL = "Use"
NAME_COL = "Shot Name"

def _range_cell(d: "ShotData") -> str:
    """Show the SET range plus the AVAILABLE total, so out-of-range is obvious."""
    total = int(getattr(d, "frames", 0) or 0)
    tot_s = f"{total}f" if total > 0 else "?f"
    fs = int(getattr(d, "frame_start", 0) or 0)
    fe = int(getattr(d, "frame_end", 0) or 0)
    if fs <= 0 and fe <= 0:
        return f"all · {tot_s}"
    return f"{fs or 1}-{fe or total or '?'} · {tot_s}"

def _visible_names(st: AppState) -> List[str]:
    """Sorted shot names honoring the search filter (substring, case-insensitive)."""
    q = (getattr(st, "filter_query", "") or "").strip().lower()
    names = sorted(st.shots_data.keys())
    if q:
        names = [n for n in names if q in n.lower()]
    return names

def _refresh_table(st: AppState):
    data = []
    use_flags = []
    for name in _visible_names(st):
        d = st.shots_data[name]
        prompts = f"INC: {d.include_prompts} | EXC: {d.exclude_prompts}"
        if len(prompts) > 50: prompts = prompts[:47] + "..."
        use_flags.append(bool(d.use))
        data.append([("✅" if d.use else "—"), name, d.strategy, _quality_cell(d), prompts, d.scale, _range_cell(d), (d.track_metrics_summary or "")])
    df = pd.DataFrame(data, columns=TABLE_COLUMNS)
    # Highlight ACTIVE shots (use=True) so they stand out from the rest.
    try:
        def _hl(row):
            on = bool(use_flags[row.name]) if row.name < len(use_flags) else False
            return ["background-color: rgba(35,209,139,0.16); font-weight:600" if on else "" for _ in row]
        return df.style.apply(_hl, axis=1)
    except Exception:
        return df

def on_use_toggle(use_flag, st: AppState):
    """Live toggle of the currently-edited shot's active state -> table re-highlights."""
    name = getattr(st, "current_shot_name", "")
    if name and name in st.shots_data:
        st.shots_data[name].use = bool(use_flag)
    return _refresh_table(st), st, _sel_count_text(st)

def on_search_change(st: AppState, query: str):
    st.filter_query = query or ""
    return _refresh_table(st), st, _sel_count_text(st), gr.update(choices=_shot_choices(st))

def _sel_count_text(st: AppState) -> str:
    total = len(st.shots_data)
    n = sum(1 for d in st.shots_data.values() if getattr(d, "use", False))
    if total == 0:
        return "No shots yet — set Input Folder and press **1 · Scan inputs**."
    if n == 0:
        return f"**0 of {total}** selected — tick **Use** on the shot(s) to run."
    return f"**{n} of {total}** shots selected."

def _shot_meta_md(d: "ShotData") -> str:
    return f"`{d.res or '?'}` · `{d.frames or '?'}` frames · mask `{d.mask_mode}` · est VRAM `{d.vram}`"

def _status_md() -> str:
    if _job_running():
        prog = f" · {LAST_PROGRESS}" if LAST_PROGRESS else ""
        return f"⏳ **{CURRENT_JOB_NAME}** running…{prog}"
    return "🟢 Idle"

def _step_btn_states():
    """Returns (steps_update, stop_update): disable steps while a job runs."""
    running = _job_running()
    return gr.update(interactive=not running), gr.update(interactive=running)

EDITOR_NOSEL = lambda st: (st, "#### Select a shot", "", "", "100%", gr.update(choices=[], value=[]),
                           "Select a shot to see its AI analysis.", False, gr.update(value=0), gr.update(value=0),
                           gr.update(), "")

def _load_editor(st: AppState, name: str):
    """Build the 12 editor outputs for a shot name. Shared by row-click + dropdown."""
    if not name or name not in st.shots_data:
        return EDITOR_NOSEL(st)
    data = st.shots_data[name]
    st.current_shot_name = name
    total = int(getattr(data, "frames", 0) or 0)
    fmax = total if total > 0 else None
    avail = f" · {total} frames (1-{total})" if total > 0 else ""
    return (
        st,
        f"#### ✏️ {name}{avail}",
        data.include_prompts,
        data.exclude_prompts,
        data.scale,
        gr.update(choices=data.detected_things, value=[]),
        _analysis_markdown(data),
        bool(data.use),
        gr.update(value=int(getattr(data, "frame_start", 0) or 0), maximum=fmax),
        gr.update(value=int(getattr(data, "frame_end", 0) or 0), maximum=fmax),
        gr.update(open=True),
        _shot_meta_md(data),
    )

def on_select_row(evt: gr.SelectData, st: AppState):
    # Outputs: [state_store, md_shot_title, txt_inc, txt_exc, dd_scale, cb_things,
    #           md_analysis, cb_use, num_fstart, num_fend, acc_edit, md_shotmeta]
    if not st or not st.shots_data:
        return [gr.update() for _ in range(12)]
    names = _visible_names(st)
    try:
        idx = evt.index[0]
    except Exception:
        return [gr.update() for _ in range(12)]
    if 0 <= idx < len(names):
        return _load_editor(st, names[idx])
    return EDITOR_NOSEL(st)

def on_pick_edit(name: str, st: AppState):
    # Same 12 editor outputs, selected via the dropdown (reliable vs dataframe click).
    return _load_editor(st, name)

def _shot_choices(st: AppState) -> List[str]:
    return _visible_names(st)

def on_add_to_include(st: AppState, selected_items, current_text):
    if not selected_items: return current_text
    current_list = [x.strip() for x in current_text.split(",") if x.strip()]
    new_list = current_list + [x for x in selected_items if x not in current_list]
    return ",".join(new_list)

def on_add_to_exclude(st: AppState, selected_items, current_text):
    if not selected_items: return current_text
    current_list = [x.strip() for x in current_text.split(",") if x.strip()]
    new_list = current_list + [x for x in selected_items if x not in current_list]
    return ",".join(new_list)

def on_save_overdrive(st: AppState, inc, exc, scale, use_flag, fstart, fend):
    # Overdrive edits prompts + downscale + use + frame range. Mask Mode is NOT editable here.
    name = st.current_shot_name
    if name and name in st.shots_data:
        d = st.shots_data[name]
        d.include_prompts = inc
        d.exclude_prompts = exc
        d.scale = scale
        d.use = bool(use_flag)
        total = int(getattr(d, "frames", 0) or 0)
        try: fs = max(0, int(fstart or 0))
        except Exception: fs = 0
        try: fe = max(0, int(fend or 0))
        except Exception: fe = 0
        note = ""
        if total > 0:
            if fs > total: fs = total
            if fe > total: fe = total; note = f" (clamped end to {total})"
        if fs and fe and fs > fe:
            fs, fe = fe, fs  # swap if reversed
        d.frame_start, d.frame_end = fs, fe
        return _refresh_table(st), st, f"Saved {name}.{note}"
    return gr.update(), st, "Error: No shot selected."

def run_step_thread(target_fn, args, name: str = "Job"):
    global CURRENT_JOB_THREAD, CURRENT_JOB_NAME, CURRENT_JOB_START
    if _job_running(): return f"{CURRENT_JOB_NAME} already running — wait or press Stop."
    STOP_EVENT.clear()
    _set_progress("")
    CURRENT_JOB_NAME = name
    CURRENT_JOB_START = time.time()
    CURRENT_JOB_THREAD = threading.Thread(target=target_fn, args=args, daemon=True)
    CURRENT_JOB_THREAD.start()
    return f"Started: {name}…"

def on_stop_job():
    STOP_EVENT.set()
    return "Stop requested — will halt after current shot/pass finishes."

def stream_logs(st: AppState):
    new_logs = []
    refresh_path = None
    refresh_table_now = False
    while True:
        try:
            msg = JOB_QUEUE.get_nowait()
            if msg.startswith("GUIDE_PATH_UPDATE:"):
                refresh_path = msg.split(":", 1)[1].strip()
                new_logs.append(f"System: Auto-reload triggered for {Path(refresh_path).name}")
            elif "DONE_ANALYSIS" in msg: pass
            elif "DONE_TRACKING" in msg:
                refresh_table_now = True
                new_logs.append(msg)
                continue
            elif "DONE_MASKING" in msg:
                refresh_table_now = True
                new_logs.append(msg)
                continue
            else: new_logs.append(msg)
            if len(new_logs) > 500: new_logs.pop(0)
        except queue.Empty: break
        
    if new_logs:
        st.log_history.extend(new_logs)
        if len(st.log_history) > 200: st.log_history = st.log_history[-200:]
            
    log_out = "\n".join(st.log_history) if st.log_history else gr.update()
    table_out = gr.update()

    if refresh_table_now: table_out = _refresh_table(st)
    if refresh_path and os.path.exists(refresh_path):
        try:
            st.guide_path = refresh_path
            with open(refresh_path, "r", encoding="utf-8") as f: data = json.load(f)
            for s in data.get("shots", []):
                name = s.get("shot_name") or s.get("shot") or s.get("name")
                if name and name in st.shots_data:
                    st.shots_data[name].strategy = (_derive_strategy(s))
                    st.shots_data[name].include_prompts = _extract_prompt_list(s, ["mask_includes", "include_prompts", "sam3_include_prompt"])
                    st.shots_data[name].exclude_prompts = _extract_prompt_list(s, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
                    raw_things = s.get("qwen2_things", [])
                    if isinstance(raw_things, list): st.shots_data[name].detected_things = raw_things
                    elif isinstance(raw_things, str): st.shots_data[name].detected_things = [x.strip() for x in raw_things.split(",")]
                    _apply_analysis_fields(st.shots_data[name], s)
            table_out = _refresh_table(st)
        except Exception as e: traceback.print_exc()

    steps_u, stop_u = _step_btn_states()
    changed = refresh_table_now or bool(refresh_path)
    dd_u = gr.update(choices=_shot_choices(st)) if changed else gr.update()
    return (log_out, table_out, st, _status_md(), _sel_count_text(st),
            steps_u, steps_u, steps_u, steps_u, stop_u, dd_u)

# -----------------------------------------------------------------------------
# 7. UI
# -----------------------------------------------------------------------------
DARK_CSS = r'''
:root{
  --bg: #0b0d10;
  --panel: #12151b;
  --panel2:#171b22;
  --text:#e7eaf0;
  --muted:#a9b1bf;
  --border:#252b36;
  --accent:#ff2d2d;          /* Vibrant Red */
  --accent2:#ff5252;
  --danger:#ff2d2d;
  --ok:#23d18b;
  --shadow: 0 10px 30px rgba(0,0,0,.45);
}

/* App background + base typography */
.gradio-container, body{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji";
}

/* Headings */
h1,h2,h3,h4,h5,h6, .prose h1,.prose h2,.prose h3{
  color: var(--text) !important;
}

/* Panels / groups */
.gr-panel, .gr-group, .wrap, .gr-box, .gr-form, .gr-accordion, .gr-row, .gr-column{
  border-color: var(--border) !important;
}
.gr-panel, .gr-group, .gr-box, .gr-form, .gr-accordion{
  background: var(--panel) !important;
  box-shadow: var(--shadow);
  border-radius: 14px !important;
}

/* Inputs */
input, textarea, select{
  background: var(--panel2) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
  border-radius: 12px !important;
}
input::placeholder, textarea::placeholder{
  color: rgba(169,177,191,.75) !important;
}
label, .gr-label, .wrap .label{
  color: var(--muted) !important;
}

/* Dataframe */
table, .dataframe, .gr-dataframe{
  background: var(--panel2) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
.gr-dataframe table thead th{
  background: #0f1217 !important;
  color: var(--muted) !important;
  border-color: var(--border) !important;
}
.gr-dataframe table td{
  border-color: var(--border) !important;
}

/* Buttons */
.btn-red button, button.primary, .gr-button-primary{
  background: var(--accent) !important;
  border: 1px solid rgba(255,45,45,.65) !important;
  color: #fff !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
}
.btn-red button:hover, button.primary:hover, .gr-button-primary:hover{
  background: var(--accent2) !important;
}
.btn-red-outline button{
  background: transparent !important;
  border: 1px solid rgba(255,45,45,.65) !important;
  color: #fff !important;
  border-radius: 14px !important;
  font-weight: 650 !important;
}
.btn-red-outline button:hover{
  background: rgba(255,45,45,.12) !important;
}
.btn-ghost button{
  background: rgba(255,255,255,.04) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 14px !important;
}
.btn-ghost button:hover{
  background: rgba(255,255,255,.06) !important;
}

/* Small icon browse buttons */
.btn-icon button{
  min-width: 44px !important;
  padding: 0.55rem 0.65rem !important;
  border-radius: 14px !important;
}

/* Make markdown blocks read well */
.prose, .markdown{
  color: var(--text) !important;
}
'''


# Hover tooltips: native browser title= set on each control by elem_id, on load.
TOOLTIP_JS = r'''
() => {
  const tips = {
    "tip_in_dir": "Folder holding your shots — either subfolders per shot or flat .mp4/.mov/.avi/.mkv files.",
    "tip_browse_in": "Browse to pick the Input Folder.",
    "tip_out_dir": "Where masks, tracks and batch JSON get written.",
    "tip_browse_out": "Browse to pick the Output Folder.",
    "tip_req_file": "Optional client brief (.xlsx/.txt): per-shot notes that steer the AI track strategy.",
    "tip_browse_req": "Browse to pick the client requirements file.",
    "tip_qwen_fps": "Nudges how many frames (6-8) the VLM sees, spread across the whole clip. More images dilute the VLM, so it plateaus at 8.",
    "tip_grid_size": "Density of the fallback tracking point grid (NxN) when feature seeding finds too few points.",
    "tip_seed_count": "Max number of feature points to track per shot. Higher = denser tracks, more VRAM/time.",
    "tip_seed_min_dist": "Minimum pixels between seeded points. Higher = sparser, more spread-out tracks.",
    "tip_scan": "Step 1: scan Input Folder, list shots, read resolution/frames, load any previous analysis.",
    "tip_analyze": "Step 2: run Qwen2 (describe shots) then LLaMA/Ollama (decide camera/object track strategy + mask prompts).",
    "tip_mask": "Step 4: run SAM3 to generate keep/ignore alpha masks from the include/exclude prompts.",
    "tip_track": "Step 5: run the tracker (SynthEyes, or TAPNext++ fallback) per shot, filter tracks against masks, export 3D Equalizer .txt.",
    "tip_stop": "Request stop. Halts after the current shot/pass finishes (a tracking pass can't be cut mid-way).",
    "tip_scale": "Downscale factor for processing this shot. Lower = faster/less VRAM; coords still exported at full res.",
    "tip_things": "Objects Qwen2 detected in the selected shot. Check items, then add them to Include or Exclude.",
    "tip_add_inc": "Add checked objects to Include prompts (track INSIDE these — object tracking).",
    "tip_add_exc": "Add checked objects to Exclude prompts (mask OUT these — camera tracking ignores them).",
    "tip_txt_inc": "Comma-separated things to keep/track inside the mask (green).",
    "tip_txt_exc": "Comma-separated things to mask out / ignore (red).",
    "tip_save": "Save the include/exclude prompts and downscale for the selected shot.",
    "tip_logs": "Live log of the running step.",
    "tip_analysis_panel": "Qwen2.5-VL analysis for the selected shot: movers, occluders, bad-track regions, quality, depth, parallax.",
    "tip_search": "Narrows the shots list. Tick the Use box on rows you want.",
    "tip_use": "Include this shot in Mask + Track. Uncheck to skip.",
    "tip_fstart": "1-based first frame to track (0 = from start). Exported track frame numbers stay aligned to the original shot.",
    "tip_fend": "1-based last frame to track, inclusive (0 = to end)."
  };
  const apply = () => {
    for (const [id, text] of Object.entries(tips)) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.title = text;
      el.querySelectorAll('button, input, textarea, select, label').forEach(c => c.title = text);
    }
  };
  apply();
  setTimeout(apply, 500);
  setTimeout(apply, 1500);
  setTimeout(apply, 3000);
  setTimeout(apply, 6000);
}
'''


def build_ui():
    with gr.Blocks(title="Unified Batch Tracker", css=DARK_CSS) as app:
        state_store = gr.State(AppState())
        ollama_url = gr.State(DEFAULT_OLLAMA_URL)
        sam_weights = gr.State(DEFAULT_SAM3_WEIGHTS)

        gr.Markdown("# Batch Tracker")

        with gr.Row():
            # ---------- LEFT RAIL: project, settings, run steps, find/select ----------
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Project")
                with gr.Row():
                    in_dir = gr.Textbox(label="Input Folder", placeholder=r"D:\shots\IN", scale=5, elem_id="tip_in_dir",
                        info="Folder of shots — subfolders per shot, or flat .mp4/.mov/.avi/.mkv files.")
                    btn_browse_in = gr.Button("📂", scale=1, min_width=1, elem_classes=["btn-icon","btn-ghost"], elem_id="tip_browse_in")
                with gr.Row():
                    out_dir = gr.Textbox(label="Output Folder", placeholder=r"D:\shots\OUT", scale=5, elem_id="tip_out_dir",
                        info="Where masks, tracks and batch JSON get written.")
                    btn_browse_out = gr.Button("📂", scale=1, min_width=1, elem_classes=["btn-icon","btn-ghost"], elem_id="tip_browse_out")
                with gr.Row():
                    req_file = gr.Textbox(label="Client Requirements (optional)", placeholder=r"D:\shots\reqs.xlsx", scale=5, elem_id="tip_req_file",
                        info="Optional .xlsx/.txt client brief: per-shot notes that steer AI track strategy.")
                    btn_browse_req = gr.Button("📄", scale=1, min_width=1, elem_classes=["btn-icon","btn-ghost"], elem_id="tip_browse_req")

                with gr.Accordion("Settings", open=False):
                    qwen_fps = gr.Slider(1, 8, value=4, step=1, label="Qwen2 Sample Density", elem_id="tip_qwen_fps",
                        info="How many frames (6-8) the VLM sees, spread across the whole clip. Plateaus at 8.")
                    grid_size = gr.Slider(4, 20, value=10, label="Grid Size", elem_id="tip_grid_size",
                        info="Fallback NxN point grid density when feature seeding finds too few points.")
                    seed_count = gr.Slider(100, 3000, value=1200, step=50, label="Seed Count (Max Tracks)", elem_id="tip_seed_count",
                        info="Max feature points tracked per shot. Higher = denser tracks, more VRAM/time.")
                    seed_min_dist = gr.Slider(0, 50, value=12, step=1, label="Min Seed Distance (px)", elem_id="tip_seed_min_dist",
                        info="Minimum pixels between seeded points. Higher = sparser, more spread-out tracks.")

                gr.Markdown("### Run  ·  do in order")
                btn_scan = gr.Button("1 · Scan inputs", elem_classes=["btn-ghost"], elem_id="tip_scan")
                btn_analyze = gr.Button("2 · Analyze (AI)", variant="primary", elem_classes=["btn-red"], elem_id="tip_analyze")
                btn_mask = gr.Button("3 · Generate masks", elem_classes=["btn-red-outline"], elem_id="tip_mask")
                btn_track = gr.Button("4 · Start tracking", variant="primary", elem_classes=["btn-red"], elem_id="tip_track")
                btn_stop = gr.Button("⏹ Stop", variant="stop", interactive=False, elem_classes=["btn-red-outline"], elem_id="tip_stop")

                gr.Markdown("### Find shots")
                txt_search = gr.Textbox(label="Find shots", placeholder="type part of a shot name…", elem_id="tip_search",
                    info="Narrows the shots list below. Tick the Use box on the rows you want.")

            # ---------- MAIN AREA: status, shots table, editor, logs ----------
            with gr.Column(scale=3):
                with gr.Row():
                    md_status = gr.Markdown("🟢 Idle", elem_id="tip_status")
                    md_selcount = gr.Markdown("No shots yet — set Input Folder and press **1 · Scan inputs**.", elem_id="tip_selcount")
                gr.Markdown("### Shots  ·  click a row to edit · active shots (✅) are highlighted")
                master_table = gr.Dataframe(
                    headers=TABLE_COLUMNS,
                    datatype=["str"] * len(TABLE_COLUMNS),
                    column_count=(len(TABLE_COLUMNS), "fixed"),
                    interactive=False,
                    type="pandas",
                    wrap=True,
                )

                dd_edit = gr.Dropdown(label="✏️ Edit shot", choices=[], value=None, interactive=True, elem_id="tip_edit_pick",
                    info="Pick a shot to open it in the editor below (reliable way to select).")
                with gr.Accordion("Edit selected shot", open=False) as acc_edit:
                    md_shot_title = gr.Markdown("#### Select a shot", elem_id="tip_shot_title")
                    md_shotmeta = gr.Markdown("", elem_id="tip_shotmeta")
                    with gr.Row():
                        cb_use = gr.Checkbox(label="Use this shot", value=False, interactive=True, elem_id="tip_use",
                            info="Include in Mask + Track. Uncheck to skip.")
                        dd_scale = gr.Dropdown(["100%", "75%", "50%", "25%"], label="Downscale", value="100%", interactive=True, elem_id="tip_scale",
                            info="Lower res = faster/less VRAM. Coords still exported at full res.")
                        num_fstart = gr.Number(label="Frame Start (0 = first)", value=0, precision=0, minimum=0, interactive=True, elem_id="tip_fstart",
                            info="1-based first frame to track. 0 = from the start.")
                        num_fend = gr.Number(label="Frame End (0 = last)", value=0, precision=0, minimum=0, interactive=True, elem_id="tip_fend",
                            info="1-based last frame, inclusive. 0 = to the end.")
                    with gr.Row():
                        txt_inc = gr.Textbox(label="Include Prompts (track inside)", interactive=True, elem_id="tip_txt_inc",
                            info="Comma-separated things to keep/track inside the mask.")
                        txt_exc = gr.Textbox(label="Exclude Prompts (mask out)", interactive=True, elem_id="tip_txt_exc",
                            info="Comma-separated things to mask out / ignore.")
                    with gr.Accordion("AI object suggestions", open=False):
                        cb_things = gr.CheckboxGroup(label="Detected objects — check, then add", choices=[], interactive=True, elem_id="tip_things",
                            info="Objects Qwen found. Check, then add to Include or Exclude.")
                        with gr.Row():
                            btn_add_inc = gr.Button("Add checked → Include", variant="secondary", elem_classes=["btn-red-outline"], elem_id="tip_add_inc")
                            btn_add_exc = gr.Button("Add checked → Exclude", variant="stop", elem_classes=["btn-red"], elem_id="tip_add_exc")
                    btn_save_overdrive = gr.Button("💾 Save shot settings", variant="primary", elem_classes=["btn-red"], elem_id="tip_save")
                    md_analysis = gr.Markdown("Select a shot to see its AI analysis.", elem_id="tip_analysis_panel")

                with gr.Accordion("Logs", open=True):
                    log_box = gr.Textbox(label="Logs", lines=14, interactive=False, show_label=False, elem_id="tip_logs")

        # --- EVENT WIRING FOR BROWSE BUTTONS ---
        btn_browse_in.click(on_browse_folder, inputs=[in_dir], outputs=[in_dir])
        btn_browse_out.click(on_browse_folder, inputs=[out_dir], outputs=[out_dir])
        btn_browse_req.click(on_browse_file, inputs=[req_file], outputs=[req_file])

        # --- EXISTING ACTIONS ---
        EDITOR_OUTPUTS = [state_store, md_shot_title, txt_inc, txt_exc, dd_scale, cb_things, md_analysis, cb_use, num_fstart, num_fend, acc_edit, md_shotmeta]
        btn_scan.click(on_scan, inputs=[in_dir, out_dir, state_store], outputs=[master_table, state_store, log_box, dd_edit])
        btn_analyze.click(lambda *args: run_step_thread(worker_analyze, args, "Analyze"), inputs=[in_dir, out_dir, req_file, qwen_fps, ollama_url, state_store], outputs=[log_box])
        master_table.select(on_select_row, inputs=[state_store], outputs=EDITOR_OUTPUTS)
        dd_edit.change(on_pick_edit, inputs=[dd_edit, state_store], outputs=EDITOR_OUTPUTS)
        btn_add_inc.click(on_add_to_include, inputs=[state_store, cb_things, txt_inc], outputs=[txt_inc])
        btn_add_exc.click(on_add_to_exclude, inputs=[state_store, cb_things, txt_exc], outputs=[txt_exc])
        btn_save_overdrive.click(on_save_overdrive, inputs=[state_store, txt_inc, txt_exc, dd_scale, cb_use, num_fstart, num_fend], outputs=[master_table, state_store, log_box])

        # --- ACTIVE TOGGLE (edit panel) + SEARCH FILTER ---
        cb_use.change(on_use_toggle, inputs=[cb_use, state_store], outputs=[master_table, state_store, md_selcount])
        txt_search.change(on_search_change, inputs=[state_store, txt_search], outputs=[master_table, state_store, md_selcount, dd_edit])
        btn_mask.click(lambda *args: run_step_thread(worker_mask, args, "Generate masks"), inputs=[in_dir, out_dir, sam_weights, state_store], outputs=[log_box])
        btn_track.click(lambda *args: run_step_thread(worker_track, args, "Tracking"), inputs=[in_dir, out_dir, grid_size, seed_count, seed_min_dist, state_store], outputs=[log_box])
        btn_stop.click(on_stop_job, inputs=[], outputs=[log_box])

        # Apply hover tooltips on page load (and keep them via MutationObserver in the JS).
        app.load(None, inputs=None, outputs=None, js=TOOLTIP_JS)

        timer = gr.Timer(1)
        timer.tick(stream_logs, inputs=[state_store],
                   outputs=[log_box, master_table, state_store, md_status, md_selcount,
                            btn_scan, btn_analyze, btn_mask, btn_track, btn_stop, dd_edit])

    return app

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    app = build_ui()
    app.queue()
    app.launch(server_name="127.0.0.1", inbrowser=True, theme=gr.themes.Base())