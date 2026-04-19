# -*- coding: utf-8 -*-
"""
Unified Batch Tracker — Automated 5-Step Workflow
Merges Qwen2/LLaMA Analysis, SAM3 Masking, and CoTracker Execution into one UI.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import queue
import traceback
import importlib
import importlib.util
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

# -----------------------------------------------------------------------------
# Fixed runtime endpoints / weights (kept out of the UI)
# You can override these via environment variables if needed.
# -----------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.environ.get("BTR_OLLAMA_URL", "http://localhost:11434")
DEFAULT_SAM3_WEIGHTS = os.environ.get(
    "BTR_SAM3_WEIGHTS",
    r"D:\Jefrin\BTr\batch_tracker_v001_starter\SAM3\weights\sam3.pt",
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
        Path(__file__).resolve().parent,
        Path(r"D:\Jefrin\BTr\batch_tracker_v001_starter"),
        Path.cwd()
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
        Path(__file__).parent,
        Path(os.getcwd()),
        Path(r"D:\Jefrin\BTr\batch_tracker_v001_starter"), 
    ]

    tracker_path: Path | None = None
    meta_path: Path | None = None

    preferred_tracker = Path(__file__).parent / "app" / "syntheyes_runner.py"
    preferred_meta = Path(__file__).parent / "app" / "video_meta.py"
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


probe_video_meta, BatchTrackerRunner, RunnerConfig, TRACKER_IMPORT_ERROR = _load_tracker_direct()

if TRACKER_IMPORT_ERROR:
    print(f"CRITICAL: Tracker Import Failed: {TRACKER_IMPORT_ERROR}")

# --- 2.3 SAM3 Imports ---
SamConfig = None
load_masking_guide = None
run_sam3_batch = None
SAM3_IMPORT_ERROR = None

def _load_sam3_direct():
    roots = [Path(os.getcwd()), Path(__file__).parent, Path(r"D:\Jefrin\BTr\batch_tracker_v001_starter")]
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

SamConfig, load_masking_guide, run_sam3_batch, SAM3_IMPORT_ERROR = _load_sam3_direct()
if SAM3_IMPORT_ERROR:
    print(f"CRITICAL: SAM3 Import Failed: {SAM3_IMPORT_ERROR}")

# --- 2.4 Qwen2 Loader ---
run_qwen2_batch = None
def _load_qwen_robustly():
    roots = [Path(r"D:\Jefrin\BTr\batch_tracker_v001_starter"), Path(__file__).parent]
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

# -----------------------------------------------------------------------------
# 4. STATE
# -----------------------------------------------------------------------------
@dataclass
class ShotData:
    name: str
    use: bool = True
    res: str = ""
    frames: int = 0
    strategy: str = "Pending"
    include_prompts: str = ""
    exclude_prompts: str = ""
    detected_things: List[str] = field(default_factory=list)
    mask_mode: str = "outside" 
    scale: str = "100%"
    vram: str = "0.0 GB"
    notes: str = ""
    track_metrics_summary: str = ""
    track_metrics_full: str = ""

@dataclass
class AppState:
    shots_data: Dict[str, ShotData] = field(default_factory=dict)
    guide_path: str = ""
    current_shot_name: str = "" 
    log_history: List[str] = field(default_factory=list)

JOB_QUEUE = queue.Queue()
CURRENT_JOB_THREAD = None

def logger(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    JOB_QUEUE.put(full_msg)
    print(full_msg)

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
    except: return end_frame, tracks
    i = 1
    for _ in range(max(0, n)):
        if i >= len(lines): break
        tid = lines[i].split()
        if len(tid) != 1: break
        track_id = tid[0]
        i += 2 # skip header "0"
        if i >= len(lines): break
        try: end_frame = max(end_frame, int(float(lines[i].split()[0])))
        except: pass
        i += 1
        pts: List[Tuple[int, float, float]] = []
        while i < len(lines):
            toks = lines[i].split()
            if len(toks) == 1: break
            if len(toks) >= 3:
                try: pts.append((int(float(toks[0])), float(toks[1]), float(toks[2])))
                except: pass
            i += 1
        tracks[track_id] = pts
    return end_frame, tracks

def _fmt_pct(x: float) -> str:
    try: return f"{max(0.0, min(1.0, x)) * 100.0:.1f}%"
    except: return "0.0%"
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
def worker_analyze(in_dir, out_dir, req_path, fps, ollama_url, state: AppState):
    try:
        logger("--- Starting Step 2: Analysis & Decision ---")
        global load_requirements, load_qwen2_v1_scene_cam_things, OllamaConfig, OllamaReasoner, build_batch_tracker_json
        try:
            from core.io_parsers import load_requirements, load_qwen2_v1_scene_cam_things
            from core.ollama_backend import OllamaConfig, OllamaReasoner
            from core.bridge import build_batch_tracker_json
        except ImportError:
            sys.path.append(os.getcwd())
            from core.io_parsers import load_requirements, load_qwen2_v1_scene_cam_things
            from core.ollama_backend import OllamaConfig, OllamaReasoner
            from core.bridge import build_batch_tracker_json

        batch_dir = Path(out_dir) / "_batches" / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        if not run_qwen2_batch: raise ImportError("Qwen2 runner failed to load.")
        logger("Running Qwen2 Visual Description...")
        qwen_json = run_qwen2_batch(in_dir=in_dir, out_dir=str(batch_dir), fps=int(fps), use_int4=False, log_cb=logger)
        
        logger("Running LLaMA (Ollama) Decision Engine...")
        reqs = []
        if req_path and os.path.exists(req_path):
            try:
                reqs = load_requirements(req_path)
                logger(f"Loaded {len(reqs)} requirements.")
            except Exception as e:
                logger(f"Warning: Requirements load failed ({e}). using defaults.")
        
        qmap = load_qwen2_v1_scene_cam_things(str(qwen_json))
        cfg = OllamaConfig(base_url=ollama_url, model="llama3.1:8b", temperature=0.2)
        reasoner = OllamaReasoner(cfg)
        guide_data = build_batch_tracker_json(items=reqs, qwen2_map=qmap, reasoner=reasoner)
        
        guide_path = batch_dir / "mask_guidance.json"
        with open(guide_path, "w", encoding="utf-8") as f:
            json.dump(guide_data, f, indent=2)
            
        logger("Analysis Complete.")
        JOB_QUEUE.put(f"GUIDE_PATH_UPDATE:{str(guide_path)}")
        JOB_QUEUE.put("DONE_ANALYSIS")
    except Exception as e:
        logger(f"ERROR in Analysis: {e}")
        traceback.print_exc()
        JOB_QUEUE.put("DONE_ANALYSIS")  # Always signal done so the UI refreshes even on error

def worker_mask(in_dir, out_dir, weights, state: AppState):
    try:
        logger("--- Starting Step 3: Mask Generation ---")
        if SamConfig is None: raise ImportError(f"SAM3 module missing. {SAM3_IMPORT_ERROR}")
        if not state.guide_path:
            batch_dir = Path(out_dir) / "_batches" / f"manual_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            state.guide_path = str(batch_dir / "overdrive_guide.json")

        guide_obj = None
        try:
            if state.guide_path and os.path.isfile(state.guide_path):
                with open(state.guide_path, "r", encoding="utf-8") as rf: guide_obj = json.load(rf)
        except: pass
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
            
            tasks = sh.get("tasks")
            if isinstance(tasks, list) and tasks:
                for t in tasks:
                    if not isinstance(t, dict): continue
                    tid = str(t.get("task_id") or "").strip().lower()
                    if tid == "object" and inc:
                        t["mask_includes"] = inc; t["include_prompts"] = inc; t["track_mode"] = "track_inside_mask"
                    if tid == "camera" and exc:
                        t["mask_excludes"] = exc; t["exclude_prompts"] = exc; t["track_mode"] = "track_outside_mask"
            else:
                if inc: sh["mask_includes"] = inc; sh["track_mode"] = "track_inside_mask"
                if exc: sh["mask_excludes"] = exc; 
                if not inc and exc: sh["track_mode"] = "track_outside_mask"
                if not inc and not exc: sh["track_mode"] = "no_mask_needed"
            
        with open(state.guide_path, "w", encoding="utf-8") as f:
            json.dump(guide_obj, f, indent=2)

        if not os.path.isfile(weights): raise FileNotFoundError(f"SAM3 Weights file not found: {weights}")
        cfg = SamConfig(guide_json_path=Path(state.guide_path), input_root=Path(in_dir), output_root=Path(out_dir), weights_path=Path(weights))
        run_sam3_batch(cfg, log_cb=logger, progress_cb=lambda d,t: logger(f"Progress: {d}/{t}"), status_cb=logger)
        logger("Masking Complete.")
        JOB_QUEUE.put("DONE_MASKING")
    except Exception as e:
        logger(f"ERROR in Masking: {e}")
        traceback.print_exc()

def worker_track(in_dir, out_dir, grid, seed_count, seed_min_dist, state: AppState):
    try:
        logger("--- Starting Step 5: Tracking ---")
        if BatchTrackerRunner is None: raise ImportError(f"Tracker module missing. {TRACKER_IMPORT_ERROR}")
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
            except: pass

        any_ran = False
        for shot_name, data in state.shots_data.items():
            if not getattr(data, "use", False): continue
            
            # Resolve Video
            video_dir, filename = None, None
            # Check for folder Layout
            shot_dir = in_root / shot_name
            if shot_dir.exists() and shot_dir.is_dir():
                mp4s = sorted([p for p in shot_dir.iterdir() if p.suffix.lower() == ".mp4"])
                if mp4s: video_dir, filename = shot_dir, mp4s[0].name
            # Check for flat Layout
            if not video_dir:
                exact = in_root / f"{shot_name}.mp4"
                if exact.exists(): video_dir, filename = in_root, exact.name
            
            if not video_dir:
                logger(f"Skip {shot_name}: No video found.")
                continue

            # Resolve Tasks
            tasks = shot_tasks_map.get(shot_name)
            if not tasks:
                tasks = [{"task_id": "camera", "track_mode": "track_inside_mask" if data.mask_mode=="inside" else "track_outside_mask", "mask_subdir": "masks"}]

            qc_parts = []
            for t in tasks:
                task_id = str(t.get("task_id") or "task").strip()
                tm = (t.get("track_mode") or "").strip().lower()
                mode = "inside" if tm=="track_inside_mask" else "outside"
                mask_subdir = str(t.get("mask_subdir") or ("masks_" + task_id)).strip()
                output_tag = "" if task_id.lower() == "camera" else task_id
                
                logger(f"Tracking: {shot_name} | {task_id} | {mode.upper()}")
                cfg = RunnerConfig(
                    input_dir=str(video_dir), output_dir=str(out_root), mask_root_dir=str(out_root),
                    mask_mode=mode, mask_polarity="auto", mask_subdir=mask_subdir, output_tag=output_tag,
                    grid_size=int(grid), seeding_mode="features",
                    max_tracks=int(seed_count), min_feature_dist=int(seed_min_dist),
                    flip_y_for_3de=True, selected_files=[filename], selected_scales={filename: float(data.scale.strip('%'))/100.0 if '%' in data.scale else 1.0}
                )
                runner = BatchTrackerRunner(cfg, on_status=lambda m: logger(f"TRACK: {m}"))
                runner.run()
                any_ran = True

                try:
                    stem = Path(filename).stem
                    out_base = f"{stem}__cotracker3_bidir.txt" if not output_tag else f"{stem}__{output_tag}__cotracker3_bidir.txt"
                    summ, _ = compute_track_metrics(os.path.join(str(out_root), out_base))
                    qc_parts.append(f"{task_id}: {summ}")
                except: pass
            
            if qc_parts: data.track_metrics_summary = " | ".join(qc_parts)

        if not any_ran: logger("Nothing to track.")
        else: logger("Tracking Complete.")
        JOB_QUEUE.put("DONE_TRACKING")
    except Exception as e:
        logger(f"ERROR in Tracking: {e}")
        traceback.print_exc()

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
            fpath = next((f for f in p.glob(f"{s}.*") if f.suffix in {'.mp4','.mov'}), None)
            if fpath:
                meta = probe_video_meta(str(fpath))
                w, h = int(meta.get("width",0)), int(meta.get("height",0))
                frames = int(meta.get("total_frames",0))
        st.shots_data[s] = ShotData(name=s, res=f"{w}x{h}", frames=frames, scale="100%", vram=est_vram(w,h,frames))
    
    msg_extra = ""
    if out_dir and os.path.exists(out_dir):
        batches_root = Path(out_dir) / "_batches"
        if batches_root.exists():
            try:
                batch_dirs = sorted([d for d in batches_root.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
                if batch_dirs:
                    latest_guide = batch_dirs[0] / "mask_guidance.json"
                    if latest_guide.exists():
                        with open(latest_guide, "r", encoding="utf-8") as f: data = json.load(f)
                        count_loaded = 0
                        for s_item in data.get("shots", []):
                            name = s_item.get("shot_name") or s_item.get("shot") or s_item.get("name")
                            if name and name in st.shots_data:
                                shot = st.shots_data[name]
                                shot.strategy = (_derive_strategy(s_item))
                                shot.include_prompts = _extract_prompt_list(s_item, ["mask_includes", "include_prompts", "sam3_include_prompt"])
                                shot.exclude_prompts = _extract_prompt_list(s_item, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
                                raw_things = s_item.get("qwen2_things", [])
                                if isinstance(raw_things, list): shot.detected_things = raw_things
                                elif isinstance(raw_things, str): shot.detected_things = [x.strip() for x in raw_things.split(",")]
                                count_loaded += 1
                        st.guide_path = str(latest_guide)
                        msg_extra = f" (Loaded {count_loaded} from previous analysis)"
                        st.log_history.append(f"System: Loaded previous analysis from {latest_guide.name}")
            except Exception as e: print(f"Error loading existing JSON: {e}")

    return _refresh_table(st), st, f"Found {len(shots)} shots.{msg_extra}"

def _refresh_table(st: AppState):
    data = []
    for name in sorted(st.shots_data.keys()):
        d = st.shots_data[name]
        prompts = f"INC: {d.include_prompts} | EXC: {d.exclude_prompts}"
        if len(prompts) > 50: prompts = prompts[:47] + "..."
        data.append([d.use, name, d.strategy, d.mask_mode, prompts, d.scale, d.vram, (d.track_metrics_summary or "")])
    return pd.DataFrame(data, columns=["Use", "Shot Name", "Strategy", "Mask Mode", "Prompts (Preview)", "Scale", "Est VRAM", "Track Metrics"])

def on_select_row(evt: gr.SelectData, st: AppState):
    # Outputs: [state_store, lbl_shot, txt_inc, txt_exc, dd_scale, cb_things]
    if not st or not st.shots_data:
        return [gr.update() for _ in range(6)]
    sorted_names = sorted(st.shots_data.keys())
    if evt.index[0] < len(sorted_names):
        name = sorted_names[evt.index[0]]
        data = st.shots_data[name]
        st.current_shot_name = name
        return (
            st,
            f"Editing: {name}",
            data.include_prompts,
            data.exclude_prompts,
            data.scale,
            gr.update(choices=data.detected_things, value=[]),
        )
    return st, "Select a shot", "", "", "100%", gr.update(choices=[], value=[])

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

def on_save_overdrive(st: AppState, inc, exc, scale):
    # Overdrive now only edits prompts + downscale. Mask Mode/Tracking Mode is NOT editable here.
    name = st.current_shot_name
    if name and name in st.shots_data:
        st.shots_data[name].include_prompts = inc
        st.shots_data[name].exclude_prompts = exc
        st.shots_data[name].scale = scale
        return _refresh_table(st), st, f"Saved overrides for {name}."
    return gr.update(), st, "Error: No shot selected."

def run_step_thread(target_fn, args):
    global CURRENT_JOB_THREAD
    if CURRENT_JOB_THREAD and CURRENT_JOB_THREAD.is_alive(): return "Job already running."
    CURRENT_JOB_THREAD = threading.Thread(target=target_fn, args=args, daemon=True)
    CURRENT_JOB_THREAD.start()
    return "Job started..."

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
            elif "DONE_ANALYSIS" in msg:
                refresh_table_now = True
                new_logs.append("[System] Analysis finished — refreshing table...")
                continue
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
                if not name:
                    continue
                # If shot wasn't scanned yet, add it anyway so it shows in the table
                if name not in st.shots_data:
                    st.shots_data[name] = ShotData(name=name)
                st.shots_data[name].strategy = (_derive_strategy(s))
                st.shots_data[name].include_prompts = _extract_prompt_list(s, ["mask_includes", "include_prompts", "sam3_include_prompt"])
                st.shots_data[name].exclude_prompts = _extract_prompt_list(s, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
                raw_things = s.get("qwen2_things", [])
                if isinstance(raw_things, list): st.shots_data[name].detected_things = raw_things
                elif isinstance(raw_things, str): st.shots_data[name].detected_things = [x.strip() for x in raw_things.split(",")]
            table_out = _refresh_table(st)
        except Exception as e: traceback.print_exc()
    return log_out, table_out, st

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


def build_ui():
    with gr.Blocks(title="Unified Batch Tracker", css=DARK_CSS) as app:
        state_store = gr.State(AppState())
        ollama_url = gr.State(DEFAULT_OLLAMA_URL)
        sam_weights = gr.State(DEFAULT_SAM3_WEIGHTS)

        gr.Markdown("# Bot Wrapper Tool")
        gr.Markdown("Workflow: Scan → Analyze → Review/Overdrive → Mask → Track")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Setup")
                # Modified Layout: Row with Textbox + Button for each path
                with gr.Row():
                    in_dir = gr.Textbox(label="Input Folder", placeholder=r"D:\Jefrin\BTr\IN", scale=5)
                    btn_browse_in = gr.Button("📂", scale=1, min_width=1, elem_classes=["btn-icon","btn-ghost"])
                
                with gr.Row():
                    out_dir = gr.Textbox(label="Output Folder", placeholder=r"D:\Jefrin\BTr\OUT", scale=5)
                    btn_browse_out = gr.Button("📂", scale=1, min_width=1, elem_classes=["btn-icon","btn-ghost"])
                
                with gr.Row():
                    req_file = gr.Textbox(label="Client Requirements (Optional)", placeholder=r"D:\Jefrin\BTr\reqs.xlsx", scale=5)
                    btn_browse_req = gr.Button("📄", scale=1, min_width=1, elem_classes=["btn-icon","btn-ghost"])

            with gr.Column(scale=2):
                gr.Markdown("## Tracking Settings")
                grid_size = gr.Slider(4, 20, value=10, label="Grid Size")
                seed_count = gr.Slider(100, 3000, value=1200, step=50, label="Seed Count (Max Tracks)")
                seed_min_dist = gr.Slider(0, 50, value=12, step=1, label="Min Seed Distance (px)")

        with gr.Row(variant="panel"):
            btn_scan = gr.Button("1. Scan Inputs", scale=1, elem_classes=["btn-ghost"])
            btn_analyze = gr.Button("2. Analyze (AI)", variant="primary", scale=2, elem_classes=["btn-red"])
            btn_mask = gr.Button("4. Generate Masks", scale=2, elem_classes=["btn-red-outline"])
            btn_track = gr.Button("5. Start Tracking", variant="primary", scale=2, elem_classes=["btn-red"])

        gr.Markdown("## Review & Overdrive")
        master_table = gr.Dataframe(
            headers=["Use", "Shot Name", "Strategy", "Mask Mode", "Prompts (Preview)", "Scale", "Est VRAM", "Track Metrics"],
            datatype=["bool", "str", "str", "str", "str", "str", "str", "str"],
            column_count=(8, "fixed"),
            interactive=False,
            type="pandas",
            wrap=True,
        )

        with gr.Group(visible=True):
            gr.Markdown("### Shot Overdrive (select a row above to edit)")
            with gr.Row():
                with gr.Column(scale=1):
                    lbl_shot = gr.Label(value="No shot selected", label="Current Shot")
                    dd_scale = gr.Dropdown(["100%", "75%", "50%", "25%"], label="Downscale", value="100%")
                with gr.Column(scale=2):
                    gr.Markdown("Detected Objects (AI Suggestions)")
                    cb_things = gr.CheckboxGroup(label="Check items to add to mask lists", choices=[])
                    with gr.Row():
                        btn_add_inc = gr.Button("Add checked to Include (+)", variant="secondary", elem_classes=["btn-red-outline"])
                        btn_add_exc = gr.Button("Add checked to Exclude (-)", variant="stop", elem_classes=["btn-red"])
            with gr.Row():
                txt_inc = gr.Textbox(label="Include Prompts (Green Mask)")
                txt_exc = gr.Textbox(label="Exclude Prompts (Red Mask)")
            btn_save_overdrive = gr.Button("Save Overdrive Settings", variant="primary", elem_classes=["btn-red"])

        log_box = gr.Textbox(label="Logs", lines=12, interactive=False)

        # --- EVENT WIRING FOR BROWSE BUTTONS ---
        btn_browse_in.click(on_browse_folder, inputs=[in_dir], outputs=[in_dir])
        btn_browse_out.click(on_browse_folder, inputs=[out_dir], outputs=[out_dir])
        btn_browse_req.click(on_browse_file, inputs=[req_file], outputs=[req_file])

        # --- EXISTING ACTIONS ---
        btn_scan.click(on_scan, inputs=[in_dir, out_dir, state_store], outputs=[master_table, state_store, log_box])
        btn_analyze.click(lambda *args: run_step_thread(worker_analyze, args), inputs=[in_dir, out_dir, req_file, gr.State(1), ollama_url, state_store], outputs=[log_box])
        master_table.select(on_select_row, inputs=[state_store], outputs=[state_store, lbl_shot, txt_inc, txt_exc, dd_scale, cb_things])
        btn_add_inc.click(on_add_to_include, inputs=[state_store, cb_things, txt_inc], outputs=[txt_inc])
        btn_add_exc.click(on_add_to_exclude, inputs=[state_store, cb_things, txt_exc], outputs=[txt_exc])
        btn_save_overdrive.click(on_save_overdrive, inputs=[state_store, txt_inc, txt_exc, dd_scale], outputs=[master_table, state_store, log_box])
        btn_mask.click(lambda *args: run_step_thread(worker_mask, args), inputs=[in_dir, out_dir, sam_weights, state_store], outputs=[log_box])
        btn_track.click(lambda *args: run_step_thread(worker_track, args), inputs=[in_dir, out_dir, grid_size, seed_count, seed_min_dist, state_store], outputs=[log_box])

        timer = gr.Timer(1)
        timer.tick(stream_logs, inputs=[state_store], outputs=[log_box, master_table, state_store])

    return app

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    app = build_ui()
    app.queue()
    app.launch(server_name="127.0.0.1", inbrowser=True, theme=gr.themes.Base())