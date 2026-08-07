# -*- coding: utf-8 -*-
"""
Unified Batch Tracker — Automated 5-Step Workflow
Merges Qwen2/LLaMA Analysis, SAM3 Masking, and SynthEyes/TAPNext++ Execution into one UI.
"""

from __future__ import annotations

import json
import os
# OpenCV only reads this at cv2 init, so it MUST be set before cv2 is imported anywhere
# in the process — otherwise EXR decode fails with "OpenEXR codec is disabled".
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
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
from concurrent.futures import ThreadPoolExecutor
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
# Studio plate servers.
# The studio runs more than one plate server and will add more. Every path helper
# below already takes shows_root as its first argument, so a new server needs NO
# code change — only a new entry in config/servers.json:
#     [{"name": "LIV1", "root": "\\\\liv1\\shows"},
#      {"name": "LIV2", "root": "\\\\liv2"}]
# The first entry is the default selection in the UI.
#
# The two live servers are not the same depth — liv1 keeps its shows one level down
# under 'shows', liv2 holds them at the share root. That difference is absorbed
# entirely by the root string; everything under <root>/<show>/<shot>/ is identical.
#
# Env override BTR_SHOWS_SERVERS beats the file, for a workstation that needs a
# one-off root (semicolon-separated NAME=ROOT pairs, ROOT alone also accepted):
#     set BTR_SHOWS_SERVERS=LIV2=\\liv2;TEST=D:\fake_studio
# -----------------------------------------------------------------------------
SERVERS_FILE = _HERE / "config" / "servers.json"

# Used only when config/servers.json is missing or unreadable, so a fresh checkout
# and a broken config both still reach the two real servers.
_FALLBACK_SERVERS: List[Dict[str, str]] = [
    {"name": "LIV1", "root": r"\\liv1\shows"},
    {"name": "LIV2", "root": r"\\liv2"},
]

def _clean_servers(items) -> List[Dict[str, str]]:
    """Keep well-formed {name, root} entries, in order, first-wins on duplicate root.
    A malformed entry is dropped rather than raising — one bad line in the config must
    not stop the app from reaching the other servers."""
    out, seen = [], set()
    for it in (items or []):
        if isinstance(it, str):
            root, name = it.strip(), ""
        elif isinstance(it, dict):
            root, name = str(it.get("root", "")).strip(), str(it.get("name", "")).strip()
        else:
            continue
        root = root.rstrip("/\\") if len(root.rstrip("/\\")) > 2 else root
        if not root or root.lower() in seen:
            continue
        seen.add(root.lower())
        out.append({"name": name or root, "root": root})
    return out

def load_servers() -> List[Dict[str, str]]:
    """Configured plate servers, in display order. Never raises and never returns
    empty — falls back to the built-in list so the Shows Root box is always usable."""
    env = os.environ.get("BTR_SHOWS_SERVERS", "").strip()
    if env:
        items = []
        for part in env.split(";"):
            part = part.strip()
            if not part:
                continue
            # NAME=ROOT, but a bare UNC/drive root is fine too. Split on the FIRST '='
            # only, and only when the left side is not itself a path (guards 'D:\x').
            name, sep, root = part.partition("=")
            items.append({"name": name, "root": root} if sep and "\\" not in name and "/" not in name
                         else {"name": "", "root": part})
        items = _clean_servers(items)
        if items:
            return items
    try:
        with open(SERVERS_FILE, "r", encoding="utf-8") as f:
            items = _clean_servers(json.load(f))
        if items:
            return items
    except Exception:
        pass
    return list(_FALLBACK_SERVERS)

def server_root_for(name: str) -> str:
    """Root path for a server name (case-insensitive). '' when unknown."""
    if not name:
        return ""
    for s in load_servers():
        if s["name"].lower() == name.strip().lower():
            return s["root"]
    return ""

def under(root: str, *parts: str) -> Path:
    """Join under a shows root and return a Path.

    NOT the same as Path(root) / part. pathlib only recognises a UNC path once it has
    BOTH a server and a share, so Path(r'\\\\liv2') parses as a plain rooted path and
    Path(r'\\\\liv2') / 'ABC' silently yields '\\liv2\\ABC' — one backslash short, and a
    path that does not exist. os.path.join concatenates the text first, so a root that
    is only a server name still produces a real UNC once the show is appended.
    Every studio-tree helper goes through here so the join is right for any root shape.
    """
    return Path(os.path.join(root, *[p for p in parts if p]))

def is_bare_server_root(root: str) -> bool:
    """True for a root like \\\\liv2 — a server with no share. On liv2 each SHOW is its
    own top-level share, so the root has nothing below it to iterdir; the show list has
    to come from the server's share table instead (see _list_server_shares). Per-shot
    paths built from such a root are fine once the show name is appended, because that
    appended name IS the share."""
    r = (root or "").replace("/", "\\")
    return r.startswith("\\\\") and "\\" not in r[2:].rstrip("\\")

# Shares that are never a show: admin/hidden shares end in '$', and these are the
# standard service shares a Windows file server exposes alongside real content.
_NON_SHOW_SHARES = {"ipc$", "print$", "netlogon", "sysvol", "users", "admin$"}

def _list_server_shares(root: str, timeout: float = 20.0) -> List[str]:
    """Share names on a bare server root (\\\\liv2), which is how the new server stores
    shows — one share per show, no parent folder to list.

    Uses `net view`, the only share enumerator available here (pywin32 is not in the
    embeddable runtime). Splits on runs of 2+ spaces rather than single whitespace so a
    share name containing a space survives. Returns [] on any failure — an unreachable
    server must leave the dropdown empty, not raise into the UI thread."""
    import subprocess
    server = (root or "").replace("/", "\\").rstrip("\\")
    if not server.startswith("\\\\"):
        return []
    try:
        # CREATE_NO_WINDOW: the app runs windowed, and a console flashing on every
        # refresh looks like a crash.
        proc = subprocess.run(["net", "view", server, "/all"],
                              capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    names, started = [], False
    for line in (proc.stdout or "").splitlines():
        if set(line.strip()) == {"-"}:       # header underline, any locale
            started = True
            continue
        if not started:
            continue
        if not line.strip():
            continue
        name = re.split(r"\s{2,}", line.strip())[0].strip()
        # Trailing status line ("The command completed successfully.") has no columns
        # and never names a share; a real share name has no sentence spacing.
        if not name or name.endswith(".") or name.lower() in _NON_SHOW_SHARES:
            continue
        if name.endswith("$"):
            continue
        names.append(name)
    return sorted(set(names))

# -----------------------------------------------------------------------------
# Studio network plate structure:
#   <shows_root>/<show>/<shot>/in/plates/<version>/<plate.exr | .jpg | .jpeg>
# The show dropdown scans <shows_root>; each shot exposes its own version list
# (folders vNNN under in/plates), defaulting to the highest = latest.
# shows_root is whichever server is selected — see load_servers() above.
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
    """Shows under a root. Normally the root's immediate subfolders (liv1:
    \\\\liv1\\shows\\<show>). When the root is a bare server (liv2: \\\\liv2\\<show>, where
    each show is its own share) there is no folder to list, so the shares are
    enumerated instead — same result, different source."""
    if not shows_root:
        return []
    if os.path.exists(shows_root):
        names = sorted(_iter_subdir_names(Path(shows_root)))
        if names:
            return names
    if is_bare_server_root(shows_root):
        return _list_server_shares(shows_root)
    return []

def list_shots_for_show(shows_root: str, show: str) -> List[str]:
    """Shot folders under <shows_root>/<show>, excluding studio pipeline dirs and any
    folder we can't stat (admin-only). Resilient: skips bad entries, never returns []
    just because one sibling folder is locked."""
    if not shows_root or not show:
        return []
    base = under(shows_root, show)
    if not base.exists():
        return []
    return sorted(n for n in _iter_subdir_names(base)
                  if n.lower() not in _NON_SHOT_DIRS)

def list_shot_versions(shows_root: str, show: str, shot: str) -> List[str]:
    """Version folders under <show>/<shot>/in/plates, sorted ascending by number.
    e.g. ['v001','v002','v010'] — the last (highest) is the latest."""
    if not (shows_root and show and shot):
        return []
    plates = under(shows_root, show, shot, "in", "plates")
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

def probe_plate_range(plate_dir: str):
    """(frame_count, start_frame, end_frame) for a plate/version dir. Uses the SAME
    scanner the SynthEyes tracker uses (find_shot_frames), so the table matches what
    will actually load: it finds frames directly in the dir OR one subfolder deep and
    parses real frame numbers from the filenames (e.g. plate.1001.exr -> 1001). Falls
    back to a flat count when the scanner is unavailable or finds nothing."""
    if not plate_dir or not os.path.exists(plate_dir):
        return 0, 0, 0
    try:
        if se_find_shot_frames:
            seq = se_find_shot_frames(str(plate_dir))
            if seq:
                return (int(seq.get("frame_count", 0)),
                        int(seq.get("start_frame", 0)),
                        int(seq.get("end_frame", 0)))
    except Exception:
        pass
    return count_plate_frames(plate_dir), 0, 0

def find_frames_subdir(version_dir: str):
    """Descend from a version dir to the folder that actually holds the frames.
    Studio plates nest several levels below vNNN (clip-id / resolution folders), e.g.
      <version>/<clip_id>/<6144x3240_exr>/plate.####.exr
    find_shot_frames only looks one level deep, so we walk the whole subtree and pick
    the folder with the MOST image frames (the real sequence). Returns
    (frames_dir, frame_count, start_frame, end_frame); frames_dir falls back to
    version_dir when nothing is found. This resolved dir is what every stage consumes,
    so SynthEyes (1-level scan), SAM3 and Qwen all get the exact frames folder."""
    if not version_dir or not os.path.exists(version_dir):
        return version_dir or "", 0, 0, 0
    best_dir, best_n = None, 0
    try:
        for dirpath, _dirnames, filenames in os.walk(version_dir, onerror=lambda e: None):
            n = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in _SEQ_EXTS)
            if n > best_n:
                best_n, best_dir = n, dirpath
    except OSError:
        pass
    if not best_dir:
        return version_dir, 0, 0, 0
    cnt, s, e = probe_plate_range(best_dir)
    return best_dir, cnt, s, e

# --- Batched (thread-pooled) variants of the two per-shot network probes above -------
# A 100+ shot show meant 100+ serial UNC round-trips before the table was usable. Both
# probes are pure IO (dir listing / os.walk), so a small thread pool is a straight win
# over the network. Each worker swallows its own error and returns a neutral value, so
# one unreadable shot never aborts the scan (same contract as the serial callers had).
_SCAN_WORKERS = 8

def _safe_list_shot_versions(args) -> List[str]:
    try:
        return list_shot_versions(*args)
    except Exception:
        return []

def _safe_find_frames(version_dir: str):
    try:
        return find_frames_subdir(version_dir)
    except Exception:
        return (version_dir or "", 0, 0, 0)

def list_versions_batch(shows_root: str, show: str, shots: List[str],
                        workers: int = _SCAN_WORKERS) -> List[List[str]]:
    """Plate versions for many shots at once. Returns one list per shot, in order."""
    if not shots:
        return []
    args = [(shows_root, show, s) for s in shots]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(args)))) as ex:
        return list(ex.map(_safe_list_shot_versions, args))

def resolve_frames_batch(version_dirs: List[str], workers: int = _SCAN_WORKERS):
    """find_frames_subdir for many version dirs at once. One tuple per dir, in order."""
    if not version_dirs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(version_dirs)))) as ex:
        return list(ex.map(_safe_find_frames, version_dirs))

# --- Per-shot progress, read from the shot's OWN folder ------------------------------
# Derived from the real published artifacts rather than a separate status file, so it can
# never drift out of sync with what is actually on disk -- and because the artifacts live
# in <show>/<shot>/mid/cmm/bot_tracks, the bot shows the same progress from any workstation
# on the share. ONE listdir per shot answers all three (publish_shot only creates
# analysis/ and masks/ when it actually copied something into them).
_BLANK_STATUS = {"analyzed": False, "masked": False, "tracked": False}

def _safe_shot_status(studio_dir: str) -> dict:
    try:
        if not studio_dir or not os.path.isdir(studio_dir):
            return dict(_BLANK_STATUS)
        analyzed = masked = tracked = False
        for e in os.listdir(studio_dir):
            low = e.lower()
            if low == "analysis":
                analyzed = True
            elif low == "masks":
                masked = True
            elif low.endswith(".txt") and "_2dtracks" in low:
                tracked = True
        return {"analyzed": analyzed, "masked": masked, "tracked": tracked}
    except Exception:
        return dict(_BLANK_STATUS)

def shot_status_batch(studio_dirs: List[str], workers: int = _SCAN_WORKERS) -> List[dict]:
    """{'analyzed','masked','tracked'} per shot, in order. Thread-pooled: these are UNC
    listings, and one per shot serially is what made a 100+ shot scan crawl."""
    if not studio_dirs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(studio_dirs)))) as ex:
        return list(ex.map(_safe_shot_status, studio_dirs))

_RENDER_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def render_sequence_to_mp4(seq_dir: str, out_mp4: str, log_cb=None, fps: int = 24,
                           max_side: int = 0) -> bool:
    """Encode an 8-bit image sequence (jpg/jpeg/png/tif) into an mp4 so the TAPNext
    (mp4-only) tracker can read it. Descends nested folders, orders frames by trailing
    number. max_side>0 downscales before encoding (mp4v stalls/crashes above ~4096 wide).
    Idempotent: reuses an existing non-empty mp4. Returns True on success."""
    def _log(m):
        if log_cb:
            try: log_cb(m)
            except Exception: pass
    if not seq_dir or not os.path.exists(seq_dir):
        return False
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
        return True
    try:
        import cv2
    except Exception:
        _log("cv2 unavailable — cannot render sequence to mp4.")
        return False
    fdir, _n, _s, _e = find_frames_subdir(seq_dir)
    def _num(p: Path):
        m = re.search(r"(\d+)$", p.stem)
        return int(m.group(1)) if m else 0
    try:
        imgs = sorted((f for f in Path(fdir).iterdir()
                       if f.is_file() and f.suffix.lower() in _RENDER_EXTS), key=_num)
    except OSError:
        imgs = []
    if not imgs:
        _log(f"No 8-bit frames (jpg/png) to render in {fdir}. Point at a JPEG/PNG render.")
        return False
    ms = int(max_side or 0)
    def _fit(img):
        if ms > 0:
            h0, w0 = img.shape[:2]
            m = max(h0, w0)
            if m > ms:
                s = ms / float(m)
                return cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))),
                                  interpolation=cv2.INTER_AREA)
        return img
    first = cv2.imread(str(imgs[0]))
    if first is None:
        _log(f"Could not read first frame {imgs[0].name}.")
        return False
    first = _fit(first)
    h, w = first.shape[:2]
    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (w, h))
    if not vw.isOpened():
        # Broken codec on this box — fail gracefully instead of risking a hard crash.
        _log(f"VideoWriter could not open (codec issue) for {out_mp4}; skipping mp4 render.")
        return False
    written = 0
    for f in imgs:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = _fit(img)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        vw.write(img); written += 1
    vw.release()
    ok = os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0 and written > 0
    _log(f"Rendered {written} frame(s) @ {w}x{h} -> {out_mp4}" if ok else "mp4 render failed.")
    return ok

# -----------------------------------------------------------------------------
# Studio output publishing.
# The pipeline computes in a LOCAL work dir (fast, keeps the cross-shot guide/SAM3
# logic intact); finished per-shot artifacts are then bulk-copied into the studio
# tree at <show>/<shot>/mid/cmm/bot_tracks/{tracks, masks/, analysis/, logs/}.
# -----------------------------------------------------------------------------
def work_dir_for_show(show: str) -> str:
    """Local scratch root the pipeline runs in, stable per show so 'reuse existing
    masks' works across sessions. Override root with env BTR_WORK."""
    root = os.environ.get("BTR_WORK", "") or str(_HERE / "runtime" / "_work")
    return str(Path(root) / (show or "_nofolder"))

def shot_bot_tracks_dir(shows_root: str, show: str, shot: str) -> str:
    """Publish target for a shot: <shows_root>/<show>/<shot>/mid/cmm/bot_tracks."""
    if not (shows_root and show and shot):
        return ""
    return str(under(shows_root, show, shot, "mid", "cmm", "bot_tracks"))

def shot_cache_dir(studio_dir: str, work_out: str = "", shot: str = "") -> str:
    """Persistent per-shot cache for bot-made JPG proxies + mp4 renders, kept in the
    shot's OWN folder (<studio>/cache) so they survive and are reused on future runs.
    Keyed further by plate version inside ensure_plate_proxies / render paths. Falls
    back to a local per-shot cache when the studio dir is unknown (legacy flat flow)."""
    if studio_dir:
        return str(Path(studio_dir) / "cache")
    if work_out:
        return str(Path(work_out) / "_cache" / (shot or "_shot"))
    return work_out or ""

def clear_shot_cache(studio_dir: str, work_out: str = "", shot: str = "") -> int:
    """Delete a shot's bot cache (proxies + renders). Returns the number of files
    removed. Safe if the cache doesn't exist."""
    import shutil
    removed = 0
    for cdir in {shot_cache_dir(studio_dir), shot_cache_dir("", work_out, shot)}:
        if cdir and os.path.isdir(cdir):
            try:
                removed += sum(1 for p in Path(cdir).rglob("*") if p.is_file())
                shutil.rmtree(cdir, ignore_errors=True)
            except Exception:
                pass
    return removed

def publish_shot(work_out: str, studio_dir: str, shot: str, backend: str,
                 scope: str = "all", log_cb=None) -> None:
    """Copy a shot's finished artifacts from the local work dir into its studio
    bot_tracks tree, organized. scope = 'analysis' | 'mask' | 'track' | 'all'.
    Skips anything missing so partial stages are fine; per-file failures are logged
    and do not abort."""
    import shutil
    def _log(m):
        if log_cb:
            try: log_cb(m)
            except Exception: pass
    if not studio_dir or not work_out:
        return
    do_track = scope in ("track", "all")
    do_mask = scope in ("mask", "all")
    do_analysis = scope in ("analysis", "all")

    def _copy_file(src, dst):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as ex:
            _log(f"publish: copy failed {os.path.basename(src)} -> {dst}: {ex}")
            return False

    def _copy_tree(src_dir, dst_dir):
        n = 0
        for dp, _dn, fns in os.walk(src_dir):
            rel = os.path.relpath(dp, src_dir)
            for f in fns:
                out = os.path.join(dst_dir, rel, f) if rel != "." else os.path.join(dst_dir, f)
                if _copy_file(os.path.join(dp, f), out):
                    n += 1
        return n

    # --- tracks: <work>/<shot>__*<backend>.txt -> <studio>/<shot>_2Dtracks[__task]__<backend>.txt
    if do_track:
        try:
            for f in os.listdir(work_out):
                low = f.lower()
                if not low.endswith(".txt"):
                    continue
                if not (low.startswith(f"{shot.lower()}__") and low.endswith(f"__{backend}.txt")):
                    continue
                mid = f[len(shot) + 2:-(len(backend) + 6)]  # strip "<shot>__" and "__<backend>.txt"
                tag = f"__{mid}" if mid else ""
                dst = os.path.join(studio_dir, f"{shot}_2Dtracks{tag}__{backend}.txt")
                if _copy_file(os.path.join(work_out, f), dst):
                    _log(f"published tracks -> {dst}")
        except OSError:
            pass

        # --- the shot's tracking log -> <studio>/logs/ (so a failure is diagnosable from
        # the shot folder, not only from whatever was on screen at the time).
        slog = os.path.join(work_out, f"{shot}__track.log")
        if os.path.isfile(slog):
            if _copy_file(slog, os.path.join(studio_dir, "logs", f"{shot}__track.log")):
                _log(f"published log -> {os.path.join(studio_dir, 'logs')}")

    # --- masks: <work>/<shot>/masks* -> <studio>/masks/
    # A camera+object shot has TWO mask dirs (masks_camera, masks_object) whose files are
    # named identically (mask_000001.png ...). Flattening both into one masks/ made the
    # second overwrite the first, so the studio tree silently kept only one task's mattes.
    # With more than one source, each keeps its own subfolder named for its task; a
    # single-task shot still publishes flat, which is what every existing shot looks like.
    if do_mask:
        mdirs = _shot_mask_dirs(work_out, shot)
        mask_root = os.path.join(studio_dir, "masks")
        for md in mdirs:
            # "masks_camera" -> "camera"; a dir named plain "masks" has no task suffix.
            task = os.path.basename(md.rstrip("/\\"))[len("masks"):].lstrip("_-")
            dst = os.path.join(mask_root, task) if (len(mdirs) > 1 and task) else mask_root
            n = _copy_tree(md, dst)
            if n:
                _log(f"published {n} mask file(s) -> {dst}")

    # --- analysis: guide slice for this shot + manual note -> <studio>/analysis/
    if do_analysis:
        g = _latest_guide_file(work_out)
        adir = os.path.join(studio_dir, "analysis")
        if g and os.path.isfile(g):
            try:
                with open(g, "r", encoding="utf-8") as f:
                    gd = json.load(f)
                shots = [s for s in (gd.get("shots") or [])
                         if _norm_shot_name(_guide_shot_name(s)) == _norm_shot_name(shot)]
                if shots:
                    os.makedirs(adir, exist_ok=True)
                    with open(os.path.join(adir, f"{shot}_guide.json"), "w", encoding="utf-8") as f:
                        json.dump({"shots": shots}, f, indent=2)
                    _log(f"published analysis -> {adir}")
            except Exception as ex:
                _log(f"publish: analysis slice failed: {ex}")
        notes = load_manual_notes(work_out)
        note = (notes or {}).get(shot, "")
        if note:
            _copy_note = os.path.join(adir, "manual_note.txt")
            try:
                os.makedirs(adir, exist_ok=True)
                with open(_copy_note, "w", encoding="utf-8") as f:
                    f.write(note)
            except Exception as ex:
                _log(f"publish: note failed: {ex}")

def resolve_plate_dir(shows_root: str, show: str, shot: str, version: str) -> str:
    """Absolute frames dir: <shows_root>/<show>/<shot>/in/plates/<version>."""
    if not (shows_root and show and shot and version):
        return ""
    return str(under(shows_root, show, shot, "in", "plates", version))


_EXR_EXTS = {".exr"}
_PROXY_OK_EXTS = {".jpg", ".jpeg", ".png"}

def ensure_plate_proxies(plate_dir: str, cache_root: str, log_cb=None,
                         max_side: int = 0, workers: int = 8,
                         lossless: bool = False) -> str:
    """SAM3 (PIL) and Qwen (cv2.imread) cannot decode EXR. If `plate_dir` holds .exr
    frames, tonemap each to an 8-bit JPG under <cache_root>/_proxies/<key>/ and return
    that dir. If frames are already jpg/jpeg/png, return `plate_dir` unchanged.

    max_side>0 downscales each proxy to that longest edge (analyse only needs ~1280px,
    so it avoids writing GBs of 6K JPGs to the share and never feeds a 6K frame to the
    mp4 encoder). Generation is parallel (cv2 releases the GIL) so 200+ frames don't
    take minutes. Idempotent: skips regen when the proxy count already matches source.
    Different max_side values cache separately.

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

    # Lossless (PNG) proxies for TRACKING: JPEG artefacts sit on the 8x8 DCT block grid and
    # do not travel with the image, so they inject sub-pixel noise -- which is exactly the
    # "track wobbles in place" symptom. Analysis and masking are not sub-pixel sensitive and
    # keep JPEG, since PNG costs several times the disk.
    ext = ".png" if lossless else ".jpg"
    import hashlib
    # The format MUST be part of the key, and the existing-file scan MUST match the chosen
    # extension: miss either and a shot that already has JPEG proxies silently keeps using
    # them for tracking, so the fix appears to do nothing.
    key = hashlib.md5(
        f"{src.resolve()}@{int(max_side)}@{ext}".encode("utf-8", "ignore")).hexdigest()[:16]
    pdir = Path(cache_root) / "_proxies" / key
    pdir.mkdir(parents=True, exist_ok=True)
    existing = [f for f in pdir.iterdir() if f.is_file() and f.suffix.lower() == ext] if pdir.exists() else []
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
            try:
                img = _cv2.imread(fp, _cv2.IMREAD_UNCHANGED)
            except Exception as ex:
                # e.g. "OpenEXR codec is disabled" when the env var wasn't set at init.
                _log(f"cv2 EXR read failed ({os.path.basename(fp)}: {ex}); trying imageio.")
                img = None
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

    ms = int(max_side or 0)

    def _one(f):
        lin = _read_exr_linear(str(f))
        if lin is None:
            return 0
        # FIXED linear->sRGB-ish (gamma 2.2) tonemap. A per-frame max normalization would
        # vary exposure frame-to-frame (flicker) and wreck NCC/pattern-lock, so use a fixed
        # curve on clipped-linear — temporally stable local contrast is what tracking needs.
        disp = _np.power(_np.clip(lin, 0.0, 1.0), 1.0 / 2.2)
        out8 = _np.clip(disp * 255.0, 0, 255).astype(_np.uint8)
        if ms > 0:
            h, w = out8.shape[:2]
            m = max(h, w)
            if m > ms:
                s = ms / float(m)
                out8 = _cv2.resize(out8, (int(round(w * s)), int(round(h * s))),
                                   interpolation=_cv2.INTER_AREA)
        outfp = pdir / (f.stem + ext)
        try:
            # PNG level 1: still lossless, and much faster to write than the default 3 --
            # these are a per-shot cache, not something we ship.
            params = ([int(_cv2.IMWRITE_PNG_COMPRESSION), 1] if lossless
                      else [int(_cv2.IMWRITE_JPEG_QUALITY), 95])
            _cv2.imwrite(str(outfp), out8, params)
            return 1
        except Exception as ex:
            _log(f"EXR proxy write failed for {f.name}: {ex}")
            return 0

    n_workers = max(1, min(int(workers or 1), len(exr)))
    _log(f"Generating {len(exr)} {'PNG (lossless)' if lossless else 'JPG'} proxy frame(s) from EXR "
         f"({'downscaled ' + str(ms) + 'px' if ms else 'full-res'}, {n_workers} threads) → {pdir}")
    made = 0
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as ex_pool:
        for r in ex_pool.map(_one, exr):
            made += int(r)
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

def sync_shots_from_guide(state: "AppState", guide_path: str) -> int:
    """Copy a guide's per-shot analysis onto state.shots_data. Returns shots updated.

    The UI normally does this in poll() when it sees GUIDE_PATH_UPDATE, but poll() only
    runs on the event loop. worker_pipeline chains Analyze -> Mask inside ONE worker
    thread, so masking could reach _shots_data_ before poll() had copied anything across
    and read the PRE-analysis (empty) include/exclude prompts. worker_mask treats those
    as authoritative and writes them back verbatim, which silently erased the prompts
    Qwen had just produced and left every task on 'no_mask_needed'. Calling this at the
    end of worker_analyze makes the handoff deterministic instead of a race with poll().
    """
    if not (state and guide_path and os.path.isfile(guide_path)):
        return 0
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return 0
    n = 0
    for s_item in (data.get("shots") or []):
        if not isinstance(s_item, dict):
            continue
        nm = s_item.get("shot_name") or s_item.get("shot") or s_item.get("name")
        shot = state.shots_data.get(nm) if nm else None
        if not shot:
            continue
        shot.strategy = _derive_strategy(s_item)
        shot.include_prompts = _extract_prompt_list(s_item, ["mask_includes", "include_prompts", "sam3_include_prompt"])
        shot.exclude_prompts = _extract_prompt_list(s_item, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
        rt = s_item.get("qwen2_things", [])
        if isinstance(rt, list):
            shot.detected_things = rt
        _apply_analysis_fields(shot, s_item)
        n += 1
    return n

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
    plate_start: int = 0   # absolute first frame number parsed from filenames
    plate_end: int = 0     # absolute last frame number
    render_path: str = ""  # TAPNext: user-pointed .mp4 file OR jpeg/png render folder
    studio_dir: str = ""   # publish target: <show>/<shot>/mid/cmm/bot_tracks
    # What this shot already has, read from studio_dir at scan time (shot_status_batch).
    # Kept separate from the in-memory signals (strategy / track_metrics_summary) so the
    # badges survive a restart and read the same from any workstation on the share.
    has_analysis: bool = False
    has_masks: bool = False
    has_tracks: bool = False

@dataclass
class AppState:
    shots_data: Dict[str, ShotData] = field(default_factory=dict)
    guide_path: str = ""
    current_shot_name: str = ""
    log_history: List[str] = field(default_factory=list)
    filter_query: str = ""
    # CV optical-flow masking of movers (4th backstop). OFF by default: it masks anything
    # moving independently of the camera, including objects the artist deliberately removed
    # from Exclude, so it overrides a decision made on purpose. Opt in per shot in the UI.
    motion_backstop: bool = False
    chunk_long_shots: bool = True  # SynthEyes: blip/peel long shots per window (avoid OOM)
    chunk_threshold: int = 1000    # frames above which SynthEyes chunks blip/peel
    reuse_existing_masks: bool = True   # if a shot already has masks in OUT, skip re-running SAM3
    track_chunks: int = 0          # TAPNext temporal chunks: 0=auto (VRAM-sized), >=1 forces
    track_spacing_px: int = 60     # TAPNext: min spacing between kept tracks, quoted @1920
    spread_ref_frames: int = 5     # TAPNext: frames sampled to measure that spacing ON SCREEN
    spread_scale_with_res: bool = True  # TAPNext: scale that spacing by plate width / 1920
    # Export a solve-ready set rather than everything that survived: 1000+ tracks just moves
    # the cleanup onto the artist. Best-scoring first, so the cap keeps the good ones.
    # The quality bar decides the count; this is only a ceiling. Quality comes from tracking
    # better (multi-frame template, upsampled peak, iterated refine), not from discarding.
    track_max_output: int = 600    # TAPNext: safety ceiling on exported tracks, 0=unlimited
    min_export_tracks: int = 40    # top a thin export up from the best rejects, flagged
    # Wobble gate: drop tracks deviating from their own smooth path by more than this
    # multiple of the shot's OWN median. OFF by default -- measured on real footage it cut
    # 14 of 29 tracks and left the worst one (20.51px) untouched, because that track drifts
    # SMOOTHLY and wobble only sees deviation from a track's own smooth path. Kept as a
    # slider for shots where visible jitter is the complaint. 0 = off.
    wobble_rel: float = 0.0
    max_track_gaps: int = 2        # more holes than this -> split into continuous runs
    template_frames: int = 5       # frames averaged into the reference pattern (1 = off)
    refine_iterations: int = 3     # match/polish passes per frame (1 = off)
    min_track_frames: int = 24     # TAPNext: drop tracks shorter than this (scaled on short shots)
    min_track_score: float = 0.35  # TAPNext: quality floor in [0,1]; 0 = off
    mask_dilation_px: int = 10     # SAM3: grow the exclude region at MASK time (soft edges)
    mask_margin_px: int = 8        # TAPNext: pull seeding/gating in from the matte edge
    min_corner_anisotropy: float = 0.08  # TAPNext: reject 1-D edge points that slide (0=off)
    moving_tile: bool = True       # TAPNext: native moving-tile re-track before NCC (4K accuracy)
    reseed: bool = True            # TAPNext: periodic re-seeding (replenish tracks on fast shots)
    reseed_every: int = 30         # TAPNext: max frames between re-seeds (window cap)
    edge_track: bool = True        # TAPNext: keep refining points to the frame border (edge tracks)
    gap_aware_refine: bool = True  # TAPNext: keep disappear/reappear points as one track (per-segment refine)
    pattern_refine: bool = True    # TAPNext: 3DE-style NCC/affine pattern lock at native res
    refine_patch_px: int = 31      # pattern-box size (px) for the refine pass
    # Sub-pixel accuracy (the "track wobbles in place" fixes)
    refine_ecc_polish: bool = True      # gradient sub-pixel polish after the NCC peak
    mt_overlap: int = 4                 # blend moving-tile window seams (0 = old butt-joint)
    refine_ncc_reref: float = 0.68      # how hard the point locks to its seeded pattern
    # Band-pass sigma before pattern matching. NCC removes a patch's mean but not its
    # low-frequency shape, so on a defocused feature the smooth ramp dominates and the
    # correlation peak goes broad. Subtracting a blurred copy leaves the detail that
    # localises. 0 = off.
    refine_bandpass: float = 0.0
    # Per-shot auto-tune. A batch tool cannot be hand-tuned per shot, so the bot measures each
    # plate (sharpness, grain, texture, motion) and derives the settings below from it.
    # Anything the user explicitly moves is recorded in auto_tune_overrides and always wins.
    auto_tune: bool = True
    auto_tune_overrides: Dict[str, object] = field(default_factory=dict)
    # Per-track policy: measure what each seed is sitting on (corner / blob / edge / dense)
    # and track it with parameters chosen for that, instead of one setting for the whole
    # shot. TAPNext backend only: SynthEyes blips and peels internally, so there is no
    # per-point loop to steer there.
    #
    # OFF by default -- turned on during 2026-08 development, then turned back off when it
    # was measured on a real soft plate instead of a synthetic sharp one.
    #
    # classify_seed judges cornerness against the FRAME'S OWN percentiles, which is right for
    # ranking and wrong for sizing. On SH004 (texture 12.6, 21% of frame in focus -- a sharp
    # subject on a fully defocused background) shot_profile correctly chose a 41px pattern
    # box for a soft plate, and the per-track policy then labelled 34 of 47 seeds "corner"
    # -- the top quartile of a soft distribution is still soft -- and shrank their box to
    # 21px on 35 tracks and 25px on 12. A soft feature needs a BIGGER box to average over
    # what little detail it has, which is what policy_for's own `blob` branch says; the
    # relative classifier never reaches that branch on a plate where everything is soft. So
    # the feature fought the shot-level auto-tune and lost, on exactly the shot type it was
    # enabled to help.
    #
    # bench/lab03 could not have caught this: its plate measures sharp (texture 84-606), so
    # the classification is correct there and the A/B came back neutral (every class within
    # 0.01px). That neutrality is what the decision to enable rested on, and it did not
    # transfer. Re-enabling needs classify_seed to respect ABSOLUTE softness -- at minimum,
    # never shrinking the box below the shot-level value on a plate shot_profile calls soft.
    per_track_policy: bool = False
    lossless_track_proxies: bool = True # PNG (not JPEG) proxies for the tracking route
    # Occlusion continuity: a mover crossing a point breaks the track instead of deleting it
    occlusion_continuity: bool = True
    min_occlusion_run: int = 3         # ignore 1-2 frame mask-edge chatter (anti-flicker)
    refine_ncc_hold: float = 0.45      # hysteresis: lower bar to HOLD a lock than to lose it
    reacquire_max_gap: int = 24        # frames to keep trying to re-find the point; 0 = off
    refine_ncc_reacquire: float = 0.75  # match vs the pre-occlusion patch to call it the same
    # Accuracy passes
    refine_fb_max_px: float = 1.5      # forward-backward consistency tolerance; 0 = off
    refine_drift_floor: float = 0.55   # min match vs the ORIGINAL patch when re-referencing
    # Uniform track starts + corner-feature precision
    seed_stagger: int = 4              # entry times a window's fresh seeds are split across
    spread_max_starts_per_window: int = 0   # cap tracks STARTING together; 0 = unlimited
    match_ambiguity_ratio: float = 0.90     # reject a match a rival peak nearly ties; 1 = off
    refine_search_max: int = 64             # ceiling for the adaptive NCC search radius
    # Run the SAME native-res NCC/ECC pattern refine over the SynthEyes export.
    #
    # Every accuracy pass above is TAPNext-only, because it lives in tracker_core's per-point
    # loop and SynthEyes blips and peels internally. But `pattern_refine.refine_tracks` does
    # not need that loop: it takes finished tracks, a plate and a config, which is exactly
    # what a SynthEyes export is. So the measured gains (4.88 -> 1.30px against 17 manual 4K
    # tracks) were reaching only the FALLBACK backend while the default one shipped raw.
    #
    # Measured on bench/synth/lab02 against exact ground truth, feeding it the errors a raw
    # correlation tracker makes (self-anchored scoring, i.e. bench/score_synth's model):
    #
    #     raw input                     median          p90
    #     jitter 0.3px             0.505 -> 0.108   0.554 -> 0.658
    #     jitter 0.3 + drift 1.5   0.917 -> 0.107   1.140 -> 0.713
    #     jitter 0.6 + drift 3.0   1.835 -> 0.124   2.281 -> 0.757
    #
    # It converges the median to ~0.11px on this plate whatever it is handed, which is the
    # point: it re-localises against the plate rather than tidying the input. Two things the
    # table also says, and neither is a detail:
    #   * the TAIL barely moves, and on already-clean input it gets slightly WORSE. ~0.11px
    #     is a floor, not a guarantee, so refining a track that is already better than that
    #     costs accuracy. (Feeding it a finished TAPNext export -- 0.06px -- measured as a
    #     clear regression. That is double-refining, not a bug.)
    #   * it cannot fix WHERE a track was seeded. The template comes from the track's own
    #     reported position, so a seed 1px off its corner stays 1px off; what refine fixes is
    #     everything after the anchor.
    #
    # Still off by default: that is a synthetic plate and synthetic input. The gate for
    # turning it on is a real SynthEyes export, refined vs unrefined on the same shot, via
    # tools/eval_refs.py and tools/verify_against_lk.py.
    refine_syntheyes: bool = False
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

def disable_console_quickedit() -> bool:
    """Stop a stray click in the launcher window from freezing the whole run.

    Windows consoles ship with QuickEdit ON (HKCU\\Console\\QuickEdit=1). While text is
    selected -- which is what a single click or drag in the window does -- the console
    BLOCKS every write to stdout. logger() ends in print(), so the worker thread stops on
    its next log line and the app looks hung with no error anywhere. Pressing Ctrl+C or Esc
    clears the selection and everything resumes exactly where it left off, which is the
    give-away: a long stall between two ADJACENT log lines that ends the moment the window
    is touched.

    Observed on a real batch: 51 minutes between the two halves of one auto-tune log, then
    another stall, both cleared by Ctrl+C.

    Clearing ENABLE_QUICK_EDIT_MODE removes the hazard (the window can still be scrolled;
    selecting text needs the Edit menu / Ctrl+Shift+drag). Best-effort: no console (pythonw,
    a service, a redirected pipe) simply means nothing to do.
    """
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)          # STD_INPUT_HANDLE
        if not h or h == ctypes.c_void_p(-1).value:
            return False
        mode = ctypes.c_uint()
        if not k32.GetConsoleMode(h, ctypes.byref(mode)):
            return False                   # not a console (redirected) -- nothing to disable
        ENABLE_QUICK_EDIT, ENABLE_EXTENDED = 0x0040, 0x0080
        if not (mode.value & ENABLE_QUICK_EDIT):
            return True                    # already safe
        new = (mode.value & ~ENABLE_QUICK_EDIT) | ENABLE_EXTENDED
        return bool(k32.SetConsoleMode(h, new))
    except Exception:
        return False


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

# Per-shot tracking log. The UI log view is the only record today, so an overnight failure
# can't be diagnosed afterward. While a shot is tracking, logger() also tees to this file;
# publish_shot copies it into the shot's studio logs/ folder.
_SHOT_LOG = {"path": "", "fh": None}

def open_shot_log(out_dir: str, shot: str) -> str:
    """Start teeing logger() to <out_dir>/<shot>__track.log. Safe to call repeatedly."""
    close_shot_log()
    if not (out_dir and shot):
        return ""
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{shot}__track.log")
        _SHOT_LOG["fh"] = open(path, "a", encoding="utf-8")
        _SHOT_LOG["path"] = path
        _SHOT_LOG["fh"].write(f"\n===== {shot} @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        _SHOT_LOG["fh"].flush()
        return path
    except Exception as e:
        print(f"shot log open failed for {shot}: {e}")
        _SHOT_LOG["fh"] = None
        _SHOT_LOG["path"] = ""
        return ""

def close_shot_log() -> None:
    fh = _SHOT_LOG.get("fh")
    if fh:
        try:
            fh.close()
        except Exception:
            pass
    _SHOT_LOG["fh"] = None
    _SHOT_LOG["path"] = ""

def logger(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    JOB_QUEUE.put(full_msg)   # UI log keeps full unicode - the browser renders it fine
    fh = _SHOT_LOG.get("fh")
    if fh:
        # Never let a logging problem kill the worker mid-stage (same rule as the console
        # copy below): a full disk or a closed handle must not abort tracking.
        try:
            fh.write(full_msg + "\n")
            fh.flush()
        except Exception:
            pass
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

def _refine_exported_tracks(txt_path: str, state, plate_path: str, width: int = 0,
                            height: int = 0, quality_flags=None) -> int:
    """Native-res NCC/ECC pattern-refine an exported 3DE track file, in place.

    The accuracy stack (moving-tile, pattern-refine, the certainty and identity gates) grew
    inside tracker_core's per-point loop, so it only ever ran on TAPNext -- the FALLBACK
    backend. SynthEyes, which is the default and therefore what actually ships, got the
    track_filter selection and nothing else. `pattern_refine.refine_tracks` never needed that
    loop though: it takes finished tracks, a plate and a config, and a SynthEyes export is
    exactly that. This is the whole port.

    Two things differ from the TAPNext call and both are load-bearing:

      * Y CONVENTION. tracker_core exports 3DE-flipped y (bottom-left origin) and hands
        refine `flip_y_for_3de=True` so it can un-flip. The Sizzle export writes
        `0.5*(1-v)*height`, which is ALREADY image space (top-left origin), so it must pass
        False. Flipping twice does not fail loudly -- it silently refines every point
        against the mirrored part of the plate, so nothing locks and the whole shot comes
        back empty or worse.
      * MOVING-TILE IS NOT PORTED. It re-runs TAPNext inside the tile, which would load the
        torch stack and take the GPU back while SynthEyes still holds it.

    Returns the track count after refining, or -1 if nothing was done (caller keeps its own).
    Best-effort throughout, for the same reason as the SAM3 post-filter and the track filter:
    the export is the expensive part of a SynthEyes run and must survive a failure here.
    """
    if not bool(getattr(state, "refine_syntheyes", False)):
        return -1
    try:
        from app.pattern_refine import refine_tracks
        from app.export_3de import write_tracks_txt
        from app.video_io import FrameSource, estimate_clip_bytes
        from app import track_filter as _tf
        from app.tracker_core import RunnerConfig as _RC     # torch import; CUDA stays lazy
    except Exception as e:
        logger(f"  pattern-refine unavailable ({e}); leaving the SynthEyes export as tracked")
        return -1
    if not plate_path or not os.path.exists(plate_path):
        logger(f"  pattern-refine skipped: plate not readable ({plate_path or 'unset'})")
        return -1
    try:
        end_frame, tracks = _parse_tracks_txt(txt_path)
        if not tracks:
            return -1

        cfg = _RC(
            # Required by RunnerConfig, unread by refine_tracks: nothing here drives the
            # tracker, only the refine parameters below.
            input_dir=os.path.dirname(txt_path), output_dir=os.path.dirname(txt_path),
            # SynthEyes coords are already image-space -- see the docstring. This one line is
            # the difference between a refine and a mirror.
            flip_y_for_3de=False,
            enable_pattern_refine=True,
            refine_patch_px=int(getattr(state, "refine_patch_px", 31) or 31),
            refine_ecc_polish=bool(getattr(state, "refine_ecc_polish", True)),
            refine_ncc_reref=float(getattr(state, "refine_ncc_reref", 0.68) or 0.68),
            refine_ncc_hold=float(getattr(state, "refine_ncc_hold", 0.45) or 0.45),
            refine_ncc_reacquire=float(getattr(state, "refine_ncc_reacquire", 0.75) or 0.75),
            refine_bandpass=float(getattr(state, "refine_bandpass", 0.0) or 0.0),
            refine_fb_max_px=float(getattr(state, "refine_fb_max_px", 1.5) or 0.0),
            refine_drift_floor=float(getattr(state, "refine_drift_floor", 0.55) or 0.0),
            refine_gap_aware=bool(getattr(state, "gap_aware_refine", True)),
            refine_search_max=int(getattr(state, "refine_search_max", 64) or 24),
            reacquire_max_gap=int(getattr(state, "reacquire_max_gap", 24) or 0),
            match_ambiguity_ratio=float(getattr(state, "match_ambiguity_ratio", 0.90) or 1.0),
            template_frames=int(getattr(state, "template_frames", 5) or 1),
            refine_iterations=int(getattr(state, "refine_iterations", 3) or 1),
            min_corner_anisotropy=float(getattr(state, "min_corner_anisotropy", 0.08) or 0.0),
            min_track_certainty=float(getattr(state, "min_track_certainty", 0.0) or 0.0),
            auto_tune=bool(getattr(state, "auto_tune", True)),
            auto_tune_overrides=dict(getattr(state, "auto_tune_overrides", {}) or {}),
            quality_flags=list(quality_flags or []),
        )

        # Hold the plate whole when it fits, stream it when it does not. FrameSource -- not
        # pattern_refine's own _FrameGray -- because an image SEQUENCE is the normal SynthEyes
        # input and _FrameGray's streaming path goes through cv2.VideoCapture, which cannot
        # open a directory of EXRs.
        stream = False
        try:
            import psutil  # type: ignore
            need = int(estimate_clip_bytes(plate_path, 1.0))
            stream = need > int(psutil.virtual_memory().available * 0.5)
        except Exception:
            stream = False
        src = FrameSource(plate_path, scale=1.0, stream=stream)
        total = int(src.total or 0) or int(end_frame or 0)
        W0 = int(width or src.w0 or 0)
        H0 = int(height or src.h0 or 0)

        # Same per-shot measurement the TAPNext runner does (tracker_core._auto_tune): the
        # refine parameters that matter most here -- pattern box size and band-pass -- are
        # exactly the ones shot_profile derives, and a soft plate wants both different.
        if cfg.auto_tune:
            try:
                from app import shot_profile as _sp
                prof = _sp.profile_shot(src.get(0, min(6, max(2, total))),
                                        flags=list(cfg.quality_flags or []))
                for k, v in _sp.tune(prof, overrides=dict(cfg.auto_tune_overrides)).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                logger(f"  auto-tune: {prof.describe()}")
            except Exception as e:
                logger(f"  auto-tune skipped ({e}); refining with the configured values")

        logger(f"  Pattern-refine (SynthEyes): {len(tracks)} track(s) at {W0}x{H0}"
               f"{' [streamed]' if stream else ''}")
        refined, info = refine_tracks(tracks, plate_path, W0, H0, total, cfg,
                                      status=lambda m: logger(f"  {m}"), bgr_source=src)
        if not refined:
            logger("  pattern-refine kept nothing; leaving the export as tracked")
            return -1

        # The two gates that need refine's measurements and so could never run here before.
        # identity_gate is the only test in the repo that catches a track which drifted
        # SMOOTHLY off its feature -- the failure an artist finds last and hates most.
        before = len(refined)
        refined = _tf.identity_gate(refined, getattr(refine_tracks, "last_identity", {}) or {},
                                    cfg, log=logger)
        refined = _tf.certainty_gate(refined, getattr(refine_tracks, "last_certainty", {}) or {},
                                     cfg, log=logger)
        if not refined:
            logger("  refine gates kept nothing; leaving the export as tracked")
            return -1

        write_tracks_txt(txt_path, refined, end_frame=int(end_frame or total))
        logger(f"  pattern-refine: {len(tracks)} -> {before} refined -> {len(refined)} kept "
               f"({info})")
        return len(refined)
    except Exception as e:
        logger(f"  pattern-refine failed ({e}); leaving the SynthEyes export as tracked")
        return -1


def _filter_exported_tracks(txt_path: str, state, width: int = 0, height: int = 0,
                            n_frames: int = 0, fallback: int = 0) -> int:
    """Cut an exported 3DE track file down to the solve-ready set. Returns the new count.

    Runs the SAME score -> gate -> spread -> cap selection the TAPNext path uses, via
    app/track_filter.py. It lives here rather than in syntheyes_engine because it is
    pipeline policy, not SynthEyes driving, and because it applies to any backend that
    hands back a finished 3DE file.

    Best-effort: on any failure the file is left exactly as SynthEyes wrote it and the
    original count is returned. Losing the filter is an inconvenience; losing the tracks
    after a long SynthEyes run is not acceptable.
    """
    try:
        from app.track_filter import FilterConfig, filter_tracks
        from app.export_3de import write_tracks_txt
    except Exception as e:
        logger(f"  track filter unavailable ({e}); exporting unfiltered")
        return fallback
    try:
        end_frame, tracks = _parse_tracks_txt(txt_path)
        if not tracks:
            return fallback
        T = int(n_frames or end_frame or 0) or max(
            (p[0] for pts in tracks.values() for p in pts), default=1)
        cfg = FilterConfig.from_state(state)
        kept = filter_tracks(tracks, T, int(width), int(height), cfg, log=logger)
        if not kept:
            logger("  track filter kept nothing; leaving the export untouched")
            return fallback
        if len(kept) >= len(tracks):
            return len(kept)
        write_tracks_txt(txt_path, kept, end_frame=int(end_frame or T))
        logger(f"  track filter: {len(tracks)} -> {len(kept)} solve-ready track(s)")
        return len(kept)
    except Exception as e:
        logger(f"  track filter failed ({e}); exporting unfiltered")
        return fallback


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


def clear_shot_artifacts(work_out: str, studio_dir: str, shot: str, *,
                         analysis: bool = False, masks: bool = False,
                         tracks: bool = False) -> Dict[str, int]:
    """Delete a shot's finished work, per scope, from BOTH the local work dir and the shot's
    own studio folder. Returns {'analysis': n, 'masks': n, 'tracks': n} file counts.

    DESTRUCTIVE and reaches the network share: the studio copies are the ones that make the
    UI badges (and another workstation's view) read 'done', so clearing a badge has to remove
    them too. Each scope is independent so masks -- by far the most expensive stage -- can be
    kept while tracking is redone. Missing paths are a no-op, never an error.
    """
    import shutil
    counts = {"analysis": 0, "masks": 0, "tracks": 0}

    def _rm_file(p) -> int:
        try:
            if p and os.path.isfile(p):
                os.remove(p)
                return 1
        except Exception as e:
            logger(f"clear '{shot}': could not delete {p}: {e}")
        return 0

    def _rm_tree(d) -> int:
        try:
            if not d or not os.path.isdir(d):
                return 0
            n = sum(1 for p in Path(d).rglob("*") if p.is_file())
            shutil.rmtree(d, ignore_errors=True)
            return n
        except Exception as e:
            logger(f"clear '{shot}': could not delete {d}: {e}")
            return 0

    if analysis:
        # Guide entry + manual note in the work dir, then the published slice.
        try:
            clear_shot_memory(work_out, shot)
        except Exception as e:
            logger(f"clear '{shot}': analysis memory: {e}")
        if studio_dir:
            counts["analysis"] += _rm_tree(os.path.join(studio_dir, "analysis"))

    if masks:
        for md in _shot_mask_dirs(work_out, shot):
            counts["masks"] += _rm_tree(md)
        if studio_dir:
            counts["masks"] += _rm_tree(os.path.join(studio_dir, "masks"))

    if tracks:
        # Work dir: "<shot>__*.txt" for either backend (same rule publish_shot matches on).
        for backend in ("syntheyes", "tapnext"):
            for f in _shot_track_files(work_out, shot, backend):
                counts["tracks"] += _rm_file(os.path.join(work_out, f))
        # Studio: the published "<shot>_2Dtracks*.txt".
        try:
            if studio_dir and os.path.isdir(studio_dir):
                pre = f"{shot.lower()}_2dtracks"
                for f in os.listdir(studio_dir):
                    if f.lower().startswith(pre) and f.lower().endswith(".txt"):
                        counts["tracks"] += _rm_file(os.path.join(studio_dir, f))
        except OSError as e:
            logger(f"clear '{shot}': listing studio dir: {e}")

    done = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    logger(f"Cleared {shot}: {done}" if done else f"Cleared {shot}: nothing to remove")
    return counts


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
        # Analyse only needs small frames for Qwen — make downscaled proxies (fast, tiny
        # on the share, and never a 6K frame into the mp4 encoder). Full-res proxies for
        # masking are made separately at Mask time. Cached per shot + reused.
        plate_map = {n: ensure_plate_proxies(d.plate_dir, shot_cache_dir(d.studio_dir, str(batch_dir), n),
                                             logger, max_side=1280)
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
        # Same reason: push the new prompts/strategy onto shots_data HERE rather than
        # waiting for poll() to do it, so a chained Mask reads this analysis and not the
        # empty pre-run values (which it would then write back over the guide).
        try:
            n_sync = sync_shots_from_guide(state, str(guide_path))
            if n_sync:
                logger(f"Applied analysis to {n_sync} shot(s).")
        except Exception as e:
            logger(f"Warning: could not apply guide to the shot table: {e}")
        # Publish each analyzed shot's guide slice + note to its studio tree.
        for nm, d in state.shots_data.items():
            if getattr(d, "use", False) and getattr(d, "studio_dir", ""):
                publish_shot(out_dir, d.studio_dir, nm, "", scope="analysis", log_cb=logger)
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
                sh["input_dir"] = ensure_plate_proxies(pd, shot_cache_dir(data.studio_dir, out_dir, name), logger)

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
        # Grow the exclude region so trackers stay off soft matte edges (hair, motion-blur
        # fringe). sam3_runner has always implemented this but nothing ever set it, so it sat
        # at 0 and tracks kept landing in the halo just outside the matte and sliding.
        mdil = max(0, int(getattr(state, "mask_dilation_px", 10) or 0))
        logger(f"Mask edge dilation: {mdil}px" if mdil else "Mask edge dilation: OFF")
        base = dict(guide_json_path=Path(run_guide_path), input_root=Path(in_dir),
                    output_root=Path(out_dir), weights_path=Path(weights))
        try:
            cfg = SamConfig(**base, motion_backstop=mb, mask_dilation_px=mdil)
        except TypeError:
            # Older SamConfig without motion_backstop / mask_dilation_px
            try:
                cfg = SamConfig(**base, motion_backstop=mb)
                if mdil:
                    logger("  (this SAM3 build has no mask_dilation_px - edge margin skipped)")
            except TypeError:
                cfg = SamConfig(**base)
        run_sam3_batch(cfg, log_cb=logger, progress_cb=lambda d,t: _set_progress(f"{d}/{t} frames", d, t), status_cb=logger)
        _free_vram("after SAM3")
        logger("Masking Complete.")
        # Publish masks to each selected shot's studio tree.
        for nm, d in state.shots_data.items():
            if getattr(d, "use", False) and getattr(d, "studio_dir", ""):
                publish_shot(out_dir, d.studio_dir, nm, "mask", scope="mask", log_cb=logger)
        JOB_QUEUE.put("DONE_MASKING")
        return True
    except Exception as e:
        logger(f"ERROR in Masking: {e}")
        traceback.print_exc()
        JOB_QUEUE.put("DONE_MASKING")   # signal end on failure too (also refreshes the table)
        return False

def _shot_track_files(work_out, shot: str, backend: str) -> List[str]:
    """Track .txt files a backend produced for a shot, using the SAME match rule as
    publish_shot ("<shot>__*__<backend>.txt"). Lets the caller confirm a backend really
    produced output before publishing under its name."""
    out = []
    try:
        for f in os.listdir(str(work_out)):
            low = f.lower()
            if (low.endswith(f"__{backend}.txt") and low.startswith(f"{shot.lower()}__")):
                out.append(f)
    except OSError:
        pass
    return out


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


def _track_shots_tapnext(in_root, out_root, shot_tasks_map, state, grid, seed_count,
                         seed_min_dist, only_shots=None):
    """TAPNext++ tracking path (Apache-2.0 GPU tracker, fallback backend).

    only_shots: restrict the run to this set of shot names (same convention as
    run_qwen2_batch). Used for the retry pass over shots SynthEyes could not track.
    """
    any_ran = False
    stopped = False
    sel_names = [n for n, d in state.shots_data.items() if getattr(d, "use", False)
                 and (only_shots is None or n in only_shots)]
    sel_total = len(sel_names)
    sel_done = 0
    for shot_name, data in state.shots_data.items():
        if STOP_EVENT.is_set():
            logger("Tracking stopped by user."); stopped = True; break
        if not getattr(data, "use", False): continue
        if only_shots is not None and shot_name not in only_shots: continue
        open_shot_log(str(out_root), shot_name)

        video_dir, filename = None, None
        # Renders persist in the shot's own cache folder (studio tree) and are reused.
        cache = shot_cache_dir(getattr(data, "studio_dir", ""), str(out_root), shot_name)
        renders = Path(cache) / "renders"
        # True plate resolution: the mp4 fed to TAPNext is a downscaled proxy, so the
        # tracker must export coords at the full plate size (else they land in a corner).
        _pd0 = str(getattr(data, "plate_dir", "") or "")
        plate_w = plate_h = 0
        try:
            if _pd0 and os.path.isdir(_pd0):
                _imgs = sorted(f for f in os.listdir(_pd0)
                               if os.path.splitext(f)[1].lower() in _SEQ_EXTS)
                if _imgs:
                    import cv2 as _cv2
                    _im = _cv2.imread(os.path.join(_pd0, _imgs[0]), _cv2.IMREAD_UNCHANGED)
                    if _im is not None:
                        plate_h, plate_w = int(_im.shape[0]), int(_im.shape[1])
        except Exception:
            plate_w = plate_h = 0
        if plate_w and plate_h:
            data.width, data.height = plate_w, plate_h
        # Prefer a full-res image SEQUENCE (native plate res, no mp4/codec, no downscale).
        # A user-pointed .mp4 keeps the video route; a user folder or the plate proxies go
        # in as a sequence so tracking + native-res refine run at true plate resolution.
        seq_path = ""
        rp = str(getattr(data, "render_path", "") or "").strip()
        if rp and os.path.exists(rp):
            if os.path.isfile(rp) and rp.lower().endswith(".mp4"):
                video_dir, filename = Path(rp).parent, Path(rp).name
            elif os.path.isdir(rp):
                seq_path = rp
            else:
                logger(f"{shot_name}: render_path '{rp}' is not an .mp4 or a folder.")
        if not video_dir and not seq_path:
            shot_dir = in_root / shot_name
            if shot_dir.exists() and shot_dir.is_dir():
                mp4s = sorted([p for p in shot_dir.iterdir() if p.suffix.lower() == ".mp4"])
                if mp4s: video_dir, filename = shot_dir, mp4s[0].name
        if not video_dir and not seq_path:
            exact = in_root / f"{shot_name}.mp4"
            if exact.exists(): video_dir, filename = in_root, exact.name
        if not video_dir and not seq_path:
            # Track the full-res JPG proxy sequence directly — reuses the Mask proxy (no EXR
            # re-decode), no mp4 encode, native plate resolution.
            pd = str(getattr(data, "plate_dir", "") or "")
            if pd:
                # Lossless for TRACKING: JPEG block artefacts don't move with the image, so
                # they read as sub-pixel jitter on an otherwise static feature.
                lossless = bool(getattr(state, "lossless_track_proxies", True))
                jpgdir = ensure_plate_proxies(pd, cache, logger, lossless=lossless)
                if jpgdir and os.path.isdir(jpgdir):
                    seq_path = jpgdir
                    logger(f"{shot_name}: tracking full-res "
                           f"{'lossless ' if lossless else ''}proxy sequence -> {jpgdir}")
        if not video_dir and not seq_path:
            logger(f"Skip {shot_name}: no footage. Point the shot at a render (.mp4 or JPEG "
                   f"folder) in the editor, or set a plate version."); continue
        if seq_path:
            filename = shot_name   # scale-key + output naming for the sequence route

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
                input_dir=(str(video_dir) if video_dir else str(out_root)), output_dir=str(out_root), mask_root_dir=str(out_root),
                # SAM3 contract: white=keep/track, black=ignore. Derive polarity from that + mode
                # (inside seeds the white keep-region; outside seeds background, excluding black
                # movers). Beats "auto" pixel-majority guessing, which flips per-frame and unions to
                # a 100% exclude region on ~50/50 masks (movers filling half the frame) -> 0 seeds.
                mask_mode=mode, mask_polarity=("white" if mode == "inside" else "black"),
                mask_subdir=mask_subdir, output_tag=output_tag,
                grid_size=int(grid), seeding_mode="features",
                max_tracks=int(seed_count), min_feature_dist=int(seed_min_dist),
                flip_y_for_3de=True, selected_files=[filename], selected_scales={filename: float(data.scale.strip('%'))/100.0 if '%' in data.scale else 1.0},
                sequence_path=(seq_path or ""), sequence_name=(shot_name if seq_path else ""),
                # Sequence route is already native plate res; only the mp4-proxy route needs upscaling.
                out_w=(0 if seq_path else int(plate_w or 0)), out_h=(0 if seq_path else int(plate_h or 0)),
                frame_start=int(getattr(data, "frame_start", 0) or 0), frame_end=int(getattr(data, "frame_end", 0) or 0),
                chunks=int(getattr(state, "track_chunks", 0) or 0),
                filter_max_jump_px=float(getattr(state, "filter_max_jump_px", 0.0) or 0.0),
                filter_max_jitter_px=float(getattr(state, "filter_max_jitter_px", 0.0) or 0.0),
                spread_min_dist_px=int(getattr(state, "track_spacing_px", 60) or 60),
                spread_ref_frames=int(getattr(state, "spread_ref_frames", 5) or 5),
                spread_scale_with_res=bool(getattr(state, "spread_scale_with_res", True)),
                max_output_tracks=int(getattr(state, "track_max_output", 600) or 0),
                min_export_tracks=int(getattr(state, "min_export_tracks", 40) or 0),
                wobble_rel=float(getattr(state, "wobble_rel", 1.5) or 0.0),
                max_track_gaps=int(getattr(state, "max_track_gaps", 2)),
                template_frames=int(getattr(state, "template_frames", 5) or 1),
                refine_iterations=int(getattr(state, "refine_iterations", 3) or 1),
                # Final gate: ship a solve-ready set, not everything that survived.
                min_track_frames=int(getattr(state, "min_track_frames", 24) or 0),
                min_track_score=float(getattr(state, "min_track_score", 0.35) or 0.0),
                # Pull seeds/gating in from the matte edge, and reject 1-D (edge) features
                # that NCC can only slide along. See RunnerConfig for the reasoning.
                mask_margin_px=int(getattr(state, "mask_margin_px", 8) or 0),
                min_corner_anisotropy=float(getattr(state, "min_corner_anisotropy", 0.08) or 0.0),
                enable_moving_tile=bool(getattr(state, "moving_tile", True)),
                enable_reseed=bool(getattr(state, "reseed", True)),
                reseed_every=int(getattr(state, "reseed_every", 30) or 30),
                mt_edge_track=bool(getattr(state, "edge_track", True)),
                refine_gap_aware=bool(getattr(state, "gap_aware_refine", True)),
                enable_pattern_refine=bool(getattr(state, "pattern_refine", True)),
                refine_patch_px=int(getattr(state, "refine_patch_px", 31) or 31),
                refine_ecc_polish=bool(getattr(state, "refine_ecc_polish", True)),
                mt_overlap=int(getattr(state, "mt_overlap", 4) or 0),
                refine_ncc_reref=float(getattr(state, "refine_ncc_reref", 0.68) or 0.68),
                refine_bandpass=float(getattr(state, "refine_bandpass", 2.0) or 0.0),
                # Auto-tune reads the plate; Qwen's quality_flags (already computed by
                # Analyze, and previously used for nothing but a table cell) cross-check it.
                auto_tune=bool(getattr(state, "auto_tune", True)),
                auto_tune_overrides=dict(getattr(state, "auto_tune_overrides", {}) or {}),
                per_track_policy=bool(getattr(state, "per_track_policy", False)),
                quality_flags=list(getattr(data, "quality_flags", []) or []),
                # Occlusion continuity + the two accuracy passes. See RunnerConfig for why
                # each exists; all are no-ops at 0/False.
                occlusion_continuity=bool(getattr(state, "occlusion_continuity", True)),
                min_occlusion_run=int(getattr(state, "min_occlusion_run", 3) or 1),
                refine_ncc_hold=float(getattr(state, "refine_ncc_hold", 0.45) or 0.45),
                reacquire_max_gap=int(getattr(state, "reacquire_max_gap", 24) or 0),
                refine_ncc_reacquire=float(getattr(state, "refine_ncc_reacquire", 0.75) or 0.75),
                refine_fb_max_px=float(getattr(state, "refine_fb_max_px", 1.5) or 0.0),
                refine_drift_floor=float(getattr(state, "refine_drift_floor", 0.55) or 0.0),
                # Even track starts + the distinctiveness test that stops a point snapping to
                # an identical neighbour (bolts, window grids). See RunnerConfig for why.
                seed_stagger=int(getattr(state, "seed_stagger", 4) or 1),
                spread_max_starts_per_window=int(getattr(state, "spread_max_starts_per_window", 0) or 0),
                match_ambiguity_ratio=float(getattr(state, "match_ambiguity_ratio", 0.90) or 1.0),
                refine_search_max=int(getattr(state, "refine_search_max", 64) or 24),
            )
            runner = BatchTrackerRunner(cfg, on_status=lambda m: logger(f"TRACK: {m}"))
            runner.run()
            any_ran = True
            # The tracker names its output after the video stem (e.g. WLF0070_v001__tapnext.txt).
            # Rename to the SHOT name so it publishes as <shot>_2Dtracks[__task]__tapnext.txt.
            stem = Path(filename).stem
            src_base = f"{stem}__tapnext.txt" if not output_tag else f"{stem}__{output_tag}__tapnext.txt"
            out_base = f"{shot_name}__tapnext.txt" if not output_tag else f"{shot_name}__{output_tag}__tapnext.txt"
            if src_base != out_base:
                sp = os.path.join(str(out_root), src_base)
                dp = os.path.join(str(out_root), out_base)
                if os.path.exists(sp):
                    try:
                        os.replace(sp, dp)
                        logger(f"  tracks: {src_base} -> {out_base}")
                    except Exception as e:
                        logger(f"  rename tracks failed: {e}"); out_base = src_base
            try:
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
    close_shot_log()
    return any_ran, stopped


def _track_shots_syntheyes(in_root, out_root, shot_tasks_map, state, seed_count):
    """SynthEyes tracking path: drive one SynthEyes instance over SyPy3 across shots.

    Returns (any_ran, stopped, failed) where `failed` maps shot -> reason for every shot that
    produced no usable tracks. worker_track retries those on TAPNext once SynthEyes has closed
    and given the GPU back.
    """
    any_ran = False
    stopped = False
    failed = {}
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
        return any_ran, stopped, failed
    engine = SynthEyesEngine(settings, on_log=lambda m: logger(f"SE: {m}"))
    if not engine.setup_sypy():
        logger("ERROR: SyPy3 not found — install SynthEyes / check its bundled Python.")
        return any_ran, stopped, failed
    # Force a CLEAN instance for the batch: reusing a stale/hung SynthEyes left over from a
    # prior run desyncs the socket and makes process_shot fail (shots then get swallowed ->
    # a bland "Nothing to track"). launch() kills any existing instance first, so this
    # guarantees a fresh, in-sync SynthEyes. (connect_or_launch would silently reuse it.)
    logger("Starting a clean SynthEyes instance for this batch…")
    if not engine.launch() or not engine.connect():
        logger("ERROR: Could not start/connect to SynthEyes.")
        return any_ran, stopped, failed

    prev_was_heavy = False
    try:
        for shot_name, data in state.shots_data.items():
            if STOP_EVENT.is_set():
                logger("Tracking stopped by user."); stopped = True; break
            if not getattr(data, "use", False): continue
            open_shot_log(str(out_root), shot_name)

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
                    failed[shot_name] = "no image sequence or movie found"
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

            # The PRESET decides the count unless it is "Custom", in which case the Seed Count
            # slider is used. Say which one won: the slider sits above the backend selector and
            # reads as global, so a preset quietly overriding it looks like a broken slider.
            track_count = se_preset_track_count(preset, seed_count) if se_preset_track_count else int(seed_count)
            if int(track_count) != int(seed_count):
                logger(f"  max tracks: {track_count} (from preset '{preset}'; the Seed Count "
                       f"slider value {int(seed_count)} is ignored — choose the 'Custom' "
                       f"preset to use the slider)")
            else:
                logger(f"  max tracks: {track_count} (preset '{preset}')")

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
                    # Only a shot that actually produced tracks counts as "ran". A 0-track
                    # shot used to report success AND publish an empty .txt over a previously
                    # good one; now it is a failure and goes to the TAPNext retry pass.
                    if n_trk > 0:
                        # Sub-pixel first, selection second. Refine changes the positions the
                        # scores are computed from, and it is what produces the certainty and
                        # identity numbers -- so running the filter first would grade tracks
                        # on measurements the refine is about to replace.
                        _ref = _refine_exported_tracks(
                            os.path.join(str(out_root), out_base), state,
                            # dirname(first_frame), not shot_dir: that is the exact folder
                            # SynthEyes read, so refine indexes the same frames the export
                            # numbers refer to. shot_dir can hold more than one version, and
                            # FrameSource would rediscover the sequence its own way.
                            plate_path=(movie_path or os.path.dirname(str(first_frame))),
                            width=int(img_w or 0), height=int(img_h or 0),
                            quality_flags=list(getattr(data, "quality_flags", []) or []))
                        if _ref > 0:
                            n_trk = _ref
                        # Same solve-ready selection the TAPNext path runs. Without it
                        # SynthEyes exported everything it tracked (800 on the 'Normal'
                        # preset) and the cleanup landed on the artist -- and since
                        # SynthEyes is the DEFAULT backend, that was the common case.
                        n_trk = _filter_exported_tracks(
                            os.path.join(str(out_root), out_base), state,
                            width=int(img_w or 0), height=int(img_h or 0),
                            n_frames=int(frame_count or 0), fallback=n_trk)
                        any_ran = True
                        logger(f"  DONE: {shot_name}/{task_id} — {n_trk} trackers")
                    else:
                        failed[shot_name] = f"{task_id}: exported 0 tracks"
                        logger(f"  FAILED: {shot_name}/{task_id} — 0 tracks "
                               f"(queued for the TAPNext retry pass)")
                        continue
                except Exception as e:
                    logger(f"ERROR tracking '{shot_name}/{task_id}': {e}")
                    traceback.print_exc()
                    failed[shot_name] = f"{task_id}: {e}"
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
        close_shot_log()
        logger("Closing SynthEyes...")
        try:
            engine.kill_syntheyes()
        except Exception as e:
            logger(f"  Could not close SynthEyes: {e}")
    return any_ran, stopped, failed


def worker_track(in_dir, out_dir, grid, seed_count, seed_min_dist, state: AppState):
    try:
        logger("--- Starting Step 5: Tracking ---")
        n_sel = sum(1 for d in state.shots_data.values() if getattr(d, "use", False))
        if n_sel == 0:
            logger("No shots selected. Tick the 'Use' box on at least one shot, then retry.")
            JOB_QUEUE.put("DONE_TRACKING")
            return False
        # Studio flow tracks from each shot's per-shot plate_dir, so the legacy Input
        # Folder is optional; only require it when NO selected shot has a plate_dir.
        in_root = Path(in_dir) if in_dir else Path("")
        out_root = Path(out_dir) if out_dir else None
        has_plate = any(getattr(d, "use", False) and getattr(d, "plate_dir", "")
                        for d in state.shots_data.values())
        if not has_plate and (not in_dir or not in_root.exists()):
            raise RuntimeError("Input folder does not exist (and no per-shot plate dir set).")
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
        if backend in ("syntheyes", "both") and SynthEyesEngine is None:
            logger(f"SynthEyes backend unavailable ({SYNTHEYES_IMPORT_ERROR}); falling back to TAPNext++.")
            backend = "tapnext"

        # Which backend(s) actually produced each shot's tracks -> shot: [backend, ...].
        # publish_shot matches "<shot>__*__<backend>.txt" and is called once per backend, so
        # this must be a LIST: a batch can mix (some shots fall back to TAPNext) and "both"
        # deliberately produces two sets for the same shot. One global backend would
        # silently publish nothing for the odd ones out.
        backend_by_shot = {}

        def _mark(nm, bk):
            backend_by_shot.setdefault(nm, [])
            if bk not in backend_by_shot[nm]:
                backend_by_shot[nm].append(bk)

        if backend in ("syntheyes", "both"):
            logger("Tracking backend: SynthEyes" + (" (then TAPNext++ as well)" if backend == "both" else ""))
            _free_vram("before SynthEyes")  # release torch cache so SynthEyes can use the GPU
            any_ran, stopped, failed = _track_shots_syntheyes(in_root, out_root, shot_tasks_map,
                                                              state, seed_count)
            for nm, d in state.shots_data.items():
                if getattr(d, "use", False) and nm not in failed:
                    _mark(nm, "syntheyes")
            if backend == "both" and not stopped:
                # Run EVERY selected shot on TAPNext too, not just the SynthEyes failures:
                # the point of "both" is two independent sets to compare. Same deferred
                # placement as the retry below -- SynthEyes holds the GPU until its finally.
                logger("=== Second pass: TAPNext++ on all selected shot(s) ===")
                if BatchTrackerRunner is None:
                    _ensure_tracker_loaded()
                if BatchTrackerRunner is None:
                    logger(f"TAPNext++ unavailable, second pass skipped: {TRACKER_IMPORT_ERROR}")
                else:
                    _free_vram("before TAPNext pass")
                    t_ran, t_stopped = _track_shots_tapnext(
                        in_root, out_root, shot_tasks_map, state, grid, seed_count, seed_min_dist)
                    _free_vram("after TAPNext pass")
                    stopped = stopped or t_stopped
                    if t_ran:
                        any_ran = True
                        for nm, d in state.shots_data.items():
                            if getattr(d, "use", False) and _shot_track_files(out_root, nm, "tapnext"):
                                _mark(nm, "tapnext")
                    else:
                        logger("TAPNext++ second pass produced no tracks.")
                failed = {}   # 'both' already ran TAPNext everywhere; no retry to do
            # Deferred TAPNext retry. It has to run AFTER the SynthEyes pass, not inline:
            # SynthEyes holds the GPU until _track_shots_syntheyes kills it in its finally.
            if failed and not stopped:
                logger(f"=== {len(failed)} shot(s) produced no tracks in SynthEyes — "
                       f"retrying on TAPNext++ ===")
                for nm, why in sorted(failed.items()):
                    logger(f"   {nm}: {why}")
                if BatchTrackerRunner is None:
                    _ensure_tracker_loaded()
                if BatchTrackerRunner is None:
                    logger(f"TAPNext++ unavailable, cannot retry: {TRACKER_IMPORT_ERROR}")
                else:
                    _free_vram("before TAPNext retry")
                    t_ran, t_stopped = _track_shots_tapnext(
                        in_root, out_root, shot_tasks_map, state, grid, seed_count,
                        seed_min_dist, only_shots=set(failed))
                    _free_vram("after TAPNext retry")
                    stopped = stopped or t_stopped
                    if t_ran:
                        any_ran = True
                        # Only shots with a real TAPNext .txt publish as tapnext.
                        for nm in failed:
                            if _shot_track_files(out_root, nm, "tapnext"):
                                _mark(nm, "tapnext")
                        recovered = [n for n in failed if "tapnext" in backend_by_shot.get(n, ())]
                        logger(f"TAPNext++ retry recovered {len(recovered)}/{len(failed)} shot(s)"
                               + (f": {', '.join(sorted(recovered))}" if recovered else ""))
                    else:
                        logger("TAPNext++ retry produced no tracks either.")
        else:
            logger("Tracking backend: TAPNext++")
            _ensure_tracker_loaded()
            _free_vram("before TAPNext")
            if BatchTrackerRunner is None: raise ImportError(f"Tracker module missing. {TRACKER_IMPORT_ERROR}")
            any_ran, stopped = _track_shots_tapnext(in_root, out_root, shot_tasks_map, state, grid, seed_count, seed_min_dist)
            _free_vram("after TAPNext")
            for nm, d in state.shots_data.items():
                if getattr(d, "use", False):
                    _mark(nm, "tapnext")

        if stopped: logger("Tracking halted.")
        elif not any_ran:
            logger("Nothing to track — no selected shot produced tracks. Check the lines above: "
                   "a shot was skipped (no image sequence/movie under the Input Folder) or "
                   "SynthEyes errored on it. Confirm the Input Folder is set and the shot is ticked.")
        else: logger("Tracking Complete.")
        if any_ran:
            # Publish the 2D tracks to each selected shot's studio bot_tracks folder, once
            # per backend that actually tracked it (two files for a 'both' run).
            for nm, d in state.shots_data.items():
                if getattr(d, "use", False) and getattr(d, "studio_dir", ""):
                    for bk in backend_by_shot.get(nm, ()):
                        publish_shot(str(out_root), d.studio_dir, nm, bk,
                                     scope="track", log_cb=logger)
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
    ps = int(getattr(d, "plate_start", 0) or 0)
    pe = int(getattr(d, "plate_end", 0) or 0)
    if fs <= 0 and fe <= 0:
        # No user sub-range: show the real plate frame range if we parsed it.
        if pe > 0:
            return f"{ps}-{pe} · {tot_s}"
        return f"all · {tot_s}"
    if ps > 0:
        # Sub-range stored as positions; display it back in absolute frame numbers.
        a = ps + (fs or 1) - 1
        b = ps + (fe or total) - 1
        return f"{a}-{b} · {tot_s}"
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