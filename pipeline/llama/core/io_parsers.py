from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

def _norm_shot(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).strip())

@dataclass
class ShotItem:
    shot: str
    client_note: str

def load_requirements_txt(path: str) -> List[ShotItem]:
    items: List[ShotItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                raise ValueError(f"Bad line (expected SHOT|NOTE): {line}")
            shot, note = line.split("|", 1)
            items.append(ShotItem(shot=shot.strip(), client_note=note.strip()))
    return items

def load_requirements_xlsx(path: str) -> List[ShotItem]:
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for XLSX. Install pandas+openpyxl in your venv.") from e

    df = pd.read_excel(path)
    cols = {c.lower().strip(): c for c in df.columns}

    shot_col = None
    note_col = None
    for k in ["shot", "shot_name", "shotid", "shot_id", "name", "id"]:
        if k in cols:
            shot_col = cols[k]; break
    for k in ["client_note", "requirement", "requirements", "note", "clientnote", "client note"]:
        if k in cols:
            note_col = cols[k]; break

    if not shot_col or not note_col:
        raise ValueError(f"XLSX must contain columns shot + client_note (or aliases). Found: {list(df.columns)}")

    items: List[ShotItem] = []
    for _, row in df.iterrows():
        shot = str(row[shot_col]).strip()
        note = str(row[note_col]).strip()
        if not shot or shot.lower() == "nan":
            continue
        items.append(ShotItem(shot=shot, client_note=note))
    return items

def load_requirements(path: str) -> List[ShotItem]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".csv"]:
        return load_requirements_txt(path)
    if ext in [".xlsx", ".xls"]:
        return load_requirements_xlsx(path)
    raise ValueError("Requirements file must be .txt or .xlsx/.xls")

def load_qwen2_v1_scene_cam_things(qwen2_json_path: str) -> Dict[str, Dict[str, Any]]:
    """Return map: {normalized_shot: {name, scene_elements, things, camera_movement}}"""
    if not os.path.isfile(qwen2_json_path):
        raise FileNotFoundError(f"Qwen2 JSON not found: {qwen2_json_path}")

    with open(qwen2_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shots = data.get("shots") if isinstance(data, dict) else None
    if not isinstance(shots, list):
        raise ValueError("Qwen2 JSON must contain top-level key 'shots' as a list.")

    out: Dict[str, Dict[str, Any]] = {}
    for s in shots:
        if not isinstance(s, dict):
            continue
        name = s.get("name") or s.get("shot") or s.get("shot_name")
        if not name:
            continue
        key = _norm_shot(name).lower()
        def _lst(key_name: str) -> List[str]:
            v = s.get(key_name)
            return v if isinstance(v, list) else []
        dl = s.get("depth_layers")
        if not isinstance(dl, dict):
            dl = {"fg": "", "mg": "", "bg": ""}
        out[key] = {
            "name": str(name),
            "scene_elements": str(s.get("scene_elements") or "").strip(),
            "things": s.get("things") if isinstance(s.get("things"), list) else [],
            "camera_movement": str(s.get("camera_movement") or "").strip(),
            "status": str(s.get("status") or "").strip(),
            "input_path": str(s.get("input_path") or "").strip(),
            # matchmove analysis fields (Qwen2.5-VL 7B)
            "moving_things": _lst("moving_things"),
            "bad_track_regions": _lst("bad_track_regions"),
            "foreground_occluders": _lst("foreground_occluders"),
            "quality_flags": _lst("quality_flags"),
            "depth_layers": dl,
            "parallax": str(s.get("parallax") or "").strip(),
        }
    return out

def get_qwen2_shot(qwen2_map: Dict[str, Dict[str, Any]], shot: str) -> Optional[Dict[str, Any]]:
    return qwen2_map.get(_norm_shot(shot).lower())
