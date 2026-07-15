from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .io_parsers import ShotItem, get_qwen2_shot, _norm_shot
from .heuristics import (
    build_sam3_prompts,
    filter_things_by_scene,
    allowed_mask_terms,
    parse_client_intent,
    moving_foreground_terms,
    cap_excludes,
    pick_object_subject,
    dynamic_subjects,
)


def _filter_list(xs: List[Any], allowed: set) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in (xs or []):
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in allowed and k not in seen:
            out.append(s)
            seen.add(k)
    return out


def _task_dict(task_id: str, track_mode: str, inc: List[str], exc: List[str]) -> Dict[str, Any]:
    sam = build_sam3_prompts(inc, exc)
    return {
        "task_id": task_id,
        "track_mode": track_mode,
        "mask_includes": inc,
        "mask_excludes": exc,
        "sam3_include_prompt": sam["sam3_include_prompt"],
        "sam3_exclude_prompt": sam["sam3_exclude_prompt"],
        "mask_subdir": f"masks_{task_id}",
        "preview_subdir": f"preview_{task_id}",
    }


def _build_object_task(client_note: str, things_filtered: List[str], scene: str, allowed: set) -> Dict[str, Any]:
    """Single-subject object mask."""
    subject = pick_object_subject(client_note, things_filtered, scene)
    inc = _filter_list([subject] if subject else [], allowed)
    exc: List[str] = []  # keep object masks clean for SAM3
    tm = "track_inside_mask" if inc else "no_mask_needed"
    t = _task_dict("object", tm, inc, exc)
    t["subject"] = inc[0] if inc else ""
    t["notes"] = "Single-subject object mask."
    return t


def _dedup_keep_order(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _build_camera_task(scene: str, things_filtered: List[str], allowed: set, q: Dict[str, Any] = None) -> Dict[str, Any]:
    """Camera task: outside-mask excludes movers + VLM bad-track signals (if any).

    Heuristic movers are gated by `allowed` (must be real scene nouns). VLM-derived
    signals (moving_things / foreground_occluders / bad_track_regions) are trusted
    as-is since the 7B model reports them directly per shot.
    """
    movers = _filter_list(moving_foreground_terms(scene, things_filtered), allowed)

    vlm: List[str] = []
    dyn: List[str] = []
    if q:
        vlm += q.get("moving_things") or []
        vlm += q.get("foreground_occluders") or []
        vlm += q.get("bad_track_regions") or []
        # Deterministic backstop: exclude inherently dynamic subjects from the RAW
        # things list, even if Qwen never flagged them moving / omitted them from prose.
        dyn = dynamic_subjects(q.get("things") or [])

    combined = cap_excludes(_dedup_keep_order(vlm + dyn + movers), cap=6)

    if combined:
        t = _task_dict("camera", "track_outside_mask", [], combined)
        t["notes"] = "Camera safety: exclude movers + dynamic subjects + bad-track regions (VLM + heuristic, capped)."
        return t

    t = _task_dict("camera", "no_mask_needed", [], [])
    t["notes"] = "No movers / bad-track regions detected => no mask for camera solve."
    return t


def _qwen_fields(q: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through of Qwen2.5-VL matchmove analysis fields for display + downstream."""
    if not q:
        q = {}
    dl = q.get("depth_layers")
    if not isinstance(dl, dict):
        dl = {"fg": "", "mg": "", "bg": ""}
    return {
        "qwen2_things": q.get("things") if isinstance(q.get("things"), list) else [],
        "qwen2_moving_things": q.get("moving_things") or [],
        "qwen2_bad_track_regions": q.get("bad_track_regions") or [],
        "qwen2_foreground_occluders": q.get("foreground_occluders") or [],
        "qwen2_quality_flags": q.get("quality_flags") or [],
        "qwen2_depth_layers": dl,
        "qwen2_parallax": str(q.get("parallax") or "").strip(),
    }


def build_batch_tracker_json(items: List[ShotItem], qwen2_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build mask guidance JSON.

    Output schema v2.1:
      - shots[].tasks is authoritative.
      - top-level mask_includes/mask_excludes/track_mode remain for backward compatibility.

    Qwen-only decisioning: no text LLM (Ollama/LLaMA) is consulted. camera/object/both
    intents are deterministic heuristics over Qwen's structured matchmove signals;
    ambiguous intent defaults to a camera task via _build_camera_task (same as the
    uncovered-shot synthesis below), which carries Qwen's mover signals.
    """

    out: Dict[str, Any] = {
        "version": "2.1-dual-tasks",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "shots": [],
    }

    for it in items:
        q = get_qwen2_shot(qwen2_map, it.shot)
        if not q:
            out["shots"].append(
                {
                    "shot": it.shot,
                    "shot_name": it.shot,
                    "client_note": it.client_note,
                    "qwen2_description": "",
                    "tasks": [],
                    "mask_includes": [],
                    "mask_excludes": [],
                    "track_mode": "no_mask_needed",
                    "tracking_targets": ["other"],
                    "confidence": 0.0,
                    "sam3_include_prompt": "",
                    "sam3_exclude_prompt": "",
                    "notes": "Missing shot entry in Qwen2 JSON for this shot name.",
                }
            )
            continue

        scene = (q.get("scene_elements") or "").strip()
        cam = (q.get("camera_movement") or "").strip()
        things_raw = q.get("things", [])

        things_filtered = filter_things_by_scene(things_raw, scene)
        allowed = allowed_mask_terms(scene, things_filtered)

        targets, intent = parse_client_intent(it.client_note)

        tasks: List[Dict[str, Any]] = []
        confidence = 0.8
        notes = ""

        if intent == "camera":
            cam_task = _build_camera_task(scene, things_filtered, allowed, q)
            tasks = [cam_task]
            notes = cam_task.get("notes", "")
            confidence = 0.85

        elif intent == "object":
            obj_task = _build_object_task(it.client_note, things_filtered, scene, allowed)
            tasks = [obj_task]
            notes = "Object-only => single-subject inside-mask."
            confidence = 0.85 if obj_task.get("mask_includes") else 0.6

        elif intent == "both":
            cam_task = _build_camera_task(scene, things_filtered, allowed, q)
            obj_task = _build_object_task(it.client_note, things_filtered, scene, allowed)
            tasks = [cam_task, obj_task]
            notes = "Dual-task => run camera + object as two separate passes."
            confidence = 0.9 if (obj_task.get("mask_includes")) else 0.75

        else:
            # Ambiguous client intent (no clear camera/object/both cue): default to a
            # camera solve. _build_camera_task folds in Qwen's structured mover signals
            # (moving_things / foreground_occluders / bad_track_regions / dynamic_subjects),
            # which is richer + more deterministic than the old text-LLM prose pass.
            cam_task = _build_camera_task(scene, things_filtered, allowed, q)
            cam_task["notes"] = "Ambiguous intent => default camera task (Qwen signals, deterministic)."
            tasks = [cam_task]
            notes = cam_task["notes"]
            confidence = 0.7

        primary = tasks[0] if tasks else _task_dict("other", "no_mask_needed", [], [])

        out["shots"].append(
            {
                "shot": it.shot,
                "shot_name": it.shot,
                "client_note": it.client_note,
                "qwen2_description": scene,
                "qwen2_camera_movement": cam,
                "qwen2_things_filtered": things_filtered,
                **_qwen_fields(q),
                "intent": intent,
                "tracking_targets": targets,
                "tasks": tasks,
                "mask_includes": primary.get("mask_includes", []),
                "mask_excludes": primary.get("mask_excludes", []),
                "track_mode": primary.get("track_mode", "no_mask_needed"),
                "sam3_include_prompt": primary.get("sam3_include_prompt", ""),
                "sam3_exclude_prompt": primary.get("sam3_exclude_prompt", ""),
                "confidence": confidence,
                "notes": notes,
            }
        )

    # Synthesize default (camera) entries for Qwen shots not covered by requirements,
    # so the guide — and the GUI — always list every shot, even with no/partial brief.
    covered = {_norm_shot(it.shot).lower() for it in items}
    for key, q in qwen2_map.items():
        if key in covered:
            continue
        name = str(q.get("name") or key)
        scene = (q.get("scene_elements") or "").strip()
        cam = (q.get("camera_movement") or "").strip()
        things_raw = q.get("things") if isinstance(q.get("things"), list) else []
        things_filtered = filter_things_by_scene(things_raw, scene)
        allowed = allowed_mask_terms(scene, things_filtered)

        cam_task = _build_camera_task(scene, things_filtered, allowed, q)
        primary = cam_task

        out["shots"].append(
            {
                "shot": name,
                "shot_name": name,
                "client_note": "",
                "qwen2_description": scene,
                "qwen2_camera_movement": cam,
                "qwen2_things_filtered": things_filtered,
                **_qwen_fields(q),
                "intent": "camera",
                "tracking_targets": ["camera"],
                "tasks": [cam_task],
                "mask_includes": primary.get("mask_includes", []),
                "mask_excludes": primary.get("mask_excludes", []),
                "track_mode": primary.get("track_mode", "no_mask_needed"),
                "sam3_include_prompt": primary.get("sam3_include_prompt", ""),
                "sam3_exclude_prompt": primary.get("sam3_exclude_prompt", ""),
                "confidence": 0.7,
                "notes": "No client brief => default camera task (deterministic).",
            }
        )

    return out
