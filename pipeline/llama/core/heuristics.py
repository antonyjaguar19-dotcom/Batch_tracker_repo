from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Motion / deforming / unstable foreground cues for camera solve safety
MOVING_TERMS = [
    # humans/animals
    "person","people","individual","man","woman","boy","girl","child","actor",
    "hand","hands","arm","arms","leg","legs","face","head","hair",
    "dog","cat","bird","bear","crowd",
    # vehicles
    "car","vehicle","truck","bus","bike","bicycle",
    # fluids/particles/fx
    "smoke","fire","water","splash","rain","snow","particles","fx","explosion",
    # interaction / deformation cues
    "glove","gloves","sponge","cloth","towel",
    # verbs commonly in descriptions (motion cues)
    "washing","scrubbing","rinsing","pouring","running","holding","waving","throwing","jogging",
    "adjusting","examining",
    # specular / reflections
    "reflection","reflections","mirror","glass","shiny","glossy","specular",
    # fast-moving objects
    "paraglider","parachute","kite",
    # robots/creatures (often animated)
    "robot","robotic","creature"
]

# Inherently dynamic subjects. For a CAMERA solve these are excluded BY DEFAULT —
# whether or not the VLM flagged them moving, and whether or not they appear in the
# scene prose. Rationale: you rarely want camera-track points ON a living/vehicle
# subject; excluding a stationary one costs a few background points, tracking a
# moving one wrecks the solve. This is the deterministic backstop for cases where
# Qwen detects the thing but never says it is moving.
DYNAMIC_SUBJECTS = set([
    # people
    "person","people","man","woman","boy","girl","child","kid","actor","human","pedestrian","crowd","worker","player",
    # animals
    "dog","cat","bird","horse","cow","sheep","goat","pig","animal","pet","deer","fox","bear","rabbit","squirrel","duck","chicken","fish",
    # vehicles
    "car","vehicle","truck","bus","van","bike","bicycle","motorcycle","motorbike","scooter","train","tram","boat","ship","plane","aircraft","helicopter","drone",
    # fluids / particles / fx
    "smoke","fire","flame","water","splash","wave","rain","snow","fountain","spray","dust","flag","balloon","leaves",
])

def dynamic_subjects(things: List[Any]) -> List[str]:
    """Raw detected things that are inherently dynamic (camera-solve hazards).

    Scans the RAW things list (NOT scene-filtered), so a subject the VLM listed but
    did not mention in its prose — and did not flag as moving — is still caught.
    """
    out: List[str] = []
    seen = set()
    for x in (things or []):
        s = str(x).strip()
        if not s:
            continue
        toks = [t for t in re.split(r"[\s_/,-]+", s.lower()) if t]
        if any(tok in DYNAMIC_SUBJECTS for tok in toks):
            k = s.lower()
            if k not in seen:
                out.append(s)
                seen.add(k)
    return out

# Terms we should NEVER pass to SAM3 as mask nouns (verbs / actions / adjectives)
BAD_MASK_TERMS = set([
    "running","jogging","holding","washing","scrubbing","rinsing","pouring",
    "adjusting","examining","clear","blue","sunny","shadows","outdoor"
])

# Only these tokens are allowed to match as a prefix word (robot -> robotic).
# This prevents false matches like "car" -> "cartoon".
PREFIX_MATCH_WHITELIST = set(["robot", "robotic", "creature"])

STATIC_PATTERNS = [
    r"\bstatic\b", r"\bstationary\b", r"\bstill\b",
    r"not\s+moving", r"no\s+movement", r"nothing\s+is\s+moving",
    r"no\s+actions?", r"no\s+other\s+objects?\s+or\s+actions?",
    r"remains\s+in\s+place", r"does\s+not\s+move",
    r"appears\s+to\s+be\s+stationary", r"\blocked[-\s]?off\b",
]

SPECULAR_TERMS = ["reflection","reflections","mirror","glass","shiny","glossy","specular"]

ENV_BLACKLIST = set([
    "background","environment","env","scene","set","plate",
    "floor","ground","road","sky","wall","ceiling",
    "table","countertop","surface","pavement","grass",
    "mountains","hills","trees","village","houses","buildings",
    "ocean","waves","beach","sand"
])

# Generic mask terms allowed only if present in scene text
GENERIC_ALLOWED = [
    "person","people","individual","man","woman","child",
    "hands","hand","arm","arms",
    "gloves","glove",
    "smoke","fire","water","rain","snow",
    "crowd","vehicle","car","bear","cat","dog",
    "glass","reflection","mirror"
]

def _term_in_text(text: str, term: str) -> bool:
    """Word-boundary match for single words; substring match for multiword phrases."""
    t = (text or "").lower()
    term_l = term.lower()
    if " " in term_l:
        return term_l in t
    return re.search(rf"\b{re.escape(term_l)}\b", t) is not None

def _prefix_term_in_text(text: str, term: str) -> bool:
    """Prefix word match, only to be used for whitelisted tokens."""
    t = (text or "").lower()
    term_l = term.lower()
    if " " in term_l:
        return term_l in t
    return re.search(rf"\b{re.escape(term_l)}\w*\b", t) is not None

def _contains_any_term(text: str, terms: List[str]) -> bool:
    return any(_term_in_text(text, term) for term in terms)

def _matches_any_regex(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)

def parse_client_intent(note: str) -> Tuple[List[str], str]:
    n = (note or "").lower()
    targets: List[str] = []
    camera_wanted = ("camera" in n) or ("matchmove" in n) or ("environment" in n) or ("env" in n) or ("bg" in n) or ("set" in n)
    # Object/rotomation cues: explicit terms + common single-subject track requests
    # ("face track", "track the actor", "head track", etc.).
    _OBJECT_TERMS = ("object", "roto", "rotomation", "face", "head", "person",
                     "people", "character", "actor", "subject", "hero", "body", "hand")
    object_wanted = any(t in n for t in _OBJECT_TERMS)

    if camera_wanted:
        targets.append("camera")
    if object_wanted or ((" and " in n) and ("track" in n) and camera_wanted):
        if "object" not in targets:
            targets.append("object")
    if not targets:
        targets = ["other"]

    if "object" in targets and "camera" in targets:
        intent = "both"
    elif "object" in targets:
        intent = "object"
    elif "camera" in targets:
        intent = "camera"
    else:
        intent = "other"
    return targets, intent

def build_sam3_prompts(mask_includes: List[str], mask_excludes: List[str]) -> Dict[str, str]:
    inc = ", ".join([str(x).strip() for x in (mask_includes or []) if str(x).strip()])
    exc = ", ".join([str(x).strip() for x in (mask_excludes or []) if str(x).strip()])
    return {"sam3_include_prompt": inc, "sam3_exclude_prompt": exc}

def is_static_scene(scene_elements: str, things: List[Any]) -> bool:
    txt = scene_elements or ""
    if not _matches_any_regex(txt, STATIC_PATTERNS):
        return False
    if _contains_any_term(txt, MOVING_TERMS):
        return False
    return True

def _is_bad_mask_term(s: str) -> bool:
    return s.strip().lower() in BAD_MASK_TERMS

def filter_things_by_scene(things: List[Any], scene_elements: str) -> List[str]:
    """
    Remove hallucinated 'things' not supported by scene text.
    Uses prefix matching ONLY for whitelisted tokens (robot->robotic).
    """
    txt = (scene_elements or "")
    out: List[str] = []
    for x in (things or []):
        s = str(x).strip()
        if not s:
            continue
        if _is_bad_mask_term(s):
            continue
        tokens = [t for t in re.split(r"[\s_/,-]+", s.lower()) if t]
        if not tokens:
            continue

        keep = False
        for tok in tokens:
            if _term_in_text(txt, tok):
                keep = True
                break
            if tok in PREFIX_MATCH_WHITELIST and _prefix_term_in_text(txt, tok):
                keep = True
                break

        if keep:
            out.append(s)

    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            uniq.append(s); seen.add(k)
    return uniq

def allowed_mask_terms(scene_elements: str, filtered_things: List[str]) -> set:
    allowed = set([t.lower() for t in filtered_things if not _is_bad_mask_term(t)])
    for g in GENERIC_ALLOWED:
        if _term_in_text(scene_elements, g):
            allowed.add(g.lower())
    return allowed

def _priority_rank(term: str) -> int:
    """Lower is higher priority."""
    t = term.lower()
    # 1) hands/gloves (interaction)
    if any(k in t for k in ["hand", "hands", "glove", "gloves"]):
        return 1
    # 2) person/people/man/woman/crowd
    if any(k in t for k in ["person", "people", "individual", "man", "woman", "child", "crowd"]):
        return 2
    # 3) vehicles
    if any(k in t for k in ["car", "vehicle", "truck", "bus", "bike", "bicycle"]):
        return 3
    # 4) animals
    if any(k in t for k in ["bear", "dog", "cat", "bird"]):
        return 4
    # 5) fluids/particles/fx
    if any(k in t for k in ["smoke", "fire", "water", "rain", "snow", "particles", "explosion"]):
        return 5
    # 6) specular/reflections
    if any(k in t for k in ["reflection", "mirror", "glass", "specular", "glossy", "shiny"]):
        return 6
    return 9

def moving_foreground_terms(scene_elements: str, things_filtered: List[str]) -> List[str]:
    """
    Conservative movers for camera-safety.
    Always add a generic human term if described, even if other movers exist.
    """
    movers: List[str] = []
    for t in (things_filtered or []):
        if _is_bad_mask_term(t):
            continue
        if _contains_any_term(t, MOVING_TERMS):
            movers.append(t)

    # Force-add human if described
    for human in ["person", "man", "woman", "people", "crowd"]:
        if _term_in_text(scene_elements, human) and human not in [m.lower() for m in movers]:
            movers.append(human)
            break

    # If still empty, fallback to other generic movers present in scene
    if not movers:
        for g in GENERIC_ALLOWED:
            if _term_in_text(scene_elements, g) and _contains_any_term(g, MOVING_TERMS):
                movers.append(g)

    # De-dup + sort by priority
    seen=set(); uniq=[]
    for x in movers:
        k=x.lower().strip()
        if k and k not in seen and not _is_bad_mask_term(k):
            uniq.append(x); seen.add(k)
    uniq.sort(key=_priority_rank)
    return uniq

def cap_excludes(excludes: List[str], cap: int = 4) -> List[str]:
    """Cap to keep SAM3 prompts stable."""
    if not excludes:
        return []
    # Already priority-sorted by moving_foreground_terms; keep first N
    return excludes[:max(1, cap)]

def pick_object_subject(client_note: str, things_filtered: List[str], scene_elements: str) -> str:
    """
    Single-subject object selection, conservative.
    """
    note = (client_note or "").lower()

    # 1) explicit mention in note
    for t in things_filtered:
        if _is_bad_mask_term(t):
            continue
        toks = [x for x in re.split(r"[\s_/,-]+", t.lower()) if x]
        if any(tok in note for tok in toks):
            return t

    # 2) most important mover
    movers = moving_foreground_terms(scene_elements, things_filtered)
    for m in movers:
        # prefer concrete thing if it exists in filtered list
        for tf in things_filtered:
            if tf.lower() == m.lower():
                return tf
    if movers:
        return movers[0]

    # 3) first non-environment noun
    for t in things_filtered:
        if _is_bad_mask_term(t):
            continue
        toks = [x for x in re.split(r"[\s_/,-]+", t.lower()) if x]
        if toks and all(tok not in ENV_BLACKLIST for tok in toks):
            return t

    return things_filtered[0] if things_filtered else "hero object"

def propose_mask_plan(client_note: str, scene_elements: str, things: List[Any], camera_movement: str) -> Dict[str, Any]:
    targets, intent = parse_client_intent(client_note)
    things_filtered = filter_things_by_scene(things, scene_elements)
    scene_text = scene_elements or ""

    is_static = is_static_scene(scene_text, things_filtered)
    movers = moving_foreground_terms(scene_text, things_filtered)
    movers = cap_excludes(movers, cap=4)
    has_moving_cues = bool(movers) or _contains_any_term(scene_text, MOVING_TERMS)
    has_specular_risk = _contains_any_term(scene_text, SPECULAR_TERMS) or any(_contains_any_term(t, SPECULAR_TERMS) for t in things_filtered)

    if is_static and not has_moving_cues:
        return {
            "suggested_mask_includes": [],
            "suggested_mask_excludes": [],
            "suggested_track_mode": "no_mask_needed",
            "suggested_tracking_targets": targets,
            "is_static_override": True,
            "has_moving_foreground": False,
            "has_specular_risk": bool(has_specular_risk),
            "intent": intent,
            "things_filtered": things_filtered,
            "moving_terms": movers,
        }

    track_mode = "no_mask_needed"
    excludes: List[str] = []

    # Safe defaults:
    if intent in ["camera", "both"]:
        if has_moving_cues:
            excludes = movers
            track_mode = "track_outside_mask"

    elif intent == "object":
        subject = pick_object_subject(client_note, things_filtered, scene_text)
        track_mode = "track_inside_mask" if subject else "no_mask_needed"

    return {
        "suggested_mask_includes": [],
        "suggested_mask_excludes": excludes,
        "suggested_track_mode": track_mode,
        "suggested_tracking_targets": targets,
        "is_static_override": False,
        "has_moving_foreground": bool(has_moving_cues),
        "has_specular_risk": bool(has_specular_risk),
        "intent": intent,
        "things_filtered": things_filtered,
        "moving_terms": movers,
    }
