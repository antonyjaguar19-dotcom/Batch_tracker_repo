# -*- coding: utf-8 -*-
"""
Batch Tracker — NiceGUI front-end.

Reuses the ENTIRE backend from app.py (loaders, workers, state, helpers) — only the
UI layer is reimplemented in NiceGUI. The Gradio app (app.py) stays as a fallback.

Run:  runtime/python311/python.exe app_nicegui.py   (or launch_nicegui.bat)
"""
from __future__ import annotations

import os
# Enable OpenCV's EXR codec before ANY cv2 import (OpenCV reads this only at init).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import re
import sys
import time
import queue
import importlib.util
from pathlib import Path

_APP_DIR = os.path.dirname(os.path.abspath(__file__))   # the app/ package dir
_HERE = os.path.dirname(_APP_DIR)                        # repo root (parent of app/)
# Embeddable Python's isolated ._pth does not auto-add the script dir to sys.path.
# Repo root must be on sys.path so `from app.*` and the `core` package resolve.
sys.path.insert(0, _HERE)

# Reuse the ENTIRE backend from app.py (sibling of this file in app/). NOTE: there is
# also an `app/` PACKAGE here, which would shadow `import app`. So load the app.py FILE
# explicitly, registered under its own sys.modules name (required for @dataclass hints).
_spec = importlib.util.spec_from_file_location("btr_backend", os.path.join(_APP_DIR, "app.py"))
be = importlib.util.module_from_spec(_spec)
sys.modules["btr_backend"] = be
_spec.loader.exec_module(be)  # builds no UI: app.py only launches under __main__

from nicegui import ui, run, app as nicegui_app  # noqa: E402


# -----------------------------------------------------------------------------
# State (single-user local tool -> one shared AppState)
# -----------------------------------------------------------------------------
state = be.AppState()

TABLE_COLS = [
    {"name": "name", "label": "Shot", "field": "name", "align": "left", "sortable": True},
    {"name": "version", "label": "Version", "field": "version", "align": "left"},
    {"name": "strategy", "label": "Strategy", "field": "strategy", "align": "left"},
    {"name": "quality", "label": "Quality", "field": "quality", "align": "left"},
    {"name": "prompts", "label": "Prompts", "field": "prompts", "align": "left"},
    {"name": "scale", "label": "Scale", "field": "scale", "align": "left"},
    {"name": "range", "label": "Range / Frames", "field": "range", "align": "left"},
    {"name": "metrics", "label": "Track Metrics", "field": "metrics", "align": "left"},
    {"name": "clear", "label": "", "field": "clear", "align": "right"},
]

_suppress_select = False  # guard so programmatic selection doesn't re-trigger handlers

# Status chip above the table. Unlike the search box (which Quasar applies in the
# browser), a chip changes which rows exist server-side, so it does rebuild the table.
# Only signals we actually hold in memory — mask state lives on disk and probing it per
# shot would reintroduce the network stall we just removed.
STATUS_CHIPS = ["All", "No plate", "Pending", "Analyzed", "Tracked"]
_status_chip = {"v": "All"}


def _passes_status(d) -> bool:
    s = _status_chip["v"]
    if s == "No plate":
        return not d.plate_dir or not d.frames
    if s == "Pending":
        return (d.strategy or "Pending") == "Pending"
    if s == "Analyzed":
        return (d.strategy or "Pending") != "Pending"
    if s == "Tracked":
        return bool(d.track_metrics_summary)
    return True


# -----------------------------------------------------------------------------
# Row building / table refresh
# -----------------------------------------------------------------------------
def _all_names():
    """Every shot passing the status chip. The search box is NOT applied here — it is
    a client-side Quasar filter, so filtered-out rows stay in table.rows and keep their
    tick. That is what lets a selection survive searching for the next batch."""
    return [n for n in sorted(state.shots_data) if _passes_status(state.shots_data[n])]


def _row_for(name: str) -> dict:
    d = state.shots_data[name]
    prompts = f"INC: {d.include_prompts} | EXC: {d.exclude_prompts}"
    if len(prompts) > 60:
        prompts = prompts[:57] + "..."
    return {
        "name": name,
        "version": d.version,
        "versions": d.versions or [],
        "strategy": d.strategy,
        "quality": be._quality_cell(d),
        "prompts": prompts,
        "scale": d.scale,
        "range": be._range_cell(d),
        "metrics": d.track_metrics_summary or "",
        "clear": "",
    }


# Fields Quasar's default filter scans: every declared column, `versions` excluded
# because it is row data with no column of its own.
_FILTER_FIELDS = [c["field"] for c in TABLE_COLS]


def _matching_names():
    """Names a user would see right now: status chip AND the search box. Built from the
    SAME row dicts the table holds, so it can't drift from Quasar's in-browser filter
    (substring over every column's cell value). Backs Select all / Invert."""
    q = (state.filter_query or "").strip().lower()
    names = _all_names()
    if not q:
        return names
    out = []
    for n in names:
        row = _row_for(n)
        if any(q in str(row[f]).lower() for f in _FILTER_FIELDS):
            out.append(n)
    return out


def _build_rows():
    return [_row_for(name) for name in _all_names()]


def refresh_table():
    """Rebuild rows + restore selection (active shots) without firing the select handler.
    Replaces the whole row array, so it resets scroll — call it only when the row SET
    changes (scan boundaries, status chip, version pick), never on a keystroke."""
    global _suppress_select
    rows = _build_rows()
    _suppress_select = True
    table.rows = rows
    table.selected = [r for r in rows if state.shots_data.get(r["name"]) and state.shots_data[r["name"]].use]
    table.update()
    _suppress_select = False
    lbl_selcount.set_content(be._sel_count_text(state))


def refresh_selection_only():
    """Push tick state without rebuilding rows — used by the bulk-select buttons so a
    'select all' on a 300-shot show doesn't re-render (and re-scroll) the table."""
    global _suppress_select
    _suppress_select = True
    table.selected = [r for r in table.rows
                      if state.shots_data.get(r["name"]) and state.shots_data[r["name"]].use]
    table.update()
    _suppress_select = False
    lbl_selcount.set_content(be._sel_count_text(state))


# -----------------------------------------------------------------------------
# Selection (active shots) + editing
# -----------------------------------------------------------------------------
def on_selection_change():
    """Reconcile ONLY the shots currently in the table. Shots excluded by the status
    chip are not represented in table.selected, so touching their `use` here would
    silently untick them (that was the old bug)."""
    if _suppress_select:
        return
    sel = {r["name"] for r in (table.selected or [])}
    for r in (table.rows or []):
        d = state.shots_data.get(r["name"])
        if d:
            d.use = r["name"] in sel
    lbl_selcount.set_content(be._sel_count_text(state))


def select_all_matching():
    names = _matching_names()
    for n in names:
        state.shots_data[n].use = True
    refresh_selection_only()
    ui.notify(f"Selected {len(names)} shot(s)", type="positive")


def select_none():
    for d in state.shots_data.values():
        d.use = False
    refresh_selection_only()


def select_invert():
    for n in _matching_names():
        d = state.shots_data[n]
        d.use = not d.use
    refresh_selection_only()


_PASTE_SPLIT = re.compile(r"[\s,;]+")


def apply_pasted_list(text: str):
    """Tick every shot named in a pasted production list. Names are matched
    case-insensitively and also by basename, so a pasted path or 'show/shot' still
    lands. Anything unmatched is reported rather than silently dropped."""
    wanted = [t.strip() for t in _PASTE_SPLIT.split(text or "") if t.strip()]
    if not wanted:
        ui.notify("Nothing to match — paste some shot names first.", type="warning")
        return
    by_key = {}
    for n in state.shots_data:
        by_key.setdefault(n.lower(), n)
    hit, missed = [], []
    for w in wanted:
        key = w.lower()
        name = by_key.get(key) or by_key.get(key.replace("\\", "/").rstrip("/").split("/")[-1])
        if name:
            hit.append(name)
        else:
            missed.append(w)
    for n in hit:
        state.shots_data[n].use = True
    refresh_selection_only()
    # A pasted shot the status chip is hiding still gets ticked (it will run), but the
    # user would only see the count move — say so rather than let it look like a bug.
    hidden = sum(1 for n in hit if not _passes_status(state.shots_data[n]))
    notes = []
    if missed:
        shown = ", ".join(missed[:8]) + (f" … (+{len(missed) - 8} more)" if len(missed) > 8 else "")
        notes.append(f"{len(missed)} not in this show: {shown}")
    if hidden:
        notes.append(f"{hidden} hidden by the '{_status_chip['v']}' filter")
    if notes:
        ui.notify(f"Ticked {len(hit)} · " + " · ".join(notes),
                  type="warning", multi_line=True, timeout=10000)
    else:
        ui.notify(f"Ticked {len(hit)} shot(s) — all matched", type="positive")


def on_status_chip(e):
    _status_chip["v"] = e.value or "All"
    refresh_table()


def load_editor(name: str):
    if not name or name not in state.shots_data:
        return
    d = state.shots_data[name]
    state.current_shot_name = name
    total = int(getattr(d, "frames", 0) or 0)
    ps = int(getattr(d, "plate_start", 0) or 0)
    pe = int(getattr(d, "plate_end", 0) or 0)
    fs_pos = int(getattr(d, "frame_start", 0) or 0)
    fe_pos = int(getattr(d, "frame_end", 0) or 0)
    if ps > 0 and pe >= ps:
        # Studio plate: user enters ABSOLUTE frame numbers (e.g. 1100-1200).
        avail = f" · {ps}-{pe} ({total}f)"
        ed_fstart.set_value(ps + fs_pos - 1 if fs_pos > 0 else ps)
        ed_fend.set_value(ps + fe_pos - 1 if fe_pos > 0 else pe)
        ed_fstart.props(f"min={ps} max={pe}")
        ed_fend.props(f"min={ps} max={pe}")
        ed_frange_hint.set_text(f"Absolute frame numbers ({ps}-{pe}). Leave at the ends for the full clip.")
    else:
        avail = f" · {total} frames (1-{total})" if total > 0 else ""
        ed_fstart.set_value(fs_pos)
        ed_fend.set_value(fe_pos)
        if total > 0:
            ed_fstart.props(f"min=0 max={total}")
            ed_fend.props(f"min=0 max={total}")
        ed_frange_hint.set_text("Frame positions (1 = first). 0 = full clip.")
    ed_title.set_content(f"#### ✏️ {name}{avail}")
    ed_meta.set_content(be._shot_meta_md(d))
    ed_scale.set_value(d.scale)
    ed_render.set_value(getattr(d, "render_path", "") or "")
    ed_req.set_value(state.manual_notes.get(name, ""))
    ed_inc.set_value(d.include_prompts)
    ed_exc.set_value(d.exclude_prompts)
    ed_things.set_options(list(d.detected_things or []), value=[])
    ed_analysis.set_content(be._analysis_markdown(d))
    editor.open()


def on_row_click(e):
    # e.args = [event, row, index]
    try:
        row = e.args[1]
        name = row.get("name") if isinstance(row, dict) else None
    except Exception:
        name = None
    if name:
        ed_pick.set_value(name)  # keeps dropdown in sync; its change loads the editor
        load_editor(name)


def _reset_shot_analysis(d):
    """Wipe a shot's analysis/scope in memory (keep identity + tracking metrics)."""
    d.strategy = "Pending"
    d.include_prompts = ""
    d.exclude_prompts = ""
    d.detected_things = []
    d.frame_start = 0
    d.frame_end = 0
    d.moving_things = []
    d.bad_track_regions = []
    d.foreground_occluders = []
    d.quality_flags = []
    d.depth_layers = {"fg": "", "mg": "", "bg": ""}
    d.parallax = ""
    d.notes = ""


def on_clear_mem(row):
    """Per-shot 'clear memory': confirm, then drop the shot's stored analysis (guide entry +
    manual note) and reset it in the table."""
    if isinstance(row, list):
        row = row[0] if row else None
    name = (row or {}).get("name") if isinstance(row, dict) else None
    if not name or name not in state.shots_data:
        return
    dlg = ui.dialog()
    with dlg, ui.card():
        ui.markdown(f"**Clear analysis memory for `{name}`?**")
        ui.label("Removes its scope, prompts and Qwen analysis (guide entry + manual brief). "
                 "Tracking metrics and the shot itself stay. This cannot be undone.")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat")

            def _do():
                try:
                    be.clear_shot_memory(out_dir.value, name)
                except Exception as ex:
                    print(f"clear_shot_memory: {ex}")
                _reset_shot_analysis(state.shots_data[name])
                if isinstance(getattr(state, "manual_notes", None), dict):
                    state.manual_notes.pop(name, None)
                refresh_table()
                dlg.close()
                ui.notify(f"Cleared memory for {name}", type="warning")

            ui.button("Clear memory", on_click=_do, color="negative")
    dlg.open()


def on_pick_edit(e):
    name = e.value if hasattr(e, "value") else None
    if name:
        load_editor(name)


def add_things_to(target_input):
    chosen = ed_things.value or []
    if not chosen:
        return
    cur = [x.strip() for x in (target_input.value or "").split(",") if x.strip()]
    for c in chosen:
        if c not in cur:
            cur.append(c)
    target_input.set_value(",".join(cur))


def save_shot():
    name = state.current_shot_name
    if not name or name not in state.shots_data:
        ui.notify("No shot selected", type="warning")
        return
    d = state.shots_data[name]
    d.include_prompts = ed_inc.value or ""
    d.exclude_prompts = ed_exc.value or ""
    d.scale = ed_scale.value or "100%"
    total = int(getattr(d, "frames", 0) or 0)
    ps = int(getattr(d, "plate_start", 0) or 0)
    pe = int(getattr(d, "plate_end", 0) or 0)
    try: a = max(0, int(ed_fstart.value or 0))
    except Exception: a = 0
    try: b = max(0, int(ed_fend.value or 0))
    except Exception: b = 0
    if ps > 0 and pe >= ps:
        # Inputs are ABSOLUTE frame numbers -> convert to 1-based positions the
        # downstream stages expect. Clamp to the plate range; the ends mean "full".
        a = min(max(a, ps), pe) if a > 0 else 0
        b = min(max(b, ps), pe) if b > 0 else 0
        fs = 0 if (a == 0 or a <= ps) else (a - ps + 1)
        fe = 0 if (b == 0 or b >= pe) else (b - ps + 1)
    else:
        fs = min(a, total) if total > 0 else a
        fe = min(b, total) if total > 0 else b
    if fs and fe and fs > fe:
        fs, fe = fe, fs
    d.frame_start, d.frame_end = fs, fe
    d.render_path = (ed_render.value or "").strip()
    # Persist the manual client requirement so it survives restarts (reloaded on Scan).
    note = (ed_req.value or "").strip()
    if note:
        state.manual_notes[name] = note
    else:
        state.manual_notes.pop(name, None)
    be.save_manual_notes(out_dir.value, state.manual_notes)
    refresh_table()
    ui.notify(f"Saved {name}", type="positive")


async def clear_cache_current():
    """Delete the currently-edited shot's bot cache (JPG proxies + mp4 renders)."""
    name = state.current_shot_name
    d = state.shots_data.get(name) if name else None
    if not d:
        ui.notify("No shot selected", type="warning")
        return
    n = await run.io_bound(be.clear_shot_cache, getattr(d, "studio_dir", ""), out_dir.value, name)
    ui.notify(f"Cleared {n} cache file(s) for {name}" if n else f"No cache for {name}",
              type="positive" if n else "info")


# -----------------------------------------------------------------------------
# Folder / file pickers (tkinter on the local machine, off the event loop)
# -----------------------------------------------------------------------------
async def pick_folder(target_input):
    p = await run.io_bound(be._tk_pick_folder, target_input.value or "")
    if p:
        target_input.set_value(p)


async def pick_file(target_input):
    p = await run.io_bound(be._tk_pick_file, target_input.value or "")
    if p:
        target_input.set_value(p)


# -----------------------------------------------------------------------------
# Run steps
# -----------------------------------------------------------------------------
def do_scan():
    rows = be.list_shots(in_dir.value)
    state.shots_data = {}
    state.log_history = []
    for s in rows:
        w = h = frames = 0
        if be.probe_video_meta:
            pdir = Path(in_dir.value)
            fpath = next((f for f in pdir.glob(f"{s}.*") if f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}), None)
            if fpath:
                meta = be.probe_video_meta(str(fpath))
                w, h = int(meta.get("width", 0)), int(meta.get("height", 0))
                frames = int(meta.get("total_frames", 0))
        state.shots_data[s] = be.ShotData(name=s, res=f"{w}x{h}", width=w, height=h, frames=frames,
                                          scale="100%", vram=be.est_vram(w, h, frames))
    # Restore manually-fed client requirements saved in a previous session.
    state.manual_notes = be.load_manual_notes(out_dir.value)
    # Load any previous analysis guide (same behavior as the Gradio scan)
    try:
        _scan_load_prev_guide()
    except Exception as ex:
        print(f"prev guide load: {ex}")
    ed_pick.set_options(sorted(state.shots_data))
    refresh_table()
    ui.notify(f"Found {len(rows)} shots", type="info")


async def load_shows():
    """Scan the shows root for show folders and fill the show dropdown. Runs the
    (possibly slow UNC) scan off the event loop."""
    root = shows_root.value
    shows = await run.io_bound(be.list_shows, root)
    show_sel.set_options(shows)
    ui.notify(f"{len(shows)} show(s) under {root}" if shows else f"No shows under {root}",
              type="info" if shows else "warning")


_scan_token = {"n": 0}

async def do_scan_show():
    """Studio flow: list shots under the picked show, then resolve each shot's plate
    versions + frame range. Two-phase so the shot list appears immediately on a show
    switch; the heavier per-shot frame walk fills the range afterwards. Resilient: a
    single unreadable shot never aborts the list, and the table always refreshes."""
    show = show_sel.value
    # New scan supersedes any in-flight one (fast show switching).
    _scan_token["n"] += 1
    token = _scan_token["n"]
    # Clear immediately so the switch is visible even before the network responds.
    state.shots_data = {}
    state.log_history = []
    refresh_table()
    if not (shows_root.value and show):
        return
    # Pipeline runs in a hidden local cache; results publish to the studio tree per shot.
    out_dir.set_value(be.work_dir_for_show(show))
    # Show scan progress so it's obvious the bot is working (not idle).
    scan_prog.set_visibility(True); scan_lbl.set_visibility(True)
    scan_prog.props("indeterminate"); scan_lbl.set_text(f"Scanning {show}…")
    try:
        shots = await run.io_bound(be.list_shots_for_show, shows_root.value, show)
    except Exception as ex:
        scan_prog.set_visibility(False); scan_lbl.set_visibility(False)
        ui.notify(f"Scan failed for {show}: {ex}", type="negative")
        return
    if token != _scan_token["n"]:
        return  # a newer show was picked while we were listing (it owns the bar now)
    # Phase 1 (cheap): show names + version dropdowns right away. One thread-pooled
    # batch instead of one UNC round-trip per shot — on a 100+ shot show that was the
    # main reason the list took so long to appear.
    scan_lbl.set_text(f"Reading plate versions · {len(shots)} shot(s)…")
    vers_per_shot = await run.io_bound(be.list_versions_batch, shows_root.value, show, shots)
    if token != _scan_token["n"]:
        return
    for s, vers in zip(shots, vers_per_shot):
        latest = vers[-1] if vers else ""
        state.shots_data[s] = be.ShotData(
            name=s, scale="100%", show=show, versions=vers, version=latest,
            plate_dir=be.resolve_plate_dir(shows_root.value, show, s, latest) if latest else "",
            studio_dir=be.shot_bot_tracks_dir(shows_root.value, show, s))
    state.manual_notes = be.load_manual_notes(out_dir.value)
    try:
        _scan_load_prev_guide()
    except Exception as ex:
        print(f"prev guide load: {ex}")
    ed_pick.set_options(sorted(state.shots_data))
    refresh_table()
    ui.notify(f"{show}: {len(shots)} shot(s)", type="info")
    # Phase 2 (heavier): descend to the real frames dir + parse the range per shot.
    # Chunked so progress still moves, but with ~1 websocket push per chunk instead of
    # two per shot (a 100-shot scan used to fire ~200 UI updates at the browser).
    todo = [s for s in shots if state.shots_data.get(s) and state.shots_data[s].plate_dir]
    total = len(todo)
    scan_prog.props(remove="indeterminate"); scan_prog.set_value(0.0)
    CHUNK = 16
    for i in range(0, total, CHUNK):
        if token != _scan_token["n"]:
            return  # superseded by a newer show switch (it owns the bar now)
        chunk = todo[i:i + CHUNK]
        scan_lbl.set_text(f"Resolving frames {min(i + CHUNK, total)}/{total}")
        scan_prog.set_value(min(i + CHUNK, total) / total if total else 1.0)
        results = await run.io_bound(be.resolve_frames_batch,
                                     [state.shots_data[s].plate_dir for s in chunk])
        for s, (pdir, frames, pstart, pend) in zip(chunk, results):
            d = state.shots_data.get(s)
            if d:
                d.plate_dir, d.frames, d.plate_start, d.plate_end = pdir, frames, pstart, pend
    if token == _scan_token["n"]:
        refresh_table()
        scan_prog.set_visibility(False); scan_lbl.set_visibility(False)


async def on_pick_version(args):
    """A shot row's version dropdown changed — re-resolve that shot's plate dir.
    The walk is a UNC os.walk, so it runs off the event loop; doing it inline used to
    freeze the whole UI for as long as the share took to answer."""
    if isinstance(args, list):
        args = args[0] if args else None
    name = (args or {}).get("name") if isinstance(args, dict) else None
    ver = (args or {}).get("version") if isinstance(args, dict) else None
    d = state.shots_data.get(name)
    if not d or not ver:
        return
    d.version = ver
    vdir = be.resolve_plate_dir(shows_root.value, d.show, name, ver)
    d.plate_dir, d.frames, d.plate_start, d.plate_end = await run.io_bound(be.find_frames_subdir, vdir)
    refresh_table()
    ui.notify(f"{name} → {ver} · {d.frames}f", type="info")


def _scan_load_prev_guide():
    out = out_dir.value
    if not out or not os.path.exists(out):
        return
    broot = Path(out) / "_batches"
    if not broot.exists():
        return
    bdirs = sorted([d for d in broot.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    if not bdirs:
        return
    # Newest guide across batches; frame range is saved at Mask time.
    cands = []
    for d in bdirs:
        for fn in ("mask_guidance.json", "overdrive_guide.json"):
            g = d / fn
            if g.exists(): cands.append(g)
    if not cands:
        return
    guide = max(cands, key=os.path.getmtime)
    import json
    with open(guide, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s_item in data.get("shots", []):
        nm = s_item.get("shot_name") or s_item.get("shot") or s_item.get("name")
        if nm and nm in state.shots_data:
            d = state.shots_data[nm]
            d.strategy = be._derive_strategy(s_item)
            d.include_prompts = be._extract_prompt_list(s_item, ["mask_includes", "include_prompts", "sam3_include_prompt"])
            d.exclude_prompts = be._extract_prompt_list(s_item, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
            # Restore previously saved frame range (0 = full), if present.
            try:
                fs = int(s_item.get("frame_start", 0) or 0)
                fe = int(s_item.get("frame_end", 0) or 0)
                if fs or fe:
                    d.frame_start, d.frame_end = fs, fe
            except Exception:
                pass
            rt = s_item.get("qwen2_things", [])
            if isinstance(rt, list): d.detected_things = rt
            be._apply_analysis_fields(d, s_item)
    state.guide_path = str(guide)


def _launch_analyze():
    msg = be.run_step_thread(be.worker_analyze,
                             (in_dir.value, out_dir.value, req_file.value, int(qwen_fps.value), be.DEFAULT_OLLAMA_URL, state),
                             "Analyze")
    ui.notify(msg)


def _norm_shot(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).strip()).lower()


def _prompt_missing_requirements(missing, on_done):
    """Ask the user for a client requirement for each missing shot, one by one,
    in a text field. Notes saved to <OUT>/manual_requirements.json for reuse.
    on_done() runs once every shot has been answered or skipped — it is what
    actually launches the job, so both Analyze and Run Pipeline can share this."""
    idx = {"i": 0}
    dlg = ui.dialog().props("persistent")
    with dlg, ui.card().classes("w-96"):
        title = ui.label().classes("text-bold")
        sub = ui.label().classes("text-caption text-orange")
        field = ui.textarea(
            placeholder="e.g. track camera, exclude the running person"
        ).classes("w-full")
        with ui.row().classes("w-full justify-end"):
            skip_btn = ui.button("Skip").props("flat")
            next_btn = ui.button("Save & Next").props("color=primary")

    def render():
        i = idx["i"]
        shot = missing[i]
        title.text = f"Requirement for: {shot}"
        sub.text = f"Shot {i+1} of {len(missing)} missing a client requirement"
        field.value = state.manual_notes.get(shot, "")
        next_btn.text = "Save & Finish" if i == len(missing) - 1 else "Save & Next"

    def advance(save):
        shot = missing[idx["i"]]
        if save:
            note = (field.value or "").strip()
            if note:
                state.manual_notes[shot] = note
        idx["i"] += 1
        if idx["i"] >= len(missing):
            be.save_manual_notes(out_dir.value, state.manual_notes)
            dlg.close()
            on_done()
        else:
            render()

    skip_btn.on_click(lambda: advance(False))
    next_btn.on_click(lambda: advance(True))
    render()
    dlg.open()


def _gate_client_scope(shots, on_ready):
    """Make sure every selected shot has a client requirement, then run on_ready().

    Analysis is only as good as the client scope it is given: with no requirement the
    guide comes back with nothing usable, masking falls through to 'no_mask_needed' and
    the run is wasted. Both Analyze and Run Pipeline go through here so the prompt can
    never be skipped by taking the other route.
    """
    # Reuse any notes saved in a previous session (saved for later use).
    saved = be.load_manual_notes(out_dir.value)
    state.manual_notes = dict(saved)
    have = be.requirement_shot_names(req_file.value)
    have |= {_norm_shot(k) for k, v in saved.items() if str(v).strip()}
    missing = [s for s in shots if _norm_shot(s) not in have]
    if not missing:
        on_ready()
        return
    ui.notify(f"{len(missing)} shot(s) missing a client requirement — enter them.",
              type="warning")
    _prompt_missing_requirements(missing, on_ready)


def start_analyze():
    shots = [n for n, d in state.shots_data.items() if getattr(d, "use", False)]
    if not shots:
        ui.notify("Select at least one shot (tick 'Use'), then Analyze.", type="warning")
        return
    _gate_client_scope(shots, _launch_analyze)


def _launch_mask():
    msg = be.run_step_thread(be.worker_mask,
                             (in_dir.value, out_dir.value, be.DEFAULT_SAM3_WEIGHTS, state),
                             "Generate masks")
    ui.notify(msg)


def start_mask():
    # If selected shots already have masks in OUT, ask: reuse or regenerate.
    existing = be.shots_with_existing_masks(out_dir.value, state)
    if not existing:
        state.reuse_existing_masks = True
        _launch_mask()
        return

    dlg = ui.dialog()
    with dlg, ui.card():
        ui.label(f"Masks already exist for {len(existing)} selected shot(s):").classes("text-bold")
        ui.label(", ".join(sorted(existing))).classes("text-caption")
        ui.label("Reuse them (skip), or regenerate (overwrite)?")
        with ui.row().classes("w-full justify-end"):
            def _reuse():
                state.reuse_existing_masks = True
                dlg.close(); _launch_mask()
            def _regen():
                state.reuse_existing_masks = False
                dlg.close(); _launch_mask()
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Regenerate", on_click=_regen).props("outline color=negative")
            ui.button("Reuse existing", on_click=_reuse).props("color=primary")
    dlg.open()


def _launch_track():
    msg = be.run_step_thread(be.worker_track,
                             (in_dir.value, out_dir.value, int(grid_size.value), int(seed_count.value), int(seed_min_dist.value), state),
                             "Tracking")
    ui.notify(msg)


def start_track():
    # Guard: shots with no masks track the FULL PLATE (no mover exclusion) — warn first.
    # Shots that DO have masks still use them, so a mixed selection is fine to continue.
    missing = be.shots_missing_masks(out_dir.value, state)
    if not missing:
        _launch_track()
        return

    unanalyzed = set(be.shots_missing_analysis(out_dir.value, state))
    never = sorted(s for s in missing if s in unanalyzed)
    masked_ok = [n for n, d in state.shots_data.items()
                 if getattr(d, "use", False) and n not in missing]

    dlg = ui.dialog()
    with dlg, ui.card():
        ui.label(f"No masks for {len(missing)} selected shot(s):").classes("text-bold")
        ui.label(", ".join(sorted(missing))).classes("text-caption")
        if never:
            ui.label(f"Never analyzed: {', '.join(never)}").classes("text-caption text-orange")
        ui.label("These will be tracked on the FULL PLATE with no masking — moving objects "
                 "(people, vehicles) get tracked too, which can wreck a camera solve.")
        if masked_ok:
            ui.label(f"Shots that do have masks still use them: {', '.join(sorted(masked_ok))}"
                     ).classes("text-caption")
        ui.label("Run Analyze + Generate masks first, or continue anyway?").classes("text-caption")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")

            def _go():
                dlg.close(); _launch_track()

            ui.button("Track full plate anyway", on_click=_go).props("color=negative")
    dlg.open()


def _launch_pipeline():
    msg = be.run_step_thread(
        be.worker_pipeline,
        (in_dir.value, out_dir.value, req_file.value, int(qwen_fps.value), be.DEFAULT_OLLAMA_URL,
         be.DEFAULT_SAM3_WEIGHTS, int(grid_size.value), int(seed_count.value),
         int(seed_min_dist.value), state),
        "Run Pipeline")
    ui.notify(msg)


def start_pipeline():
    sel = [n for n, d in state.shots_data.items() if getattr(d, "use", False)]
    if not sel:
        ui.notify("Tick at least one shot first.", type="warning")
        return
    dlg = ui.dialog()
    with dlg, ui.card():
        ui.label(f"Run the full pipeline on {len(sel)} shot(s)?").classes("text-bold")
        ui.label(", ".join(sorted(sel))).classes("text-caption")
        ui.label("Analyze (Qwen) → Generate masks (SAM3) → Track, back-to-back. "
                 "Shots that already have masks reuse them. Stop halts between stages.")
        ui.label("Shots with no client requirement are asked for one first — the same "
                 "prompt Analyze shows.").classes("text-caption")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")

            def _go():
                # Same client-scope gate the Analyze button uses. Without it the pipeline
                # ran Qwen with zero requirements, so the guide came back empty, masking
                # fell through to 'no_mask_needed' and the chain died before Track ever
                # published anything to the shot's bot_tracks folder.
                dlg.close(); _gate_client_scope(sel, _launch_pipeline)

            ui.button("Run pipeline", on_click=_go, color="primary")
    dlg.open()


def stop_job():
    ui.notify(be.on_stop_job())


def shutdown_bot():
    """Confirm, then stop any running job and shut the server down."""
    dlg = ui.dialog()
    with dlg, ui.card():
        ui.label("Shut down Batch Tracker?").classes("text-bold")
        ui.label("Stops the server. The browser tab will go dead.").classes("text-caption")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            def _do():
                dlg.close()
                ui.notify("Shutting down…", type="warning")
                try:
                    be.on_stop_job()
                except Exception:
                    pass
                # Defer so the notify/close can flush before the server dies.
                ui.timer(0.5, lambda: nicegui_app.shutdown(), once=True)
            ui.button("Shut down", on_click=_do).props("color=negative")
    dlg.open()


# -----------------------------------------------------------------------------
# Poll background job queue (logs, progress, table refresh, button states)
# -----------------------------------------------------------------------------
# Last values poll() actually pushed, so an idle tick sends nothing.
_poll_last = {"status": None, "running": None, "indeterminate": False}


def poll():
    new_logs = []
    refresh_now = False
    refresh_path = None
    while True:
        try:
            m = be.JOB_QUEUE.get_nowait()
        except queue.Empty:
            break
        if m.startswith("GUIDE_PATH_UPDATE:"):
            refresh_path = m.split(":", 1)[1].strip()
            new_logs.append(f"System: reloading {Path(refresh_path).name}")
        elif m in ("DONE_ANALYSIS",):
            pass
        elif m in ("DONE_MASKING", "DONE_TRACKING"):
            refresh_now = True
            new_logs.append(m)
        else:
            new_logs.append(m)

    for ln in new_logs:
        log_view.push(ln)

    if refresh_path and os.path.exists(refresh_path):
        try:
            # Same helper worker_analyze calls, so the table and a chained Mask can never
            # disagree about what the analysis said.
            state.guide_path = refresh_path
            be.sync_shots_from_guide(state, refresh_path)
            refresh_now = True
        except Exception as ex:
            print(f"guide reload: {ex}")

    if refresh_now:
        refresh_table()
        ed_pick.set_options(sorted(state.shots_data))

    # status + progress bar + elapsed + button busy-state
    running = be._job_running()
    if bar.visible != running:
        bar.visible = running
    if running:
        prog = f" · {be.LAST_PROGRESS}" if be.LAST_PROGRESS else ""
        start = getattr(be, "CURRENT_JOB_START", 0.0) or 0.0
        el = ""
        if start:
            secs = int(time.time() - start)
            el = f" · {secs // 60}:{secs % 60:02d}"
        frac = getattr(be, "LAST_PROGRESS_FRAC", None)
        if frac is None:
            # unknown total (e.g. Qwen) -> keep it moving rather than fake a number
            if not _poll_last["indeterminate"]:
                bar.props("indeterminate"); _poll_last["indeterminate"] = True
        else:
            if _poll_last["indeterminate"]:
                bar.props(remove="indeterminate"); _poll_last["indeterminate"] = False
            bar.value = float(frac)
            prog += f" · {int(frac * 100)}%"
        status_md = f"⏳ **{be.CURRENT_JOB_NAME}** running…{prog}{el}"
    else:
        status_md = "🟢 Idle"
    # Only push what actually changed — this runs at 1 Hz forever, and blindly
    # rewriting the status + five button states kept a websocket update going out
    # every second even while the bot sat idle.
    if status_md != _poll_last["status"]:
        lbl_status.set_content(status_md)
        _poll_last["status"] = status_md
    if running != _poll_last["running"]:
        for b in (btn_pipe, btn_analyze, btn_mask, btn_track):
            b.set_enabled(not running)
        btn_stop.set_enabled(running)
        _poll_last["running"] = running


def on_search(e):
    """Client-side filter: Quasar hides non-matching rows in the browser, so the row
    array (and every tick in it) is untouched. No table rebuild, no scroll reset, and a
    selection built under one search survives the next one."""
    state.filter_query = e.value or ""   # kept so 'Select all matching' agrees with the view
    table.filter = state.filter_query


# -----------------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------------
ui.dark_mode(True)
ui.colors(primary="#4c8dff", secondary="#2dd4bf", accent="#a78bfa",
          positive="#22c55e", negative="#f43f5e", warning="#f59e0b", info="#38bdf8",
          dark="#12151c", dark_page="#0d1017")

ui.add_head_html("""
<style>
  body { background: #0d1017; }
  .bt-header  { background: linear-gradient(90deg,#141a26 0%,#101623 60%,#0d1017 100%);
                border-bottom: 1px solid rgba(255,255,255,.07); }
  .bt-rail    { background: #10141d; border-right: 1px solid rgba(255,255,255,.07); }
  .bt-card    { background: #141924; border: 1px solid rgba(255,255,255,.07);
                border-radius: 10px; box-shadow: none; }
  .bt-card > .q-card__section { padding: 12px 14px; }
  .bt-section { font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
                color: #8a94a6; font-weight: 600; }
  .bt-status  { font-size: 13px; color: #cbd5e1; }
  .bt-status p { margin: 0; }
  .bt-hint    { font-size: 12px; color: #7c8698; }
  .bt-chip    { background: rgba(76,141,255,.14); color: #9dc0ff; border-radius: 999px;
                padding: 2px 10px; font-size: 12px; }
  .bt-chip p  { margin: 0; }
  .bt-editor  { width: 780px; max-width: 94vw; background: #141924; }
  .bt-paste   { width: 520px; max-width: 94vw; background: #141924; }
  /* Run bar: pipeline button, then the four stages as one evenly-sized sequence. */
  .bt-run     { min-width: 150px; height: 36px; font-weight: 600; }
  .bt-step    { min-width: 122px; height: 36px; color: #9dc0ff; }
  .bt-vsep    { height: 24px; margin: 0 6px; background: rgba(255,255,255,.12); }
  /* Shots table: virtual-scrolled so a 300-shot show puts ~20 rows in the DOM, not 300.
     Quasar needs the scroll container height-bounded, and the header pinned over it. */
  .bt-shots .q-table__middle { max-height: 58vh; }
  .bt-shots thead tr th { position: sticky; top: 0; z-index: 2; background: #141924; }
</style>
""")

# ---------- HEADER: identity, live status, kill switches ----------
with ui.header().classes("bt-header items-center px-4 py-2"):
    with ui.row().classes("w-full items-center justify-between no-wrap"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.button(icon="menu", on_click=lambda: rail.toggle()).props("flat dense round color=white")
            ui.icon("track_changes").classes("text-2xl text-blue-4")
            ui.label("Batch Tracker").classes("text-h6 text-weight-bold")
        with ui.row().classes("items-center gap-3 no-wrap"):
            lbl_status = ui.markdown("🟢 Idle").classes("bt-status")
            btn_stop = ui.button("Stop", icon="stop", on_click=stop_job).props("flat dense color=white")
            btn_stop.set_enabled(False)
            ui.button(icon="power_settings_new", on_click=shutdown_bot
                      ).props("flat dense round color=red-4").tooltip("Shut down Batch Tracker")

# ---------- LEFT RAIL: project paths + settings + find ----------
with ui.left_drawer(value=True, fixed=False).props("width=340 bordered").classes("bt-rail p-3 gap-3") as rail:
    with ui.card().classes("w-full bt-card"):
        ui.label("Project").classes("bt-section")
        # ---- Studio network plate fetch: <shows_root>/<show>/<shot>/in/plates/<version> ----
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            shows_root = ui.input("Shows Root", value=r"\\liv1\shows",
                                  placeholder=r"\\liv1\shows").props("dense outlined").classes("grow")
            ui.button(icon="folder", on_click=lambda: pick_folder(shows_root)).props("flat dense")
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            show_sel = ui.select([], label="Show", with_input=True,
                                 on_change=do_scan_show).props("dense outlined").classes("grow")
            ui.button(icon="refresh", on_click=load_shows).props("flat dense").tooltip("Load shows")
        # Scan progress (driven only by do_scan_show; poll() never touches it).
        scan_prog = ui.linear_progress(value=0, show_value=False).props("rounded").classes("w-full")
        scan_lbl = ui.label("").classes("bt-hint")
        scan_prog.set_visibility(False)
        scan_lbl.set_visibility(False)
        ui.separator().classes("q-my-xs")
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            in_dir = ui.input("Input Folder (legacy / fallback)", placeholder=r"D:\shots\IN"
                              ).props("dense outlined").classes("grow")
            ui.button(icon="folder", on_click=lambda: pick_folder(in_dir)).props("flat dense")
        # Output is no longer user-set: the pipeline runs in a hidden local cache and
        # publishes each shot's results to <show>/<shot>/mid/cmm/bot_tracks. Kept as a
        # hidden holder so the existing out_dir.value plumbing is unchanged.
        out_dir = ui.input("Output Folder").props("dense outlined").classes("grow")
        out_dir.set_visibility(False)
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            req_file = ui.input("Client Requirements (optional)", placeholder=r"D:\shots\reqs.xlsx"
                                ).props("dense outlined").classes("grow")
            ui.button(icon="description", on_click=lambda: pick_file(req_file)).props("flat dense")

    with ui.card().classes("w-full bt-card"):
        ui.label("Find shots").classes("bt-section")
        # Filters live in the browser (Quasar), so typing never touches the server or
        # the ticks. Matches any column: shot name, version, strategy, range, metrics.
        ui.input(placeholder="shot name, version, strategy…"
                 ).props("dense outlined clearable").classes("w-full").on_value_change(on_search)
        ui.label("Matches any column · ticks survive searching").classes("bt-hint")

    with ui.card().classes("w-full bt-card"):
        with ui.expansion("Settings", icon="tune", value=False).classes("w-full"):
            # ---- Universal (both backends / AI pipeline) ----
            ui.label("Analysis & seeding").classes("bt-section q-mt-sm")
            ui.label("Qwen2 Sample Density")
            qwen_fps = ui.slider(min=1, max=8, value=4).props("label-always")
            ui.label("Seed Count (Max Tracks)")
            seed_count = ui.slider(min=100, max=3000, value=1200, step=50).props("label-always")
            seed_count.tooltip(
                "Always used by TAPNext++. On SynthEyes it only applies when the SynthEyes "
                "preset below is set to 'Custom' — every other preset has its own fixed count "
                "(Locked 100 / Slow 500 / Normal 800 / Fast 2000) and ignores this slider. "
                "The tracking log prints the count actually used and where it came from.")

            sw_motion = ui.switch("CV motion backstop", value=True,
                                  on_change=lambda e: setattr(state, "motion_backstop", bool(e.value)))
            sw_motion.tooltip("Masks objects moving independently of the camera (even ones NOT in your Exclude list). "
                              "Turn OFF if it masks things you removed from Exclude.")

            ui.label("Mask edge dilation (px · applies at Mask time)")
            mask_dilate = ui.slider(min=0, max=30, value=int(getattr(state, "mask_dilation_px", 10)), step=1).props("label-always")
            mask_dilate.on_value_change(lambda e: setattr(state, "mask_dilation_px", int(e.value or 0)))
            mask_dilate.tooltip("Grows the excluded (mover) region when masks are GENERATED, so trackers stay off soft "
                                "edges — hair, motion-blur fringe. Mirrors SynthEyes Mask ML's Mask Dilation and helps "
                                "both backends. Only affects masks made from now on: re-run Generate masks to apply it "
                                "to existing shots (or use the track-time margin in TAPNext++ settings instead).")

            ui.separator().classes("q-my-sm")
            ui.label("Tracking backend").classes("bt-section")
            backend_sel = ui.select(["syntheyes", "tapnext"], value=state.track_backend,
                                    on_change=lambda e: setattr(state, "track_backend", e.value or "syntheyes")
                                    ).props("dense outlined").classes("w-full")
            backend_sel.tooltip("SynthEyes = drive SynthEyes over SyPy3 (default). TAPNext++ = Apache-2.0 GPU tracker fallback. "
                                "The settings below switch to match the selected backend.")

            # ---- SynthEyes-only (shown when backend = syntheyes) ----
            with ui.column().classes("w-full gap-2") as syn_box:
                ui.label("SynthEyes settings").classes("bt-section q-mt-sm")
                with ui.row().classes("w-full no-wrap items-end gap-1"):
                    se_exe = ui.input("SynthEyes .exe", value=state.syntheyes_exe,
                                      placeholder=r"C:\Program Files\Andersson Technologies LLC\SynthEyes\SynthEyes64.exe"
                                      ).props("dense outlined").classes("grow")
                    ui.button(icon="folder", on_click=lambda: pick_file(se_exe)).props("flat dense")
                se_exe.on_value_change(lambda e: setattr(state, "syntheyes_exe", e.value or ""))

                se_preset = ui.select(be.SE_PRESET_NAMES or ["Normal / Handheld"], value=state.track_preset,
                                      label="SynthEyes preset (max tracks)",
                                      on_change=lambda e: setattr(state, "track_preset", e.value or "Normal / Handheld")
                                      ).props("dense outlined").classes("w-full")
                se_preset.tooltip("Locked 100 / Slow 500 / Normal 800 / Fast 2000. Custom = use Seed Count slider above.")

                sw_matte = ui.switch("Use SAM3 masks as matte", value=state.use_sam3_matte,
                                     on_change=lambda e: setattr(state, "use_sam3_matte", bool(e.value)))
                sw_matte.tooltip("Feed SAM3 per-frame masks into SynthEyes so trackers avoid masked regions. Off = track full frame.")

                sw_3de = ui.switch("Auto-create .3de project", value=state.auto_3de,
                                   on_change=lambda e: setattr(state, "auto_3de", bool(e.value)))
                sw_3de.tooltip("After export, build a 3DEqualizer .3de project from the 2D tracks. Needs the 3DE4 exe below.")
                with ui.row().classes("w-full no-wrap items-end gap-1"):
                    tde_exe = ui.input("3DEqualizer4 .exe (for auto .3de)", value=state.tde4_exe,
                                       placeholder=r"C:\Program Files\3DE4\bin\3DE4.exe"
                                       ).props("dense outlined").classes("grow")
                    ui.button(icon="folder", on_click=lambda: pick_file(tde_exe)).props("flat dense")
                tde_exe.on_value_change(lambda e: setattr(state, "tde4_exe", e.value or ""))
            syn_box.bind_visibility_from(backend_sel, "value", value="syntheyes")

            # ---- TAPNext++-only (shown when backend = tapnext) ----
            with ui.column().classes("w-full gap-2") as tap_box:
                ui.label("TAPNext++ settings").classes("bt-section q-mt-sm")
                ui.label("Grid Size")
                grid_size = ui.slider(min=4, max=20, value=10).props("label-always")
                ui.label("Min Seed Distance (px)")
                seed_min_dist = ui.slider(min=0, max=50, value=12).props("label-always")
                ui.label("Track chunks (0 = Auto)")
                track_chunks = ui.number(value=0, min=0, max=16, precision=0).props("dense outlined").classes("w-full")
                track_chunks.on_value_change(lambda e: setattr(state, "track_chunks", int(e.value or 0)))
                track_chunks.tooltip("Split long/high-res shots into N overlapping chunks to avoid GPU OOM "
                                     "(track IDs are chained across chunks). 0 = pick automatically from free VRAM.")

                ui.label("Track spacing (px · density dial)")
                track_spacing = ui.slider(min=10, max=120, value=int(getattr(state, "track_spacing_px", 40)), step=5).props("label-always")
                track_spacing.on_value_change(lambda e: setattr(state, "track_spacing_px", int(e.value or 40)))
                track_spacing.tooltip("Min pixel gap between kept tracks, measured ON SCREEN at several sampled frames "
                                      "(not on each track's average position), so tracks can't clump at the start or end "
                                      "of a moving shot. Small = denser/more tracks, large = sparser/fewer.")

                ui.label("Mask safety margin at track time (px)")
                mask_margin = ui.slider(min=0, max=40, value=int(getattr(state, "mask_margin_px", 8)), step=1).props("label-always")
                mask_margin.on_value_change(lambda e: setattr(state, "mask_margin_px", int(e.value or 0)))
                mask_margin.tooltip("Pulls seeding and mask gating IN from the matte edge by this many pixels. "
                                    "SAM3 mattes often stop just short of hair / motion-blur fringe, and a track left in "
                                    "that halo sticks to the character and slides. Applies to the masks you ALREADY have "
                                    "— no re-masking needed. 0 = use the matte exactly as-is.")

                ui.label("Reject edge-like tracks (0 = off)")
                aniso = ui.slider(min=0.0, max=0.4, value=float(getattr(state, "min_corner_anisotropy", 0.08)), step=0.01).props("label-always")
                aniso.on_value_change(lambda e: setattr(state, "min_corner_anisotropy", float(e.value or 0.0)))
                aniso.tooltip("Drops points sitting on a straight edge (rope, plate lines, TV-screen borders). Such a point "
                              "can only be pinned across the edge, never along it, so it slides left/right. Higher = stricter "
                              "(fewer, more corner-like tracks). Try 0.15 if sliding persists, 0.04 if too many tracks are lost.")
                ui.label("Max tracks per task (0 = auto)")
                track_max = ui.number(value=int(getattr(state, "track_max_output", 0)), min=0, max=500, precision=0
                                      ).props("dense outlined").classes("w-full")
                track_max.on_value_change(lambda e: setattr(state, "track_max_output", int(e.value or 0)))
                track_max.tooltip("Soft ceiling on exported tracks per shot/task after spread selection. 0 = unlimited.")

                ui.label("Bad-track filter (plate px · 0 = off)")
                with ui.row().classes("w-full no-wrap gap-2"):
                    flt_jump = ui.number("Max jump", value=int(getattr(state, "filter_max_jump_px", 0) or 0),
                                         min=0, precision=0).props("dense outlined").classes("grow")
                    flt_jitter = ui.number("Max jitter", value=int(getattr(state, "filter_max_jitter_px", 0) or 0),
                                           min=0, precision=0).props("dense outlined").classes("grow")
                flt_jump.on_value_change(lambda e: setattr(state, "filter_max_jump_px", float(e.value or 0)))
                flt_jitter.on_value_change(lambda e: setattr(state, "filter_max_jitter_px", float(e.value or 0)))
                flt_jump.tooltip("Drops a track if ANY single-frame jump exceeds this many plate pixels "
                                 "(teleport = mistrack). 0 = off. Try ~15 on a clean plate, tighten if needed.")
                flt_jitter.tooltip("Drops a track whose mean frame-to-frame jitter (change in velocity) exceeds "
                                   "this many plate pixels. Self-consistency only, so a fast SMOOTH point passes. 0 = off. Try ~3.")

                sw_movtile = ui.switch("Moving-tile native re-track (4K accuracy)", value=getattr(state, "moving_tile", True),
                                       on_change=lambda e: setattr(state, "moving_tile", bool(e.value)))
                sw_movtile.tooltip("Before the NCC lock, re-track each selected point inside a NATIVE 256px crop that "
                                   "follows it, so the model sees full-res pixels instead of the whole frame squashed to 256 (~15x on 4K). "
                                   "Fixes the coarse position NCC alone can't recover. Measured 4.03px -> 1.30px vs manual on a 4K plate.")
                sw_reseed = ui.switch("Re-seed tracks (fast/low-angle shots)", value=getattr(state, "reseed", True),
                                      on_change=lambda e: setattr(state, "reseed", bool(e.value)))
                sw_reseed.tooltip("Seed fresh features periodically across the clip (not just at frame 0) so the "
                                  "frame stays populated when the initial points sweep out on fast/low-angle shots. Re-seeded tracks "
                                  "pass through the same mover-gating + motion filter + spread selection, so movers/junk are still dropped.")
                ui.label("Re-seed interval (frames)")
                reseed_every = ui.slider(min=10, max=90, value=int(getattr(state, "reseed_every", 30)), step=5).props("label-always")
                reseed_every.on_value_change(lambda e: setattr(state, "reseed_every", int(e.value or 30)))
                reseed_every.tooltip("Max frames between re-seeds. Smaller = denser replenishment (better on very fast shots), more compute.")
                sw_edge = ui.switch("Track to frame edge", value=getattr(state, "edge_track", True),
                                    on_change=lambda e: setattr(state, "edge_track", bool(e.value)))
                sw_edge.tooltip("Keep refining a point right up to the frame border instead of trimming it when "
                                "the NCC search box / native tile clamps against the edge. Preserves the edge tracks that anchor "
                                "lens distortion and solve corners.")
                sw_gap = ui.switch("Keep disappear/reappear as one track", value=getattr(state, "gap_aware_refine", True),
                                   on_change=lambda e: setattr(state, "gap_aware_refine", bool(e.value)))
                sw_gap.tooltip("When a point is occluded then reappears, refine each visible segment on its own "
                               "reference patch and keep them under ONE track id -> the reappeared frames are re-acquired and kept, "
                               "not trimmed away by the pre-occlusion pattern.")
                sw_refine = ui.switch("3DE-style pattern lock (NCC/affine, full-res)", value=getattr(state, "pattern_refine", True),
                                      on_change=lambda e: setattr(state, "pattern_refine", bool(e.value)))
                sw_refine.tooltip("After selection, re-track each point at NATIVE resolution with an NCC pattern box + "
                                  "affine (rotation/scale) refine, like a 3DE pattern/search box. Locks to the contrast pattern, "
                                  "sub-pixel; trims a track where it loses lock. Breaks the 256px precision ceiling.")
                ui.label("Pattern box (px · odd)")
                refine_patch = ui.slider(min=15, max=61, value=int(getattr(state, "refine_patch_px", 31)), step=2).props("label-always")
                refine_patch.on_value_change(lambda e: setattr(state, "refine_patch_px", int(e.value or 31)))
                refine_patch.tooltip("Pattern-box size for the NCC lock. Larger = more stable on low contrast, less local; smaller = tighter to fine detail.")
            tap_box.bind_visibility_from(backend_sel, "value", value="tapnext")

# ---------- MAIN COLUMN ----------
with ui.column().classes("w-full gap-3 p-3"):
    # Progress bar: determinate when the stage reports done/total (masking = frames,
    # tracking = shots), indeterminate otherwise (Qwen exposes no numeric hook).
    bar = ui.linear_progress(value=0, show_value=False, size="6px").classes("w-full")
    bar.visible = False

    # ---- Run bar: one-click pipeline, then the same stages individually ----
    with ui.card().classes("w-full bt-card"):
        with ui.row().classes("w-full items-center no-wrap gap-2"):
            btn_pipe = ui.button("Run Pipeline", icon="play_arrow", on_click=start_pipeline, color="primary"
                                 ).props("unelevated no-caps").classes("bt-run")
            btn_pipe.tooltip("Runs all three stages back-to-back on the ticked shots, so you don't "
                             "have to click 1/2/3 and wait between each. Existing masks are reused "
                             "(use 2 · Masks to force a regen). Stop halts between stages.")
            ui.separator().props("vertical").classes("bt-vsep")
            btn_analyze = ui.button("1 · Analyze", icon="auto_awesome", on_click=start_analyze
                                    ).props("outline no-caps").classes("bt-step")
            btn_mask = ui.button("2 · Masks", icon="layers", on_click=start_mask
                                 ).props("outline no-caps").classes("bt-step")
            btn_track = ui.button("3 · Track", icon="my_location", on_click=start_track
                                  ).props("outline no-caps").classes("bt-step")

    # ---- Shots table ----
    with ui.card().classes("w-full bt-card"):
        with ui.row().classes("w-full items-center justify-between no-wrap gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Shots").classes("bt-section")
                ui.label("tick to include · click a row to edit · trash clears that shot's memory"
                         ).classes("bt-hint")
            lbl_selcount = ui.markdown("No shots yet — set **Shows Root**, load shows, then pick a **Show**."
                                       ).classes("bt-chip")
            ed_pick = ui.select([], label="Edit shot", with_input=True).props("dense outlined").classes("w-64")
            ed_pick.on_value_change(on_pick_edit)

        # ---- Bulk selection + status filter (the 100+ shot workflow) ----
        with ui.row().classes("w-full items-center no-wrap gap-2 q-mb-xs"):
            ui.button("Select all matching", icon="done_all", on_click=select_all_matching
                      ).props("outline dense no-caps"
                              ).tooltip("Ticks every shot passing the status chip AND the search box.")
            ui.button("None", icon="remove_done", on_click=select_none
                      ).props("outline dense no-caps").tooltip("Unticks every shot in the show.")
            ui.button("Invert", icon="swap_horiz", on_click=select_invert
                      ).props("outline dense no-caps").tooltip("Flips the tick on every matching shot.")
            ui.button("Paste list", icon="content_paste", on_click=lambda: paste_dlg.open()
                      ).props("outline dense no-caps"
                              ).tooltip("Paste a production shot list to tick them all at once.")
            ui.separator().props("vertical").classes("bt-vsep")
            ui.toggle(STATUS_CHIPS, value="All", on_change=on_status_chip
                      ).props("dense no-caps unelevated toggle-color=primary size=sm")

        table = ui.table(columns=TABLE_COLS, rows=[], row_key="name", selection="multiple",
                         pagination={"rowsPerPage": 0}
                         ).props("flat dense virtual-scroll hide-bottom").classes("w-full bt-shots")
        table.on("selection", on_selection_change)
        table.on("rowClick", on_row_click)
        # Per-shot "clear memory" button (right end). @click.stop so it doesn't also open the
        # editor via rowClick; emits the row up to the Python handler.
        table.add_slot("body-cell-clear", r'''
          <q-td :props="props" auto-width>
            <q-btn dense flat round color="grey-6" icon="delete_outline"
                   @click.stop="() => $parent.$emit('clearmem', props.row)">
              <q-tooltip>Clear this shot's analysis memory</q-tooltip>
            </q-btn>
          </q-td>
        ''')
        table.on("clearmem", lambda e: on_clear_mem(e.args))
        # Per-shot plate-version dropdown. Emits {name, version} up to Python, which
        # re-resolves that shot's plate_dir. @click.stop so it doesn't open the editor.
        table.add_slot("body-cell-version", r'''
          <q-td :props="props" auto-width @click.stop>
            <q-select dense options-dense borderless
                      v-model="props.row.version" :options="props.row.versions"
                      @update:model-value="(val) => $parent.$emit('pickversion', {name: props.row.name, version: val})"
                      style="min-width:88px">
              <template v-slot:no-option>
                <q-item><q-item-section class="text-grey">no versions</q-item-section></q-item>
              </template>
            </q-select>
          </q-td>
        ''')
        table.on("pickversion", lambda e: on_pick_version(e.args))

    # ---- Paste shot list (dialog; the "Paste list" button opens it) ----
    with ui.dialog() as paste_dlg, ui.card().classes("bt-paste"):
        ui.markdown("#### 📋 Paste a shot list").classes("q-mb-none")
        ui.label("One per line, or comma/space separated. Matched case-insensitively; "
                 "anything not in this show is reported back.").classes("bt-hint")
        paste_box = ui.textarea(placeholder="ABC_0010\nABC_0020\nABC_0030"
                                ).props("outlined autogrow input-style=min-height:180px").classes("w-full")
        with ui.row().classes("w-full justify-end no-wrap gap-2"):
            ui.button("Cancel", on_click=paste_dlg.close).props("flat no-caps")

            def _apply_paste():
                apply_pasted_list(paste_box.value or "")
                paste_dlg.close()

            ui.button("Tick these shots", icon="done_all", on_click=_apply_paste,
                      color="primary").props("unelevated no-caps")

    # ---- Logs ----
    with ui.card().classes("w-full bt-card"):
        with ui.expansion("Logs", icon="article", value=True).classes("w-full"):
            log_view = ui.log(max_lines=400).classes("w-full h-64")

# ---------- SHOT EDITOR (dialog; load_editor() calls editor.open()) ----------
with ui.dialog() as editor, ui.card().classes("bt-editor"):
    with ui.row().classes("w-full items-center justify-between no-wrap"):
        ed_title = ui.markdown("#### Select a shot")
        ui.button(icon="close", on_click=editor.close).props("flat dense round")
    ed_meta = ui.markdown("").classes("bt-hint")
    ui.separator()

    ui.label("Scope").classes("bt-section")
    with ui.row().classes("w-full no-wrap gap-2"):
        ed_scale = ui.select(["100%", "75%", "50%", "25%"], value="100%", label="Downscale"
                             ).props("dense outlined").classes("grow")
        ed_fstart = ui.number("Frame Start", value=0, min=0, precision=0
                              ).props("dense outlined").classes("grow")
        ed_fend = ui.number("Frame End", value=0, min=0, precision=0
                            ).props("dense outlined").classes("grow")
    ed_frange_hint = ui.label("").classes("bt-hint")

    ui.label("TAPNext render (optional)").classes("bt-section q-mt-sm")
    with ui.row().classes("w-full no-wrap items-end gap-1"):
        ed_render = ui.input("Render (.mp4 or JPEG/PNG folder)",
                             placeholder=r"…\render_folder  or  …\shot.mp4"
                             ).props("dense outlined").classes("grow")
        ui.button(icon="folder", on_click=lambda: pick_folder(ed_render)
                  ).props("flat dense").tooltip("Pick a JPEG/PNG render folder")
        ui.button(icon="movie", on_click=lambda: pick_file(ed_render)
                  ).props("flat dense").tooltip("Pick an .mp4 render")
    ui.label("TAPNext is mp4-only. Point it at an .mp4 or a JPEG/PNG render folder "
             "(auto-encoded to mp4). SynthEyes ignores this.").classes("bt-hint")

    ui.label("Brief").classes("bt-section q-mt-sm")
    ed_req = ui.textarea(placeholder="e.g. Camera track / Face track / track car, exclude crowd"
                         ).props("dense outlined autogrow").classes("w-full")
    ed_req.tooltip("Per-shot brief used by Analyze. Saved to "
                   "<OUT>/manual_requirements.json and reloaded on Scan.")

    ui.label("Prompts").classes("bt-section q-mt-sm")
    with ui.row().classes("w-full no-wrap gap-2"):
        ed_inc = ui.input("Include Prompts (track inside)").props("dense outlined").classes("grow")
        ed_exc = ui.input("Exclude Prompts (mask out)").props("dense outlined").classes("grow")
    with ui.expansion("AI object suggestions", icon="auto_awesome").classes("w-full"):
        ed_things = ui.select([], multiple=True, label="Detected objects").props("dense outlined").classes("w-full")
        with ui.row().classes("gap-2"):
            ui.button("Add → Include", on_click=lambda: add_things_to(ed_inc)).props("outline dense")
            ui.button("Add → Exclude", on_click=lambda: add_things_to(ed_exc)).props("outline dense color=negative")
    with ui.expansion("AI analysis", icon="psychology", value=False).classes("w-full"):
        ed_analysis = ui.markdown("Select a shot to see its AI analysis.")

    with ui.row().classes("w-full justify-between gap-2 q-mt-sm"):
        ui.button("Clear cache", icon="delete_sweep", on_click=clear_cache_current
                  ).props("flat color=orange").tooltip("Delete this shot's bot cache (JPG proxies + mp4 renders). Rebuilt on next run.")
        with ui.row().classes("gap-2"):
            ui.button("Close", on_click=editor.close).props("flat")
            ui.button("Save shot settings", icon="save", on_click=save_shot, color="primary")

ui.timer(1.0, poll)

ui.run(title="Batch Tracker (NiceGUI)", port=8080, reload=False, show=True, dark=True)
