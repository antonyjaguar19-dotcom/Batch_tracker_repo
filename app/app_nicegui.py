# -*- coding: utf-8 -*-
"""
Batch Tracker — NiceGUI front-end.

Reuses the ENTIRE backend from app.py (loaders, workers, state, helpers) — only the
UI layer is reimplemented in NiceGUI. The Gradio app (app.py) stays as a fallback.

Run:  runtime/python311/python.exe app_nicegui.py   (or launch_nicegui.bat)
"""
from __future__ import annotations

import os
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


# -----------------------------------------------------------------------------
# Row building / table refresh
# -----------------------------------------------------------------------------
def _visible_names():
    return be._visible_names(state)


def _build_rows():
    rows = []
    for name in _visible_names():
        d = state.shots_data[name]
        prompts = f"INC: {d.include_prompts} | EXC: {d.exclude_prompts}"
        if len(prompts) > 60:
            prompts = prompts[:57] + "..."
        rows.append({
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
        })
    return rows


def refresh_table():
    """Rebuild rows + restore selection (active shots) without firing the select handler."""
    global _suppress_select
    rows = _build_rows()
    _suppress_select = True
    table.rows = rows
    table.selected = [r for r in rows if state.shots_data.get(r["name"]) and state.shots_data[r["name"]].use]
    table.update()
    _suppress_select = False
    lbl_selcount.set_content(be._sel_count_text(state))


# -----------------------------------------------------------------------------
# Selection (active shots) + editing
# -----------------------------------------------------------------------------
def on_selection_change():
    if _suppress_select:
        return
    sel = {r["name"] for r in (table.selected or [])}
    for name, d in state.shots_data.items():
        d.use = name in sel
    lbl_selcount.set_content(be._sel_count_text(state))


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
    # Persist the manual client requirement so it survives restarts (reloaded on Scan).
    note = (ed_req.value or "").strip()
    if note:
        state.manual_notes[name] = note
    else:
        state.manual_notes.pop(name, None)
    be.save_manual_notes(out_dir.value, state.manual_notes)
    refresh_table()
    ui.notify(f"Saved {name}", type="positive")


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
    ed_pick.set_options(_visible_names())
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


async def do_scan_show():
    """Studio flow: list shots under the picked show, resolve each shot's plate
    versions (default = latest vNNN) and plate dir. Replaces the flat-folder scan."""
    show = show_sel.value
    if not (shows_root.value and show):
        return
    shots = await run.io_bound(be.list_shots_for_show, shows_root.value, show)
    state.shots_data = {}
    state.log_history = []
    for s in shots:
        vers = await run.io_bound(be.list_shot_versions, shows_root.value, show, s)
        latest = vers[-1] if vers else ""
        vdir = be.resolve_plate_dir(shows_root.value, show, s, latest) if latest else ""
        # Descend past clip-id / resolution subfolders to the actual frames dir.
        pdir, frames, pstart, pend = await run.io_bound(be.find_frames_subdir, vdir) if vdir else ("", 0, 0, 0)
        state.shots_data[s] = be.ShotData(name=s, scale="100%", show=show,
                                          versions=vers, version=latest, plate_dir=pdir,
                                          frames=frames, plate_start=pstart, plate_end=pend)
    state.manual_notes = be.load_manual_notes(out_dir.value)
    try:
        _scan_load_prev_guide()
    except Exception as ex:
        print(f"prev guide load: {ex}")
    ed_pick.set_options(_visible_names())
    refresh_table()
    ui.notify(f"{show}: {len(shots)} shot(s)", type="info")


def on_pick_version(args):
    """A shot row's version dropdown changed — re-resolve that shot's plate dir."""
    if isinstance(args, list):
        args = args[0] if args else None
    name = (args or {}).get("name") if isinstance(args, dict) else None
    ver = (args or {}).get("version") if isinstance(args, dict) else None
    d = state.shots_data.get(name)
    if not d or not ver:
        return
    d.version = ver
    vdir = be.resolve_plate_dir(shows_root.value, d.show, name, ver)
    d.plate_dir, d.frames, d.plate_start, d.plate_end = be.find_frames_subdir(vdir)
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


def _prompt_missing_requirements(missing):
    """Ask the user for a client requirement for each missing shot, one by one,
    in a text field. Notes saved to <OUT>/manual_requirements.json for reuse."""
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
            _launch_analyze()
        else:
            render()

    skip_btn.on_click(lambda: advance(False))
    next_btn.on_click(lambda: advance(True))
    render()
    dlg.open()


def start_analyze():
    shots = [n for n, d in state.shots_data.items() if getattr(d, "use", False)]
    if not shots:
        ui.notify("Select at least one shot (tick 'Use'), then Analyze.", type="warning")
        return
    # Reuse any notes saved in a previous session (saved for later use).
    saved = be.load_manual_notes(out_dir.value)
    state.manual_notes = dict(saved)
    have = be.requirement_shot_names(req_file.value)
    have |= {_norm_shot(k) for k, v in saved.items() if str(v).strip()}
    missing = [s for s in shots if _norm_shot(s) not in have]
    if missing:
        ui.notify(f"{len(missing)} shot(s) missing a client requirement — enter them.",
                  type="warning")
        _prompt_missing_requirements(missing)
        return
    _launch_analyze()


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
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")

            def _go():
                dlg.close(); _launch_pipeline()

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
            import json
            state.guide_path = refresh_path
            with open(refresh_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("shots", []):
                nm = s.get("shot_name") or s.get("shot") or s.get("name")
                if nm and nm in state.shots_data:
                    d = state.shots_data[nm]
                    d.strategy = be._derive_strategy(s)
                    d.include_prompts = be._extract_prompt_list(s, ["mask_includes", "include_prompts", "sam3_include_prompt"])
                    d.exclude_prompts = be._extract_prompt_list(s, ["mask_excludes", "exclude_prompts", "sam3_exclude_prompt"])
                    rt = s.get("qwen2_things", [])
                    if isinstance(rt, list): d.detected_things = rt
                    be._apply_analysis_fields(d, s)
            refresh_now = True
        except Exception as ex:
            print(f"guide reload: {ex}")

    if refresh_now:
        refresh_table()
        ed_pick.set_options(_visible_names())

    # status + progress bar + elapsed + button busy-state
    running = be._job_running()
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
            bar.props("indeterminate")
        else:
            bar.props(remove="indeterminate")
            bar.value = float(frac)
            prog += f" · {int(frac * 100)}%"
        lbl_status.set_content(f"⏳ **{be.CURRENT_JOB_NAME}** running…{prog}{el}")
    else:
        lbl_status.set_content("🟢 Idle")
    for b in (btn_pipe, btn_analyze, btn_mask, btn_track):
        b.set_enabled(not running)
    btn_stop.set_enabled(running)


def on_search(e):
    state.filter_query = e.value or ""
    ed_pick.set_options(_visible_names())
    refresh_table()


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
  /* Run bar: pipeline button, then the four stages as one evenly-sized sequence. */
  .bt-run     { min-width: 150px; height: 36px; font-weight: 600; }
  .bt-step    { min-width: 122px; height: 36px; color: #9dc0ff; }
  .bt-vsep    { height: 24px; margin: 0 6px; background: rgba(255,255,255,.12); }
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
        ui.separator().classes("q-my-xs")
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            in_dir = ui.input("Input Folder (legacy / fallback)", placeholder=r"D:\shots\IN"
                              ).props("dense outlined").classes("grow")
            ui.button(icon="folder", on_click=lambda: pick_folder(in_dir)).props("flat dense")
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            out_dir = ui.input("Output Folder", placeholder=r"D:\shots\OUT").props("dense outlined").classes("grow")
            ui.button(icon="folder", on_click=lambda: pick_folder(out_dir)).props("flat dense")
        with ui.row().classes("w-full no-wrap items-end gap-1"):
            req_file = ui.input("Client Requirements (optional)", placeholder=r"D:\shots\reqs.xlsx"
                                ).props("dense outlined").classes("grow")
            ui.button(icon="description", on_click=lambda: pick_file(req_file)).props("flat dense")

    with ui.card().classes("w-full bt-card"):
        ui.label("Find shots").classes("bt-section")
        ui.input(placeholder="type part of a shot name…"
                 ).props("dense outlined clearable").classes("w-full").on_value_change(on_search)

    with ui.card().classes("w-full bt-card"):
        with ui.expansion("Settings", icon="tune", value=False).classes("w-full"):
            # ---- Universal (both backends / AI pipeline) ----
            ui.label("Analysis & seeding").classes("bt-section q-mt-sm")
            ui.label("Qwen2 Sample Density")
            qwen_fps = ui.slider(min=1, max=8, value=4).props("label-always")
            ui.label("Seed Count (Max Tracks)")
            seed_count = ui.slider(min=100, max=3000, value=1200, step=50).props("label-always")

            sw_motion = ui.switch("CV motion backstop", value=True,
                                  on_change=lambda e: setattr(state, "motion_backstop", bool(e.value)))
            sw_motion.tooltip("Masks objects moving independently of the camera (even ones NOT in your Exclude list). "
                              "Turn OFF if it masks things you removed from Exclude.")

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
                track_spacing.tooltip("Min pixel gap between kept tracks. Small = denser/more tracks, large = sparser/fewer. "
                                      "Collapses duplicate passes and spreads the strongest tracks evenly; count floats with footage texture/parallax.")
                ui.label("Max tracks per task (0 = auto)")
                track_max = ui.number(value=int(getattr(state, "track_max_output", 0)), min=0, max=500, precision=0
                                      ).props("dense outlined").classes("w-full")
                track_max.on_value_change(lambda e: setattr(state, "track_max_output", int(e.value or 0)))
                track_max.tooltip("Soft ceiling on exported tracks per shot/task after spread selection. 0 = unlimited.")

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

        table = ui.table(columns=TABLE_COLS, rows=[], row_key="name", selection="multiple"
                         ).props("flat dense").classes("w-full")
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

    with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):
        ui.button("Close", on_click=editor.close).props("flat")
        ui.button("Save shot settings", icon="save", on_click=save_shot, color="primary")

ui.timer(1.0, poll)

ui.run(title="Batch Tracker (NiceGUI)", port=8080, reload=False, show=True, dark=True)
