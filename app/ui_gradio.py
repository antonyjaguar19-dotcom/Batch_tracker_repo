# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import threading
import queue
from typing import Dict, Any, List, Optional

import gradio as gr
import pandas as pd

from app.video_meta import probe_video_meta
from app.tracker_core import BatchTrackerRunner, RunnerConfig


def _gb(bytes_count: float) -> float:
    return float(bytes_count) / (1024.0 ** 3)


def est_cpu_ram_gb(w: int, h: int, frames: int) -> float:
    if w <= 0 or h <= 0 or frames <= 0:
        return 0.0
    return _gb(float(w) * float(h) * 3.0 * float(frames))


def est_gpu_vram_gb(w: int, h: int, frames: int, grid_size: int) -> float:
    if w <= 0 or h <= 0 or frames <= 0:
        return 0.0
    grid = max(1, int(grid_size))
    n = float(grid * grid)
    video_bytes = float(w) * float(h) * float(frames) * 3.0 * 4.0
    tracks_bytes = float(frames) * n * 2.0 * 4.0
    return _gb(video_bytes + tracks_bytes)


def cpu_risk_tag(gb: float) -> str:
    if gb >= 6.0:
        return "DANGER"
    if gb >= 2.0:
        return "WARN"
    return "OK"


def gpu_risk_tag(gb: float) -> str:
    if gb >= 12.0:
        return "DANGER"
    if gb >= 6.0:
        return "WARN"
    return "OK"


def parse_scale(s: Any) -> float:
    try:
        if isinstance(s, str):
            s2 = s.strip().replace("%", "")
            return float(s2) / 100.0
        return float(s)
    except Exception:
        return 1.0


def format_scale(scale: float) -> str:
    for v in (1.0, 0.75, 0.5, 0.25):
        if abs(scale - v) < 1e-6:
            return f"{int(v * 100)}%"
    return f"{scale * 100:.0f}%"


def suggest_scale(w: int, h: int) -> float:
    px = w * h
    if px >= 3840 * 2160:
        return 0.5
    if px >= 2560 * 1440:
        return 0.75
    return 1.0


def _find_child_dir_case_insensitive(parent: str, child_name: str) -> str | None:
    if not parent or not os.path.isdir(parent) or not child_name:
        return None
    direct = os.path.join(parent, child_name)
    if os.path.isdir(direct):
        return direct
    want = child_name.lower()
    try:
        for d in os.listdir(parent):
            p = os.path.join(parent, d)
            if os.path.isdir(p) and d.lower() == want:
                return p
    except Exception:
        return None
    return None


def _resolve_mask_dir(mask_root: str, shot_name: str) -> str | None:
    """
    Matches your OUT layout:
      MASK_ROOT\Shot_01\masks\*.png
    Fallback:
      MASK_ROOT\Shot_01\*.png
    """
    root = (mask_root or "").strip()
    if not root or not os.path.isdir(root):
        return None
    shot_dir = _find_child_dir_case_insensitive(root, shot_name)
    if not shot_dir:
        return None
    masks_dir = _find_child_dir_case_insensitive(shot_dir, "masks")
    return masks_dir if masks_dir else shot_dir


def _count_masks(mask_root: str, shot_name: str) -> int:
    d = _resolve_mask_dir(mask_root, shot_name)
    if not d:
        return 0
    try:
        return len([f for f in os.listdir(d) if f.lower().endswith(".png")])
    except Exception:
        return 0


def build_table(input_dir: str, mask_root: str, grid_size: int) -> pd.DataFrame:
    if not input_dir or not os.path.isdir(input_dir):
        return pd.DataFrame()

    rows = []
    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith(".mp4"):
            continue
        shot = os.path.splitext(fn)[0]
        meta = probe_video_meta(os.path.join(input_dir, fn))
        w = int(meta.get("width", 0) or 0)
        h = int(meta.get("height", 0) or 0)
        T = int(meta.get("total_frames", 0) or 0)
        fps = float(meta.get("fps", 0.0) or 0.0)

        scale = suggest_scale(w, h)
        ws = int(round(w * scale)) if w else 0
        hs = int(round(h * scale)) if h else 0

        cpu = est_cpu_ram_gb(ws, hs, T)
        gpu = est_gpu_vram_gb(ws, hs, T, grid_size)

        masks = _count_masks((mask_root or "").strip(), shot)

        rows.append(
            {
                "Select": False if (cpu_risk_tag(cpu) == "DANGER" or gpu_risk_tag(gpu) == "DANGER") else True,
                "File": fn,
                "Res": f"{w}x{h}" if w and h else "-",
                "Frames": f"1-{T}" if T else "-",
                "FPS": round(fps, 3) if fps else 0.0,
                "Masks": masks,
                "Scale": format_scale(scale),
                "ScaledRes": f"{ws}x{hs}" if ws and hs else "-",
                "CPU_GB": round(cpu, 3),
                "CPU_Risk": cpu_risk_tag(cpu),
                "GPU_GB": round(gpu, 3),
                "GPU_Risk": gpu_risk_tag(gpu),
            }
        )

    return pd.DataFrame(rows)


def recompute_table(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df2 = df.copy()
    for idx, r in df2.iterrows():
        try:
            res = str(r.get("Res", "-"))
            if "x" in res:
                w_str, h_str = res.split("x", 1)
                w = int(w_str)
                h = int(h_str)
            else:
                w = h = 0

            frames = str(r.get("Frames", "-"))
            T = int(frames.split("-", 1)[1]) if "-" in frames else 0

            scale = max(0.25, min(1.0, parse_scale(r.get("Scale", "100%"))))
            ws = int(round(w * scale)) if w else 0
            hs = int(round(h * scale)) if h else 0

            cpu = est_cpu_ram_gb(ws, hs, T)
            gpu = est_gpu_vram_gb(ws, hs, T, grid_size)

            df2.at[idx, "Scale"] = format_scale(scale)
            df2.at[idx, "ScaledRes"] = f"{ws}x{hs}" if ws and hs else "-"
            df2.at[idx, "CPU_GB"] = round(cpu, 3)
            df2.at[idx, "CPU_Risk"] = cpu_risk_tag(cpu)
            df2.at[idx, "GPU_GB"] = round(gpu, 3)
            df2.at[idx, "GPU_Risk"] = gpu_risk_tag(gpu)
        except Exception:
            continue
    return df2


class JobState:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.runner: Optional[BatchTrackerRunner] = None
        self.q: "queue.Queue[str]" = queue.Queue()
        self.logs: List[str] = []
        self.running: bool = False


JOBS: Dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()


def _get_job(session_id: str) -> JobState:
    sid = session_id or "default"
    with JOBS_LOCK:
        job = JOBS.get(sid)
        if job is None:
            job = JobState()
            JOBS[sid] = job
        return job


def _status_cb(job: JobState):
    def _cb(msg: str):
        job.q.put(msg)

    return _cb


def _run_job(job: JobState, runner: BatchTrackerRunner):
    try:
        runner.run()
    finally:
        job.running = False
        job.q.put("JOB_FINISHED")


def start_job(
    input_dir: str,
    output_dir: str,
    mask_root: str,
    mask_mode: str,
    grid_size: int,
    flip_y: bool,
    df: pd.DataFrame,
    session_id: str,
):
    job = _get_job(session_id)
    if job.running:
        raise RuntimeError("A job is already running. Stop it first.")
    if not input_dir or not os.path.isdir(input_dir):
        raise RuntimeError("Invalid input folder.")
    if not output_dir or not os.path.isdir(output_dir):
        raise RuntimeError("Invalid output folder.")
    if df is None or len(df) == 0:
        raise RuntimeError("No shots. Click Scan first.")

    selected_files: List[str] = []
    scales: Dict[str, float] = {}
    for _, r in df.iterrows():
        if bool(r.get("Select", False)):
            fn = str(r.get("File", "")).strip()
            if not fn:
                continue
            selected_files.append(fn)
            scales[fn] = max(0.25, min(1.0, parse_scale(r.get("Scale", "100%"))))

    if not selected_files:
        raise RuntimeError("No shots selected.")

    # UPDATED CONFIG: FP16/Features/Bidir defaults are in tracker_core, 
    # but here we pass the UI inputs.
    cfg = RunnerConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        mask_root_dir=(mask_root or "").strip(),
        mask_mode=(mask_mode or "outside").strip().lower(),
        mask_polarity="auto",
        enable_mask_gating=True,
        inside_ratio=0.80,
        grid_size=int(grid_size),
        flip_y_for_3de=bool(flip_y),
        selected_files=selected_files,
        selected_scales=scales,
    )

    job.logs = [f"Starting batch with {len(selected_files)} shots..."]
    job.running = True
    job.runner = BatchTrackerRunner(cfg, on_status=_status_cb(job))
    job.thread = threading.Thread(target=_run_job, args=(job, job.runner), daemon=True)
    job.thread.start()


def stop_job(session_id: str):
    job = _get_job(session_id)
    if job.runner:
        job.runner.request_stop()


def stream_logs(session_id: str):
    job = _get_job(session_id)
    while True:
        try:
            msg = job.q.get(timeout=0.2)
        except queue.Empty:
            msg = None

        if msg:
            if msg == "JOB_FINISHED":
                while True:
                    try:
                        m2 = job.q.get_nowait()
                        if m2 != "JOB_FINISHED":
                            job.logs.append(m2)
                    except queue.Empty:
                        break
                break
            job.logs.append(msg)

        yield "\n".join(job.logs)

        if not job.running and msg is None:
            break


SCALE_CHOICES = ["100%", "75%", "50%", "25%"]
MASK_MODE_CHOICES = [
    ("Track OUTSIDE mask (exclude mask region)", "outside"),
    ("Track INSIDE mask (keep mask region only)", "inside"),
]


def build_app():
    with gr.Blocks(title="Batch Tracker (CoTracker3 FP16) - Localhost") as demo:
        gr.Markdown(
            "## Batch Tracker (CoTracker3 FP16) — Localhost\n"
            "**Features:** Bidirectional Tracking, Half-Precision (Low VRAM), Feature Seeding.\n"
            "### Masks (optional)\n"
            "Tool reads: `MASK_ROOT\\Shot_XX\\masks\\*.png` (fallback: `MASK_ROOT\\Shot_XX\\*.png`).\n"
        )

        session_state = gr.State("")
        selected_row_state = gr.State(-1)

        def init_session(request: gr.Request):
            return request.session_hash

        demo.load(init_session, inputs=None, outputs=[session_state])

        with gr.Row():
            input_dir = gr.Textbox(label="Input Folder (mp4)", placeholder=r"D:\shots\in", scale=3)
            output_dir = gr.Textbox(label="Output Folder", placeholder=r"D:\shots\out", scale=3)

        mask_root = gr.Textbox(label="Mask Root Folder (optional)", placeholder=r"D:\shots\OUT", scale=3)
        mask_mode = gr.Radio(
            choices=[x[0] for x in MASK_MODE_CHOICES],
            value=MASK_MODE_CHOICES[0][0],
            label="Mask Mode",
        )

        def _mask_mode_value(label: str) -> str:
            for text, val in MASK_MODE_CHOICES:
                if text == label:
                    return val
            return "outside"

        with gr.Row():
            # Renamed label to imply fallback use
            grid_size = gr.Slider(4, 40, value=10, step=1, label="Grid Size (Fallback)", scale=2)
            flip_y = gr.Checkbox(value=True, label="Flip Y for 3DE import", scale=1)

        with gr.Row():
            btn_scan = gr.Button("Scan")
            btn_update = gr.Button("Update Estimates")
            btn_start = gr.Button("Start", variant="primary")
            btn_stop = gr.Button("Stop", variant="stop")

        table = gr.Dataframe(
            headers=["Select", "File", "Res", "Frames", "FPS", "Masks", "Scale", "ScaledRes", "CPU_GB", "CPU_Risk", "GPU_GB", "GPU_Risk"],
            datatype=["bool", "str", "str", "str", "number", "number", "str", "str", "number", "str", "number", "str"],
            interactive=True,
            label="Shots",
            wrap=True,
            row_count=(0, "dynamic"),
            # FIX: Changed col_count -> column_count
            column_count=(12, "fixed"),
        )

        with gr.Row():
            picked = gr.Textbox(label="Selected shot (click a row in the table)", interactive=False, scale=2)
            scale_dd = gr.Dropdown(choices=SCALE_CHOICES, value="100%", label="Downscale for selected shot", scale=1)
            btn_apply_one = gr.Button("Apply to Selected Shot", scale=1)
            btn_apply_sel = gr.Button("Apply to ALL shots where Select=True", scale=1)

        log = gr.Textbox(label="Live Log", lines=18, interactive=False)

        def do_scan(in_dir, mask_dir, gsize):
            df = build_table(in_dir, (mask_dir or "").strip(), int(gsize))
            if df is None or len(df) == 0:
                return df, "", "No .mp4 files found (or invalid folder)."
            return df, "", f"Scanned {len(df)} shots."

        def do_update(df_val, gsize):
            df = pd.DataFrame(df_val) if not isinstance(df_val, pd.DataFrame) else df_val
            return recompute_table(df, int(gsize)), "Updated estimates."

        def on_table_select(evt: gr.SelectData, df_val):
            try:
                row = int(evt.index[0])
            except Exception:
                row = -1
            df = pd.DataFrame(df_val) if not isinstance(df_val, pd.DataFrame) else df_val
            shot = ""
            if row >= 0 and row < len(df):
                shot = str(df.iloc[row].get("File", ""))
            return row, shot

        def apply_scale_one(df_val, gsize, row_idx: int, scale_str: str):
            df = pd.DataFrame(df_val) if not isinstance(df_val, pd.DataFrame) else df_val
            if df is None or len(df) == 0:
                return df, "No rows."
            if row_idx is None or int(row_idx) < 0 or int(row_idx) >= len(df):
                return df, "No shot selected. Click a row in the table first."
            df.at[int(row_idx), "Scale"] = str(scale_str)
            df2 = recompute_table(df, int(gsize))
            fn = str(df2.iloc[int(row_idx)].get("File", ""))
            return df2, f"Applied {scale_str} to {fn}."

        def apply_scale_selected(df_val, gsize, scale_str: str):
            df = pd.DataFrame(df_val) if not isinstance(df_val, pd.DataFrame) else df_val
            if df is None or len(df) == 0:
                return df, "No rows."
            count = 0
            for i in range(len(df)):
                if bool(df.iloc[i].get("Select", False)):
                    df.at[i, "Scale"] = str(scale_str)
                    count += 1
            df2 = recompute_table(df, int(gsize))
            return df2, f"Applied {scale_str} to {count} selected shots."

        def do_start(in_dir, out_dir, mask_dir, mask_mode_label, gsize, flip, df_val, session_id: str):
            df = pd.DataFrame(df_val) if not isinstance(df_val, pd.DataFrame) else df_val
            mode_val = _mask_mode_value(mask_mode_label)
            start_job(in_dir, out_dir, (mask_dir or "").strip(), mode_val, int(gsize), bool(flip), df, session_id)
            return f"Started. mask_mode={mode_val}. Streaming logs..."

        def do_stop(session_id: str):
            stop_job(session_id)
            return "Stop requested (safe stop)."

        btn_scan.click(do_scan, inputs=[input_dir, mask_root, grid_size], outputs=[table, picked, log])
        btn_update.click(do_update, inputs=[table, grid_size], outputs=[table, log])

        table.select(on_table_select, inputs=[table], outputs=[selected_row_state, picked])

        btn_apply_one.click(apply_scale_one, inputs=[table, grid_size, selected_row_state, scale_dd], outputs=[table, log])
        btn_apply_sel.click(apply_scale_selected, inputs=[table, grid_size, scale_dd], outputs=[table, log])

        btn_start.click(
            do_start,
            inputs=[input_dir, output_dir, mask_root, mask_mode, grid_size, flip_y, table, session_state],
            outputs=[log],
        ).then(stream_logs, inputs=[session_state], outputs=[log])

        btn_stop.click(do_stop, inputs=[session_state], outputs=[log])

    return demo


def launch():
    demo = build_app()
    # FIX: Removed server_port=7860 to allow auto-port selection if 7860 is busy.
    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    launch()