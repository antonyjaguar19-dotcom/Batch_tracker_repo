# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from app.tracker_core import BatchTrackerRunner, RunnerConfig
from app.video_meta import probe_video_meta


def _gb(bytes_count: float) -> float:
    return float(bytes_count) / (1024.0 ** 3)

def _fmt_gb(gb: float) -> str:
    if gb <= 0:
        return "-"
    if gb < 0.1:
        return f"{gb*1024:.0f}MB"
    return f"{gb:.2f}GB"

def _estimate_cpu_ram_gb(w: int, h: int, frames: int) -> float:
    if w <= 0 or h <= 0 or frames <= 0:
        return 0.0
    return _gb(float(w) * float(h) * 3.0 * float(frames))

def _estimate_gpu_vram_gb(w: int, h: int, frames: int, grid_size: int) -> float:
    if w <= 0 or h <= 0 or frames <= 0:
        return 0.0
    grid_size = max(1, int(grid_size))
    n = float(grid_size * grid_size)
    video_bytes = float(w) * float(h) * float(frames) * 3.0 * 4.0
    tracks_bytes = float(frames) * n * 2.0 * 4.0
    return _gb(video_bytes + tracks_bytes)

def _risk_cpu(est_gb: float) -> tuple[str, str]:
    if est_gb >= 6.0:
        return ("CPU:DANGER", "red")
    if est_gb >= 2.0:
        return ("CPU:WARN", "#b36b00")
    return ("CPU:OK", "black")

def _risk_gpu(est_gb: float) -> tuple[str, str]:
    if est_gb >= 12.0:
        return ("GPU:DANGER", "red")
    if est_gb >= 6.0:
        return ("GPU:WARN", "#b36b00")
    return ("GPU:OK", "black")

def _combined_color(c1: str, c2: str) -> str:
    if c1 == "red" or c2 == "red":
        return "red"
    if c1 == "#b36b00" or c2 == "#b36b00":
        return "#b36b00"
    return "black"


class _ShotRow:
    def __init__(self, parent: tk.Frame, filename: str, meta: dict, grid_var: tk.IntVar):
        self.filename = filename
        self.meta = meta
        self.grid_var = grid_var

        self.var_selected = tk.BooleanVar(value=True)
        self.var_scale = tk.StringVar(value="100%")

        self.frame = tk.Frame(parent)
        self.frame.pack(fill="x", padx=6, pady=2)

        self.chk = tk.Checkbutton(self.frame, variable=self.var_selected)
        self.chk.pack(side="left")

        self.lbl = tk.Label(self.frame, text="", anchor="w", justify="left")
        self.lbl.pack(side="left", fill="x", expand=True)

        opts = ["100%", "75%", "50%", "25%"]
        self.menu = tk.OptionMenu(self.frame, self.var_scale, *opts, command=lambda _v: self.update_text())
        self.menu.config(width=6)
        self.menu.pack(side="right", padx=(6, 0))

        w0 = int(meta.get("width", 0) or 0)
        h0 = int(meta.get("height", 0) or 0)
        if w0 * h0 >= 3840 * 2160:
            self.var_scale.set("50%")
        elif w0 * h0 >= 2560 * 1440:
            self.var_scale.set("75%")

        self.update_text()

    def scale_factor(self) -> float:
        v = (self.var_scale.get() or "100%").strip()
        if v == "75%":
            return 0.75
        if v == "50%":
            return 0.5
        if v == "25%":
            return 0.25
        return 1.0

    def update_text(self):
        w0 = int(self.meta.get("width", 0) or 0)
        h0 = int(self.meta.get("height", 0) or 0)
        T = int(self.meta.get("total_frames", 0) or 0)
        fps = float(self.meta.get("fps", 0.0) or 0.0)

        s = self.scale_factor()
        ws = max(1, int(round(w0 * s))) if w0 else 0
        hs = max(1, int(round(h0 * s))) if h0 else 0

        grid = int(self.grid_var.get() or 1)
        cpu_gb = _estimate_cpu_ram_gb(ws, hs, T)
        gpu_gb = _estimate_gpu_vram_gb(ws, hs, T, grid)

        cpu_tag, cpu_col = _risk_cpu(cpu_gb)
        gpu_tag, gpu_col = _risk_gpu(gpu_gb)
        col = _combined_color(cpu_col, gpu_col)

        res0 = f"{w0}x{h0}" if w0 and h0 else "-"
        frames_txt = f"1-{T}" if T else "-"
        fps_txt = f"{fps:.2f}" if fps else "-"
        res_s = f"{ws}x{hs}" if ws and hs and s != 1.0 else "orig"

        txt = (
            f"{self.filename} | {res0:<11} | {frames_txt:<10} | {fps_txt:<6} | "
            f"Scale:{self.var_scale.get():<4} ({res_s:<9}) | "
            f"CPU:{_fmt_gb(cpu_gb):<7} {cpu_tag:<10} | "
            f"GPU:{_fmt_gb(gpu_gb):<7} {gpu_tag}"
        )
        self.lbl.config(text=txt, fg=col)


class BatchTrackerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch Tracker (Native CoTracker3 Grid)")
        self.root.geometry("1200x620")
        self.root.resizable(True, True)

        self._runner: BatchTrackerRunner | None = None
        self._thread: threading.Thread | None = None

        self.var_in = tk.StringVar(value="")
        self.var_out = tk.StringVar(value="")
        self.var_grid = tk.IntVar(value=10)
        self.var_flip_y = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(value="Idle.")

        self._rows: list[_ShotRow] = []

        self._build()
        self.var_grid.trace_add("write", lambda *_: self._update_all_rows())

    def _build(self):
        pad = {"padx": 10, "pady": 6}
        top = tk.Frame(self.root)
        top.pack(fill="x")

        tk.Label(top, text="Input Folder (mp4):").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(top, textvariable=self.var_in, width=95).grid(row=0, column=1, sticky="we", **pad)
        tk.Button(top, text="Browse", command=self._browse_in).grid(row=0, column=2, **pad)

        tk.Label(top, text="Output Folder:").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(top, textvariable=self.var_out, width=95).grid(row=1, column=1, sticky="we", **pad)
        tk.Button(top, text="Browse", command=self._browse_out).grid(row=1, column=2, **pad)

        tk.Label(top, text="Grid Size:").grid(row=2, column=0, sticky="w", **pad)
        tk.Spinbox(top, from_=4, to=40, textvariable=self.var_grid, width=10).grid(row=2, column=1, sticky="w", **pad)

        tk.Checkbutton(top, text="Flip Y for 3DE import", variable=self.var_flip_y).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=10, pady=4
        )

        note = (
            "Per-shot warnings update when Grid Size changes. "
            "Choose 75%/50%/25% downscale to reduce memory. "
            "Tracks are exported back to ORIGINAL resolution."
        )
        tk.Label(top, text=note, fg="#444").grid(row=4, column=0, columnspan=3, sticky="w", padx=10, pady=(2, 8))

        mid = tk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        hdr = tk.Frame(mid)
        hdr.pack(fill="x", pady=(2, 4))
        tk.Label(hdr, text="Shots (tick to select):").pack(side="left")
        tk.Button(hdr, text="Select All", command=self._select_all).pack(side="right", padx=(6, 0))
        tk.Button(hdr, text="Select None", command=self._select_none).pack(side="right")

        self.canvas = tk.Canvas(mid, borderwidth=1, highlightthickness=0)
        self.scroll = tk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.list_frame = tk.Frame(self.canvas)

        self.list_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        bot = tk.Frame(self.root)
        bot.pack(fill="x", padx=10, pady=8)

        self.btn_start = tk.Button(bot, text="Start", width=12, command=self._start)
        self.btn_stop = tk.Button(bot, text="Stop", width=12, command=self._stop, state="disabled")
        self.btn_start.pack(side="left")
        self.btn_stop.pack(side="left", padx=(8, 0))

        tk.Label(bot, textvariable=self.var_status, fg="blue").pack(side="left", padx=14)

        top.grid_columnconfigure(1, weight=1)

    def _browse_in(self):
        p = filedialog.askdirectory(title="Select Input Folder")
        if p:
            self.var_in.set(p)
            self._refresh_shots()

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select Output Folder")
        if p:
            self.var_out.set(p)

    def _clear_shot_list(self):
        for child in list(self.list_frame.winfo_children()):
            child.destroy()
        self._rows.clear()

    def _refresh_shots(self):
        self._clear_shot_list()
        in_dir = self.var_in.get().strip()
        if not in_dir or not os.path.isdir(in_dir):
            return

        shots = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".mp4")])
        if not shots:
            tk.Label(self.list_frame, text="(No .mp4 files found in this folder)").pack(anchor="w", padx=6, pady=6)
            return

        for fn in shots:
            meta = probe_video_meta(os.path.join(in_dir, fn))
            row = _ShotRow(self.list_frame, fn, meta, self.var_grid)
            self._rows.append(row)

            # Auto-unselect if currently DANGER on either CPU or GPU (based on suggested scale)
            w0 = int(meta.get("width", 0) or 0)
            h0 = int(meta.get("height", 0) or 0)
            T = int(meta.get("total_frames", 0) or 0)
            s = row.scale_factor()
            ws = max(1, int(round(w0 * s))) if w0 else 0
            hs = max(1, int(round(h0 * s))) if h0 else 0
            cpu_gb = _estimate_cpu_ram_gb(ws, hs, T)
            gpu_gb = _estimate_gpu_vram_gb(ws, hs, T, int(self.var_grid.get() or 1))
            _, cpu_col = _risk_cpu(cpu_gb)
            _, gpu_col = _risk_gpu(gpu_gb)
            if cpu_col == "red" or gpu_col == "red":
                row.var_selected.set(False)

    def _update_all_rows(self):
        for r in self._rows:
            r.update_text()

    def _select_all(self):
        for r in self._rows:
            r.var_selected.set(True)

    def _select_none(self):
        for r in self._rows:
            r.var_selected.set(False)

    def _selected(self) -> tuple[list[str], dict[str, float]]:
        files = []
        scales: dict[str, float] = {}
        for r in self._rows:
            if bool(r.var_selected.get()):
                files.append(r.filename)
                scales[r.filename] = float(r.scale_factor())
        return files, scales

    def _start(self):
        in_dir = self.var_in.get().strip()
        out_dir = self.var_out.get().strip()
        if not in_dir or not os.path.isdir(in_dir):
            messagebox.showerror("Error", "Please select a valid Input Folder.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror("Error", "Please select a valid Output Folder.")
            return

        selected, scales = self._selected()
        if not selected:
            messagebox.showerror("Error", "No shots selected. Tick at least one shot.")
            return

        cfg = RunnerConfig(
            input_dir=in_dir,
            output_dir=out_dir,
            grid_size=int(self.var_grid.get()),
            flip_y_for_3de=bool(self.var_flip_y.get()),
            selected_files=selected,
            selected_scales=scales,
        )

        self._runner = BatchTrackerRunner(cfg, on_status=self._set_status)
        self._thread = threading.Thread(target=self._runner.run, daemon=True)
        self._thread.start()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._set_status("Running...")

    def _stop(self):
        if self._runner:
            self._runner.request_stop()
            self._set_status("Stop requested (safe stop)...")
        self.btn_stop.config(state="disabled")

    def _set_status(self, msg: str):
        def _apply():
            self.var_status.set(msg)
            if msg.startswith("Done") or msg.startswith("Stopped") or msg.startswith("Error"):
                self.btn_start.config(state="normal")
                self.btn_stop.config(state="disabled")
        try:
            self.root.after(0, _apply)
        except Exception:
            pass

    def run(self):
        self.root.mainloop()
