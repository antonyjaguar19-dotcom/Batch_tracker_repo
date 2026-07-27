from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from qwen2_shot_describer_core import (
    ModelConfig,
    QwenShotDescriber,
    list_shots_in_folder,
    sequence_folder_to_temp_mp4,
)

try:
    from PySide6 import QtCore, QtWidgets
    PYSIDE_AVAILABLE = True
except Exception:
    PYSIDE_AVAILABLE = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
OUT_JSON_NAME = "shot_descriptions.json"


def _resolve_model_dir() -> Path:
    """Pick the best local Qwen2.5-VL model folder.

    Priority: env override -> 7B AWQ (int4, preferred) -> full 7B -> 3B (legacy).
    All must sit under <this script dir>/models/ to stay offline. Returns the preferred
    7B-AWQ path even if absent, so the not-found error points users at the recommended weights.
    """
    base = Path(__file__).resolve().parent / "models"
    candidates = []
    env = os.environ.get("QWEN2_MODEL_DIR", "").strip()
    if env:
        candidates.append(Path(env))
    candidates += [
        base / "Qwen2.5-VL-7B-Instruct-AWQ",
        base / "Qwen2.5-VL-7B-Instruct",
        base / "Qwen2.5-VL-3B-Instruct",
    ]
    for c in candidates:
        if c.exists():
            return c
    return base / "Qwen2.5-VL-7B-Instruct-AWQ"


LOCAL_MODEL_DIR = _resolve_model_dir()


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _default_out_json(out_dir: str) -> Path:
    return Path(out_dir) / OUT_JSON_NAME


def _int4_usable() -> bool:
    """int4 is usable if weights are pre-quantized (AWQ/GPTQ) or bitsandbytes is present."""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import json as _json
        cfgp = _resolve_model_dir() / "config.json"
        if cfgp.is_file():
            d = _json.loads(cfgp.read_text(encoding="utf-8"))
            return bool(isinstance(d, dict) and d.get("quantization_config"))
    except Exception:
        pass
    return False


def _ensure_offline_env() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    hf_home = str(PROJECT_ROOT / ".hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def run_batch(in_dir: str, out_dir: str, fps: int, use_int4: bool, log_cb=None, only_shots=None,
              shot_dirs=None) -> Path:
    _ensure_offline_env()

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_json = _default_out_json(out_dir)
    run_started = _iso_now()

    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg, flush=True)

    model_dir = _resolve_model_dir()

    log("Starting… (offline mode enabled)")
    log(f"Run started: {run_started}")
    log(f"IN : {in_dir}")
    log(f"OUT: {out_dir}")
    log(f"Model (local): {model_dir}")
    log(f"FPS preset: {fps}")
    log(f"Int4: {use_int4}")

    if not model_dir.exists():
        base = Path(__file__).resolve().parent / "models"
        raise RuntimeError(
            "Local Qwen2.5-VL model folder not found.\n"
            f"Looked for (in order):\n"
            f"  - $QWEN2_MODEL_DIR (if set)\n"
            f"  - {base / 'Qwen2.5-VL-7B-Instruct-AWQ'}   (recommended: 7B int4)\n"
            f"  - {base / 'Qwen2.5-VL-7B-Instruct'}       (full 7B, int4 via bitsandbytes)\n"
            f"  - {base / 'Qwen2.5-VL-3B-Instruct'}       (legacy 3B)\n\n"
            "Offline-by-design: the tool will NOT download models.\n"
            "Place model files in one of those folders, or set QWEN2_MODEL_DIR."
        )

    cfg = ModelConfig(model_id_or_path=str(model_dir), fps=int(fps), use_torchao_int4=bool(use_int4))
    describer = QwenShotDescriber(cfg)
    log(f"Model loaded. Precision: {getattr(describer, 'quant_mode', 'unknown')}")

    def _norm(s): return re.sub(r"[\s_\-]+", "", str(s).strip()).lower()
    if shot_dirs:
        # Explicit per-shot dirs (studio network plate fetch): frames live outside in_dir.
        shots = [(n, Path(d)) for n, d in shot_dirs.items() if d]
        log(f"Using {len(shots)} explicit per-shot plate dir(s).")
    else:
        shots = list_shots_in_folder(in_dir)
    if only_shots:
        want = {_norm(s) for s in only_shots}
        shots = [(n, p) for (n, p) in shots if _norm(n) in want]
        log(f"Restricted to {len(shots)} selected shot(s).")
    payload: Dict[str, Any] = {
        "meta": {
            "run_started": run_started,
            "in_dir": str(Path(in_dir).resolve()),
            "out_dir": str(Path(out_dir).resolve()),
            "model_local_path": str(model_dir),
            "fps": int(fps),
            "use_torchao_int4": bool(use_int4),
            "format_version": "v1_scene_cam_things",
        },
        "shots": [],
    }
    _atomic_write_json(out_json, payload)

    with tempfile.TemporaryDirectory(prefix="qwen2_tmp_") as tmp_dir:
        tmp_dir_p = Path(tmp_dir)

        for i, (shot_name, shot_path) in enumerate(shots, start=1):
            log(f"[{i}/{len(shots)}] {shot_name}")
            item: Dict[str, Any] = {"name": shot_name, "input_path": str(shot_path), "status": "ok"}

            try:
                if shot_path.is_dir():
                    # Sample frames straight from the sequence — no mp4 encode (the OpenCV
                    # VideoWriter can hard-crash on some builds, esp. at high resolution).
                    log("  - Image sequence detected -> sampling frames directly (no mp4)…")
                    d = describer.describe_sequence_struct(str(shot_path))
                else:
                    d = describer.describe_video_struct(str(shot_path))
                item.update({
                    "scene_elements": d.get("scene_elements", ""),
                    "camera_movement": d.get("camera_movement", ""),
                    "things": d.get("things", []),
                    "moving_things": d.get("moving_things", []),
                    "bad_track_regions": d.get("bad_track_regions", []),
                    "foreground_occluders": d.get("foreground_occluders", []),
                    "quality_flags": d.get("quality_flags", []),
                    "depth_layers": d.get("depth_layers", {"fg": "", "mg": "", "bg": ""}),
                    "parallax": d.get("parallax", ""),
                })

            except Exception as e:
                item["status"] = "error"
                item["error"] = str(e)
                item["traceback"] = traceback.format_exc()
                log("FAILED:")
                log(str(e))

            payload["shots"].append(item)
            _atomic_write_json(out_json, payload)

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Free Qwen2.5-VL VRAM now that description is done (the Ollama LLM stays resident
    # and is unloaded later, at the masking step).
    try:
        import gc, torch
        del describer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        log("Freed Qwen2.5-VL from VRAM.")
    except Exception as e:
        log(f"Qwen VRAM free skipped ({e}).")

    log(f"Done. Wrote JSON: {out_json}")
    return out_json


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen2 Shot Describer (Offline / Single JSON Output)")

        self.in_edit = QtWidgets.QLineEdit()
        self.out_edit = QtWidgets.QLineEdit()

        self.fps_combo = QtWidgets.QComboBox()
        self.fps_combo.addItem("1 (safe)", 1)
        self.fps_combo.addItem("2 (better action)", 2)
        self.fps_combo.addItem("3 (heavy)", 3)
        self.fps_combo.setCurrentIndex(0)

        self.int4_chk = QtWidgets.QCheckBox("Use int4 (4-bit: AWQ/GPTQ weights or bitsandbytes)")
        self._int4_ok = _int4_usable()
        if not self._int4_ok:
            self.int4_chk.setChecked(False)
            self.int4_chk.setEnabled(False)
            self.int4_chk.setToolTip("Disabled: no pre-quantized weights found and bitsandbytes is not installed.")
        else:
            self.int4_chk.setChecked(True)
            self.int4_chk.setEnabled(True)
            self.int4_chk.setToolTip("Load the model in 4-bit to save VRAM (recommended for 7B on 16GB).")

        self.model_label = QtWidgets.QLabel(f"Model (local): {LOCAL_MODEL_DIR}")
        self.model_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        b_in = QtWidgets.QPushButton("Browse IN…")
        b_out = QtWidgets.QPushButton("Browse OUT…")
        b_run = QtWidgets.QPushButton("Run")
        b_run.setDefault(True)

        b_in.clicked.connect(self._pick_in)
        b_out.clicked.connect(self._pick_out)
        b_run.clicked.connect(self._run)

        form = QtWidgets.QFormLayout()
        row_in = QtWidgets.QHBoxLayout()
        row_in.addWidget(self.in_edit)
        row_in.addWidget(b_in)
        form.addRow("IN folder", row_in)

        row_out = QtWidgets.QHBoxLayout()
        row_out.addWidget(self.out_edit)
        row_out.addWidget(b_out)
        form.addRow("OUT folder", row_out)

        form.addRow("FPS preset", self.fps_combo)
        form.addRow("", self.int4_chk)
        form.addRow("", self.model_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(b_run)
        layout.addWidget(self.log)
        self._running = False

    def _append_log(self, msg: str):
        self.log.appendPlainText(msg)
        QtCore.QCoreApplication.processEvents()

    def _pick_in(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select IN folder")
        if d:
            self.in_edit.setText(d)

    def _pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select OUT folder")
        if d:
            self.out_edit.setText(d)

    def _run(self):
        if self._running:
            return
        in_dir = self.in_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not in_dir or not out_dir:
            QtWidgets.QMessageBox.warning(self, "Missing paths", "Please set both IN and OUT folder paths.")
            return

        self._running = True
        self._append_log("")

        try:
            fps = int(self.fps_combo.currentData())
            use_int4 = bool(self.int4_chk.isChecked() and self.int4_chk.isEnabled())
            out_json = run_batch(in_dir=in_dir, out_dir=out_dir, fps=fps, use_int4=use_int4, log_cb=self._append_log)
            QtWidgets.QMessageBox.information(self, "Done", f"JSON written:\n{out_json}")
        except Exception as e:
            self._append_log("FAILED (top-level):")
            self._append_log(str(e))
            self._append_log(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            self._running = False


def main():
    if PYSIDE_AVAILABLE:
        app = QtWidgets.QApplication([])
        w = MainWindow()
        w.resize(980, 560)
        w.show()
        app.exec()
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--in_dir", required=True)
        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--fps", type=int, choices=[1, 2, 3], default=1, help="FPS preset: 1, 2, or 3")
        ap.add_argument("--int4", action="store_true", help="Use int4 (torchao) if available")
        args = ap.parse_args()
        run_batch(args.in_dir, args.out_dir, args.fps, bool(args.int4), log_cb=None)


if __name__ == "__main__":
    main()
