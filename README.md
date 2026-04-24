# Batch Tracker

A production-oriented local tool for **batch point tracking across multiple shots** using **CoTracker3**, with a **Gradio UI**, mask-aware seeding, per-shot scaling, and SynthEyes-friendly export.

This repository is designed for VFX / matchmove workflows where you need to process many `.mp4` shots in one pass and export track data reliably.

---

## Key Features

- **Batch processing** of multiple shots from an input folder.
- **Interactive Gradio UI** for scanning, selecting, and running jobs.
- **Mask-aware tracking** (`outside` or `inside`) with automatic mask-folder resolution.
- **Per-shot scale control** to reduce RAM/VRAM pressure on heavy shots.
- **Resource estimation** (CPU RAM / GPU VRAM risk tags) before launch.
- **3DE/SynthEyes-friendly track export** plus run logs (`track_log.txt`, `track_log.csv`).
- **One-shot installer** (`install.py`) that can create a venv, install dependencies, and fetch model weights.

---

## Repository Layout

```text
Batch_tracker_repo/
├─ app/
│  ├─ ui_gradio.py          # Main local Gradio batch UI
│  ├─ tracker_core.py       # Batch runner + filtering + mask gating
│  ├─ cotracker_engine.py   # CoTracker3 integration
│  ├─ export_3de.py         # 3DE/SynthEyes export formatting
│  ├─ video_io.py           # Video decode + scaling utilities
│  └─ video_meta.py         # Metadata probing helpers
├─ run_batch_tracker.py     # Launches the Gradio batch UI
├─ app.py                   # Unified integrated workflow UI (advanced)
├─ install.py               # One-shot setup/install script
├─ requirements.txt
└─ run_batch_tracker.bat    # Windows launcher
```

---

## Requirements

- **Python 3.10+** (recommended)
- **NVIDIA GPU** recommended for CoTracker performance
- **PyTorch + TorchVision** (CUDA wheel recommended; installer handles this)
- OS: Windows / macOS / Linux (Windows launchers included)

> Note: Actual runtime also depends on external components referenced by your workflow (for example local model weights and optional auxiliary runners).

---

## Quick Start

### 1) Clone and enter the repo

```bash
git clone <your-repo-url>
cd Batch_tracker_repo
```

### 2) Run the installer

```bash
python install.py
```

The installer can:

1. Create `.venv` (unless `BTR_SKIP_VENV=1`)
2. Upgrade pip
3. Install PyTorch/TorchVision (unless `BTR_SKIP_TORCH=1`)
4. Install `requirements.txt`
5. Download weights (unless `BTR_SKIP_WEIGHTS=1`)

### 3) Launch the app

```bash
python run_batch_tracker.py
```

On Windows, you can also use:

- `run_batch_tracker.bat`
- `launch_integrated_gradio.bat` (runs `app.py`)

---

## Input / Output Conventions

### Video Input

- Put source shots as `.mp4` files in your selected **Input Folder**.

### Mask Input (optional but supported)

Expected structure:

```text
<MASK_ROOT>/
  Shot_01/
    masks/
      0001.png
      0002.png
  Shot_02/
    masks/
      ...
```

Fallback is also supported:

```text
<MASK_ROOT>/Shot_01/*.png
```

### Output

The runner writes tracking outputs to your selected output folder and appends run summaries to:

- `track_log.txt`
- `track_log.csv`

---

## Environment Variables

Useful configuration knobs:

- `BTR_SKIP_VENV=1` — install into current interpreter
- `BTR_SKIP_TORCH=1` — skip PyTorch install
- `BTR_SKIP_WEIGHTS=1` — skip weight downloads
- `BTR_SAM3_WEIGHTS_URL=<url>` — URL for SAM3 weights
- `BTR_COTRACKER_WEIGHTS_URL=<url>` — URL override for CoTracker weights
- `BTR_PROJECT_ROOT=<path>` — override project root resolution in integrated app
- `BTR_OLLAMA_URL=<url>` — override Ollama endpoint used by integrated workflow
- `BTR_SAM3_WEIGHTS=<path>` — override integrated SAM3 weight path

---

## Typical Workflow

1. Open UI (`python run_batch_tracker.py`)
2. Select input/output/mask folders
3. Click **Scan** to build shot table and inspect CPU/GPU risk
4. Adjust per-shot scale if needed
5. Start tracking and monitor live logs
6. Validate exported tracks in downstream matchmove pipeline

---

## Troubleshooting

- **Import errors for tracker/SAM modules**: ensure you run from repo root and dependencies are installed in the active interpreter.
- **OOM / slow runs**: lower per-shot scale, reduce grid size, and avoid high-resolution batches in one pass.
- **No masks found**: verify folder names match shot names and PNG files are inside `masks/` (or shot root fallback).
- **Windows launcher can’t find venv**: run `python install.py` first to create `.venv`.

---

## Development Notes

- Main batch UI entrypoint: `run_batch_tracker.py` -> `app/ui_gradio.py`.
- Advanced integrated pipeline UI: `app.py`.
- Keep pathing relative to repo root to preserve portability across workstations.

---

## License

No license file is currently included in this repository. Add a `LICENSE` file to define usage terms.
