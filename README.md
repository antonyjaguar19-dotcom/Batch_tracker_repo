# Batch Tracker

Batch Tracker is a Python + Gradio workflow app for video tracking pipelines. It brings multiple AI-assisted steps into one interface so you can go from scene analysis to tracking execution in a single run.

## AI / Models used

This project is built around a combined workflow that references these components:

- **Qwen2 / LLaMA (via Ollama)** for scene/camera reasoning and JSON planning.
- **SAM3** for mask generation/segmentation.
- **CoTracker3** for batch point/object tracking.

> Note: Model weights are not fully stored in this repo by default. The installer can download required checkpoints when URLs are configured.

## Repository layout

- `app.py` — unified app entrypoint and runtime bootstrap.
- `run_batch_tracker.py` — minimal launcher for the Gradio interface.
- `install.py` — one-shot installer for environment setup + optional model downloads.
- `requirements.txt` — Python dependency list for the app stack.
- `app/` — UI, tracking logic, and helper modules.
- `thirdparty/` — external code/assets such as CoTracker-related resources.

## Build from scratch

If you are starting from a fresh machine:

### 1) Prerequisites

- Python **3.10+** recommended.
- `git` installed.
- (Recommended) NVIDIA GPU + CUDA-compatible driver for faster tracking.
- Local **Ollama** server if using Qwen/LLaMA reasoning features.

### 2) Clone

```bash
git clone <your-repo-url>
cd Batch_tracker_repo
```

### 3) Install (recommended path)

Run the bundled installer:

```bash
python install.py
```

The installer will:

1. Create `.venv` (unless disabled).
2. Upgrade `pip`.
3. Install PyTorch + TorchVision from CUDA 12.1 index (unless disabled).
4. Install `requirements.txt`.
5. Download configured model weights.

### 4) Run the app

```bash
python run_batch_tracker.py
```

## Manual dependency install (advanced)

If you prefer full manual control:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dependency requirements

From `requirements.txt`:

- `numpy>=1.24.0`
- `opencv-python>=4.8.0`
- `gradio>=4.0.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`
- `psutil>=5.9.0`

Optional/commented dependencies may be needed depending on how your local SAM3/Ollama integrations are configured.

## Environment variables

Common runtime/install toggles:

- `BTR_SKIP_VENV=1` — install into current interpreter.
- `BTR_SKIP_TORCH=1` — skip PyTorch installation.
- `BTR_SKIP_WEIGHTS=1` — skip model weight downloads.
- `BTR_SAM3_WEIGHTS_URL` — URL for SAM3 checkpoint.
- `BTR_COTRACKER_WEIGHTS_URL` — URL for CoTracker checkpoint.
- `BTR_OLLAMA_URL` — Ollama endpoint (default: `http://localhost:11434`).

## Quick start summary

```bash
python install.py
python run_batch_tracker.py
```
