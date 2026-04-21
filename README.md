# Batch Tracker Repo

Batch Tracker is a Python-based Gradio application for running a unified, multi-step tracking workflow. It combines model-assisted scene analysis, mask generation, and batch tracking execution from a single UI.

## What’s in this repository

- `app.py` — main unified entrypoint with path bootstrapping and module loading.
- `run_batch_tracker.py` — minimal launcher that starts the Gradio UI.
- `install.py` — one-shot installer that creates a virtual environment, installs dependencies, and optionally downloads model weights.
- `app/` — application modules for UI, tracking core logic, and utilities.

## Quick start

1. **Install dependencies**# Batch Tracker Repo

Batch Tracker is a Python-based Gradio application for running a unified, multi-step tracking workflow. It combines model-assisted scene analysis, mask generation, and batch tracking execution from a single UI.

## What’s in this repository

- `app.py` — main unified entrypoint with path bootstrapping and module loading.
- `run_batch_tracker.py` — minimal launcher that starts the Gradio UI.
- `install.py` — one-shot installer that creates a virtual environment, installs dependencies, and optionally downloads model weights.
- `app/` — application modules for UI, tracking core logic, and utilities.

## Quick start

1. **Install dependencies**

   ```bash
   python install.py
   ```

2. **Run the app**

   ```bash
   python run_batch_tracker.py
   ```

## Notes

- The installer supports environment-variable toggles such as `BTR_SKIP_VENV`, `BTR_SKIP_TORCH`, and `BTR_SKIP_WEIGHTS`.
- Weight download URLs can be configured with environment variables, including `BTR_SAM3_WEIGHTS_URL` and `BTR_COTRACKER_WEIGHTS_URL`.
- The app defaults to Ollama at `http://localhost:11434` (override with `BTR_OLLAMA_URL`).

   ```bash
   python install.py
   ```

2. **Run the app**

   ```bash
   python run_batch_tracker.py
   ```

## Notes

- The installer supports environment-variable toggles such as `BTR_SKIP_VENV`, `BTR_SKIP_TORCH`, and `BTR_SKIP_WEIGHTS`.
- Weight download URLs can be configured with environment variables, including `BTR_SAM3_WEIGHTS_URL` and `BTR_COTRACKER_WEIGHTS_URL`.
- The app defaults to Ollama at `http://localhost:11434` (override with `BTR_OLLAMA_URL`).
