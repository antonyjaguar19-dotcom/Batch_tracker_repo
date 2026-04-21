# 🎯 Batch Tracker

Batch Tracker is a powerful Python + Gradio workflow app designed for video tracking pipelines. It brings multiple AI-assisted steps into one seamless interface, allowing you to go from scene analysis to tracking execution in a single run.

## ✨ Features

- **Unified Interface:** Gradio-based UI to manage your entire tracking workflow.
- **Advanced Tracking:** Utilizes CoTracker3 with support for Bidirectional Tracking, Half-Precision (FP16) for low VRAM usage, and Feature Seeding.
- **Resource Estimation:** Real-time CPU RAM and GPU VRAM footprint estimation to prevent out-of-memory crashes before you start tracking.
- **Mask Support:** Option to track *inside* or *outside* of designated mask regions.
- **3D Equalizer / SynthEyes Ready:** Includes toggles like "Flip Y" to seamlessly export tracking data to standard VFX software.

## 🧠 AI / Models Used

This project is built around a combined workflow that references these components:

- **Qwen2 / LLaMA (via Ollama):** For scene/camera reasoning and JSON planning.
- **SAM3:** For high-quality mask generation and segmentation.
- **CoTracker3:** For batch point and object tracking.

> **Note:** Model weights are not fully stored in this repo by default. The bundled installer can download required checkpoints when the appropriate environment variables/URLs are configured.

## 📂 Repository Layout

- `app.py` — Unified app entrypoint and runtime bootstrap.
- `run_batch_tracker.py` — Minimal launcher for the Gradio interface.
- `install.py` — One-shot installer for environment setup + optional model downloads.
- `requirements.txt` — Python dependency list for the app stack.
- `app/` — Core UI, tracking logic, and helper modules.
- `thirdparty/` — External code and assets (e.g., CoTracker-related resources).

## 🚀 Quick Start Summary

If you just want to get up and running immediately, use the bundled installer:

```bash
python install.py
python run_batch_tracker.py
```

## 🛠️ Build from Scratch (Recommended Path)

If you are starting from a fresh machine, follow these steps:

### 1. Prerequisites
- Python **3.10+** recommended.
- `git` installed.
- (Recommended) NVIDIA GPU + CUDA-compatible driver for faster tracking.
- Local **Ollama** server if using Qwen/LLaMA reasoning features.

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd Batch_tracker_repo
```

### 3. Run the Installer
```bash
python install.py
```
*The installer automatically handles creating a `.venv`, upgrading `pip`, installing PyTorch (CUDA 12.1), installing dependencies, and downloading model weights.*

### 4. Launch the App
```bash
python run_batch_tracker.py
```
*(Alternatively, Windows users can double-click the `launch_integrated_gradio.bat` file!)*

## ⚙️ Manual Dependency Install (Advanced)

If you prefer full manual control over your environment:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install core packages
pip install --upgrade pip
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install remaining requirements
pip install -r requirements.txt
```

## 🌍 Environment Variables

You can customize the runtime and install process using these environment variables:

- `BTR_SKIP_VENV=1` — Install into the current interpreter instead of making a virtual environment.
- `BTR_SKIP_TORCH=1` — Skip PyTorch installation.
- `BTR_SKIP_WEIGHTS=1` — Skip model weight downloads.
- `BTR_SAM3_WEIGHTS_URL` — Custom URL for the SAM3 checkpoint.
- `BTR_COTRACKER_WEIGHTS_URL` — Custom URL for the CoTracker checkpoint.
- `BTR_OLLAMA_URL` — Ollama endpoint (Defaults to `http://localhost:11434`).
```
