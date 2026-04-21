#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Tracker — one-shot installer.

After cloning the repo, run this script once from the repo root:

    python install.py

It will:
  1. Create the .venv virtual environment in the repo root (if missing).
  2. Upgrade pip inside that venv.
  3. Install PyTorch + TorchVision from the CUDA 12.1 wheel index.
  4. Install the rest of requirements.txt.
  5. Download the model weights (.safetensors / .pt) into the expected folders.

Nothing in this script is tied to a specific machine or user folder — the repo
can live anywhere on any OS (Windows / macOS / Linux).

Weight download URLs can be overridden via environment variables:

    BTR_SAM3_WEIGHTS_URL        (default: unset — you must set this OR drop the
                                 file into SAM3/weights/ manually)
    BTR_COTRACKER_WEIGHTS_URL   (default: Facebook Research CoTracker3 offline)

You can also skip either step with:

    BTR_SKIP_WEIGHTS=1          (skip all weight downloads)
    BTR_SKIP_TORCH=1            (skip PyTorch install — useful if already done)
    BTR_SKIP_VENV=1             (install into the current interpreter, no venv)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

# Default CoTracker3 offline checkpoint published by facebookresearch.
DEFAULT_COTRACKER_URL = (
    "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
)

# Weight targets: (env var for URL, destination path, friendly name)
WEIGHT_TARGETS = [
    (
        "BTR_SAM3_WEIGHTS_URL",
        REPO_ROOT / "SAM3" / "weights" / "sam3.pt",
        "SAM3 weights (sam3.pt / .safetensors)",
    ),
    (
        "BTR_COTRACKER_WEIGHTS_URL",
        REPO_ROOT / "thirdparty" / "co-tracker-main" / "checkpoints" / "scaled_offline.pth",
        "CoTracker3 offline checkpoint (scaled_offline.pth)",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _venv_python(venv_dir: Path) -> Path:
    if _is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: Iterable[str], cwd: Path | None = None) -> None:
    cmd_list = [str(c) for c in cmd]
    print(f"\n$ {' '.join(cmd_list)}")
    subprocess.check_call(cmd_list, cwd=str(cwd) if cwd else None)


def _banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def ensure_venv() -> Path:
    """Create .venv if missing and return the path to its python."""
    if os.environ.get("BTR_SKIP_VENV") == "1":
        print("BTR_SKIP_VENV=1 — using current interpreter, skipping venv creation.")
        return Path(sys.executable)

    py = _venv_python(VENV_DIR)
    if py.exists():
        print(f"Found existing venv: {VENV_DIR}")
        return py

    _banner(f"Creating virtual environment at {VENV_DIR}")
    # with_pip=True ensures pip is bootstrapped inside the venv.
    venv.EnvBuilder(with_pip=True, upgrade_deps=False, clear=False).create(str(VENV_DIR))

    if not py.exists():
        raise RuntimeError(f"venv creation failed — {py} not found")
    print(f"Created venv: {VENV_DIR}")
    return py


def upgrade_pip(py: Path) -> None:
    _banner("Upgrading pip")
    _run([py, "-m", "pip", "install", "--upgrade", "pip"])


def install_torch(py: Path) -> None:
    if os.environ.get("BTR_SKIP_TORCH") == "1":
        print("BTR_SKIP_TORCH=1 — skipping PyTorch install.")
        return
    _banner("Installing PyTorch + TorchVision (CUDA 12.1 wheel, ~2.5 GB)")
    _run([
        py, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])


def install_requirements(py: Path) -> None:
    if not REQUIREMENTS.exists():
        print(f"WARNING: {REQUIREMENTS} not found — skipping pip install -r.")
        return
    _banner(f"Installing {REQUIREMENTS.name}")
    _run([py, "-m", "pip", "install", "-r", str(REQUIREMENTS)])


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> downloading to {dest}")

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100.0 / total_size
        mb_done = done / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r     {pct:6.2f}%  {mb_done:8.1f} / {mb_total:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
    sys.stdout.write("\n")
    shutil.move(str(tmp), str(dest))


def download_weights() -> None:
    if os.environ.get("BTR_SKIP_WEIGHTS") == "1":
        print("BTR_SKIP_WEIGHTS=1 — skipping weight downloads.")
        return
    _banner("Downloading model weights (.pt / .safetensors)")

    for env_key, dest, label in WEIGHT_TARGETS:
        url = os.environ.get(env_key)
        if url is None and env_key == "BTR_COTRACKER_WEIGHTS_URL":
            url = DEFAULT_COTRACKER_URL

        print(f"\n- {label}")
        print(f"  dest: {dest}")
        if dest.exists() and dest.stat().st_size > 0:
            print("  already present — skipping.")
            continue
        if not url:
            print(f"  no URL configured (set {env_key}=... to auto-download).")
            print(f"  drop the file manually into: {dest.parent}")
            continue
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  ERROR downloading from {url}: {e}")
            print(f"  please download it manually into {dest}")


def summary(py: Path) -> None:
    _banner("INSTALL COMPLETE")
    print(f"Repo root : {REPO_ROOT}")
    print(f"Python    : {py}")
    print("\nNext steps:")
    if _is_windows():
        print(r"  .venv\Scripts\activate")
    else:
        print("  source .venv/bin/activate")
    print("  python run_batch_tracker.py")
    print("\nOr on Windows just double-click: run_batch_tracker.bat")


def main() -> int:
    if not REPO_ROOT.exists():
        print(f"ERROR: repo root {REPO_ROOT} does not exist", file=sys.stderr)
        return 1

    print(f"Batch Tracker installer — repo root: {REPO_ROOT}")

    py = ensure_venv()
    upgrade_pip(py)
    install_torch(py)
    install_requirements(py)
    download_weights()
    summary(py)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
