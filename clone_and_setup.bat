@echo off
setlocal EnableDelayedExpansion
REM ============================================================================
REM  Batch Tracker - one-shot clone + full setup (fresh Windows machine)
REM
REM  Run this .bat on a bare machine. It:
REM    1. git clone the repo
REM    2. bootstraps the portable Python 3.11.9 runtime + pip installs (cu121)
REM    3. downloads the model weights (Qwen2.5-VL, SAM3, TAPNext++)
REM    4. pulls the Ollama LLM (if Ollama is installed)
REM
REM  The weights + runtime are NOT in git (too big) - this script fetches them.
REM
REM  PREREQS you must install yourself first (script checks + warns):
REM    - Git            https://git-scm.com
REM    - NVIDIA driver new enough for CUDA 12.1 (no CUDA Toolkit needed)
REM    - Ollama         https://ollama.com    (optional, for the LLaMA stage)
REM ============================================================================

REM ---- CONFIG (edit if needed) ----------------------------------------------
set "REPO_URL=https://github.com/antonyjaguar19-dotcom/Batch_tracker_repo.git"
set "TARGET=%~dp0Batch_tracker_repo"
set "PY_VER=3.11.9"
set "PY_ZIP_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
REM Qwen model: 7B (default, ~16GB, matches the int4/bitsandbytes loader).
REM For a lighter box switch to: Qwen/Qwen2.5-VL-3B-Instruct  (~7GB)
set "QWEN_REPO=Qwen/Qwen2.5-VL-7B-Instruct"
REM ---------------------------------------------------------------------------

echo(
echo === Batch Tracker setup ===
echo Target: %TARGET%
echo(

REM ---- 0. prereq: git -------------------------------------------------------
where git >nul 2>&1 || (
  echo [ERROR] git not found on PATH. Install from https://git-scm.com and re-run.
  goto :fail
)

REM ---- 1. clone -------------------------------------------------------------
if exist "%TARGET%\.git" (
  echo [1/5] repo already cloned - pulling latest...
  git -C "%TARGET%" pull --ff-only || echo [warn] pull failed, continuing with existing checkout.
) else (
  echo [1/5] cloning %REPO_URL% ...
  git clone "%REPO_URL%" "%TARGET%" || goto :fail
)

set "ROOT=%TARGET%"
set "PY_DIR=%ROOT%\runtime\python311"
set "PY=%PY_DIR%\python.exe"

REM ---- 2. portable Python 3.11.9 embeddable --------------------------------
if exist "%PY%" (
  echo [2/5] runtime python already present - skipping bootstrap.
) else (
  echo [2/5] bootstrapping portable Python %PY_VER% ...
  mkdir "%PY_DIR%" 2>nul
  set "PY_ZIP=%ROOT%\runtime\python-embed.zip"
  curl -L -o "!PY_ZIP!" "%PY_ZIP_URL%" || goto :fail
  powershell -NoProfile -Command "Expand-Archive -Force '!PY_ZIP!' '%PY_DIR%'" || goto :fail
  del "!PY_ZIP!" 2>nul

  REM Enable `import site` so pip works in the embeddable build.
  REM (uncomment the '#import site' line inside python311._pth)
  powershell -NoProfile -Command "$f=Get-ChildItem '%PY_DIR%\*._pth' | Select-Object -First 1; (Get-Content $f.FullName) -replace '#\s*import site','import site' | Set-Content $f.FullName" || goto :fail

  echo       installing pip ...
  curl -L -o "%ROOT%\runtime\get-pip.py" "%GETPIP_URL%" || goto :fail
  "%PY%" "%ROOT%\runtime\get-pip.py" --no-warn-script-location || goto :fail
  del "%ROOT%\runtime\get-pip.py" 2>nul
)

REM Keep all pip cache/temp inside the project (off C:), per repo constraint.
set "PIP_CACHE_DIR=%ROOT%\runtime\pipcache"
set "TMP=%ROOT%\runtime\tmp"
set "TEMP=%ROOT%\runtime\tmp"
mkdir "%TMP%" 2>nul

echo       installing requirements (torch cu121 + all deps, this is large)...
"%PY%" -m pip install --upgrade pip --no-warn-script-location || goto :fail
"%PY%" -m pip install -r "%ROOT%\requirements.txt" --no-warn-script-location || goto :fail

REM ---- 3. weights ----------------------------------------------------------
REM  NOTE: we use the huggingface_hub Python API, NOT the `hf` CLI - the CLI
REM  imports the `venv` stdlib module, which the embeddable interpreter omits.
echo [3/5] downloading model weights ...

echo       - TAPNext++ repo (google-deepmind/tapnet, Apache-2.0) + checkpoint (public)
if exist "%ROOT%\pipeline\tapnext-main\.git" (
  git -C "%ROOT%\pipeline\tapnext-main" pull --ff-only || echo [warn] tapnet pull failed, using existing checkout.
) else (
  git clone https://github.com/google-deepmind/tapnet "%ROOT%\pipeline\tapnext-main" || goto :fail
)
mkdir "%ROOT%\pipeline\tapnext-main\checkpoints" 2>nul
curl -L -o "%ROOT%\pipeline\tapnext-main\checkpoints\tapnextpp_ckpt.pt" "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt" || goto :fail

echo       - Qwen2.5-VL  (%QWEN_REPO%, large)
for %%R in ("%QWEN_REPO%") do set "QWEN_NAME=%%~nxR"
"%PY%" -c "from huggingface_hub import snapshot_download as s; s(repo_id='%QWEN_REPO%', local_dir=r'%ROOT%\pipeline\qwen\models\!QWEN_NAME!')" || goto :fail

echo       - SAM3 sam3.pt  (GATED repo facebook/sam3 - needs a Hugging Face token)
if not defined HF_TOKEN (
  echo         facebook/sam3 is gated. Accept its license at https://huggingface.co/facebook/sam3
  echo         then paste a token from https://huggingface.co/settings/tokens
  set /p "HF_TOKEN=        HF token: "
)
"%PY%" -c "from huggingface_hub import hf_hub_download as d; d(repo_id='facebook/sam3', filename='sam3.pt', local_dir=r'%ROOT%\pipeline\sam3\weights')" || goto :fail

REM ---- 4. Ollama LLM -------------------------------------------------------
echo [4/5] Ollama LLM (llama3.1:8b) ...
where ollama >nul 2>&1 && (
  ollama pull llama3.1:8b || echo [warn] ollama pull failed - pull it manually later.
) || (
  echo [warn] Ollama not installed. Install from https://ollama.com then run: ollama pull llama3.1:8b
)

REM ---- 5. done -------------------------------------------------------------
echo [5/5] done.
echo(
echo === Setup complete ===
echo Launch the app by double-clicking:  "%ROOT%\app\BotBatchTracker.bat"
echo (opens the NiceGUI UI at http://localhost:8080)
echo(
goto :eof

:fail
echo(
echo [FAILED] setup aborted - see the error above.
exit /b 1
