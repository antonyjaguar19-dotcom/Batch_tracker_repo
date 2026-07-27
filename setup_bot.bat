@echo off
setlocal EnableDelayedExpansion
REM ============================================================================
REM  Batch Tracker - fetch the heavy bits (runtime + pipeline weights)
REM
REM  Drop this .bat in the bot root (the folder holding app\ and pipeline\) and
REM  run it. It creates the folders and downloads everything that is NOT in git:
REM    1. portable Python 3.11.9 runtime + pip installs (torch cu121 + deps)
REM    2. TAPNext++ code repo + checkpoint          -> pipeline\tapnext-main
REM    3. Qwen2.5-VL weights                         -> pipeline\qwen\models
REM    4. SAM3 sam3.pt (GATED, needs a HF token)     -> pipeline\sam3\weights
REM
REM  Idempotent: anything already present is skipped, so it is safe to re-run.
REM
REM  PREREQS you install yourself first:
REM    - Git            https://git-scm.com
REM    - NVIDIA driver new enough for CUDA 12.1 (no CUDA Toolkit needed)
REM    - A Hugging Face account (facebook/sam3 is gated - accept its license)
REM
REM  SynthEyes (default tracker) is a licensed app install - not fetched here.
REM  Ollama/LLaMA is NOT used anymore (decisioning is Qwen-only) - not fetched.
REM ============================================================================

REM ---- CONFIG (edit if needed) ----------------------------------------------
set "PY_VER=3.11.9"
set "PY_ZIP_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
REM Qwen: 7B (default, ~16GB, int4/bitsandbytes loader). Light box -> Qwen/Qwen2.5-VL-3B-Instruct (~7GB)
set "QWEN_REPO=Qwen/Qwen2.5-VL-7B-Instruct"
set "TAPNET_REPO=https://github.com/google-deepmind/tapnet"
set "TAPNEXT_CKPT_URL=https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"
REM ---------------------------------------------------------------------------

REM ROOT = the folder this .bat sits in (strip trailing backslash).
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY_DIR=%ROOT%\runtime\python311"
set "PY=%PY_DIR%\python.exe"

echo(
echo === Batch Tracker setup ===
echo Root: %ROOT%
echo(

REM ---- prereq: git ----------------------------------------------------------
where git >nul 2>&1 || (
  echo [ERROR] git not found on PATH. Install from https://git-scm.com and re-run.
  goto :fail
)
where curl >nul 2>&1 || (
  echo [ERROR] curl not found on PATH ^(ships with Windows 10+^). Cannot download.
  goto :fail
)

REM ---- 1. portable Python 3.11.9 embeddable + pip + requirements ------------
if exist "%PY%" (
  echo [1/4] runtime python already present - skipping bootstrap.
) else (
  echo [1/4] bootstrapping portable Python %PY_VER% ...
  mkdir "%PY_DIR%" 2>nul
  set "PY_ZIP=%ROOT%\runtime\python-embed.zip"
  curl -L -o "!PY_ZIP!" "%PY_ZIP_URL%" || goto :fail
  powershell -NoProfile -Command "Expand-Archive -Force '!PY_ZIP!' '%PY_DIR%'" || goto :fail
  del "!PY_ZIP!" 2>nul
  REM Enable `import site` so pip works in the embeddable build.
  powershell -NoProfile -Command "$f=Get-ChildItem '%PY_DIR%\*._pth' | Select-Object -First 1; (Get-Content $f.FullName) -replace '#\s*import site','import site' | Set-Content $f.FullName" || goto :fail
  echo       installing pip ...
  curl -L -o "%ROOT%\runtime\get-pip.py" "%GETPIP_URL%" || goto :fail
  "%PY%" "%ROOT%\runtime\get-pip.py" --no-warn-script-location || goto :fail
  del "%ROOT%\runtime\get-pip.py" 2>nul
)

REM Keep pip cache/temp inside the project (off C:).
set "PIP_CACHE_DIR=%ROOT%\runtime\pipcache"
set "TMP=%ROOT%\runtime\tmp"
set "TEMP=%ROOT%\runtime\tmp"
mkdir "%TMP%" 2>nul

echo       installing requirements (torch cu121 + all deps, large)...
"%PY%" -m pip install --upgrade pip --no-warn-script-location || goto :fail
"%PY%" -m pip install -r "%ROOT%\requirements.txt" --no-warn-script-location || goto :fail

REM ---- 2. TAPNext++ repo + checkpoint --------------------------------------
echo [2/4] TAPNext++ (google-deepmind/tapnet, Apache-2.0) ...
if exist "%ROOT%\pipeline\tapnext-main\.git" (
  echo       repo present - pulling latest ...
  git -C "%ROOT%\pipeline\tapnext-main" pull --ff-only || echo [warn] tapnet pull failed, using existing checkout.
) else (
  git clone %TAPNET_REPO% "%ROOT%\pipeline\tapnext-main" || goto :fail
)
mkdir "%ROOT%\pipeline\tapnext-main\checkpoints" 2>nul
if exist "%ROOT%\pipeline\tapnext-main\checkpoints\tapnextpp_ckpt.pt" (
  echo       checkpoint present - skipping.
) else (
  curl -L -o "%ROOT%\pipeline\tapnext-main\checkpoints\tapnextpp_ckpt.pt" "%TAPNEXT_CKPT_URL%" || goto :fail
)

REM ---- 3. Qwen2.5-VL weights ----------------------------------------------
for %%R in ("%QWEN_REPO%") do set "QWEN_NAME=%%~nxR"
if exist "%ROOT%\pipeline\qwen\models\!QWEN_NAME!\config.json" (
  echo [3/4] Qwen model already present ^(!QWEN_NAME!^) - skipping.
) else (
  echo [3/4] downloading Qwen2.5-VL (%QWEN_REPO%, large) ...
  "%PY%" -c "from huggingface_hub import snapshot_download as s; s(repo_id='%QWEN_REPO%', local_dir=r'%ROOT%\pipeline\qwen\models\!QWEN_NAME!')" || goto :fail
)

REM ---- 4. SAM3 sam3.pt (GATED) --------------------------------------------
if exist "%ROOT%\pipeline\sam3\weights\sam3.pt" (
  echo [4/4] SAM3 sam3.pt already present - skipping.
) else (
  echo [4/4] SAM3 sam3.pt ^(GATED repo facebook/sam3 - needs a Hugging Face token^) ...
  if not defined HF_TOKEN (
    echo       facebook/sam3 is gated. Accept its license at https://huggingface.co/facebook/sam3
    echo       then paste a token from https://huggingface.co/settings/tokens
    set /p "HF_TOKEN=      HF token: "
  )
  "%PY%" -c "from huggingface_hub import hf_hub_download as d; d(repo_id='facebook/sam3', filename='sam3.pt', local_dir=r'%ROOT%\pipeline\sam3\weights')" || goto :fail
)

echo(
echo === Setup complete ===
echo Launch the app:  "%ROOT%\app\BotBatchTracker.bat"   ^(NiceGUI UI at http://localhost:8080^)
echo(
goto :eof

:fail
echo(
echo [FAILED] setup aborted - see the error above.
exit /b 1
