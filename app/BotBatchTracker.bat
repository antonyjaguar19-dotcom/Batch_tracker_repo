@echo off
setlocal

REM This BAT lives in app\ ; HERE = repo root (parent of app\).
set "HERE=%~dp0..\"

REM cwd MUST be repo root: app\ contains app.py, which would shadow the `app` package.
cd /d "%HERE%"

set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%HERE%runtime\python311\python.exe"

REM Keep pip cache/temp inside the project folder, not on C:.
set "PIP_CACHE_DIR=%HERE%runtime\pipcache"
set "TMP=%HERE%runtime\tmp"
set "TEMP=%HERE%runtime\tmp"

REM Tcl/Tk libs for tkinter (browse-folder dialogs).
set "TCL_LIBRARY=%HERE%runtime\python311\tcl\tcl8.6"
set "TK_LIBRARY=%HERE%runtime\python311\tcl\tk8.6"

REM SAM3 weights default.
set "BTR_SAM3_WEIGHTS=%HERE%pipeline\sam3\weights\sam3.pt"

REM SynthEyes tracking backend (pre-fills the UI Settings each launch).
REM SynthEyes is an external app (like Ollama) and may live on C:.
set "BTR_SYNTHEYES_EXE=C:\Program Files\BorisFX\SynthEyes 2026\SynthEyes64.exe"
set "BTR_SYPY3_DIR=D:\Jefrin\Assets\SyPy3"
REM set "BTR_TDE4_EXE=C:\Program Files\3DE4\bin\3DE4.exe"
REM set "BTR_SE_PORT=2222"
REM set "BTR_SE_PIN=listen"

REM Free the UI port first so a stale/duplicate instance never blocks this launch
REM ("[Errno 10048] only one usage of each socket address"). Kills whatever is
REM LISTENING on 8080 - on this single-user box that is always the old Batch Tracker.
set "BTR_PORT=8080"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%BTR_PORT% " ^| findstr LISTENING') do (
  echo Port %BTR_PORT% busy - stopping old instance (PID %%P) ...
  taskkill /F /PID %%P >nul 2>&1
)

if exist "%PY%" (
  echo Using python: "%PY%"
  "%PY%" "%HERE%app\app_nicegui.py"
) else (
  echo ERROR: no python found at "%HERE%.venv" or "%HERE%runtime\python311"
  pause
  exit /b 1
)

pause
