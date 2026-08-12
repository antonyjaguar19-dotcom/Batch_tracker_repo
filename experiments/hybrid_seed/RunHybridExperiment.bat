@echo off
setlocal enabledelayedexpansion

REM ===================================================================================
REM  Hybrid tracking experiment: TAPNext picks the features, SynthEyes tracks them.
REM
REM  Run this on the LICENSED (Pro) SynthEyes machine. It answers, in order:
REM    1. is SynthEyes licensed (a Demo caps tracking at ~10 frames and fakes the rest)
REM    2. can a tracker be created at a mid-shot frame  -> enables staggered seeding
REM    3. can a live tracker be re-keyed mid-shot        -> enables re-acquisition
REM    4-7. full runs, a coordinate sanity check, videos, and scoring
REM
REM  Usage:
REM      RunHybridExperiment.bat
REM      RunHybridExperiment.bat "D:\path\to\SHOT.mp4"
REM      RunHybridExperiment.bat "\\liv1\shows\SHOW\SH010\in\plates\v001"    (frames dir)
REM      RunHybridExperiment.bat "D:\path\to\SHOT.mp4" refs\SH004_lk         (with scoring)
REM
REM  Results land in experiments\hybrid_seed\out\ (gitignored - they stay on this box).
REM ===================================================================================

REM This BAT lives in experiments\hybrid_seed\ ; HERE = repo root.
set "HERE=%~dp0..\..\"

REM cwd MUST be repo root: app\ contains app.py, which would shadow the `app` package.
cd /d "%HERE%"

set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%HERE%runtime\python311\python.exe"
if not exist "%PY%" (
  echo ERROR: no Python found at .venv\Scripts or runtime\python311.
  exit /b 1
)

REM Keep pip cache/temp inside the project folder, not on C:.
set "PIP_CACHE_DIR=%HERE%runtime\pipcache"
set "TMP=%HERE%runtime\tmp"
set "TEMP=%HERE%runtime\tmp"
if not exist "%TMP%" mkdir "%TMP%" >nul 2>&1

REM --- EDIT THESE TWO IF PROD INSTALLS SYNTHEYES SOMEWHERE ELSE ----------------------
if not defined BTR_SYNTHEYES_EXE set "BTR_SYNTHEYES_EXE=C:\Program Files\BorisFX\SynthEyes 2026\SynthEyes64.exe"
if not defined BTR_SYPY3_DIR     set "BTR_SYPY3_DIR=D:\Jefrin\Assets\SyPy3"
REM -----------------------------------------------------------------------------------
if not defined BTR_SE_PORT set "BTR_SE_PORT=2222"
if not defined BTR_SE_PIN  set "BTR_SE_PIN=listen"

set "EXP=experiments\hybrid_seed"
set "OUT=%HERE%%EXP%\out"

REM Plate: first argument, else the dev-box default.
set "PLATE=%~1"
if "%PLATE%"=="" set "PLATE=D:\Jefrin\IN\SH004.mp4"
REM Optional reference folder for scoring (refs\<name> with manual.txt + refs.json).
set "REFS=%~2"

set "SEEDS=400"

echo.
echo ===================================================================
echo  Hybrid tracking experiment
echo  plate   : %PLATE%
echo  python  : %PY%
echo  synth   : %BTR_SYNTHEYES_EXE%
echo  output  : %OUT%
echo ===================================================================

if not exist "%BTR_SYNTHEYES_EXE%" (
  echo.
  echo ERROR: SynthEyes not found at:
  echo   %BTR_SYNTHEYES_EXE%
  echo Set BTR_SYNTHEYES_EXE, or edit the marked line near the top of this file.
  exit /b 1
)

REM ---------------------------------------------------------------- 1. licence check
echo.
echo --- [1/7] Licence check -------------------------------------------
"%PY%" "%EXP%\check_licence.py"
if errorlevel 2 (
  echo.
  echo STOPPED: this SynthEyes is a Demo build. It tracks about 10 frames per tracker
  echo and holds a frozen coordinate for the rest, so every number below would describe
  echo the licence rather than the tracker. Fix the licence, then re-run.
  exit /b 2
)
if errorlevel 1 (
  echo STOPPED: could not talk to SynthEyes. Check the exe path and that no modal
  echo dialog is open in an existing SynthEyes window.
  exit /b 3
)

REM ------------------------------------------------------ 2. mid-shot creation probe
echo.
echo --- [2/7] Mid-shot tracker creation -------------------------------
echo     (if this PASSES, staggered seeding can be turned on)
set "STAGGER=1"
"%PY%" "%EXP%\probes.py" midshot --plate "%PLATE%"
if errorlevel 1 (
  echo     -^> keeping seeds on frame 0 only.
) else (
  set "STAGGER=4"
  echo     -^> staggered seeding ENABLED for the runs below.
)

REM ---------------------------------------------------------- 3. re-acquisition probe
echo.
echo --- [3/7] Re-acquisition (replant the same seed) ------------------
set "REACQ="
"%PY%" "%EXP%\probes.py" reacquire --plate "%PLATE%"
if errorlevel 1 (
  echo     -^> re-acquisition NOT available on this build; step 5 will be skipped.
) else (
  set "REACQ=1"
  echo     -^> re-acquisition available.
)

REM ------------------------------------------------------------- 4. baseline hybrid
echo.
echo --- [4/7] Hybrid run: baseline ------------------------------------
"%PY%" "%EXP%\run_hybrid.py" --plate "%PLATE%" --seeds %SEEDS% --stagger !STAGGER! --tag hybrid
if errorlevel 1 (
  echo ERROR: the baseline hybrid run failed. Stopping.
  exit /b 4
)

REM -------------------------------------------------------- 5. hybrid + reacquisition
echo.
echo --- [5/7] Hybrid run: with re-acquisition -------------------------
if defined REACQ (
  "%PY%" "%EXP%\run_hybrid.py" --plate "%PLATE%" --seeds %SEEDS% --stagger !STAGGER! --tag hybrid_reacq --reacquire
) else (
  echo     skipped - step 3 said re-acquisition is not available on this build.
)

REM ------------------------------------------------------------ 6. round-trip check
echo.
echo --- [6/7] Seed round-trip check -----------------------------------
echo     (proves the injected seeds landed exactly where the seeder asked)
"%PY%" "%EXP%\check_roundtrip.py" --plate "%PLATE%" --seeds %SEEDS% --stagger !STAGGER!

REM ------------------------------------------------------------- 7. videos + scoring
echo.
echo --- [7/7] Overlay videos ------------------------------------------
"%PY%" "%EXP%\render_overlay.py" --plate "%PLATE%" --seeds %SEEDS%
if defined REACQ (
  for %%F in ("%PLATE%") do set "STEM=%%~nF"
  "%PY%" "%EXP%\render_overlay.py" --plate "%PLATE%" --seeds %SEEDS% --tracks "%OUT%\!STEM!__hybrid_reacq.txt"
)

if not "%REFS%"=="" (
  echo.
  echo --- Scoring against %REFS% ---
  for %%F in ("%PLATE%") do set "STEM=%%~nF"
  "%PY%" tools\eval_refs.py "%REFS%" --bot "%OUT%\!STEM!__hybrid.txt"
  if defined REACQ (
    "%PY%" tools\eval_refs.py "%REFS%" --bot "%OUT%\!STEM!__hybrid_reacq.txt" --baseline "%OUT%\!STEM!__hybrid.txt"
  )
) else (
  echo.
  echo --- Scoring skipped: pass a reference folder as the 2nd argument to enable it ---
  echo     e.g.  RunHybridExperiment.bat "%PLATE%" refs\SH004_lk
)

echo.
echo ===================================================================
echo  Done. Exports, .szl scripts and overlay .mp4 files are in:
echo    %OUT%
echo.
echo  What to look at:
echo    * step 1 must say LICENSED
echo    * steps 2 and 3 PASS/FAIL decide what the wired-in mode can do
echo    * step 6 must say PASS (seeds landed where asked)
echo    * in the videos, points should keep moving for the WHOLE shot.
echo      A plate that goes grey around frame 10 means the licence is not active.
echo ===================================================================
endlocal
