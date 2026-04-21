@echo off
setlocal

REM Change to the directory where this .bat lives so the tool works from any folder.
set "HERE=%~dp0"
cd /d "%HERE%"

REM Activate virtual environment if present (CMD-compatible)
if exist "%HERE%.venv\Scripts\activate.bat" (
    call "%HERE%.venv\Scripts\activate.bat"
) else (
    echo WARNING: .venv not found at "%HERE%.venv"
    echo Run install.py first to create the virtual environment.
)

REM Run the script
python run_batch_tracker.py

pause
