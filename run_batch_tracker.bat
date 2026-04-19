@echo off
setlocal

REM Change to project directory
cd /d "D:\Jefrin\BTr\batch_tracker_v001_starter"

REM Activate virtual environment (CMD-compatible)
call ".venv\Scripts\activate.bat"

REM Run the script
python run_batch_tracker.py

pause
