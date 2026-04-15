@echo off
echo ============================================
echo  SyPy Auto-Tracker - Running with Python 3.9
echo ============================================
echo.

SET PYTHON39=C:\Python39\python.exe
SET SCRIPT_DIR=%~dp0
SET SYNTHEYES=C:\Program Files\BorisFX\SynthEyes 2025.5\SynthEyes64.exe

IF NOT EXIST "%PYTHON39%" (
    echo ERROR: Python 3.9 not found at %PYTHON39%
    pause & exit /b
)

echo Using Python : %PYTHON39%
echo Script folder: %SCRIPT_DIR%
echo.

REM Close any existing SynthEyes so we launch fresh with the listener
echo Closing any existing SynthEyes...
taskkill /IM SynthEyes64.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul

REM Launch SynthEyes with the correct listener flags:
REM   -l 2222       = listen on port 2222
REM   -pin listen   = connection password is "listen"
echo Launching SynthEyes with listener on port 2222...
start "" "%SYNTHEYES%" -l 2222 -pin listen

echo Waiting 10 seconds for SynthEyes to fully start...
timeout /t 10 /nobreak >nul
echo.

"%PYTHON39%" "%SCRIPT_DIR%autotrack_to_3de.py"

echo.
echo ============================================
echo  Script finished. Check your output folder.
echo ============================================
pause
