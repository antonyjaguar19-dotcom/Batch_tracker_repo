@echo off
setlocal
echo ============================================================
echo   Batch Tracker — One-Click Installer (Windows)
echo   Delegates to install.py (cross-platform single-run setup)
echo ============================================================
echo.

REM Resolve folder that contains this BAT (so the tool works on any system / any folder)
set "HERE=%~dp0"

REM Prefer Python Launcher (py), fall back to whatever is on PATH as "python".
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%HERE%install.py"
) else (
    python "%HERE%install.py"
)

echo.
pause
