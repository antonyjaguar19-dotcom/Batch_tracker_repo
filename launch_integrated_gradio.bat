\
@echo off
setlocal

REM Put this BAT in the repo root (same level as .venv) and double-click to run.
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"

if exist "%PY%" (
  echo Using venv python: "%PY%"
  "%PY%" "%HERE%app.py"
) else (
  echo ERROR: .venv not found at "%HERE%.venv"
  echo Fix: create/restore your venv at repo root, or run with the correct interpreter.
  pause
  exit /b 1
)

pause
