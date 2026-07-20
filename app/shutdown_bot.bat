@echo off
setlocal
REM Shut down Batch Tracker: kill whatever is LISTENING on the UI port (8080).
set "PORT=8080"

set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
  echo Killing PID %%P on port %PORT% ...
  taskkill /F /PID %%P >nul 2>&1
  set "FOUND=1"
)

if not defined FOUND (
  echo No Batch Tracker process found on port %PORT%.
) else (
  echo Batch Tracker stopped.
)

timeout /t 2 >nul
