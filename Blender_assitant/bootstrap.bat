@echo off
REM Build everything the sidecar needs, inside this folder. See bootstrap.py's docstring.
setlocal
set "HERE=%~dp0"
set "PY="

REM Any Python 3 will do to RUN this script -- it only downloads and shells out. The
REM interpreter it BUILDS is a different thing entirely (3.11, torch, CUDA).
if exist "%HERE%runtime\python311\python.exe" set "PY=%HERE%runtime\python311\python.exe"
if not defined PY if exist "%HERE%..\runtime\python311\python.exe" set "PY=%HERE%..\runtime\python311\python.exe"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY if exist "C:\Users\%USERNAME%\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\5.2\python\bin\python.exe" set "PY=C:\Users\%USERNAME%\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\5.2\python\bin\python.exe"

if not defined PY (
  echo [ERROR] no Python found to run the bootstrap with.
  exit /b 1
)

echo using %PY%
"%PY%" "%HERE%bootstrap.py" %*
exit /b %ERRORLEVEL%
