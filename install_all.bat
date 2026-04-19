@echo off
setlocal
echo ============================================================
echo   Batch Tracker — One-Click Installer
echo   This will install ALL Python dependencies into your .venv
echo ============================================================
echo.

REM --- Locate the .venv ---
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
set "PIP=%HERE%.venv\Scripts\pip.exe"

if not exist "%PY%" (
    echo ERROR: .venv not found at "%HERE%.venv"
    echo.
    echo FIX: Create your virtual environment first by running:
    echo     python -m venv .venv
    echo Then run this installer again.
    echo.
    pause
    exit /b 1
)

echo Found venv at: %HERE%.venv
echo.

REM --- Step 1: Upgrade pip itself ---
echo [1/3] Upgrading pip...
"%PY%" -m pip install --upgrade pip
echo.

REM --- Step 2: Install PyTorch with CUDA 12.1 ---
echo [2/3] Installing PyTorch + TorchVision (CUDA 12.1)...
echo       This is a large download (~2.5 GB). Please wait...
"%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

REM --- Step 3: Install everything else ---
echo [3/3] Installing remaining packages from requirements.txt...
"%PIP%" install -r "%HERE%requirements.txt"
echo.

echo ============================================================
echo   INSTALL COMPLETE!
echo.
echo   To verify, run:
echo     .venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
echo   Should print: True
echo.
echo   To launch the tool, double-click:
echo     launch_integrated_gradio.bat
echo ============================================================
pause
