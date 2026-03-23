@echo off
title AI-Based IDS/IPS Installer
echo ===================================================
echo     Installing AI-Based IDS and IPS System
echo ===================================================
echo.
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH! 
    echo Please install Python (https://www.python.org/downloads/) and try again.
    pause
    exit /b
)

echo Step 2: Creating virtual environment (.venv)...
python -m venv .venv

echo Step 3: Activating virtual environment...
call .venv\Scripts\activate.bat

echo Step 4: Installing required dependencies...
pip install -r requirements.txt

echo.
echo ===================================================
echo   Installation Complete! 
echo   You can now double-click 'start.bat' to run it.
echo ===================================================
pause
