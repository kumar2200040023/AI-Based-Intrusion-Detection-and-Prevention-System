@echo off
title AI-Based IDS/IPS Launcher
echo ===================================================
echo     Starting AI-Based IDS and IPS System
echo ===================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run install.bat first!
    pause
    exit /b
)

echo Activating environment...
call .venv\Scripts\activate.bat

echo.
echo Starting Backend API Server (Port 8000)...
start "AI-Based IDS API" cmd /c ".venv\Scripts\python.exe -m api.main"

echo Waiting for API to initialize...
timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend Streamlit Dashboard (Port 8501)...
start "AI-Based IDS Dashboard" cmd /c ".venv\Scripts\streamlit.exe run dashboard\app.py"

echo.
echo The application should now open in your web browser.
echo You can close this window.
pause >nul
