@echo off
REM Brain Tumor Classification - Windows Startup Script
REM This script sets up and starts the Flask backend server

echo ========================================
echo Brain Tumor Classification - Backend
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python installation...
python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo.
    echo [2/4] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo [4/4] Installing dependencies...
pip install --upgrade pip
pip install -r backend\requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Flask Server...
echo ========================================
echo.
echo Backend API will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Flask server
cd backend
python app_simple.py

pause
