@echo off
REM Brain Tumor Classification - Frontend Startup Script
REM This script starts a simple HTTP server for the frontend

echo ========================================
echo Brain Tumor Classification - Frontend
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

echo Starting HTTP server for frontend...
echo.
echo Frontend will be available at:
echo http://localhost:8000
echo.
echo IMPORTANT: Make sure the backend is running on http://localhost:5000
echo           Use start_backend.bat to start the backend server
echo.
echo Press Ctrl+C to stop the server
echo.

cd frontend
python -m http.server 8000

pause
