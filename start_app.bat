@echo off
REM Brain Tumor Classification - Complete Application Startup
REM This script starts both backend and frontend servers

echo ========================================
echo Brain Tumor Classification
echo Complete Application Startup
echo ========================================
echo.

REM Set pip cache and temp to A: drive to avoid C: drive space issues
set PIP_CACHE_DIR=A:\BRAIN_ML\.pip_cache
set TEMP=A:\BRAIN_ML\.temp
set TMP=A:\BRAIN_ML\.temp
mkdir A:\BRAIN_ML\.pip_cache 2>nul
mkdir A:\BRAIN_ML\.temp 2>nul

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo This will start both backend and frontend servers
echo in separate windows.
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:8000
echo.
pause

REM Start backend in new window
echo Starting Backend Server...
start "Brain Tumor Classification - Backend" cmd /c "cd backend && python app_simple.py && pause"

REM Wait a bit for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend in new window
echo Starting Frontend Server...
start "Brain Tumor Classification - Frontend" cmd /c start_frontend.bat

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo.
echo Backend API: http://localhost:5000
echo Frontend UI: http://localhost:8000
echo.
echo Open http://localhost:8000 in your browser
echo.
echo Close both terminal windows to stop the servers
echo.
pause
