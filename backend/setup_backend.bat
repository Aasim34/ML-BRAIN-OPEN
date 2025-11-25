@echo off
echo ========================================
echo Backend Setup - Using A: Drive Cache
echo ========================================
echo.

REM Set pip cache to A: drive to avoid C: drive space issues
set PIP_CACHE_DIR=A:\BRAIN_ML\.pip_cache
set TMPDIR=A:\BRAIN_ML\.tmp
set TEMP=A:\BRAIN_ML\.tmp
set TMP=A:\BRAIN_ML\.tmp

REM Create cache directories
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"

echo Using pip cache at: %PIP_CACHE_DIR%
echo.

echo Installing backend dependencies...
echo.

REM Install only the lightweight packages
pip install --cache-dir "%PIP_CACHE_DIR%" Flask==3.0.0 Flask-CORS==4.0.0 python-multipart==0.0.6 python-dotenv==1.0.0

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Backend dependencies installed!
    echo ========================================
    echo.
    echo Note: Using TensorFlow and other ML packages
    echo from main project installation.
    echo.
) else (
    echo.
    echo ERROR: Failed to install dependencies
    echo.
)

pause
