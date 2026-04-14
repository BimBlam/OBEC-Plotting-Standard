@echo off
REM BatteryPlot - First-time setup script
REM Run this once before using run_windows.bat

setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"

echo === BatteryPlot Setup ===
echo.

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Please install Python 3.11 or later from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Creating virtual environment at %VENV_DIR% ...
python -m venv "%VENV_DIR%"

echo Installing batteryplot and dependencies...
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip
"%VENV_DIR%\Scripts\pip.exe" install -r "%SCRIPT_DIR%requirements.txt"

echo.
echo Initializing default config...
"%VENV_DIR%\Scripts\batteryplot.exe" init-config

echo.
echo === Setup complete! ===
echo.
echo To run: double-click run_windows.bat or run:
echo   run_windows.bat run
echo   run_windows.bat run --input-dir "C:\path\to\csv\files"
echo.
pause
endlocal
