@echo off
REM BatteryPlot - Windows launcher
REM This script sets up a virtual environment if needed and runs batteryplot.
REM No admin rights required.

setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=python"

echo === BatteryPlot Windows Launcher ===

REM Check if venv exists
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.11+ is installed and on your PATH.
        pause
        exit /b 1
    )
    echo Installing dependencies...
    "%VENV_DIR%\Scripts\pip.exe" install -r "%SCRIPT_DIR%requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
    "%VENV_DIR%\Scripts\pip.exe" install "%SCRIPT_DIR%."
    if errorlevel 1 (
        echo ERROR: Failed to install batteryplot package.
        pause
        exit /b 1
    )
    echo.
    echo Setup complete!
)

REM Run batteryplot with any arguments passed to this script
"%VENV_DIR%\Scripts\batteryplot.exe" %*

if errorlevel 1 (
    echo.
    echo batteryplot exited with an error. Check logs above.
    pause
)

endlocal
