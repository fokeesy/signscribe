@echo off
rem One-click launcher for Windows: creates a virtual environment on first run, then starts SignScribe.
rem Extra arguments are passed through, e.g.  run.bat --demo
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo Setting up SignScribe for the first time. This downloads a few packages and takes a minute...
    python -m venv .venv || goto :fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    ".venv\Scripts\python.exe" -m pip install -e . || goto :fail
)

".venv\Scripts\python.exe" -m signscribe %*
exit /b %errorlevel%

:fail
echo.
echo Setup failed. Make sure Python 3.10, 3.11 or 3.12 is installed and on your PATH (https://www.python.org/downloads/).
pause
exit /b 1
