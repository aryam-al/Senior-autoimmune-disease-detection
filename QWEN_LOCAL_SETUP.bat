@echo off
cd /d "%~dp0backend\qwen_local"
echo ============================================
echo   ImmunoAI - set up Qwen on this computer
echo ============================================
echo.
set PY=
if exist ..\venv\Scripts\python.exe set PY=..\venv\Scripts\python.exe
if not defined PY (
  where py >nul 2>nul && set PY=py -3
)
if not defined PY (
  python --version >nul 2>nul && set PY=python
)
if not defined PY (
  echo Python is not installed. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)
%PY% setup_local_qwen.py
echo.
pause
