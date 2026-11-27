@echo off
cd /d "%~dp0"

rem ---- find Python (py launcher first, then python) ----
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (
  python --version >nul 2>nul && set PY=python
)
if not defined PY (
  echo.
  echo Python is not installed or not on PATH.
  echo Install it from https://www.python.org/downloads/  and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

rem ---- create the virtual environment the first time ----
if not exist venv\Scripts\activate.bat (
  echo Creating virtual environment...
  %PY% -m venv venv || goto :fail
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt || goto :fail
) else (
  call venv\Scripts\activate.bat
)

echo.
echo Backend running at http://localhost:8000/docs
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
exit /b 0

:fail
echo.
echo Setup failed - see the error above.
if exist venv rmdir /s /q venv
pause
exit /b 1
