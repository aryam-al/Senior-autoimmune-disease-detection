@echo off
cd /d "%~dp0"

where npm >nul 2>nul
if errorlevel 1 (
  echo.
  echo Node.js is not installed. Install the LTS version from https://nodejs.org and run this file again.
  echo.
  pause
  exit /b 1
)

if not exist node_modules (
  echo Installing packages, this takes a few minutes the first time...
  call npm install || goto :fail
)

echo.
echo Open http://localhost:3000  (the backend must be running on http://localhost:8000)
echo.
call npm run dev
pause
exit /b 0

:fail
echo.
echo Setup failed - see the error above.
pause
exit /b 1
