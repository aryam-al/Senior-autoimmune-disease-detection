@echo off
cd /d "%~dp0"
start "ImmunoAI backend" cmd /k "backend\start.bat"
start "ImmunoAI frontend" cmd /k "frontend\start-frontend.bat"
if exist "backend\qwen_local\llama" (
  start "ImmunoAI Qwen" cmd /k "backend\qwen_local\run_qwen_local.bat"
  echo Three windows opened: backend, frontend and Qwen.
) else (
  echo Two windows opened: backend and frontend.
)
echo When the frontend window says Ready, open http://localhost:3000
timeout /t 8 >nul
