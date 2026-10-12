@echo off
echo ==============================================================
echo   STEELSIGHT - AI Surface Inspection Platform
echo ==============================================================
echo.
echo Starting AI Engine (FastAPI on port 8000)...
start "SteelSight AI Engine" cmd /k "cd /d "%~dp0" && venv\Scripts\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 8000"

echo Starting Frontend (Vite on port 5173)...
start "SteelSight Frontend" cmd /k "cd /d "%~dp0steelsight" && npm run dev"

echo.
echo Both servers starting...
echo   API:      http://localhost:8000
echo   Frontend: http://localhost:5173
echo.
echo Press any key to open the application...
pause >nul
start http://localhost:5173
