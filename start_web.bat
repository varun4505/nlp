@echo off
REM Starts the DriftLens Python API (port 8000) and the Next.js frontend (port 3000).
REM Double-click this file, wait ~15 s, then open http://localhost:3000
cd /d "%~dp0"
start "DriftLens API" cmd /k python -m uvicorn api.server:app --port 8000
cd web
if not exist node_modules call npm install
start "DriftLens Web" cmd /k npm run dev
timeout /t 12 >nul
start http://localhost:3000
