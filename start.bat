@echo off
echo Starting Free LLM Gateway...
echo.

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.9+
    pause
    exit /b 1
)

echo Installing backend dependencies...
cd backend
pip install -r requirements.txt -q

echo.
echo Starting backend server...
start "Backend API" cmd /k "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo Starting frontend server...
cd ..\frontend
start "Frontend" cmd /k "python -m http.server 3000"

echo.
echo ========================================
echo   Free LLM Gateway is running!
echo   Backend API: http://localhost:8000
echo   Frontend:    http://localhost:3000
echo ========================================
echo.
echo Press any key to open browser...
pause >nul
start http://localhost:3000
