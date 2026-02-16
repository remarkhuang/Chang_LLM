@echo off
setlocal
echo Starting Free LLM Gateway (with uv virtualenv)...

:: 啟動後端
echo [1/2] Starting backend on port 8000...
cd /d "%~dp0backend"
start "FreeLLM-Backend" cmd /k ".\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

:: 啟動前端
echo [2/2] Starting frontend on port 3000...
cd /d "%~dp0frontend"
start "FreeLLM-Frontend" cmd /k "python -m http.server 3000"

echo.
echo ========================================
echo   Free LLM Gateway 已啟動！
echo   後端 API: http://localhost:8000
echo   前端界面: http://localhost:3000
echo ========================================
echo.
echo 按任意鍵打開瀏覽器...
pause >nul
start http://localhost:3000
