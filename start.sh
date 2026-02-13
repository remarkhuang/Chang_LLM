#!/bin/bash
echo "Starting Free LLM Gateway..."
echo

echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python not found! Please install Python 3.9+"
    exit 1
fi

echo "Installing backend dependencies..."
cd backend
pip3 install -r requirements.txt -q

echo
echo "Starting backend server..."
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo
echo "Starting frontend server..."
cd ../frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!

echo
echo "========================================"
echo "  Free LLM Gateway is running!"
echo "  Backend API: http://localhost:8000"
echo "  Frontend:    http://localhost:3000"
echo "========================================"
echo
echo "Press Ctrl+C to stop..."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
