#!/bin/bash

# Kill child processes on exit
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo "🚀 Starting Polymarket Bot Dashboard..."

# 1. Start Backend API
echo "🔌 Starting Backend API (port 8000)..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/dashboard/api/main.py &
BACKEND_PID=$!

# Wait for backend to be ready (naive check)
sleep 2

# 2. Start Frontend
echo "💻 Starting Frontend (port 5173)..."
cd src/dashboard/frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo "✅ Dashboard launched!"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo "   Press Ctrl+C to stop both."

wait
