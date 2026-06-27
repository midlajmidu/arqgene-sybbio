#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Clear screen if TERM is set and output is a terminal
if [ -n "$TERM" ] && [ -t 1 ]; then
    clear
fi
echo -e "\033[1;34m============================================================\033[0m"
echo -e "\033[1;36m🧬  SynB · Metabolic Engineering & Genome Reconstruction  🧬\033[0m"
echo -e "\033[1;34m============================================================\033[0m"
echo ""

# 1. Check for python3
if ! command -v python3 &> /dev/null; then
    echo -e "\033[1;31m❌ Error: Python 3 is not installed or not in your PATH.\033[0m"
    echo -e "Please install Python 3.9+ before running this script."
    exit 1
fi

# 2. Check/Create virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\033[1;33m📦 Creating a Python virtual environment in $VENV_DIR...\033[0m"
    python3 -m venv "$VENV_DIR"
    echo -e "\033[1;32m✅ Virtual environment created.\033[0m"
else
    echo -e "\033[1;32m🐍 Found existing virtual environment in $VENV_DIR.\033[0m"
fi

# 3. Activate venv & install dependencies
echo -e "\033[1;33m⚙️  Activating virtual environment & installing requirements...\033[0m"
source "$VENV_DIR/bin/activate"

# Upgrade pip inside venv
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt
echo -e "\033[1;32m✅ Dependencies installed successfully.\033[0m"
echo ""

# 4. Define cleanup on exit
cleanup() {
    echo ""
    echo -e "\033[1;33m🧹 Shutting down servers...\033[0m"
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping FastAPI Backend (PID: $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping Streamlit Frontend (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    echo -e "\033[1;32m✅ Done. Goodbye!\033[0m"
    exit 0
}

# Trap Ctrl+C (SIGINT), SIGTERM, and exit signals to run cleanup
trap cleanup INT TERM EXIT

# 5. Start Backend (FastAPI) on port 8001
echo -e "\033[1;34m🚀 Starting FastAPI Backend on http://localhost:8001...\033[0m"
# Run uvicorn in the background and redirect output to a log file
uvicorn backend.main:app --port 8001 --reload > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to spin up and pass liveness check
echo -e "\033[1;33m⏳ Waiting for backend to start...\033[0m"
HEALTH_CHECK_URL="http://localhost:8001/api/v1/health"
for i in {1..20}; do
    if curl -s -f "$HEALTH_CHECK_URL" &> /dev/null; then
        echo -e "\033[1;32m✅ Backend is up and healthy!\033[0m"
        break
    fi
    if [ $i -eq 20 ]; then
        echo -e "\033[1;33m⚠️  Backend healthcheck timed out, but trying to start frontend anyway...\033[0m"
        echo -e "Check backend.log if you encounter connection errors."
    fi
    sleep 1
done

# 6. Start Frontend (Streamlit)
echo -e "\033[1;34m🎨 Starting Streamlit Frontend on http://localhost:8501...\033[0m"
streamlit run app.py &
FRONTEND_PID=$!

# Keep script running to let wait process events and capture Ctrl+C
wait
