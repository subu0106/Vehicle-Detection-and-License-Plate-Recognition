#!/bin/bash

# Use virtual environment Python
PYTHON="./VDLPR/bin/python"
STREAMLIT="./VDLPR/bin/streamlit"

echo "🚀 Starting Vehicle Detection & License Plate Recognition Application"
echo "================================================================"

# Initialize database
echo ""
echo "📊 Initializing database..."
$PYTHON scripts/init_database.py

if [ $? -ne 0 ]; then
    echo "❌ Database initialization failed"
    exit 1
fi

echo ""
echo "✅ Database initialized successfully"
echo ""

# Start Flask API in background
echo "🔧 Starting Flask API on port 8000..."
$PYTHON -m src.api.app &
FLASK_PID=$!

# Wait for Flask to start
sleep 5

# Check if Flask started successfully
if ps -p $FLASK_PID > /dev/null; then
    echo "✅ Flask API started successfully (PID: $FLASK_PID)"
else
    echo "❌ Failed to start Flask API"
    exit 1
fi

echo ""
echo "🎨 Starting Streamlit UI on port 8501..."
echo "📝 Access the application at: http://localhost:8501"
echo ""
$PYTHON run_streamlit.py

# Cleanup: Kill Flask when Streamlit exits
echo ""
echo "🛑 Shutting down..."
kill $FLASK_PID
echo "✅ Application stopped"
