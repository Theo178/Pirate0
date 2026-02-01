#!/bin/bash
# Startup script for Pirate0 v2.0

set -e

echo "🏴‍☠️ Pirate0 v2.0 Startup"
echo "========================"

# Set OpenMP fix for macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ Setup complete"
echo ""
echo "Starting server on http://localhost:8000"
echo "UI available at: http://localhost:8000/static/"
echo ""

# Start server
python app.py
