#!/bin/bash

echo "🚀 Fast setup - updating existing environment..."

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ No virtual environment found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Only install/update what's needed
echo "Updating dependencies (only installing missing/outdated packages)..."
pip install -r requirements.txt --upgrade --upgrade-strategy only-if-needed --prefer-binary --quiet

echo ""
echo "✅ Fast update complete!"
echo ""
echo "Virtual environment is activated. To deactivate, run: deactivate"
