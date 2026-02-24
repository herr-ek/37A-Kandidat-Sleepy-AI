#!/bin/bash

echo "Setting up Python virtual environment for 37A-Kandidat-Sleepy-AI..."

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null
then
    echo "Error: Python 3.12 is not installed."
    echo "Install it first (macOS Homebrew): brew install python@3.12"
    exit 1
fi

PYTHON_BIN="$(command -v python3.12)"
echo "Using interpreter: $PYTHON_BIN ($($PYTHON_BIN --version))"

# Remove old virtual environment if it exists
if directory=".venv" && [ -d "$directory" ]; then
    rm -rf "$directory"
    echo "Removed existing virtual environment at .venv"
fi

# Create virtual environment
echo "Creating virtual environment in .venv with Python 3.12..."
"$PYTHON_BIN" -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
