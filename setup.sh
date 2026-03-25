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

# Check if virtual environment exists and is valid
if [ -d ".venv" ]; then
    echo "Virtual environment already exists at .venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    else
        echo "Using existing virtual environment."
        source .venv/bin/activate
        
        # Just upgrade packages
        echo "Upgrading pip..."
        pip install --upgrade pip --quiet
        
        echo "Installing/upgrading dependencies..."
        pip install -r requirements.txt --upgrade --quiet
        
        echo ""
        echo "✅ Setup complete!"
        echo ""
        echo "Virtual environment is activated. To deactivate, run: deactivate"
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment in .venv with Python 3.12..."
"$PYTHON_BIN" -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies with optimizations
echo "Installing dependencies from requirements.txt..."
echo "(This may take a few minutes on first install...)"
pip install -r requirements.txt --prefer-binary

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
