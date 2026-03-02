# Setup script for Windows
Write-Host "Setting up Python virtual environment for 37A-Kandidat-Sleepy-AI..." -ForegroundColor Green

# Check if Python 3 is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion"
} catch {
    Write-Host "Error: Python 3 is not installed. Please install Python 3 first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment in .venv..." -ForegroundColor Yellow
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n✓ Setup complete!" -ForegroundColor Green
Write-Host "`nTo activate the virtual environment, run:" -ForegroundColor Cyan
Write-Host "  PowerShell: .\.venv\Scripts\Activate.ps1"
Write-Host "  CMD: .venv\Scripts\activate.bat"
Write-Host "`nTo deactivate, run:" -ForegroundColor Cyan
Write-Host "  deactivate"
