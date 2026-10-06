$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
python -m venv .venv

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing project requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Setup complete! Add your GEMINI_API_KEY into .env" -ForegroundColor Yellow
Write-Host "  Then start with: python -m backend.main" -ForegroundColor Green
Write-Host "  Browser URL: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
