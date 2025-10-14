# Setup script for Minecraft 3D Asset Generator
# Run this script to set up the environment

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Minecraft 3D Asset Generator - Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python not found! Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: Failed to install dependencies!" -ForegroundColor Red
    exit 1
}

# Check for CUDA
Write-Host "`nChecking for CUDA support..." -ForegroundColor Yellow
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaAvailable -eq "True") {
    Write-Host "CUDA is available! GPU training enabled." -ForegroundColor Green
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Host "CUDA not available. Training will use CPU (slower)." -ForegroundColor Yellow
    Write-Host "For GPU support, install CUDA and PyTorch with CUDA:" -ForegroundColor Yellow
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Cyan
}

# Create necessary directories
Write-Host "`nCreating project directories..." -ForegroundColor Yellow
$dirs = @("data", "data/processed", "data/splits", "models", "logs", "generated", "evaluation")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    }
}

# Create .gitkeep files
foreach ($dir in $dirs) {
    $gitkeep = Join-Path $dir ".gitkeep"
    if (!(Test-Path $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep | Out-Null
    }
}

Write-Host "`n=====================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Preprocess the data:" -ForegroundColor White
Write-Host "   python scripts/preprocess_data.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Train the model:" -ForegroundColor White
Write-Host "   python scripts/train.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Generate assets:" -ForegroundColor White
Write-Host "   python scripts/generate.py --checkpoint models/best.ckpt --prompt 'oak tree'" -ForegroundColor Yellow
Write-Host ""
Write-Host "For more information, see QUICKSTART.md" -ForegroundColor Cyan
