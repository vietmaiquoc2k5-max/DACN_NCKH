# ==============================================================================
# SETUP CUDA ENVIRONMENT FOR WEEK 5 GPU TRAINING HANDOFF
# Project: Computer Vision Fire/Smoke Detection (DACN_NCKH)
# ==============================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  DACN_NCKH - WEEK 5 GPU TRAINING ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python not found in PATH! Please install Python 3.10 or 3.11."
    exit 1
}
$pyVer = & python --version
Write-Host "[1/6] Found Python: $pyVer" -ForegroundColor Green

# 2. Check NVIDIA GPU via nvidia-smi
Write-Host "[2/6] Detecting NVIDIA GPU..." -ForegroundColor Yellow
$nvidiaCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaCmd) {
    Write-Warning "nvidia-smi not found! Ensure NVIDIA drivers are installed."
    $cudaIndex = "https://download.pytorch.org/whl/cu121"
} else {
    & nvidia-smi
    $cudaIndex = "https://download.pytorch.org/whl/cu121"
    Write-Host "NVIDIA GPU detected. Recommended PyTorch CUDA index: $cudaIndex" -ForegroundColor Green
}

# 3. Create Virtual Environment
$venvPath = ".venv_gpu"
if (-not (Test-Path $venvPath)) {
    Write-Host "[3/6] Creating virtual environment: $venvPath..." -ForegroundColor Yellow
    & python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "[3/6] Virtual environment $venvPath already exists." -ForegroundColor Green
}

$venvPython = "$venvPath\Scripts\python.exe"
$venvPip = "$venvPath\Scripts\pip.exe"

# 4. Upgrade pip
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

# 5. Install PyTorch with CUDA
Write-Host "[5/6] Installing PyTorch with CUDA ($cudaIndex)..." -ForegroundColor Yellow
& $venvPip install torch torchvision --index-url $cudaIndex
if ($LASTEXITCODE -ne 0) {
    Write-Warning "cu121 install failed. Trying cu118..."
    & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
}

# 6. Install Project Dependencies
Write-Host "[6/6] Installing training dependencies from requirements-training.txt..." -ForegroundColor Yellow
& $venvPip install -r requirements-training.txt

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  RUNNING VERIFICATION SCRIPT" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
& $venvPython scripts/verify_week5_handoff.py

Write-Host "Setup finished. To activate this environment in PowerShell, run:" -ForegroundColor Green
Write-Host "  .\$venvPath\Scripts\Activate.ps1" -ForegroundColor Yellow

