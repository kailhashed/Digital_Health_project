# Complete Installation Script for Emotion Recognition Project
# This script installs everything from scratch

Write-Host "Emotion Recognition Project - Complete Installation" -ForegroundColor Green
Write-Host "=" * 60

# Check if Python is installed
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = [version]$matches[1]
        Write-Host "Python $version found" -ForegroundColor Green
        
        if ($version -lt [version]"3.8") {
            Write-Host "WARNING: Python 3.8+ is recommended. Current version: $version" -ForegroundColor Red
            Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Red
    exit 1
}

# Check if pip is available
Write-Host "`nChecking pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "pip not found. Installing pip..." -ForegroundColor Yellow
    python -m ensurepip --upgrade
}

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Check CUDA installation
Write-Host "`nChecking CUDA installation..." -ForegroundColor Yellow
try {
    $cudaVersion = nvcc --version 2>&1
    if ($cudaVersion -match "release (\d+\.\d+)") {
        $cudaVer = $matches[1]
        Write-Host "CUDA $cudaVer found" -ForegroundColor Green
    }
} catch {
    Write-Host "CUDA not found in PATH, but that's okay - PyTorch will handle CUDA detection" -ForegroundColor Yellow
}

# Install PyTorch with CUDA support
Write-Host "`nInstalling PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# Install PyTorch with CUDA 11.8 (most compatible version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
Write-Host "`nVerifying PyTorch installation..." -ForegroundColor Yellow
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1e9:.1f} GB')
"

# Install other required packages
Write-Host "`nInstalling other required packages..." -ForegroundColor Yellow
pip install librosa soundfile scikit-learn xgboost pandas numpy matplotlib seaborn tqdm pyAudioAnalysis tensorboard

# Install additional audio processing packages
Write-Host "`nInstalling additional audio processing packages..." -ForegroundColor Yellow
pip install resampy numba

# Verify all installations
Write-Host "`nVerifying all installations..." -ForegroundColor Yellow
python -c "
import sys
print(f'Python version: {sys.version}')

packages = [
    'torch', 'torchaudio', 'librosa', 'soundfile', 'sklearn', 
    'xgboost', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'tqdm'
]

for package in packages:
    try:
        if package == 'sklearn':
            import sklearn
            print(f'✓ scikit-learn: {sklearn.__version__}')
        else:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f'✓ {package}: {version}')
    except ImportError as e:
        print(f'✗ {package}: NOT INSTALLED - {e}')
"

# Check if dataset is organized
Write-Host "`nChecking dataset organization..." -ForegroundColor Yellow
if (Test-Path "organized_by_emotion") {
    Write-Host "✓ Dataset is organized" -ForegroundColor Green
    
    # Count files in each emotion folder
    $emotions = @('angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised')
    $totalFiles = 0
    
    Write-Host "`nDataset statistics:" -ForegroundColor Cyan
    foreach ($emotion in $emotions) {
        $emotionPath = "organized_by_emotion\$emotion"
        if (Test-Path $emotionPath) {
            $files = (Get-ChildItem $emotionPath -Filter "*.wav").Count
            $totalFiles += $files
            Write-Host "  $emotion`: $files files" -ForegroundColor White
        } else {
            Write-Host "  $emotion`: 0 files (folder not found)" -ForegroundColor Red
        }
    }
    Write-Host "  Total: $totalFiles files" -ForegroundColor Green
} else {
    Write-Host "✗ Dataset not organized. Please run the organization script first." -ForegroundColor Red
    Write-Host "Run: PowerShell -ExecutionPolicy Bypass -File organize_emotion_dataset.ps1" -ForegroundColor Yellow
}

# Create a simple test script
Write-Host "`nCreating test script..." -ForegroundColor Yellow
$testScript = @"
#!/usr/bin/env python3
import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("Testing all installations...")

# Test PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test librosa
y, sr = librosa.load("organized_by_emotion/angry/CREMA_1001_IEO_ANG_HI.wav", sr=22050, duration=1.0)
print(f"Librosa: Loaded audio with {len(y)} samples at {sr} Hz")

# Test sklearn
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)
print(f"Scikit-learn: RandomForest accuracy = {clf.score(X, y):.3f}")

# Test XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=10)
xgb_clf.fit(X, y)
print(f"XGBoost: Accuracy = {xgb_clf.score(X, y):.3f}")

print("All tests passed!")
"@

$testScript | Out-File -FilePath "test_installation.py" -Encoding UTF8

# Run the test
Write-Host "`nRunning installation test..." -ForegroundColor Yellow
python test_installation.py

Write-Host "`n" + "=" * 60 -ForegroundColor Green
Write-Host "INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run training: python train_models.py --mode both" -ForegroundColor White
Write-Host "2. Or use the simple runner: python run_training.py" -ForegroundColor White
Write-Host "3. Evaluate models: python evaluate_models.py" -ForegroundColor White

Write-Host "`nIf you encounter any issues:" -ForegroundColor Yellow
Write-Host "- Check that Python 3.8+ is installed and in PATH" -ForegroundColor White
Write-Host "- Verify CUDA drivers are properly installed" -ForegroundColor White
Write-Host "- Try running: pip install --upgrade torch torchvision torchaudio" -ForegroundColor White
