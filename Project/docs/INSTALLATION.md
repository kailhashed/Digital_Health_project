# 🚀 Installation Guide

Complete guide for setting up the Audio Emotion Recognition System on your machine.

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8 or higher
- **CPU**: Multi-core processor (4+ cores recommended)

### Recommended Requirements
- **OS**: Windows 11, Ubuntu 20.04+, macOS 12+
- **RAM**: 16 GB or higher
- **Storage**: 20 GB free space (SSD preferred)
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **Python**: 3.9 or 3.10
- **CPU**: 8+ cores

---

## 🔧 Installation Methods

### Method 1: Quick Setup (Recommended)

#### Windows PowerShell
```powershell
# Clone the repository
git clone https://github.com/kailhashed/Digital_Health_project.git
cd Digital_Health_project/Project

# Run automated setup script
.\install_from_scratch.ps1
```

#### Linux/macOS
```bash
# Clone the repository
git clone https://github.com/kailhashed/Digital_Health_project.git
cd Digital_Health_project/Project

# Create virtual environment
python3 -m venv emotion_env
source emotion_env/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

### Method 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/kailhashed/Digital_Health_project.git
cd Digital_Health_project
```

#### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv emotion_env
emotion_env\Scripts\activate

# Linux/macOS
python3 -m venv emotion_env
source emotion_env/bin/activate
```

#### Step 3: Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### Step 4: Install Core Dependencies
```bash
# Navigate to Project directory
cd Project

# Install from requirements file
pip install -r requirements.txt
```

---

## 📦 Dependencies Overview

### Core Machine Learning Libraries
```
torch>=2.0.0                    # PyTorch deep learning framework
torchaudio>=2.0.0               # Audio processing for PyTorch
librosa>=0.10.0                 # Audio analysis library
numpy>=1.21.0                   # Numerical computing
scikit-learn>=1.0.0             # Machine learning utilities
pandas>=1.3.0                   # Data manipulation
```

### Pre-trained Model Libraries
```
transformers>=4.30.0            # Hugging Face transformers
accelerate>=0.20.0              # Training acceleration
datasets>=2.12.0                # Dataset utilities
tokenizers>=0.13.0              # Tokenization utilities
```

### Visualization and Analysis
```
matplotlib>=3.5.0               # Plotting library
seaborn>=0.11.0                 # Statistical visualizations
plotly>=5.0.0                   # Interactive plots
```

### Utilities
```
tqdm>=4.64.0                    # Progress bars
pyyaml>=6.0                     # Configuration files
joblib>=1.1.0                   # Model serialization
soundfile>=0.10.0               # Audio file I/O
```

---

## 🖥️ GPU Setup (Optional but Recommended)

### NVIDIA CUDA Setup

#### Check GPU Compatibility
```bash
# Check if NVIDIA GPU is available
nvidia-smi
```

#### Install CUDA Toolkit
1. **Download**: Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. **Select**: Your OS and architecture
3. **Install**: Follow platform-specific instructions
4. **Verify**: Run `nvidia-smi` and `nvcc --version`

#### Install PyTorch with CUDA
```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Verify GPU Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")
```

---

## 🔧 Platform-Specific Instructions

### Windows Setup

#### Prerequisites
```powershell
# Install Python from python.org or Microsoft Store
# Install Git for Windows
# Install Visual Studio Build Tools (for some packages)
```

#### Common Issues and Solutions
```powershell
# If pip install fails due to missing compiler
# Install Microsoft Visual Studio Build Tools

# If librosa installation fails
pip install librosa --no-use-pep517

# If torch installation is slow
pip install torch torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

### Linux (Ubuntu/Debian) Setup

#### Prerequisites
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and development tools
sudo apt install python3 python3-pip python3-venv python3-dev
sudo apt install build-essential git curl

# Install audio libraries
sudo apt install ffmpeg libsndfile1 libasound2-dev
```

#### For Older Ubuntu Versions
```bash
# Ubuntu 18.04 specific
sudo apt install python3.8 python3.8-venv python3.8-dev
python3.8 -m venv emotion_env
```

### macOS Setup

#### Prerequisites
```bash
# Install Homebrew if not available
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and audio libraries
brew install python@3.9 ffmpeg libsndfile

# Install Xcode command line tools
xcode-select --install
```

#### Common Issues
```bash
# If librosa installation fails
brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang
pip install librosa

# For M1/M2 Macs
arch -arm64 pip install torch torchaudio
```

---

## 🧪 Verification and Testing

### Test Installation
```python
# Create test_installation.py file
import sys
import pkg_resources

def test_installation():
    """Test if all required packages are installed correctly."""
    
    required_packages = [
        'torch', 'torchaudio', 'librosa', 'numpy', 
        'pandas', 'scikit-learn', 'matplotlib', 
        'seaborn', 'tqdm', 'transformers'
    ]
    
    print("🔍 Checking package installation...")
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: Not installed")
    
    print("\n🔍 Checking PyTorch functionality...")
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU device: {torch.cuda.get_device_name()}")
    
    print("\n🔍 Checking audio processing...")
    try:
        import librosa
        import torchaudio
        print(f"✅ librosa version: {librosa.__version__}")
        print(f"✅ torchaudio version: {torchaudio.__version__}")
        
        # Test basic audio loading
        sr = 16000
        duration = 1.0
        test_audio = torch.randn(int(sr * duration))
        mel_spec = torchaudio.transforms.MelSpectrogram()(test_audio)
        print(f"✅ Audio processing test passed: {mel_spec.shape}")
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
    
    print("\n🔍 Checking model imports...")
    try:
        from src.models.custom_models import EmotionResNet, EmotionLSTM, EmotionTransformer
        from src.data.dataset import EmotionDataset
        print("✅ Custom models import successful")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
    
    print("\n🎉 Installation verification complete!")

if __name__ == "__main__":
    test_installation()
```

```bash
# Run the test
python test_installation.py
```

### Quick Functionality Test
```python
# Create quick_test.py
import torch
import torchaudio
import numpy as np
from src.models.custom_models import EmotionResNet

def quick_test():
    """Quick functionality test."""
    
    print("🧪 Running quick functionality test...")
    
    # Test model creation
    model = EmotionResNet(num_classes=8)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 64, 94)  # Mel-spectrogram shape
    output = model(dummy_input)
    print(f"✅ Forward pass successful: {output.shape}")
    
    # Test audio processing
    sample_rate = 16000
    duration = 3.0
    dummy_audio = torch.randn(int(sample_rate * duration))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=64,
        hop_length=512
    )
    mel_spec = mel_transform(dummy_audio)
    print(f"✅ Audio processing successful: {mel_spec.shape}")
    
    print("🎉 All tests passed! System ready for training.")

if __name__ == "__main__":
    quick_test()
```

---

## 🔧 Troubleshooting Common Issues

### Package Installation Failures

#### Issue: `pip install` fails with compiler errors
```bash
# Solution 1: Update build tools
pip install --upgrade setuptools wheel pip

# Solution 2: Use conda instead
conda install pytorch torchaudio librosa -c pytorch -c conda-forge

# Solution 3: Use pre-compiled wheels
pip install --only-binary=all torch torchaudio librosa
```

#### Issue: CUDA/GPU not detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Audio processing libraries fail
```bash
# Linux: Install system audio libraries
sudo apt install ffmpeg libsndfile1-dev libasound2-dev

# macOS: Install with Homebrew
brew install ffmpeg libsndfile

# Windows: Use conda for complex audio packages
conda install librosa soundfile -c conda-forge
```

### Memory Issues

#### Issue: Out of memory during training
```python
# Reduce batch size in config
CONFIG = {
    'training': {
        'batch_size': 8,  # Reduce from 16
        'accumulate_grad_batches': 2,  # Simulate larger batches
    }
}

# Enable memory optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```

#### Issue: Large dataset loading
```python
# Use data loading optimizations
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # Reduce if memory limited
    pin_memory=True,  # Only if GPU available
    shuffle=True
)
```

### Permission Issues

#### Issue: Permission denied errors
```bash
# Linux/macOS: Fix permissions
chmod +x install_from_scratch.ps1
sudo chown -R $USER:$USER Digital_Health_project/

# Windows: Run as administrator or check folder permissions
```

---

## 🚀 Next Steps After Installation

### 1. Verify Installation
```bash
python test_installation.py
python quick_test.py
```

### 2. Download Datasets
```bash
# See docs/DATASETS.md for detailed instructions
python -c "print('📖 Next: Follow DATASETS.md to download emotion datasets')"
```

### 3. Run Preprocessing
```bash
# After downloading datasets
python data_preprocessing.py
```

### 4. Start Training
```bash
# Train all models (requires datasets)
python scripts/train_all_models.py
```

### 5. Test Predictions
```bash
# Make predictions on new audio
python scripts/predict_emotion.py --help
```

---

## 📱 Development Environment Setup

### IDE Configuration

#### VS Code Extensions
- Python
- Pylance
- Jupyter
- GitLens
- Python Docstring Generator

#### PyCharm Configuration
- Enable scientific mode
- Configure Python interpreter to virtual environment
- Set project root to `Digital_Health_project/`

### Code Quality Tools
```bash
# Install development tools
pip install black flake8 isort mypy pytest

# Format code
black .
isort .

# Check code quality
flake8 src/
mypy src/
```

---

## 📞 Support and Help

### Getting Help
1. **Check Documentation**: Review all files in `docs/` folder
2. **Run Diagnostics**: Use provided test scripts
3. **Check Issues**: Look for similar problems in project issues
4. **Ask Questions**: Create detailed issue reports

### Useful Commands for Debugging
```bash
# Check Python environment
python --version
pip list
which python

# Check GPU status
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Check audio capabilities
python -c "import librosa; print('librosa working')"
python -c "import torchaudio; print('torchaudio working')"

# Check project structure
ls -la src/
python -c "from src.models import custom_models; print('imports working')"
```

---

## 📊 Performance Optimization

### Training Acceleration
```python
# Enable mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# In training loop
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Memory Optimization
```python
# Gradient checkpointing for large models
torch.utils.checkpoint.checkpoint_sequential()

# Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

*Installation guide complete! Proceed to [DATASETS.md](DATASETS.md) for dataset setup instructions.*
