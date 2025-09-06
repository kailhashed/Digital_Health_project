# 🧭 Project Navigation Guide

This guide helps you navigate the Audio Emotion Recognition System codebase efficiently.

## 📁 Repository Structure Overview

```
Digital_Health_project/
├── 📄 README.md                    # Main project documentation
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 NAVIGATION.md                # This navigation guide
├── 📄 CITATION.cff                 # Academic citation format
├── 📄 LICENSE                      # MIT license and dataset attributions
├── 📄 .gitignore                   # Git ignore patterns
├── 📄 requirements.txt             # Python dependencies
├── 📁 src/                         # 🎯 Main source code (ORGANIZED STRUCTURE)
├── 📁 scripts/                     # 🎯 Execution scripts (START HERE)
├── 📁 Project/                     # 🏛️ Legacy code (REFERENCE ONLY)
├── 📁 models/                      # 💾 Saved model checkpoints
├── 📁 results/                     # 📊 Training results and reports
├── 📁 logs/                        # 📝 Training logs
└── 📁 organized_by_emotion/        # 🎵 Dataset (if available)
```

## 🎯 Quick Start Paths

### 🔥 I want to run the system immediately
1. **Start here**: `scripts/train_all_models.py` - Train all models
2. **Then run**: `scripts/compare_models.py` - Compare performance
3. **Finally**: `scripts/predict_emotion.py` - Make predictions

### 🧠 I want to understand the models
1. **Custom models**: `src/models/custom_models.py`
2. **Pre-trained models**: `src/models/pretrained_models.py`
3. **Training logic**: `src/training/trainer.py`

### 📊 I want to see results and analysis
1. **Latest results**: `results/comprehensive_comparison_report.md`
2. **Model comparison**: `results/model_comparison.png`
3. **Detailed metrics**: `results/comparisons/`

### 🔧 I want to modify the system
1. **Configuration**: `src/utils/config.py`
2. **Data processing**: `src/data/preprocessing.py`
3. **Training setup**: `src/training/trainer.py`

## 📍 Key File Locations

### 🚀 Execution Scripts (scripts/)

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_all_models.py` | Train all models | `python scripts/train_all_models.py` |
| `compare_models.py` | Compare model performance | `python scripts/compare_models.py` |
| `predict_emotion.py` | Make predictions | `python scripts/predict_emotion.py --help` |

### 🧩 Source Code (src/)

#### Models (src/models/)
- `custom_models.py` - ResNet, Transformer, LSTM architectures
- `pretrained_models.py` - Wav2Vec2, SimpleCNNAudio models
- `__init__.py` - Model registry and imports

#### Data (src/data/)
- `dataset.py` - EmotionDataset class for PyTorch
- `preprocessing.py` - Audio feature extraction
- `utils.py` - Data loading and splitting utilities
- `__init__.py` - Data module exports

#### Training (src/training/)
- `trainer.py` - CustomModelTrainer and base training logic
- `__init__.py` - Training module exports

#### Evaluation (src/evaluation/)
- `metrics.py` - Evaluation metrics and scoring
- `comparison.py` - Model comparison utilities
- `__init__.py` - Evaluation module exports

#### Utils (src/utils/)
- `config.py` - Configuration management
- `logger.py` - Logging utilities
- `__init__.py` - Utils module exports

### 🏛️ Legacy Code (Project/)

**Note**: This directory contains the original development code. Use for reference only.

| File | Purpose | Status |
|------|---------|--------|
| `emotion_recognition_models.py` | Original ML models | ✅ Reference |
| `data_preprocessing.py` | Original preprocessing | ✅ Reference |
| `train_models.py` | Original training script | ✅ Reference |
| `archive/` | Old experimental code | 🗄️ Archived |

## 🎯 Common Use Cases

### 1. Training a New Model

```bash
# Option A: Train all models
python scripts/train_all_models.py

# Option B: Train specific model type (edit script first)
python scripts/training/train_custom_models.py
```

**Files to modify**:
- `src/models/custom_models.py` - Add new model class
- `src/training/trainer.py` - Add training logic if needed
- `scripts/train_all_models.py` - Register new model

### 2. Adding New Features

**Audio Features**:
- Modify: `src/data/preprocessing.py`
- Function: `extract_features()`

**Model Architecture**:
- Add to: `src/models/custom_models.py`
- Inherit from: `torch.nn.Module`

**Training Process**:
- Extend: `src/training/trainer.py`
- Override: `train()` method

### 3. Analyzing Results

**Performance Comparison**:
```bash
python scripts/compare_models.py
```

**Detailed Analysis**:
- Check: `results/comprehensive_comparison_report.md`
- Visualizations: `results/visualizations/`

**Individual Model Results**:
- ResNet: `results/resnet/`
- LSTM: `results/lstm/`
- Transformer: `results/transformer/`

### 4. Making Predictions

**Single File**:
```bash
python scripts/predict_emotion.py \
  --model models/resnet/best_ResNet.pth \
  --audio audio.wav
```

**Batch Processing**:
```bash
python scripts/predict_emotion.py \
  --model models/resnet/best_ResNet.pth \
  --audio_dir audio_folder/ \
  --output predictions.json
```

## 🔍 Finding Specific Components

### 🎵 Audio Processing
- **Loading**: `src/data/preprocessing.py::load_audio()`
- **Features**: `src/data/preprocessing.py::extract_features()`
- **Normalization**: `src/data/preprocessing.py::normalize_audio()`

### 🧠 Model Definitions
- **ResNet**: `src/models/custom_models.py::EmotionResNet`
- **Transformer**: `src/models/custom_models.py::EmotionTransformer`
- **LSTM**: `src/models/custom_models.py::EmotionLSTM`
- **Wav2Vec2**: `src/models/pretrained_models.py::FixedWav2Vec2`

### 🎯 Training Components
- **Main Trainer**: `src/training/trainer.py::CustomModelTrainer`
- **Training Loop**: `src/training/trainer.py::train()`
- **Validation**: `src/training/trainer.py::validate()`

### 📊 Evaluation & Metrics
- **Accuracy**: `src/evaluation/metrics.py::calculate_accuracy()`
- **Classification Report**: `src/evaluation/metrics.py::classification_report()`
- **Confusion Matrix**: `src/evaluation/metrics.py::confusion_matrix()`

## 🛠️ Development Workflows

### Adding a New Model

1. **Define Model** (`src/models/custom_models.py`):
   ```python
   class NewModel(nn.Module):
       def __init__(self, num_classes=8):
           # Implementation
   ```

2. **Register Model** (`src/models/__init__.py`):
   ```python
   from .custom_models import NewModel
   ```

3. **Add Training** (`scripts/train_all_models.py`):
   ```python
   models_to_train = ['resnet', 'lstm', 'transformer', 'newmodel']
   ```

4. **Test Model**:
   ```bash
   python scripts/train_all_models.py
   ```

### Modifying Data Processing

1. **Update Preprocessing** (`src/data/preprocessing.py`)
2. **Test Changes**:
   ```python
   from src.data.preprocessing import extract_features
   features = extract_features("test_audio.wav")
   ```

3. **Retrain Models**:
   ```bash
   python scripts/train_all_models.py
   ```

### Debugging Training Issues

1. **Check Logs**: `logs/emotion_recognition_*.log`
2. **Validate Data**: `src/data/utils.py::validate_dataset()`
3. **Test Individual Components**:
   ```python
   from src.models.custom_models import EmotionResNet
   model = EmotionResNet()
   print(model)
   ```

## 📊 Understanding Results

### Model Performance Files

| File Pattern | Content |
|--------------|---------|
| `results/*/best_*.pth` | Trained model checkpoints |
| `results/*/test_results.pkl` | Detailed test results |
| `results/*/classification_report.pkl` | Sklearn classification report |
| `results/*/test_metrics.csv` | CSV format metrics |

### Comparison Reports

| File | Content |
|------|---------|
| `results/comprehensive_comparison_report.md` | Complete analysis |
| `results/model_comparison.png` | Performance visualization |
| `results/overall_comparison.csv` | Tabular comparison |

## 🎯 Performance Optimization

### GPU Utilization
- **Check**: `src/training/trainer.py::__init__()` - Device selection
- **Monitor**: Use `nvidia-smi` during training

### Memory Optimization
- **Batch Size**: `src/utils/config.py` - Reduce if OOM errors
- **Model Size**: `src/models/` - Check parameter counts

### Training Speed
- **Data Loading**: `src/data/dataset.py` - Parallel data loading
- **Mixed Precision**: `src/training/trainer.py` - Enable AMP

## 🔧 Configuration Guide

### Key Configuration Files

| File | Purpose |
|------|---------|
| `src/utils/config.py` | Main configuration |
| `requirements.txt` | Dependencies |
| `.gitignore` | Git exclusions |

### Common Configuration Changes

**Audio Settings**:
```python
# src/utils/config.py
AUDIO_CONFIG = {
    'sample_rate': 16000,  # Change sample rate
    'duration': 3.0,       # Change clip duration
    'n_mels': 64,          # Change mel bands
}
```

**Training Settings**:
```python
# src/utils/config.py
TRAINING_CONFIG = {
    'batch_size': 16,      # Adjust for memory
    'learning_rate': 0.001, # Tune learning rate
    'epochs': 30,          # Change training duration
}
```

## 🎯 Troubleshooting

### Common Issues

| Issue | Location | Solution |
|-------|----------|----------|
| CUDA out of memory | `src/utils/config.py` | Reduce batch_size |
| Audio file errors | `src/data/preprocessing.py` | Check file formats |
| Model not converging | `src/training/trainer.py` | Adjust learning rate |
| Import errors | `requirements.txt` | Update dependencies |

### Debug Commands

```bash
# Check data
python -c "from src.data.utils import validate_dataset; validate_dataset()"

# Test model
python -c "from src.models.custom_models import EmotionResNet; print(EmotionResNet())"

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 📞 Getting Help

1. **Documentation**: Start with `README.md`
2. **Issues**: Check existing GitHub issues
3. **Code Examples**: Look in `scripts/` directory
4. **Configuration**: Check `src/utils/config.py`

---

**Quick Reference Card**:
- 🚀 **Run Everything**: `python scripts/train_all_models.py`
- 🔍 **Compare Models**: `python scripts/compare_models.py`
- 🎯 **Make Prediction**: `python scripts/predict_emotion.py --help`
- 🧠 **View Models**: `src/models/`
- 📊 **Check Results**: `results/comprehensive_comparison_report.md`

Happy coding! 🎉
