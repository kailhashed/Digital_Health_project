# 🎭 Audio Emotion Recognition System - Clean Version

A comprehensive deep learning system for recognizing emotions from speech audio using multiple neural network architectures and pre-trained models.

## 🏆 Performance Results

### Top Performing Models

| Rank | Model | Type | Test Accuracy | Validation Accuracy | Parameters | Training Time |
|------|-------|------|---------------|-------------------|------------|---------------|
| 🥇 | **ResNet** | Custom Deep Learning | **76.34%** | **67.04%** | 191,416 | ~45 min |
| 🥈 | SimpleCNN | Custom Deep Learning | 68.72% | 65.41% | 191,416 | ~40 min |
| 🥉 | SimpleCNNAudio | Pre-trained Fine-tuned | 60.62% | 58.48% | 4,456,648 | ~30 min |
| 4 | FixedWav2Vec2 | Pre-trained Fine-tuned | 58.90% | 61.22% | ~95M | ~40 min |
| 5 | Transformer | Custom Deep Learning | 15.92% | 16.87% | 298,568 | ~35 min |
| 6 | LSTM | Custom Deep Learning | 15.07% | 16.95% | 191,176 | ~25 min |

**Champion**: Custom ResNet achieved **76.34% test accuracy** (511% improvement over random baseline of 12.5%)

## 📊 Dataset Information

### Combined Dataset Statistics

| Dataset | Files | Emotions | Source |
|---------|-------|----------|--------|
| **RAVDESS** | 1,440 | 8 emotions | [Zenodo](https://zenodo.org/record/1188976) |
| **CREMA-D** | 7,442 | 6 emotions | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| **TESS** | 2,800 | 7 emotions | [Kaggle](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess) |
| **Total** | **11,682** | **8 emotions** | Combined |

### Emotion Distribution

| Emotion | Files | Percentage |
|---------|-------|------------|
| Angry | 1,863 | 15.9% |
| Disgust | 1,863 | 15.9% |
| Fearful | 1,863 | 15.9% |
| Happy | 1,863 | 15.9% |
| Sad | 1,863 | 15.9% |
| Neutral | 1,583 | 13.6% |
| Surprised | 592 | 5.1% |
| Calm | 192 | 1.6% |

## 🏗️ Clean Project Structure

```
CLEAN_PROJECT/
├── 📁 src/                          # Source code modules
│   ├── 📁 models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── custom_models.py         # Transformer, LSTM, ResNet, DenseNet
│   │   ├── pretrained_models.py     # Wav2Vec2, CNN models
│   │   └── ensemble_model.py        # Ensemble models
│   ├── 📁 data/                     # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset classes
│   │   ├── preprocessing.py        # Audio preprocessing
│   │   └── utils.py                # Data utilities
│   ├── 📁 training/                # Training components
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training classes
│   │   └── densenet_trainer.py     # DenseNet specific trainer
│   ├── 📁 evaluation/              # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── comparison.py           # Model comparison
│   └── 📁 utils/                   # General utilities
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       └── logger.py               # Logging utilities
├── 📁 scripts/                     # Execution scripts
│   ├── train_all_models.py        # Train all models
│   ├── compare_models.py          # Compare performance
│   ├── predict_emotion.py         # Make predictions
│   ├── train_densenet_full_dataset.py  # DenseNet training
│   ├── train_optimized_densenet.py     # Optimized DenseNet training
│   ├── test_epoch_52_best_model.py     # Test best DenseNet
│   ├── test_ensemble_fixed.py          # Test ensemble model
│   └── test_ensemble.py               # Basic ensemble test
├── 📁 models/                      # Saved model checkpoints
│   ├── 📁 densenet/               # DenseNet models
│   │   ├── densenet_current_best.pth
│   │   ├── densenet_epoch_52_best.pth
│   │   └── densenet_epoch_54_backup.pth
│   ├── 📁 resnet/                 # ResNet models
│   │   └── best_ResNet.pth
│   ├── 📁 simplecnn/              # SimpleCNN models
│   │   └── best_SimpleCNN.pth
│   ├── 📁 lstm/                   # LSTM models
│   │   └── best_LSTM.pth
│   ├── 📁 transformer/            # Transformer models
│   │   └── best_Transformer.pth
│   └── 📁 pretrained/             # Pre-trained models
│       ├── 📁 simplecnnaudio/
│       └── 📁 wav2vec2/
├── 📁 results/                     # Training results & reports
│   ├── 📁 densenet/               # DenseNet results
│   ├── 📁 ensemble/               # Ensemble results
│   └── 📁 comparison/             # Model comparison results
├── 📁 data/                        # Dataset
│   └── 📁 organized_by_emotion/   # Processed dataset (8 emotion folders)
├── 📁 docs/                        # Documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 paths.json                   # Configuration paths
└── 📄 README_CLEAN.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Models
```bash
python scripts/train_all_models.py
```

### 3. Compare Model Performance
```bash
python scripts/compare_models.py
```

### 4. Make Predictions
```bash
python scripts/predict_emotion.py --model models/resnet/best_ResNet.pth --audio audio.wav
```

## 🧠 Model Details

### Custom Deep Learning Models
- **ResNet**: Residual neural network with skip connections
- **SimpleCNN**: Basic convolutional neural network
- **Transformer**: Self-attention based architecture
- **LSTM**: Long short-term memory network
- **DenseNet**: Densely connected convolutional network

### Pre-trained Fine-tuned Models
- **Wav2Vec2**: Pre-trained speech representation model
- **SimpleCNNAudio**: Audio-specific CNN architecture

## 📊 Results and Analysis

### Model Performance Summary
- **Best Overall**: ResNet (76.34% test accuracy)
- **Most Efficient**: SimpleCNN (68.72% accuracy, fast training)
- **Best Pre-trained**: SimpleCNNAudio (60.62% accuracy)

### Key Findings
1. **Custom models outperform pre-trained models** for this specific emotion recognition task
2. **ResNet architecture** provides the best balance of accuracy and efficiency
3. **Transformer and LSTM** showed poor performance, likely due to insufficient data or architecture mismatch
4. **Class imbalance** affects model performance, especially for underrepresented emotions

## 🔧 Development

### Adding New Models
1. Define model in `src/models/custom_models.py`
2. Add training logic in `src/training/trainer.py`
3. Register in `scripts/train_all_models.py`

### Modifying Data Processing
1. Update `src/data/preprocessing.py`
2. Test with `src/data/utils.py`
3. Retrain models to see impact

## 📝 Notes

This clean version consolidates all the best components from the original project:
- ✅ Removed duplicate files and scripts
- ✅ Organized models by type
- ✅ Consolidated results directories
- ✅ Maintained all working functionality
- ✅ Preserved all trained models and results
- ✅ Clean, maintainable structure

## 🎯 Next Steps

1. **Hyperparameter Optimization**: Fine-tune ResNet for even better performance
2. **Data Augmentation**: Implement audio augmentation techniques
3. **Ensemble Methods**: Combine best models for improved accuracy
4. **Real-time Inference**: Optimize models for real-time emotion recognition
5. **Mobile Deployment**: Convert models for mobile applications

---

**Last Updated**: September 7, 2025  
**Version**: Clean v1.0  
**Status**: Production Ready ✅
