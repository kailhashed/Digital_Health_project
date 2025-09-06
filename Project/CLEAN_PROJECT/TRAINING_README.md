# 🎭 Parallel Model Training for Emotion Recognition

This directory contains scripts for training multiple emotion recognition models in parallel with GPU acceleration and early stopping.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Models in Parallel
```bash
python run_parallel_training.py
```

### 3. Run Individual Models
```bash
# Deep Learning Models (GPU)
python train_densenet.py
python train_crnn.py

# Classical ML Models (CPU)
python train_adaboost.py
python train_xgboost.py
python train_naivebayes.py
```

## 📊 Models Trained

### Deep Learning Models (GPU)
1. **DenseNet** - Densely connected convolutional network
2. **CRNN** - Convolutional Recurrent Neural Network

### Classical ML Models (CPU)
3. **AdaBoost** - Adaptive boosting with decision trees
4. **XGBoost** - Extreme gradient boosting
5. **Naive Bayes** - Gaussian Naive Bayes classifier

## ⚙️ Training Configuration

### Data Split
- **Training**: 80% of data
- **Validation**: 10% of data  
- **Testing**: 10% of data

### Early Stopping
- **Patience**: 20 epochs
- **Max Epochs**: 200 epochs
- **Min Delta**: 0.001

### GPU Support
- Automatic GPU detection and usage
- CUDA acceleration for deep learning models
- Fallback to CPU if GPU unavailable

## 📁 Directory Structure

```
CLEAN_PROJECT/
├── data_loader.py              # Data loading and preprocessing
├── train_densenet.py          # DenseNet training script
├── train_crnn.py              # CRNN training script
├── train_adaboost.py          # AdaBoost training script
├── train_xgboost.py           # XGBoost training script
├── train_naivebayes.py        # Naive Bayes training script
├── run_parallel_training.py   # Parallel execution script
├── requirements.txt           # Dependencies
├── models/                    # Trained model checkpoints
│   ├── densenet/
│   ├── crnn/
│   ├── adaboost/
│   ├── xgboost/
│   └── naivebayes/
└── results/                   # Training results and visualizations
    ├── densenet/
    ├── crnn/
    ├── adaboost/
    ├── xgboost/
    └── naivebayes/
```

## 🎯 Features

### Data Processing
- **Audio Loading**: 22kHz sampling rate, 3-second duration
- **Feature Extraction**: Mel-spectrograms for deep learning, comprehensive features for classical ML
- **Normalization**: Audio normalization and feature scaling
- **Data Augmentation**: Built-in data augmentation support

### Training Features
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Gradient Clipping**: Prevents gradient explosion
- **Progress Monitoring**: Real-time training progress with tqdm
- **Model Checkpointing**: Automatic saving of best models

### Evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrices**: Visual performance analysis
- **Training Curves**: Loss and accuracy visualization
- **Feature Importance**: Analysis of important features (classical ML)

## 📊 Expected Results

### Performance Targets
- **DenseNet**: 70-80% accuracy
- **CRNN**: 65-75% accuracy
- **XGBoost**: 60-70% accuracy
- **AdaBoost**: 55-65% accuracy
- **Naive Bayes**: 45-55% accuracy

### Training Time (Estimated)
- **DenseNet**: 2-4 hours (GPU)
- **CRNN**: 3-5 hours (GPU)
- **XGBoost**: 1-2 hours (CPU)
- **AdaBoost**: 30-60 minutes (CPU)
- **Naive Bayes**: 10-20 minutes (CPU)

## 🔧 System Requirements

### Minimum Requirements
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **CPU**: 8 cores
- **GPU**: NVIDIA GPU with 8GB VRAM (recommended)

### Recommended Requirements
- **RAM**: 32 GB
- **Storage**: 20 GB free space
- **CPU**: 16 cores
- **GPU**: NVIDIA RTX 3080/4080 or better

## 📈 Monitoring Training

### Real-time Monitoring
- Progress bars for each model
- System resource usage
- GPU memory utilization
- Training/validation metrics

### Log Files
- Individual model logs in `results/{model}/`
- Parallel training summary in `results/parallel_training_results.json`
- Error logs for debugging

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Use gradient accumulation
   - Close other GPU applications

2. **Slow Training**
   - Check GPU utilization
   - Increase number of workers
   - Use mixed precision training

3. **Poor Performance**
   - Check data quality
   - Adjust hyperparameters
   - Increase training time

### Debug Mode
```bash
# Run with verbose output
python train_densenet.py --verbose

# Check data loading
python data_loader.py
```

## 📝 Customization

### Hyperparameters
Edit the training scripts to modify:
- Learning rates
- Batch sizes
- Model architectures
- Early stopping patience

### Data Augmentation
Modify `data_loader.py` to add:
- Time stretching
- Pitch shifting
- Noise injection
- Speed variation

### Model Architectures
Customize models in:
- `src/models/custom_models.py` (DenseNet)
- `train_crnn.py` (CRNN architecture)

## 🎉 Results Analysis

After training, check:
1. **Model Performance**: `results/{model}/{model}_results.json`
2. **Training Curves**: `results/{model}/training_history.png`
3. **Confusion Matrices**: `results/{model}/confusion_matrix.png`
4. **Best Models**: `models/{model}/best_{model}_epoch_X.pth`

## 📞 Support

For issues or questions:
1. Check the logs in `results/`
2. Verify system requirements
3. Check GPU/CUDA installation
4. Review error messages in training output

---

**Happy Training! 🚀**

*All models will train until completion or early stopping is triggered. The parallel execution ensures maximum efficiency while maintaining model quality.*
