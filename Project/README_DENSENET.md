# DenseNet Emotion Recognition Model

This implementation provides a DenseNet-based deep learning model for audio emotion recognition with an 80-10-10 train/validation/test split.

## Overview

The DenseNet (Densely Connected Convolutional Networks) model uses dense connectivity patterns where each layer receives feature maps from all preceding layers, enabling efficient feature reuse and gradient flow.

### Key Features

- **80-10-10 Data Split**: Proper stratified splitting for training (80%), validation (10%), and testing (10%)
- **Mel-Spectrogram Input**: Uses mel-spectrogram features extracted from audio files
- **Dense Connectivity**: Implements dense blocks with growth rate and transition layers
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and training curves
- **GPU Support**: Automatic CUDA detection and usage when available

## Model Architecture

### DenseNet Components

1. **Dense Blocks**: Groups of densely connected layers
   - Each layer receives inputs from all preceding layers
   - Feature map concatenation enables feature reuse
   - Bottleneck layers (1x1 conv) for computational efficiency

2. **Transition Layers**: Between dense blocks
   - Batch normalization + ReLU + 1x1 convolution + dropout + average pooling
   - Reduces feature map dimensions and controls model complexity

3. **Classification Head**: 
   - Global average pooling
   - Multi-layer perceptron with dropout
   - Final linear layer for emotion classification

### Default Configuration

```python
EmotionDenseNet(
    num_classes=8,           # 8 emotion classes
    growth_rate=32,          # Number of feature maps added per layer
    block_config=(6,12,24,16), # Layers per dense block
    num_init_features=64,    # Initial feature maps
    dropout=0.1              # Dropout rate
)
```

## File Structure

```
Project/
├── src/models/
│   └── custom_models.py      # DenseNet implementation
├── train_densenet.py         # Main training script
├── test_densenet.py          # Integration test script
└── README_DENSENET.md        # This documentation
```

## Usage

### 1. Quick Test

First, verify everything is working:

```bash
cd Project
python test_densenet.py
```

This will test:
- Model creation and forward pass
- Data loading from organized dataset
- Required dependencies
- CUDA availability

### 2. Training

Basic training with default parameters:

```bash
python train_densenet.py
```

### Training Options

```bash
python train_densenet.py \
    --data_path organized_by_emotion \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --results_dir results_densenet
```

### Available Arguments

- `--data_path`: Path to organized dataset (default: `organized_by_emotion`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--max_files`: Maximum files per emotion for testing (default: None - use all)
- `--results_dir`: Directory to save results (default: `results_densenet`)

### 3. Results

Training generates a timestamped results directory containing:

```
results_densenet_YYYYMMDD_HHMMSS/
├── best_densenet.pth         # Best model weights
├── training_history.png      # Loss and accuracy curves
├── confusion_matrix.png      # Test set confusion matrix
├── results.json              # Summary results
└── full_results.pkl          # Complete results with all data
```

## Data Requirements

### Dataset Structure

The model expects data organized by emotion:

```
organized_by_emotion/
├── angry/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── calm/
├── disgust/
├── fearful/
├── happy/
├── neutral/
├── sad/
└── surprised/
```

### Audio Preprocessing

- **Sample Rate**: 22,050 Hz
- **Duration**: 3 seconds (padded/trimmed)
- **Features**: 128-dimensional mel-spectrograms
- **Normalization**: Librosa utility normalization
- **Augmentation**: Optional (can be added to dataset class)

## Model Details

### Input Processing

1. Audio files loaded with librosa
2. Normalized and fixed to 3-second duration
3. Mel-spectrogram extraction:
   - n_mels: 128
   - n_fft: 2048
   - hop_length: 512
4. Log-scale conversion for better dynamic range

### Architecture Details

```
Input: [batch_size, 128, 130] (mel-spectrogram)
    ↓
Initial Conv: [1, 64, 7x7, stride=2] + BatchNorm + ReLU + MaxPool
    ↓
DenseBlock1: 6 layers, growth_rate=32
    ↓
Transition1: BatchNorm + ReLU + 1x1 Conv + Dropout + AvgPool
    ↓
DenseBlock2: 12 layers, growth_rate=32
    ↓
Transition2: BatchNorm + ReLU + 1x1 Conv + Dropout + AvgPool
    ↓
DenseBlock3: 24 layers, growth_rate=32
    ↓
Transition3: BatchNorm + ReLU + 1x1 Conv + Dropout + AvgPool
    ↓
DenseBlock4: 16 layers, growth_rate=32
    ↓
Global Average Pooling
    ↓
Classifier: 512 → 256 → 128 → 8 (with ReLU + Dropout)
    ↓
Output: [batch_size, 8] (emotion probabilities)
```

### Training Process

1. **Data Split**: Stratified 80-10-10 split maintains class balance
2. **Optimization**: Adam optimizer with weight decay
3. **Learning Rate**: ReduceLROnPlateau scheduler
4. **Loss Function**: CrossEntropyLoss
5. **Early Stopping**: Based on validation accuracy
6. **Batch Processing**: Efficient data loading with PyTorch DataLoader

## Performance Expectations

### Typical Results

- **Training Time**: ~2-4 hours (100 epochs, GPU)
- **Model Size**: ~15-25M parameters
- **Memory Usage**: ~2-4GB GPU memory (batch_size=32)
- **Expected Accuracy**: 60-80% on test set (depending on dataset quality)

### Monitoring Training

The training script provides:
- Real-time progress bars for each epoch
- Epoch-by-epoch metrics logging
- Best model checkpointing
- Learning rate scheduling
- Early stopping on plateau

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size (try 16 or 8)
   - Use smaller model configuration

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path for src/models

3. **Data Loading Issues**
   - Verify dataset organization
   - Check audio file formats (should be .wav)
   - Ensure sufficient disk space

4. **Poor Performance**
   - Increase training epochs
   - Adjust learning rate
   - Verify data quality and balance

### Requirements

```bash
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
numpy>=1.20.0
```

## Advanced Usage

### Custom Model Configuration

```python
from src.models.custom_models import EmotionDenseNet

# Smaller model for faster training
small_model = EmotionDenseNet(
    growth_rate=16,
    block_config=(4, 8, 12, 8),
    num_init_features=32
)

# Larger model for better accuracy
large_model = EmotionDenseNet(
    growth_rate=48,
    block_config=(8, 16, 32, 24),
    num_init_features=96
)
```

### Custom Data Loading

```python
from train_densenet import EmotionDataset

# Custom dataset with data augmentation
dataset = EmotionDataset(
    file_paths, labels,
    transform=your_transform_function,
    duration=4.0,  # Longer audio clips
    n_mels=256     # Higher resolution spectrograms
)
```

## Integration with Existing Framework

This DenseNet implementation is designed to work alongside the existing emotion recognition framework:

- Uses same emotion labels and data organization
- Compatible with existing preprocessing utilities
- Can be integrated into ensemble models
- Follows same evaluation and reporting patterns

## Future Improvements

- [ ] Data augmentation techniques
- [ ] Attention mechanisms
- [ ] Multi-scale temporal modeling
- [ ] Transfer learning from pretrained models
- [ ] Hyperparameter optimization
- [ ] Cross-validation evaluation
