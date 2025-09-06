#!/usr/bin/env python3
"""
Test and Demonstrate the Epoch 54 DenseNet Model
"""

import os
import sys
import torch
import numpy as np
import librosa
from datetime import datetime

# Add path for models
sys.path.append('src/models')
from custom_models import EmotionDenseNet


def load_epoch_54_model():
    """Load the epoch 54 DenseNet checkpoint"""
    checkpoint_path = "models/densenet_epoch_54.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    print("Loading Epoch 54 DenseNet Model...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model_config = checkpoint['model_config']
    model = EmotionDenseNet(**model_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    return model, checkpoint


def test_model_inference(model):
    """Test model inference with dummy data"""
    print("\nTesting model inference...")
    
    # Create dummy mel-spectrogram input
    dummy_input = torch.randn(1, 128, 130)  # Batch size 1, 128 mel bins, 130 time steps
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.softmax(output, dim=1)
    
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    print("✅ Inference successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\nPredicted emotion probabilities:")
    for i, emotion in enumerate(emotions):
        print(f"  {emotion:10}: {probabilities[0][i].item():.4f}")
    
    predicted_emotion = emotions[torch.argmax(probabilities, dim=1).item()]
    print(f"\nPredicted emotion: {predicted_emotion}")


def display_training_info(checkpoint):
    """Display training information from checkpoint"""
    print(f"\n{'='*50}")
    print("EPOCH 54 MODEL INFORMATION")
    print(f"{'='*50}")
    
    # Model configuration
    config = checkpoint['model_config']
    print(f"Model Configuration:")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Growth Rate: {config['growth_rate']}")
    print(f"  Block Config: {config['block_config']}")
    print(f"  Initial Features: {config['num_init_features']}")
    print(f"  Dropout: {config['dropout']}")
    
    # Training metrics
    metrics = checkpoint['performance_metrics']
    print(f"\nEpoch 54 Performance:")
    print(f"  Training Loss: {metrics['epoch_54_train_loss']:.4f}")
    print(f"  Training Accuracy: {metrics['epoch_54_train_acc']:.2f}%")
    print(f"  Validation Loss: {metrics['epoch_54_val_loss']:.4f}")
    print(f"  Validation Accuracy: {metrics['epoch_54_val_acc']:.2f}%")
    
    print(f"\nBest Performance (Epoch {metrics['best_val_acc_epoch']}):")
    print(f"  Best Validation Accuracy: {metrics['best_val_acc_value']:.2f}%")
    
    # Training history summary
    history = checkpoint['training_history']
    print(f"\nTraining History Summary:")
    print(f"  Total Epochs: {len(history['train_losses'])}")
    print(f"  Final Training Accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"  Final Validation Accuracy: {history['val_accuracies'][-1]:.2f}%")
    
    # Creation info
    creation_info = checkpoint['creation_info']
    print(f"\nCheckpoint Information:")
    print(f"  Created: {creation_info['created_at']}")
    print(f"  Source: {creation_info['created_from']}")
    print(f"  Note: {creation_info['note']}")


def analyze_model_architecture(model):
    """Analyze and display model architecture details"""
    print(f"\n{'='*50}")
    print("MODEL ARCHITECTURE ANALYSIS")
    print(f"{'='*50}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parameter Count:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    print(f"\nModel Structure:")
    print(f"  Input: Audio → Mel-Spectrogram (128 x 130)")
    print(f"  Architecture: DenseNet with 4 Dense Blocks")
    print(f"  Block Config: (6, 12, 24, 16) layers per block")
    print(f"  Growth Rate: 32 features per layer")
    print(f"  Output: 8 emotion classes")
    
    # Count layer types
    conv_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    bn_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    linear_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    
    print(f"\nLayer Breakdown:")
    print(f"  Convolutional Layers: {conv_layers}")
    print(f"  Batch Normalization Layers: {bn_layers}")
    print(f"  Linear Layers: {linear_layers}")


def main():
    """Main function"""
    print("DenseNet Epoch 54 Model Tester")
    print("=" * 60)
    print("This script loads and tests the DenseNet model saved at epoch 54")
    print("=" * 60)
    
    # Load model
    result = load_epoch_54_model()
    if result is None:
        print("Failed to load model. Exiting.")
        return
    
    model, checkpoint = result
    
    # Display information
    display_training_info(checkpoint)
    
    # Analyze architecture
    analyze_model_architecture(model)
    
    # Test inference
    test_model_inference(model)
    
    print(f"\n{'='*60}")
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print("The Epoch 54 DenseNet model is ready for use!")
    print(f"Model file: models/densenet_epoch_54.pth")
    print(f"Model size: {os.path.getsize('models/densenet_epoch_54.pth') / 1024 / 1024:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
