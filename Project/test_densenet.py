#!/usr/bin/env python3
"""
Quick test script for DenseNet model to verify everything works
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src/models')

def test_model_creation():
    """Test if DenseNet model can be created"""
    print("Testing DenseNet model creation...")
    
    try:
        from custom_models import EmotionDenseNet
        
        # Create model
        model = EmotionDenseNet(
            num_classes=8,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_init_features=64,
            dropout=0.1
        )
        
        print("✓ DenseNet model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 128, 130)  # Batch of 2 mel spectrograms
        output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output classes: {output.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating DenseNet model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test if data can be loaded"""
    print("\nTesting data loading...")
    
    data_path = "organized_by_emotion"
    
    if not os.path.exists(data_path):
        print(f"✗ Data path '{data_path}' does not exist")
        return False
    
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    total_files = 0
    
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            total_files += len(files)
            print(f"  {emotion}: {len(files)} files")
        else:
            print(f"  {emotion}: directory not found")
    
    if total_files > 0:
        print(f"✓ Data loading test passed - found {total_files} total files")
        return True
    else:
        print("✗ No audio files found")
        return False

def test_training_script():
    """Test if training script imports work"""
    print("\nTesting training script imports...")
    
    try:
        # Test if all required modules can be imported
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import librosa
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        import json
        import pickle
        
        print("✓ All required modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_device_availability():
    """Test CUDA availability"""
    print("\nTesting device availability...")
    
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("! CUDA not available - will use CPU")
    
    return True

def main():
    """Run all tests"""
    print("DenseNet Integration Test")
    print("=" * 50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training Script Imports", test_training_script),
        ("Device Availability", test_device_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Ready for training.")
        print("\nTo start training, run:")
        print("  python train_densenet.py --epochs 50 --batch_size 32")
        print("  python train_densenet.py --help  # for all options")
    else:
        print("✗ Some tests failed. Please fix issues before training.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
