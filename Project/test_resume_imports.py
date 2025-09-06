#!/usr/bin/env python3
"""
Test script to verify imports for resumed training
"""

import os
import sys

print("Testing imports for resumed training...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
models_dir = os.path.join(project_dir, 'src', 'models')
sys.path.insert(0, models_dir)
sys.path.insert(0, project_dir)

print(f"Added to path: {models_dir}")
print(f"Added to path: {project_dir}")

try:
    from custom_models import EmotionDenseNet
    print("✓ Successfully imported EmotionDenseNet")
    
    # Test model creation
    model = EmotionDenseNet(num_classes=8)
    print(f"✓ Successfully created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
except Exception as e:
    print(f"✗ Failed to import EmotionDenseNet: {e}")
    import traceback
    traceback.print_exc()

try:
    from train_densenet import EmotionDataset, DenseNetTrainer
    print("✓ Successfully imported EmotionDataset and DenseNetTrainer")
except Exception as e:
    print(f"✗ Failed to import from train_densenet: {e}")
    import traceback
    traceback.print_exc()

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

# Test checkpoint loading
checkpoint_path = "results_densenet_20250906_230232/best_densenet.pth"
if os.path.exists(checkpoint_path):
    print(f"✓ Checkpoint file found: {checkpoint_path}")
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        print(f"  - Best validation accuracy: {checkpoint.get('best_val_accuracy', 'N/A')}")
        print(f"  - Training epochs: {len(checkpoint.get('training_history', {}).get('train_losses', []))}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
else:
    print(f"✗ Checkpoint file not found: {checkpoint_path}")

# Test data path
data_path = "organized_by_emotion"
if os.path.exists(data_path):
    print(f"✓ Data directory found: {data_path}")
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    total_files = 0
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            total_files += len(files)
    print(f"  - Total audio files: {total_files}")
else:
    print(f"✗ Data directory not found: {data_path}")

print("\nImport test completed!")
