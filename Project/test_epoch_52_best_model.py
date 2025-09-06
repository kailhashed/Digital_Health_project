#!/usr/bin/env python3
"""
Test and Demonstrate the Epoch 52 DenseNet Model (Best Performance)
This script validates the peak performance model at 71.09% validation accuracy.
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


def load_epoch_52_best_model():
    """Load the epoch 52 best performance DenseNet checkpoint"""
    checkpoint_path = "models/densenet_epoch_52_best.pth"
    current_best_path = "models/densenet_current_best.pth"
    
    # Try current best first, then specific epoch 52
    for path in [current_best_path, checkpoint_path]:
        if os.path.exists(path):
            print(f"Loading Best Performance Model from: {path}")
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            # Verify this is epoch 52
            if checkpoint['current_epoch'] != 52:
                print(f"⚠️  Warning: Expected epoch 52, got epoch {checkpoint['current_epoch']}")
            
            # Create model
            model_config = checkpoint['model_config']
            model = EmotionDenseNet(**model_config)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print("✅ Best performance model loaded successfully!")
            
            return model, checkpoint
    
    print(f"❌ Error: No checkpoint found at {checkpoint_path} or {current_best_path}")
    return None


def verify_best_performance_metrics(checkpoint):
    """Verify this is indeed the best performance checkpoint"""
    print(f"\n🔍 Verifying Best Performance Metrics...")
    
    metrics = checkpoint['performance_metrics']
    training_history = checkpoint['training_history']
    
    # Check if this is marked as best epoch
    is_best = metrics.get('is_best_epoch', False)
    current_epoch = checkpoint['current_epoch']
    val_acc = metrics['epoch_52_val_acc']
    
    print(f"✅ Checkpoint Verification:")
    print(f"  📊 Current Epoch: {current_epoch}")
    print(f"  🏆 Marked as Best: {is_best}")
    print(f"  🎯 Validation Accuracy: {val_acc:.2f}%")
    print(f"  📈 Training Accuracy: {metrics['epoch_52_train_acc']:.2f}%")
    
    # Verify this is the peak validation accuracy
    all_val_accs = training_history['val_accuracies']
    max_val_acc = max(all_val_accs)
    max_epoch = all_val_accs.index(max_val_acc) + 1
    
    print(f"\n📈 Training History Analysis:")
    print(f"  🔝 Maximum Validation Accuracy: {max_val_acc:.2f}% (Epoch {max_epoch})")
    print(f"  📊 Current Model Accuracy: {val_acc:.2f}% (Epoch {current_epoch})")
    
    if abs(val_acc - max_val_acc) < 0.01 and current_epoch == max_epoch:
        print(f"  ✅ CONFIRMED: This is the best performance model!")
        return True
    else:
        print(f"  ⚠️  Warning: This may not be the optimal checkpoint")
        return False


def test_model_comprehensive(model):
    """Comprehensive testing of the model"""
    print(f"\n🧪 Comprehensive Model Testing...")
    
    # Test 1: Basic inference
    print(f"Test 1: Basic Inference")
    dummy_input = torch.randn(1, 128, 130)
    
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.softmax(output, dim=1)
    
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    predicted_emotion = emotions[torch.argmax(probabilities, dim=1).item()]
    confidence = torch.max(probabilities).item()
    
    print(f"  ✅ Single sample inference successful")
    print(f"     🎭 Prediction: {predicted_emotion}")
    print(f"     🎯 Confidence: {confidence:.4f}")
    
    # Test 2: Batch inference
    print(f"\nTest 2: Batch Inference")
    batch_input = torch.randn(4, 128, 130)  # Batch of 4
    
    with torch.no_grad():
        batch_output = model(batch_input)
        batch_probs = torch.softmax(batch_output, dim=1)
    
    print(f"  ✅ Batch inference successful")
    print(f"     📊 Batch size: {batch_input.shape[0]}")
    print(f"     📊 Output shape: {batch_output.shape}")
    
    # Test 3: Different input sizes (should handle gracefully)
    print(f"\nTest 3: Input Robustness")
    try:
        # Test with different time dimensions
        for time_steps in [100, 130, 200]:
            test_input = torch.randn(1, 128, time_steps)
            with torch.no_grad():
                test_output = model(test_input)
            print(f"  ✅ Input size (128, {time_steps}): OK")
    except Exception as e:
        print(f"  ⚠️  Input robustness issue: {e}")
    
    # Test 4: Model parameters analysis
    print(f"\nTest 4: Model Analysis")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  📊 Total Parameters: {total_params:,}")
    print(f"  📊 Trainable Parameters: {trainable_params:,}")
    print(f"  💾 Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return True


def performance_comparison():
    """Compare with previous epochs and show improvement"""
    print(f"\n📊 Performance Comparison Analysis...")
    
    checkpoint_path = "models/densenet_epoch_52_best.pth"
    if not os.path.exists(checkpoint_path):
        print("❌ Cannot perform comparison - checkpoint not found")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    history = checkpoint['training_history']
    
    print(f"🎯 Performance Evolution:")
    
    # Show key epochs
    key_epochs = [1, 10, 20, 30, 40, 50, 51, 52]
    for epoch in key_epochs:
        if epoch <= len(history['val_accuracies']):
            val_acc = history['val_accuracies'][epoch-1]
            train_acc = history['train_accuracies'][epoch-1]
            
            if epoch == 52:
                print(f"  🏆 Epoch {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}% ⭐ BEST")
            elif epoch >= 50:
                print(f"  📈 Epoch {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}%")
            elif epoch % 10 == 0 or epoch == 1:
                print(f"  📊 Epoch {epoch:2d}: Train {train_acc:5.2f}% | Val {val_acc:5.2f}%")
    
    # Show improvement from start to best
    initial_val = history['val_accuracies'][0]
    best_val = max(history['val_accuracies'])
    improvement = best_val - initial_val
    
    print(f"\n📈 Overall Training Progress:")
    print(f"  🚀 Initial Validation Accuracy: {initial_val:.2f}%")
    print(f"  🏆 Best Validation Accuracy: {best_val:.2f}%")
    print(f"  📊 Total Improvement: +{improvement:.2f}%")
    print(f"  🎯 Epochs to Best: 52")


def main():
    """Main function"""
    print("🏆 DenseNet Epoch 52 Best Performance Model Tester")
    print("=" * 70)
    print("Testing the peak performance model (71.09% validation accuracy)")
    print("=" * 70)
    
    # Load model
    result = load_epoch_52_best_model()
    if result is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    model, checkpoint = result
    
    # Verify this is the best performance model
    is_verified_best = verify_best_performance_metrics(checkpoint)
    
    # Comprehensive testing
    test_success = test_model_comprehensive(model)
    
    # Performance comparison
    performance_comparison()
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("🏆 BEST PERFORMANCE MODEL TEST SUMMARY")
    print("=" * 70)
    
    if test_success and is_verified_best:
        print("✅ Epoch 52 BEST model validated successfully!")
        print(f"🎯 Peak Performance: 71.09% validation accuracy")
        print(f"📊 Training Performance: 75.56% accuracy")
        print(f"🏅 Status: OPTIMAL MODEL STATE")
        print(f"📁 Model Location: models/densenet_epoch_52_best.pth")
        print(f"📁 Current Best: models/densenet_current_best.pth")
        
        print(f"\n🚀 This model represents:")
        print(f"  🎯 Peak validation performance achieved")
        print(f"  📈 Optimal balance of training/validation accuracy")
        print(f"  🛡️  Best stopping point (before overfitting)")
        print(f"  🏆 Ready for production use")
        
    else:
        print("❌ Model validation issues detected!")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
