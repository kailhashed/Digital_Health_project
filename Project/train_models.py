#!/usr/bin/env python3
"""
Enhanced Training Script for Emotion Recognition Models
Uses improved feature extraction and model architectures for higher accuracy.
"""

import os
import sys
import argparse
import torch
import numpy as np
from emotion_recognition_models import EnhancedEmotionRecognitionTrainer
from data_preprocessing import DatasetValidator, AudioPreprocessor
import warnings
warnings.filterwarnings('ignore')

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - will use CPU")
    
    # Check data directory
    data_path = "organized_by_emotion"
    if not os.path.exists(data_path):
        print(f"ERROR: Data directory '{data_path}' not found!")
        print("Please run the dataset organization script first.")
        return False
    
    print("Environment check passed!")
    return True

def validate_dataset():
    """Validate the dataset before training"""
    print("\nValidating dataset...")
    
    validator = DatasetValidator("organized_by_emotion")
    valid_files, invalid_files = validator.validate_audio_files()
    
    if len(valid_files) == 0:
        print("ERROR: No valid audio files found!")
        return False
    
    print(f"Found {len(valid_files)} valid files")
    if len(invalid_files) > 0:
        print(f"Found {len(invalid_files)} invalid files (will be skipped)")
    
    # Get dataset statistics
    stats = validator.get_dataset_statistics()
    validator.print_statistics(stats)
    
    return True

def train_classical_ml_only():
    """Train only classical ML models with enhanced features"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED CLASSICAL ML MODELS")
    print("="*60)
    
    trainer = EnhancedEmotionRecognitionTrainer()
    file_paths, labels = trainer.load_data(max_files_per_emotion=None)  # Use all available files
    
    if len(file_paths) == 0:
        print("ERROR: No data loaded!")
        return None
    
    print(f"Loaded {len(file_paths)} files for training")
    
    try:
        # Split data into train/validation/test (60/20/20)
        from sklearn.model_selection import train_test_split
        
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} files")
        print(f"Validation set: {len(X_val)} files") 
        print(f"Test set: {len(X_test)} files")
        
        # Train on training set
        print("\nTraining on training set...")
        train_results = trainer.train_classical_ml(X_train, y_train, test_size=0.0)  # No test split since we have separate test set
        
        if train_results is None:
            print("ERROR: Training failed!")
            return None
        
        # Test on test set
        print("\nTesting on test set...")
        test_results = trainer.test_classical_ml(X_test, y_test, train_results)
        
        # Combine all results
        combined_results = {
            'train': train_results,
            'test': test_results
        }
        
        # Save results
        print("\nSaving enhanced classical ML results...")
        os.makedirs('results', exist_ok=True)
        
        import pickle
        with open('results/enhanced_classical_ml_results.pkl', 'wb') as f:
            pickle.dump(combined_results, f)
        
        return combined_results
        
    except Exception as e:
        print(f"ERROR during enhanced classical ML training: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_deep_learning_only():
    """Train only deep learning models with enhanced architectures"""
    print("\n" + "="*60)
    print("TRAINING ENHANCED DEEP LEARNING MODELS")
    print("="*60)
    
    trainer = EnhancedEmotionRecognitionTrainer()
    file_paths, labels = trainer.load_data(max_files_per_emotion=None)  # Use all available files
    
    if len(file_paths) == 0:
        print("ERROR: No data loaded!")
        return None
    
    print(f"Loaded {len(file_paths)} files for training")
    
    try:
        # Split data into train/validation/test (60/20/20)
        from sklearn.model_selection import train_test_split
        
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Second split: 75% train, 25% validation (of the 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} files")
        print(f"Validation set: {len(X_val)} files") 
        print(f"Test set: {len(X_test)} files")
        
        # Train on training set with validation
        print("\nTraining on training set with validation...")
        train_results = trainer.train_deep_learning_with_validation(
            X_train, y_train, X_val, y_val,
            epochs=150,  # Increased for better performance with full dataset
            batch_size=128 if torch.cuda.is_available() else 64  # Larger batch size for full dataset
        )
        
        if train_results is None:
            print("ERROR: Training failed!")
            return None
        
        # Test on test set
        print("\nTesting on test set...")
        test_results = trainer.test_deep_learning(X_test, y_test, train_results)
        
        # Combine all results
        combined_results = {
            'train': train_results,
            'test': test_results
        }
        
        # Save results
        print("\nSaving enhanced deep learning results...")
        os.makedirs('results', exist_ok=True)
        
        import pickle
        with open('results/enhanced_deep_learning_results.pkl', 'wb') as f:
            pickle.dump(combined_results, f)
        
        return combined_results
        
    except Exception as e:
        print(f"ERROR during enhanced deep learning training: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_both_models():
    """Train both enhanced classical ML and deep learning models"""
    print("\n" + "="*60)
    print("TRAINING BOTH ENHANCED MODELS")
    print("="*60)
    
    trainer = EnhancedEmotionRecognitionTrainer()
    file_paths, labels = trainer.load_data(max_files_per_emotion=None)  # Use all available files
    
    if len(file_paths) == 0:
        print("ERROR: No data loaded!")
        return None, None
    
    # Train classical ML
    print("\nTraining Enhanced Classical ML models...")
    classical_results = trainer.train_classical_ml(file_paths, labels, test_size=0.2)
    
    # Train deep learning
    print("\nTraining Enhanced Deep Learning models...")
    # Split data for deep learning
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    deep_results = trainer.train_deep_learning_with_validation(
        X_train, y_train, X_val, y_val,
        epochs=150, 
        batch_size=128 if torch.cuda.is_available() else 64
    )
    
    # Test deep learning
    test_results = trainer.test_deep_learning(X_test, y_test, deep_results)
    deep_results = {'train': deep_results, 'test': test_results}
    
    # Plot comparison
    trainer.plot_results(classical_results, deep_results['train'])
    
    # Save results
    print("\nSaving enhanced results...")
    os.makedirs('results', exist_ok=True)
    
    import pickle
    with open('results/enhanced_classical_ml_results.pkl', 'wb') as f:
        pickle.dump(classical_results, f)
    
    with open('results/enhanced_deep_learning_results.pkl', 'wb') as f:
        pickle.dump(deep_results, f)
    
    return classical_results, deep_results

def print_final_summary(classical_results=None, deep_results=None):
    """Print final summary of enhanced results"""
    print("\n" + "="*60)
    print("ENHANCED TRAINING SUMMARY")
    print("="*60)
    
    if classical_results:
        print("\nEnhanced Classical ML Results:")
        print("-" * 30)
        for name, result in classical_results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                print(f"  {name:15}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            elif isinstance(result, dict) and 'cv_score' in result:
                print(f"  {name:15}: {result['cv_score']:.4f} ({result['cv_score']*100:.2f}%)")
            else:
                print(f"  {name:15}: Result format not recognized")
    
    if deep_results:
        print("\nEnhanced Deep Learning Results:")
        print("-" * 30)
        if isinstance(deep_results, dict) and 'train' in deep_results:
            for name, result in deep_results['train'].items():
                if isinstance(result, dict) and 'best_accuracy' in result:
                    print(f"  {name:15}: {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
                else:
                    print(f"  {name:15}: Result format not recognized")
        else:
            for name, result in deep_results.items():
                if isinstance(result, dict) and 'best_accuracy' in result:
                    print(f"  {name:15}: {result['best_accuracy']:.4f} ({result['best_accuracy']*100:.2f}%)")
                else:
                    print(f"  {name:15}: Result format not recognized")
    
    # Find best model
    best_accuracy = 0
    best_model = None
    
    if classical_results:
        for name, result in classical_results.items():
            acc = result.get('accuracy', result.get('cv_score', 0))
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = f"{name} (Enhanced Classical ML)"
    
    if deep_results:
        if isinstance(deep_results, dict) and 'train' in deep_results:
            for name, result in deep_results['train'].items():
                acc = result.get('best_accuracy', 0)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = f"{name} (Enhanced Deep Learning)"
        else:
            for name, result in deep_results.items():
                acc = result.get('best_accuracy', 0)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = f"{name} (Enhanced Deep Learning)"
    
    if best_model:
        print(f"\nBest Enhanced Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    print(f"\nEnhanced models saved in 'models/' directory")
    print(f"Enhanced results saved in 'results/' directory")
    print(f"Enhanced plots saved as 'enhanced_model_comparison.png'")

def main():
    """Main enhanced training function"""
    parser = argparse.ArgumentParser(description='Train Enhanced Emotion Recognition Models')
    parser.add_argument('--mode', choices=['classical', 'deep', 'both'], default='both',
                       help='Training mode: classical ML only, deep learning only, or both')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset, do not train')
    
    args = parser.parse_args()
    
    print("Enhanced Emotion Recognition Model Training")
    print("="*50)
    
    # Check environment
    if not check_environment():
        return
    
    # Validate dataset
    if not validate_dataset():
        return
    
    if args.validate_only:
        print("\nDataset validation completed. Exiting.")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train models based on mode
    classical_results = None
    deep_results = None
    
    if args.mode in ['classical', 'both']:
        print("\nStarting Enhanced Classical ML training...")
        classical_results = train_classical_ml_only()
        if classical_results is not None:
            print("✓ Enhanced Classical ML training completed successfully!")
        else:
            print("✗ Enhanced Classical ML training failed!")
    
    if args.mode in ['deep', 'both']:
        print("\nStarting Enhanced Deep Learning training...")
        deep_results = train_deep_learning_only()
        if deep_results is not None:
            print("✓ Enhanced Deep Learning training completed successfully!")
        else:
            print("✗ Enhanced Deep Learning training failed!")
    
    # Print final summary
    print_final_summary(classical_results, deep_results)
    
    if classical_results is not None or deep_results is not None:
        print("\nEnhanced training completed successfully!")
    else:
        print("\nEnhanced training failed for all models!")

if __name__ == "__main__":
    main()
