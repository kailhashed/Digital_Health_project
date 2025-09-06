#!/usr/bin/env python3
"""
Main script to train all emotion recognition models
Uses the organized codebase structure
"""

import os
import sys
import warnings
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Imports from organized structure
from models import EmotionTransformer, EmotionLSTM, EmotionResNet
from models import FixedWav2Vec2Classifier, SimpleCNNAudioClassifier
from data import EmotionDataset, AudioDataset, load_emotion_data, train_test_split
from training import CustomModelTrainer, PretrainedModelTrainer
from evaluation import ModelEvaluator, ModelComparator
from utils import Config, setup_logger

import torch
from torch.utils.data import DataLoader

def main():
    """Main training function"""
    # Setup
    print("🚀 COMPREHENSIVE EMOTION RECOGNITION TRAINING")
    print("="*70)
    
    # Initialize configuration and logger
    Config.create_directories()
    Config.print_config()
    
    logger = setup_logger("main_training")
    logger.info("Starting comprehensive model training")
    
    # Load data
    print("\n📁 Loading Data...")
    try:
        file_paths, labels = load_emotion_data()
        print(f"✓ Loaded {len(file_paths)} files")
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(
            file_paths, labels, 
            test_ratio=Config.TEST_RATIO, 
            val_ratio=Config.VAL_RATIO
        )
        
        print(f"✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Store all results
    all_results = {}
    
    # 1. CUSTOM MODELS
    print("\n" + "="*70)
    print("🧠 TRAINING CUSTOM MODELS")
    print("="*70)
    
    custom_models = [
        ('Transformer', EmotionTransformer, 'spectrogram'),
        ('LSTM', EmotionLSTM, 'spectrogram'), 
        ('ResNet', EmotionResNet, 'spectrogram')
    ]
    
    for model_name, model_class, feature_type in custom_models:
        try:
            print(f"\n🚀 Training {model_name}...")
            
            # Create datasets
            train_dataset = EmotionDataset(X_train, y_train, feature_type=feature_type)
            val_dataset = EmotionDataset(X_val, y_val, feature_type=feature_type)
            test_dataset = EmotionDataset(X_test, y_test, feature_type=feature_type)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Initialize model and trainer
            num_classes = train_dataset.get_num_classes()
            model = model_class(num_classes=num_classes)
            trainer = CustomModelTrainer(model, Config.DEVICE)
            
            print(f"  Classes: {list(train_dataset.get_classes())}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train model
            model, best_val_acc, train_losses, val_accuracies = trainer.train(
                train_loader, val_loader, model_name,
                epochs=Config.EPOCHS, lr=Config.LEARNING_RATE,
                patience=Config.PATIENCE, save_dir=Config.MODELS_DIR
            )
            
            # Test model
            test_accuracy, all_preds, all_targets = trainer.test(test_loader)
            
            # Evaluate
            evaluator = ModelEvaluator(list(train_dataset.get_classes()))
            evaluation_results = evaluator.evaluate(all_targets, all_preds)
            
            # Store results
            all_results[f"Custom-{model_name}"] = {
                'type': 'Custom Deep Learning',
                'train': {
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies
                },
                'test': {
                    'accuracy': test_accuracy,
                    'predictions': all_preds,
                    'targets': all_targets
                },
                'evaluation': evaluation_results,
                'model_info': {
                    'classes': list(train_dataset.get_classes()),
                    'num_params': sum(p.numel() for p in model.parameters())
                }
            }
            
            print(f"✅ {model_name} completed - Val: {best_val_acc:.4f}, Test: {test_accuracy:.4f}")
            logger.info(f"{model_name} training completed - Val: {best_val_acc:.4f}, Test: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # 2. PRE-TRAINED MODELS
    print("\n" + "="*70)
    print("🤗 TRAINING PRE-TRAINED MODELS")
    print("="*70)
    
    pretrained_models = [
        ('FixedWav2Vec2', FixedWav2Vec2Classifier),
        ('SimpleCNNAudio', SimpleCNNAudioClassifier)
    ]
    
    for model_name, model_class in pretrained_models:
        try:
            print(f"\n🚀 Fine-tuning {model_name}...")
            
            # Create datasets (raw audio for pre-trained models)
            train_dataset = AudioDataset(X_train, y_train)
            val_dataset = AudioDataset(X_val, y_val)
            test_dataset = AudioDataset(X_test, y_test)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=Config.PRETRAINED_BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=Config.PRETRAINED_BATCH_SIZE, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=Config.PRETRAINED_BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Initialize model and trainer
            num_classes = train_dataset.get_num_classes()
            model = model_class(num_classes=num_classes)
            trainer = PretrainedModelTrainer(model, Config.DEVICE)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  Classes: {list(train_dataset.get_classes())}")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            
            # Train model
            model, best_val_acc, train_losses, val_accuracies = trainer.train(
                train_loader, val_loader, model_name,
                epochs=Config.PRETRAINED_EPOCHS, lr=Config.PRETRAINED_LEARNING_RATE,
                patience=Config.PRETRAINED_PATIENCE, save_dir=Config.MODELS_DIR
            )
            
            # Test model
            test_accuracy, all_preds, all_targets = trainer.test(test_loader)
            
            # Evaluate
            evaluator = ModelEvaluator(list(train_dataset.get_classes()))
            evaluation_results = evaluator.evaluate(all_targets, all_preds)
            
            # Store results
            all_results[f"Pretrained-{model_name}"] = {
                'type': 'Pre-trained Fine-tuned',
                'train': {
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies
                },
                'test': {
                    'accuracy': test_accuracy,
                    'predictions': all_preds,
                    'targets': all_targets
                },
                'evaluation': evaluation_results,
                'model_info': {
                    'classes': list(train_dataset.get_classes()),
                    'total_params': total_params,
                    'trainable_params': trainable_params
                }
            }
            
            print(f"✅ {model_name} completed - Val: {best_val_acc:.4f}, Test: {test_accuracy:.4f}")
            logger.info(f"{model_name} fine-tuning completed - Val: {best_val_acc:.4f}, Test: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ Error fine-tuning {model_name}: {e}")
            logger.error(f"Error fine-tuning {model_name}: {e}")
            continue
    
    # 3. SAVE RESULTS
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    # Save comprehensive results
    results_file = Config.get_results_path('comprehensive_training_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"✓ Results saved: {results_file}")
    logger.info(f"Results saved: {results_file}")
    
    # 4. FINAL COMPARISON
    print("\n" + "="*70)
    print("🏆 FINAL COMPARISON")
    print("="*70)
    
    # Create model comparator
    comparator = ModelComparator()
    
    # Convert results to comparison format
    comparison_results = {}
    for model_name, result in all_results.items():
        comparison_results[model_name] = {
            'type': result['type'],
            'val_acc': result['train']['best_val_acc'],
            'test_acc': result['test']['accuracy'],
            'trainable_params': result['model_info'].get('trainable_params', 'All'),
            'total_params': result['model_info'].get('total_params', result['model_info'].get('num_params', 'Unknown'))
        }
    
    comparator.results = comparison_results
    
    # Run comparison
    sorted_results = comparator.create_comparison_table()
    comparator.analyze_by_type()
    champions = comparator.find_champions()
    comparator.generate_insights()
    
    # Save comparison report
    comparator.save_comparison_report(Config.get_results_path('final_training_report.md'))
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    if sorted_results:
        best_model, best_result = sorted_results[0]
        print(f"\n🏆 CHAMPION: {best_model}")
        print(f"   Test Accuracy: {best_result['test_acc']:.4f} ({best_result['test_acc']*100:.2f}%)")
        print(f"   Model Type: {best_result['type']}")
    
    print(f"\n📁 Files Generated:")
    print(f"   Models: {Config.MODELS_DIR}/")
    print(f"   Results: {Config.RESULTS_DIR}/")
    print(f"   Logs: {Config.LOGS_DIR}/")
    
    logger.info("Comprehensive training completed successfully")
    
    return True

if __name__ == "__main__":
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    success = main()
    if success:
        print("\n✨ SUCCESS: All models trained and evaluated!")
    else:
        print("\n💥 FAILED: Training encountered errors!")

