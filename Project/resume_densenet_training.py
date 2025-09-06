#!/usr/bin/env python3
"""
Resume DenseNet Training Script
Resumes training from a saved checkpoint with early stopping based on validation accuracy.
"""

import os
import sys
import argparse
import numpy as np
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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the DenseNet model and dataset
# Handle both running from Project directory and from root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
models_dir = os.path.join(project_dir, 'src', 'models')
sys.path.insert(0, models_dir)
sys.path.insert(0, project_dir)

from custom_models import EmotionDenseNet
from train_densenet import EmotionDataset, DenseNetTrainer


class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        """
        Args:
            val_score: current validation score (higher is better)
            model: model to potentially save weights from
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    print('Restoring best weights...')
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class ResumedDenseNetTrainer(DenseNetTrainer):
    """Extended trainer class with resume functionality"""
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint and training history"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        model_config = checkpoint['model_config']
        
        # Create model with same configuration
        model = EmotionDenseNet(**model_config).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded successfully!")
        print(f"Best validation accuracy from checkpoint: {checkpoint['best_val_accuracy']:.2f}%")
        print(f"Previous training epochs: {len(training_history['train_losses'])}")
        
        return model, training_history, checkpoint['best_val_accuracy']
    
    def resume_training(self, model, train_loader, val_loader, previous_history, 
                       previous_best_val_acc, max_epochs=200, learning_rate=0.001, 
                       weight_decay=1e-4, patience=10, save_path=None):
        """Resume training with early stopping"""
        
        # Setup optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Initialize training history (continuing from previous)
        train_losses = previous_history['train_losses'].copy()
        train_accuracies = previous_history['train_accuracies'].copy()
        val_losses = previous_history['val_losses'].copy()
        val_accuracies = previous_history['val_accuracies'].copy()
        
        best_val_acc = previous_best_val_acc
        best_model_state = model.state_dict().copy()
        
        start_epoch = len(train_losses)
        
        print(f"\nResuming training from epoch {start_epoch + 1}")
        print(f"Previous best validation accuracy: {best_val_acc:.2f}%")
        print(f"Early stopping patience: {patience} epochs")
        print(f"Maximum additional epochs: {max_epochs}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(start_epoch, start_epoch + max_epochs):
            current_epoch = epoch + 1
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {current_epoch} [Train]', leave=False)
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                train_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_acc:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {current_epoch} [Val]', leave=False)
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
                    
                    # Update progress bar
                    val_acc = 100. * val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{val_acc:.2f}%'
                    })
            
            # Calculate average metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = 100. * train_correct / train_total
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(avg_val_acc)
            
            # Save metrics
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)
            
            # Save best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model_state = model.state_dict().copy()
            
            # Print epoch results
            print(f'Epoch {current_epoch}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%')
            print(f'  Best Val Acc: {best_val_acc:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Check early stopping
            if early_stopping(avg_val_acc, model):
                print(f'\nEarly stopping triggered after {current_epoch} epochs!')
                print(f'Training stopped due to no improvement in validation accuracy for {patience} epochs.')
                break
            
            print('-' * 50)
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save final model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': best_model_state,
                'model_config': {
                    'num_classes': self.num_classes,
                    'growth_rate': 32,
                    'block_config': (6, 12, 24, 16),
                    'num_init_features': 64,
                    'dropout': 0.2
                },
                'training_history': {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies
                },
                'best_val_accuracy': best_val_acc,
                'total_epochs': len(train_losses),
                'resumed_from_epoch': start_epoch
            }, save_path)
            print(f"Final model saved to {save_path}")
        
        return model, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc,
            'total_epochs': len(train_losses),
            'resumed_from_epoch': start_epoch,
            'early_stopped': current_epoch < start_epoch + max_epochs
        }


def main():
    """Main function to resume training"""
    parser = argparse.ArgumentParser(description='Resume DenseNet Training with Early Stopping')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pth)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to organized dataset')
    parser.add_argument('--max_epochs', type=int, default=200,
                       help='Maximum additional epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate (reduced for resumed training)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--results_dir', type=str, default='results_densenet_resumed',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set default data path if not provided
    if args.data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_path = os.path.join(script_dir, 'organized_by_emotion')
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.results_dir}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("DenseNet Resumed Training with Early Stopping")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Early stopping patience: {args.patience} epochs")
    print(f"Results will be saved to: {results_dir}")
    
    # Initialize trainer
    trainer = ResumedDenseNetTrainer()
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
        return
    
    model, previous_history, previous_best_val_acc = trainer.load_checkpoint(args.checkpoint)
    
    # Load data (same as before)
    file_paths, labels = trainer.load_data(args.data_path)
    
    if len(file_paths) == 0:
        print("Error: No data found!")
        return
    
    # Use same split as before (important for consistency)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(file_paths, labels)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, args.batch_size
    )
    
    # Resume training
    model_save_path = os.path.join(results_dir, 'best_densenet_resumed.pth')
    model, history = trainer.resume_training(
        model, train_loader, val_loader, 
        previous_history, previous_best_val_acc,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        save_path=model_save_path
    )
    
    # Evaluate model on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    test_results = trainer.evaluate_model(model, test_loader)
    
    # Plot results
    history_plot_path = os.path.join(results_dir, 'resumed_training_history.png')
    trainer.plot_training_history(history, history_plot_path)
    
    cm_plot_path = os.path.join(results_dir, 'resumed_confusion_matrix.png')
    trainer.plot_confusion_matrix(test_results['confusion_matrix'], cm_plot_path)
    
    # Save detailed results
    results = {
        'resumed_training_history': history,
        'test_results': test_results,
        'previous_best_val_acc': previous_best_val_acc,
        'final_best_val_acc': history['best_val_accuracy'],
        'improvement': history['best_val_accuracy'] - previous_best_val_acc,
        'total_epochs': history['total_epochs'],
        'resumed_from_epoch': history['resumed_from_epoch'],
        'early_stopped': history['early_stopped'],
        'hyperparameters': {
            'max_epochs': args.max_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'patience': args.patience
        },
        'timestamp': timestamp
    }
    
    # Save results as JSON
    results_json = results.copy()
    results_json['test_results'] = {
        'test_accuracy': test_results['test_accuracy'],
        'classification_report': test_results['classification_report']
    }
    
    with open(os.path.join(results_dir, 'resumed_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save full results as pickle
    with open(os.path.join(results_dir, 'resumed_full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESUMED TRAINING SUMMARY")
    print("=" * 60)
    print(f"Previous Best Validation Accuracy: {previous_best_val_acc:.2f}%")
    print(f"Final Best Validation Accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"Improvement: {history['best_val_accuracy'] - previous_best_val_acc:+.2f}%")
    print(f"Final Test Accuracy: {test_results['test_accuracy']:.2f}%")
    print(f"Total Training Epochs: {history['total_epochs']}")
    print(f"Resumed from Epoch: {history['resumed_from_epoch'] + 1}")
    print(f"Additional Epochs: {history['total_epochs'] - history['resumed_from_epoch']}")
    
    if history['early_stopped']:
        print(f"✓ Early stopping triggered after {args.patience} epochs without improvement")
    else:
        print(f"Training completed {args.max_epochs} additional epochs")
    
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
