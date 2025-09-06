"""
DenseNet Training Module
Integrated trainer for DenseNet models with the existing training framework
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.densenet_model import create_densenet_model, create_densenet_variants
from src.data.dataset import EmotionDataset
from src.utils.logger import setup_logger
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix


class DenseNetTrainer:
    """Trainer class for DenseNet emotion recognition models"""
    
    def __init__(self, config=None, device=None, logger=None):
        self.config = config or self._get_default_config()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or setup_logger('densenet_trainer')
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # Training history
        self.history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'learning_rates': []
        }
        
        self.logger.info(f"DenseNet Trainer initialized on device: {self.device}")
    
    def _get_default_config(self):
        """Get default training configuration"""
        return {
            'model': {
                'num_classes': 8,
                'growth_rate': 32,
                'block_config': (6, 12, 24, 16),
                'num_init_features': 64,
                'dropout': 0.2
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'early_stopping_patience': 10,
                'lr_scheduler_patience': 5,
                'lr_scheduler_factor': 0.5
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'num_workers': 0  # Windows compatibility
            }
        }
    
    def create_model(self, model_type='standard'):
        """Create DenseNet model"""
        if model_type == 'standard':
            self.model = create_densenet_model(self.config['model'])
        elif model_type in ['small', 'medium', 'large']:
            variants = create_densenet_variants()
            self.model = variants[f'densenet_{model_type}']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=self.config['training']['lr_scheduler_patience'],
            factor=self.config['training']['lr_scheduler_factor']
        )
        
        model_info = self.model.get_model_info()
        self.logger.info(f"Created {model_type} DenseNet model:")
        self.logger.info(f"  Parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"  Architecture: {model_info['architecture']}")
        
        return self.model
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Update progress bar
                accuracy = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        epoch_loss = total_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, train_loader, val_loader, save_dir=None):
        """Complete training loop"""
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"results_densenet_training_{timestamp}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Early stopping patience: {early_stopping_patience}")
        self.logger.info(f"Save directory: {save_dir}")
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.history['train_losses'].append(train_loss)
            self.history['train_accuracies'].append(train_acc)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Check for best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint(save_dir, epoch + 1, best_val_acc, 'best')
                
            else:
                patience_counter += 1
            
            # Log progress
            self.logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"  Best Val Acc: {best_val_acc:.2f}%")
            self.logger.info(f"  Learning Rate: {current_lr:.6f}")
            self.logger.info(f"  Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save intermediate checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir, epoch + 1, val_acc, f'epoch_{epoch + 1}')
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Save final results
        self.save_training_results(save_dir, best_val_acc)
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.model, self.history
    
    def evaluate(self, test_loader, save_dir=None):
        """Evaluate model on test set"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        self.logger.info("Evaluating on test set...")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                total_loss += loss.item()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        test_accuracy = accuracy_score(all_targets, all_predictions)
        test_loss = total_loss / len(test_loader)
        
        # Generate detailed report
        report = classification_report(
            all_targets, all_predictions,
            target_names=self.emotions,
            output_dict=True
        )
        
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        self.logger.info(f"  Loss: {test_loss:.4f}")
        
        # Save evaluation results
        if save_dir:
            self.save_evaluation_results(results, save_dir)
        
        return results
    
    def save_checkpoint(self, save_dir, epoch, accuracy, checkpoint_type='latest'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.config,
            'model_info': self.model.get_model_info(),
            'history': self.history
        }
        
        checkpoint_path = os.path.join(save_dir, f'densenet_{checkpoint_type}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def save_training_results(self, save_dir, best_val_acc):
        """Save complete training results"""
        # Save training history plot
        self.plot_training_history(save_dir)
        
        # Save training summary
        summary = {
            'best_validation_accuracy': best_val_acc,
            'total_epochs': len(self.history['train_losses']),
            'final_train_accuracy': self.history['train_accuracies'][-1],
            'final_val_accuracy': self.history['val_accuracies'][-1],
            'config': self.config,
            'model_info': self.model.get_model_info(),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(save_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save full history as pickle
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)
        
        self.logger.info(f"Training results saved to {save_dir}")
    
    def save_evaluation_results(self, results, save_dir):
        """Save evaluation results"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotions, yticklabels=self.emotions)
        plt.title('DenseNet Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save evaluation summary
        eval_summary = {
            'test_accuracy': results['test_accuracy'],
            'test_loss': results['test_loss'],
            'classification_report': results['classification_report'],
            'timestamp': datetime.now().isoformat()
        }
        
        eval_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_summary, f, indent=2)
        
        # Save full results as pickle
        full_results_path = os.path.join(save_dir, 'full_evaluation_results.pkl')
        with open(full_results_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Evaluation results saved to {save_dir}")
    
    def plot_training_history(self, save_dir):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_losses']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_losses'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_accuracies'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(epochs, self.history['learning_rates'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Combined accuracy plot
        ax4.plot(epochs, self.history['train_accuracies'], 'b-', label='Training', alpha=0.7)
        ax4.plot(epochs, self.history['val_accuracies'], 'r-', label='Validation', alpha=0.7)
        ax4.fill_between(epochs, self.history['train_accuracies'], alpha=0.3, color='blue')
        ax4.fill_between(epochs, self.history['val_accuracies'], alpha=0.3, color='red')
        ax4.set_title('Accuracy Comparison')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved: {plot_path}")


def main():
    """Test the DenseNet trainer"""
    print("Testing DenseNet Trainer...")
    
    # Create dummy config
    config = {
        'model': {
            'num_classes': 8,
            'growth_rate': 16,  # Smaller for testing
            'block_config': (4, 8, 12, 8),
            'num_init_features': 32,
            'dropout': 0.1
        },
        'training': {
            'epochs': 2,  # Small for testing
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'early_stopping_patience': 10,
            'lr_scheduler_patience': 5,
            'lr_scheduler_factor': 0.5
        }
    }
    
    trainer = DenseNetTrainer(config)
    model = trainer.create_model('small')
    
    print("✅ DenseNet trainer test passed!")


if __name__ == "__main__":
    main()
