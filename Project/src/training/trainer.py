"""
Training utilities for emotion recognition models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelTrainer:
    """Base trainer class for emotion recognition models"""
    
    def __init__(self, model, device=None):
        """
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda/cpu)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                try:
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                except:
                    continue
        
        return correct / total if total > 0 else 0
    
    def test(self, test_loader):
        """Test model and return predictions"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                try:
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                except:
                    continue
        
        accuracy = sum(1 for t, p in zip(all_targets, all_preds) if t == p) / len(all_targets)
        return accuracy, all_preds, all_targets
    
    def save_model(self, filepath, epoch, val_acc, additional_info=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'epoch': epoch,
            'val_acc': val_acc,
            'num_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)


class CustomModelTrainer(ModelTrainer):
    """Trainer for custom deep learning models"""
    
    def train(self, train_loader, val_loader, model_name, epochs=30, lr=0.001, 
              patience=8, save_dir='models'):
        """
        Train custom model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name for saving model
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            save_dir: Directory to save models
            
        Returns:
            Tuple of (model, best_val_acc, train_losses, val_accuracies)
        """
        print(f"\nTraining {model_name}")
        print("="*50)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        patience_counter = 0
        
        save_path = os.path.join(save_dir, model_name.lower(), f'best_{model_name}.pth')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            
            # Validation
            val_acc = self.validate(val_loader)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                self.save_model(save_path, epoch, best_val_acc, {'model_name': model_name})
                print(f"✓ New best model saved (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.model, best_val_acc, train_losses, val_accuracies


class PretrainedModelTrainer(ModelTrainer):
    """Trainer for pre-trained fine-tuned models"""
    
    def train(self, train_loader, val_loader, model_name, epochs=15, lr=0.0001, 
              patience=5, save_dir='models'):
        """
        Fine-tune pre-trained model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name for saving model
            epochs: Maximum number of epochs
            lr: Learning rate (lower for pre-trained)
            patience: Early stopping patience
            save_dir: Directory to save models
            
        Returns:
            Tuple of (model, best_val_acc, train_losses, val_accuracies)
        """
        print(f"\nFine-tuning {model_name}")
        print("="*60)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        patience_counter = 0
        
        save_path = os.path.join(save_dir, f'pretrained_{model_name.lower()}', f'best_{model_name}.pth')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            
            # Validation
            val_acc = self.validate(val_loader)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                additional_info = {
                    'model_name': model_name,
                    'total_params': sum(p.numel() for p in self.model.parameters()),
                    'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                }
                self.save_model(save_path, epoch, best_val_acc, additional_info)
                print(f"✓ New best model saved (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.model, best_val_acc, train_losses, val_accuracies

