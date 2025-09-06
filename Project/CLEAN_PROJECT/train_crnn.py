#!/usr/bin/env python3
"""
CRNN (Convolutional Recurrent Neural Network) Training Script for Emotion Recognition
GPU-accelerated training with early stopping and comprehensive evaluation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from data_loader import load_emotion_data, create_data_loaders, get_device

class EmotionCRNN(nn.Module):
    """CRNN model for emotion recognition"""
    
    def __init__(self, num_classes=8, input_height=128, input_width=130):
        super(EmotionCRNN, self).__init__()
        
        # Convolutional layers (8 input channels for 8 features)
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate LSTM input size
        self.lstm_input_size = 256 * 4 * 4
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Convolutional feature extraction
        conv_out = self.conv_layers(x)  # [batch, 256, 4, 4]
        
        # Reshape for LSTM
        batch_size = conv_out.size(0)
        conv_out = conv_out.view(batch_size, -1, self.lstm_input_size)  # [batch, 1, 256*4*4]
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_out)  # [batch, 1, 512*2]
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]  # [batch, 512*2]
        
        # Classification
        output = self.classifier(lstm_out)
        
        return output

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class CRNNTrainer:
    """CRNN trainer with GPU support and early stopping"""
    
    def __init__(self, device, num_classes=8):
        self.device = device
        self.num_classes = num_classes
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # Initialize model
        self.model = EmotionCRNN(num_classes=num_classes).to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=20, min_delta=0.001)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, max_epochs=200):
        """Main training loop"""
        print(f"🚀 Starting CRNN training for {max_epochs} epochs...")
        print(f"   Early stopping patience: 20 epochs")
        print(f"   Device: {self.device}")
        
        best_val_acc = 0
        start_time = datetime.now()
        
        for epoch in range(max_epochs):
            print(f"\n📅 Epoch {epoch+1}/{max_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f"models/crnn/best_crnn_epoch_{epoch+1}.pth")
                print(f"   💾 New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"   ⏹️ Early stopping triggered after {epoch+1} epochs")
                break
        
        training_time = datetime.now() - start_time
        print(f"\n✅ Training completed in {training_time}")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        print("🔍 Evaluating on test set...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(test_loader)
        test_acc = 100. * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'emotions': self.emotions
        }, path)
    
    def plot_training_history(self, save_path="results/crnn/training_history.png"):
        """Plot training history"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('CRNN Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('CRNN Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training history saved to {save_path}")

def main():
    """Main training function"""
    print("🎭 CRNN Emotion Recognition Training")
    print("=" * 50)
    
    # Get device
    device = get_device()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32, num_workers=4
    )
    
    # Initialize trainer
    trainer = CRNNTrainer(device)
    
    # Train model
    history = trainer.train(train_loader, val_loader, max_epochs=200)
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save results
    results = {
        'model': 'CRNN',
        'training_history': history,
        'test_results': test_results,
        'timestamp': datetime.now().isoformat(),
        'device': str(device)
    }
    
    os.makedirs("results/crnn", exist_ok=True)
    with open("results/crnn/crnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 CRNN training completed!")
    print(f"   Final test accuracy: {test_results['test_accuracy']:.2f}%")
    print(f"   Results saved to results/crnn/")

if __name__ == "__main__":
    main()
