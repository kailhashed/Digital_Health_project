#!/usr/bin/env python3
"""
DenseNet Training Script for Emotion Recognition
Implements 80-10-10 train/validation/test split with comprehensive evaluation.
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

# Import the DenseNet model
sys.path.append('src/models')
from custom_models import EmotionDenseNet


class EmotionDataset(Dataset):
    """Dataset class for emotion recognition with mel-spectrogram features"""
    
    def __init__(self, file_paths, labels, transform=None, sr=22050, 
                 n_mels=128, n_fft=2048, hop_length=512, duration=3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # Normalize and pad/trim
            y = librosa.util.normalize(y)
            target_length = int(self.sr * self.duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(log_mel_spec)
            
            if self.transform:
                mel_tensor = self.transform(mel_tensor)
            
            return mel_tensor, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros in case of error
            mel_tensor = torch.zeros(self.n_mels, 130)  # Approximate shape
            return mel_tensor, label


class DenseNetTrainer:
    """Trainer class for DenseNet emotion recognition"""
    
    def __init__(self, num_classes=8, device=None):
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        print(f"Using device: {self.device}")
        
    def load_data(self, data_path="organized_by_emotion", max_files_per_emotion=None):
        """Load data from organized dataset"""
        file_paths = []
        labels = []
        
        print("Loading dataset...")
        
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_path = os.path.join(data_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"Warning: Emotion directory '{emotion}' not found")
                continue
            
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            
            if max_files_per_emotion:
                files = files[:max_files_per_emotion]
            
            for file in files:
                file_path = os.path.join(emotion_path, file)
                file_paths.append(file_path)
                labels.append(emotion_idx)
            
            print(f"{emotion}: {len([f for f in files])} files")
        
        print(f"Total files loaded: {len(file_paths)}")
        return file_paths, labels
    
    def split_data(self, file_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split data into train/validation/test sets (80-10-10)"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Convert to numpy arrays for easier handling
        file_paths = np.array(file_paths)
        labels = np.array(labels)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths, labels, 
            test_size=test_ratio, 
            random_state=42, 
            stratify=labels
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\nData split (80-10-10):")
        print(f"Training set: {len(X_train)} files ({len(X_train)/len(file_paths)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} files ({len(X_val)/len(file_paths)*100:.1f}%)")
        print(f"Test set: {len(X_test)} files ({len(X_test)/len(file_paths)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """Create PyTorch data loaders"""
        
        # Create datasets
        train_dataset = EmotionDataset(X_train, y_train)
        val_dataset = EmotionDataset(X_val, y_val)
        test_dataset = EmotionDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001, 
                   weight_decay=1e-4, save_path=None):
        """Train the DenseNet model"""
        
        # Initialize model
        model = EmotionDenseNet(
            num_classes=self.num_classes,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_init_features=64,
            dropout=0.2
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
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
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
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
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%')
            print(f'  Best Val Acc: {best_val_acc:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Save model
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
                'best_val_accuracy': best_val_acc
            }, save_path)
            print(f"Model saved to {save_path}")
        
        return model, {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc
        }
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        print("\nEvaluating on test set...")
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Testing')
            for data, target in test_pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Update progress bar
                test_acc = 100. * test_correct / test_total
                test_pbar.set_postfix({'Acc': f'{test_acc:.2f}%'})
        
        test_accuracy = 100. * test_correct / test_total
        
        # Generate detailed classification report
        report = classification_report(
            all_targets, all_predictions,
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        print(f"\nTest Results:")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, target_names=self.emotion_labels))
        
        return {
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_losses'], label='Training Loss', color='blue')
        ax1.plot(history['val_losses'], label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_accuracies'], label='Training Accuracy', color='blue')
        ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title('Confusion Matrix - DenseNet Emotion Recognition')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train DenseNet for Emotion Recognition')
    parser.add_argument('--data_path', type=str, default='organized_by_emotion',
                       help='Path to organized dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum files per emotion (for testing)')
    parser.add_argument('--results_dir', type=str, default='results_densenet',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.results_dir}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("DenseNet Emotion Recognition Training")
    print("=" * 50)
    print(f"Results will be saved to: {results_dir}")
    
    # Initialize trainer
    trainer = DenseNetTrainer()
    
    # Load data
    file_paths, labels = trainer.load_data(args.data_path, args.max_files)
    
    if len(file_paths) == 0:
        print("Error: No data found!")
        return
    
    # Split data (80-10-10)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(file_paths, labels)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, args.batch_size
    )
    
    # Train model
    model_save_path = os.path.join(results_dir, 'best_densenet.pth')
    model, history = trainer.train_model(
        train_loader, val_loader, 
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=model_save_path
    )
    
    # Evaluate model
    test_results = trainer.evaluate_model(model, test_loader)
    
    # Plot results
    history_plot_path = os.path.join(results_dir, 'training_history.png')
    trainer.plot_training_history(history, history_plot_path)
    
    cm_plot_path = os.path.join(results_dir, 'confusion_matrix.png')
    trainer.plot_confusion_matrix(test_results['confusion_matrix'], cm_plot_path)
    
    # Save detailed results
    results = {
        'training_history': history,
        'test_results': test_results,
        'data_split': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        },
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        },
        'timestamp': timestamp
    }
    
    # Save results as JSON (excluding non-serializable objects)
    results_json = results.copy()
    results_json['test_results'] = {
        'test_accuracy': test_results['test_accuracy'],
        'classification_report': test_results['classification_report']
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save full results as pickle
    with open(os.path.join(results_dir, 'full_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Best Validation Accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"Final Test Accuracy: {test_results['test_accuracy']:.2f}%")
    print(f"Results saved to: {results_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
