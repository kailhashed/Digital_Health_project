#!/usr/bin/env python3
"""
Simple DenseNet Training Script for Emotion Recognition
Restored version with 80-10-10 train/validation/test split.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add path for models
sys.path.append('src/models')
from custom_models import EmotionDenseNet


class EmotionDataset(Dataset):
    """Dataset class for emotion recognition with mel-spectrogram features"""
    
    def __init__(self, file_paths, labels, sr=22050, n_mels=128, duration=3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load and preprocess audio
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            y = librosa.util.normalize(y)
            
            # Pad or trim to exact duration
            target_length = int(self.sr * self.duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return torch.FloatTensor(log_mel_spec), label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros in case of error
            return torch.zeros(self.n_mels, 130), label


def load_data(data_path="organized_by_emotion"):
    """Load data from organized dataset"""
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    file_paths = []
    labels = []
    
    print("Loading dataset...")
    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            continue
        
        files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        for file in files:
            file_path = os.path.join(emotion_path, file)
            file_paths.append(file_path)
            labels.append(emotion_idx)
        
        print(f"{emotion}: {len(files)} files")
    
    print(f"Total files loaded: {len(file_paths)}")
    return file_paths, labels


def split_data_80_10_10(file_paths, labels):
    """Split data into 80% train, 10% validation, 10% test"""
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Second split: 80% train, 20% val (of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split (80-10-10):")
    print(f"Training set: {len(X_train)} files ({len(X_train)/len(file_paths)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} files ({len(X_val)/len(file_paths)*100:.1f}%)")
    print(f"Test set: {len(X_test)} files ({len(X_test)/len(file_paths)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, train_loader, val_loader, epochs=50, device='cuda'):
    """Train the DenseNet model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    train_history = {'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': []}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Save metrics
        train_history['train_losses'].append(avg_train_loss)
        train_history['train_accs'].append(train_acc)
        train_history['val_losses'].append(avg_val_loss)
        train_history['val_accs'].append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%')
    
    # Load best model
    model.load_state_dict(best_model_state)
    train_history['best_val_acc'] = best_val_acc
    
    return model, train_history


def main():
    """Main training function"""
    print("DenseNet Emotion Recognition Training")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split data
    file_paths, labels = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_80_10_10(file_paths, labels)
    
    # Create datasets and loaders
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    test_dataset = EmotionDataset(X_test, y_test)
    
    # Use num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    model = EmotionDenseNet(
        num_classes=8,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        dropout=0.2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, epochs=20, device=device)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/densenet_simple_{timestamp}.pth"
    os.makedirs("models", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 8,
            'growth_rate': 32,
            'block_config': (6, 12, 24, 16),
            'num_init_features': 64,
            'dropout': 0.2
        },
        'training_history': history,
        'timestamp': timestamp
    }, model_path)
    
    print(f"\nModel saved: {model_path}")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()
