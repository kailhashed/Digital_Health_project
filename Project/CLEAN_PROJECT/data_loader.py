#!/usr/bin/env python3
"""
Data Loading Utility for Emotion Recognition Training
Handles 80-10-10 train/validation/test split with GPU support
"""

import os
import sys
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EmotionDataset(Dataset):
    """Dataset class for emotion recognition with 8-channel audio features"""
    
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
    
    def extract_all_features(self, y, sr):
        """Extract all 8 audio features"""
        features = []
        
        # 1. Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.append(mel_spec_db)
        
        # 2. MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, 
                                   n_fft=self.n_fft, hop_length=self.hop_length)
        # Take mean across MFCC coefficients to match mel-spectrogram shape
        mfcc_mean = np.mean(mfcc, axis=0, keepdims=True)
        features.append(mfcc_mean)
        
        # 3. Chroma features (12 chroma bins)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, 
                                           n_fft=self.n_fft, hop_length=self.hop_length)
        # Take mean across chroma bins to match mel-spectrogram shape
        chroma_mean = np.mean(chroma, axis=0, keepdims=True)
        features.append(chroma_mean)
        
        # 4. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                            n_fft=self.n_fft, hop_length=self.hop_length)
        features.append(spectral_centroid)
        
        # 5. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, 
                                                          n_fft=self.n_fft, hop_length=self.hop_length)
        features.append(spectral_rolloff)
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.n_fft, hop_length=self.hop_length)
        features.append(zcr)
        
        # 7. RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        features.append(rms)
        
        # 8. Tonnetz (6 tonal features)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        # Take mean across tonnetz features to match mel-spectrogram shape
        tonnetz_mean = np.mean(tonnetz, axis=0, keepdims=True)
        features.append(tonnetz_mean)
        
        return features
    
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
            
            # Extract all 8 features
            feature_list = self.extract_all_features(y, sr)
            
            # Stack features into 8-channel tensor
            # Ensure all features have the same shape (128, 130)
            processed_features = []
            for feature in feature_list:
                # Resize to match mel-spectrogram shape (128, 130)
                if feature.shape[0] != self.n_mels or feature.shape[1] != 130:
                    # Use interpolation to resize
                    from scipy.ndimage import zoom
                    zoom_factors = (self.n_mels / feature.shape[0], 130 / feature.shape[1])
                    feature = zoom(feature, zoom_factors, order=1)
                processed_features.append(feature)
            
            # Stack into 8-channel tensor
            multi_channel_features = np.stack(processed_features, axis=0)
            feature_tensor = torch.FloatTensor(multi_channel_features)
            
            if self.transform:
                feature_tensor = self.transform(feature_tensor)
            
            return feature_tensor, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor and neutral label on error
            return torch.zeros(8, self.n_mels, 130), 5  # 8 channels, neutral label (index 5)

def load_emotion_data(data_dir="data/organized_by_emotion"):
    """Load all emotion data with 80-10-10 split"""
    
    print("🔄 Loading emotion data...")
    
    # Emotion mapping (8 emotions available)
    emotion_map = {
        'angry': 0, 'calm': 1, 'disgust': 2, 'fearful': 3,
        'happy': 4, 'neutral': 5, 'sad': 6, 'surprised': 7
    }
    
    file_paths = []
    labels = []
    
    # Load data from organized emotion directories
    for emotion, label in emotion_map.items():
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.exists(emotion_dir):
            files = glob.glob(os.path.join(emotion_dir, "*.wav"))
            file_paths.extend(files)
            labels.extend([label] * len(files))
            print(f"✓ Loaded {len(files)} {emotion} files")
        else:
            print(f"⚠️ Directory not found: {emotion_dir}")
    
    print(f"📊 Total files loaded: {len(file_paths)}")
    
    if len(file_paths) == 0:
        raise ValueError("No audio files found! Check data directory.")
    
    # 80-10-10 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp  # 0.111 = 0.1/0.9
    )
    
    print(f"📈 Data split:")
    print(f"   Training: {len(X_train)} files ({len(X_train)/len(file_paths)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} files ({len(X_val)/len(file_paths)*100:.1f}%)")
    print(f"   Test: {len(X_test)} files ({len(X_test)/len(file_paths)*100:.1f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                       batch_size=32, num_workers=4):
    """Create PyTorch data loaders"""
    
    print("🔄 Creating data loaders...")
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)
    test_dataset = EmotionDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"✓ Data loaders created with batch size {batch_size}")
    
    return train_loader, val_loader, test_loader

def get_device():
    """Get the best available device (GPU if available)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

if __name__ == "__main__":
    # Test data loading
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data()
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        device = get_device()
        print("✅ Data loading test successful!")
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
