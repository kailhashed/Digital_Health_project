"""
Dataset classes for emotion recognition
"""

import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset

from .utils import SimpleLabelEncoder


class EmotionDataset(Dataset):
    """Dataset for emotion recognition from audio files"""
    
    def __init__(self, file_paths, labels, sr=16000, duration=3.0, feature_type='spectrogram'):
        """
        Args:
            file_paths: List of audio file paths
            labels: List of emotion labels
            sr: Sample rate for audio loading
            duration: Duration in seconds to load
            feature_type: Type of features to extract ('spectrogram' or 'raw')
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.feature_type = feature_type
        self.label_encoder = SimpleLabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            audio_path = self.file_paths[idx]
            label = self.encoded_labels[idx]
            
            # Load and normalize audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            y = librosa.util.normalize(y)
            
            # Pad or trim to fixed length
            target_length = int(self.sr * self.duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            if self.feature_type == 'spectrogram':
                # Convert to mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=64, fmax=8000, n_fft=1024, hop_length=256
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                return torch.FloatTensor(log_mel_spec), label
            else:
                # Raw audio for some models
                return torch.FloatTensor(y), label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            if self.feature_type == 'spectrogram':
                return torch.zeros(64, 188), label
            else:
                return torch.zeros(int(self.sr * self.duration)), label

    def get_classes(self):
        """Get the emotion classes"""
        return self.label_encoder.classes_

    def get_num_classes(self):
        """Get the number of emotion classes"""
        return len(self.label_encoder.classes_)


class AudioDataset(Dataset):
    """Simple audio dataset for raw audio processing"""
    
    def __init__(self, file_paths, labels, sr=16000, duration=3.0):
        """
        Args:
            file_paths: List of audio file paths
            labels: List of emotion labels
            sr: Sample rate for audio loading
            duration: Duration in seconds to load
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.label_encoder = SimpleLabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            audio_path = self.file_paths[idx]
            label = self.encoded_labels[idx]
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            y = librosa.util.normalize(y)
            
            # Pad or trim to fixed length
            target_length = int(self.sr * self.duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            return torch.FloatTensor(y), label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            target_length = int(self.sr * self.duration)
            return torch.zeros(target_length), label

    def get_classes(self):
        """Get the emotion classes"""
        return self.label_encoder.classes_

    def get_num_classes(self):
        """Get the number of emotion classes"""
        return len(self.label_encoder.classes_)

