#!/usr/bin/env python3
"""
Data Preprocessing Utilities for Emotion Recognition
Handles audio preprocessing, feature extraction, and data augmentation.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import skew, kurtosis
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, sr=22050, duration=3.0):
        self.sr = sr
        self.duration = duration
        
    def load_audio(self, file_path, duration=None):
        """Load and preprocess audio file"""
        if duration is None:
            duration = self.duration
            
        try:
            y, sr = librosa.load(file_path, sr=self.sr, duration=duration)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Pad or trim to exact duration
            target_length = int(self.sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
                
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_mel_spectrogram(self, y, n_mels=128, n_fft=2048, hop_length=512):
        """Extract mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def extract_mfcc(self, y, n_mfcc=13):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        return mfccs
    
    def extract_spectral_features(self, y):
        """Extract comprehensive spectral features"""
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)[0]
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        
        # Spectral flatness
        features['spectral_flatness'] = librosa.feature.spectral_flatness(y=y)[0]
        
        return features
    
    def extract_rhythm_features(self, y):
        """Extract rhythm and tempo features"""
        features = {}
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr)
        features['tempo'] = tempo
        features['beat_frames'] = beats
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(y=y, sr=self.sr)
        features['onset_strength'] = onset_strength
        
        return features
    
    def extract_pitch_features(self, y):
        """Extract pitch-related features"""
        features = {}
        
        # Pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        features['pitch_values'] = np.array(pitch_values)
        
        # Fundamental frequency using YIN
        f0 = librosa.yin(y, fmin=50, fmax=4000)
        features['f0'] = f0
        
        return features
    
    def extract_chroma_features(self, y):
        """Extract chroma features"""
        features = {}
        
        # Chroma STFT
        features['chroma_stft'] = librosa.feature.chroma_stft(y=y, sr=self.sr)
        
        # Chroma CQT
        features['chroma_cqt'] = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        
        # Chroma CENS
        features['chroma_cens'] = librosa.feature.chroma_cens(y=y, sr=self.sr)
        
        return features
    
    def extract_tonnetz(self, y):
        """Extract tonnetz features"""
        tonnetz = librosa.feature.tonnetz(y=y, sr=self.sr)
        return tonnetz
    
    def extract_poly_features(self, y):
        """Extract polynomial features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        
        # Fit polynomial
        coeffs = np.polyfit(range(len(spectral_centroids)), spectral_centroids, 2)
        features['poly_coeffs'] = coeffs
        
        return features

class FeatureAggregator:
    """Aggregate statistical features from time-series data"""
    
    @staticmethod
    def aggregate_features(feature_matrix):
        """Aggregate features using statistical measures"""
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        
        aggregated = []
        
        for i in range(feature_matrix.shape[0]):
            row = feature_matrix[i]
            # Remove NaN values
            row = row[~np.isnan(row)]
            
            if len(row) > 0:
                aggregated.extend([
                    np.mean(row),
                    np.std(row),
                    np.median(row),
                    np.min(row),
                    np.max(row),
                    np.var(row),
                    skew(row),
                    kurtosis(row),
                    np.percentile(row, 25),
                    np.percentile(row, 75)
                ])
            else:
                aggregated.extend([0] * 10)
        
        return np.array(aggregated)
    
    @staticmethod
    def extract_delta_features(feature_matrix):
        """Extract delta (first derivative) features"""
        if feature_matrix.ndim == 1:
            return np.array([0])
        
        delta = np.diff(feature_matrix, axis=1)
        return FeatureAggregator.aggregate_features(delta)
    
    @staticmethod
    def extract_delta_delta_features(feature_matrix):
        """Extract delta-delta (second derivative) features"""
        if feature_matrix.ndim == 1:
            return np.array([0])
        
        delta = np.diff(feature_matrix, axis=1)
        delta_delta = np.diff(delta, axis=1)
        return FeatureAggregator.aggregate_features(delta_delta)

class DataAugmentation:
    """Audio data augmentation techniques"""
    
    def __init__(self, sr=22050):
        self.sr = sr
    
    def add_noise(self, y, noise_factor=0.005):
        """Add random noise to audio"""
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    
    def time_shift(self, y, shift_max=0.2):
        """Randomly shift audio in time"""
        shift = np.random.randint(-int(shift_max * len(y)), int(shift_max * len(y)))
        return np.roll(y, shift)
    
    def pitch_shift(self, y, n_steps=2):
        """Shift pitch of audio"""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
    
    def time_stretch(self, y, rate=1.1):
        """Stretch or compress time"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def add_reverb(self, y, room_scale=0.3):
        """Add reverb effect"""
        # Simple reverb simulation
        reverb = np.zeros_like(y)
        for i in range(1, len(y)):
            reverb[i] = y[i] + room_scale * reverb[i-1]
        return reverb
    
    def augment_audio(self, y, num_augmentations=3):
        """Apply multiple augmentations"""
        augmented = [y]  # Original audio
        
        for _ in range(num_augmentations):
            # Randomly choose augmentation
            aug_type = np.random.choice(['noise', 'time_shift', 'pitch_shift', 'time_stretch'])
            
            if aug_type == 'noise':
                aug_y = self.add_noise(y)
            elif aug_type == 'time_shift':
                aug_y = self.time_shift(y)
            elif aug_type == 'pitch_shift':
                aug_y = self.pitch_shift(y, n_steps=np.random.uniform(-2, 2))
            elif aug_type == 'time_stretch':
                aug_y = self.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
            
            augmented.append(aug_y)
        
        return augmented

class DatasetValidator:
    """Validate and clean dataset"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    def validate_audio_files(self):
        """Validate all audio files in the dataset"""
        print("Validating audio files...")
        
        valid_files = []
        invalid_files = []
        
        for emotion in self.emotions:
            emotion_path = os.path.join(self.data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
            
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            
            for file in tqdm(files, desc=f"Validating {emotion}"):
                file_path = os.path.join(emotion_path, file)
                
                try:
                    # Try to load the file
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Check if file is not empty
                    if len(y) > 0 and np.max(np.abs(y)) > 0:
                        valid_files.append((file_path, emotion))
                    else:
                        invalid_files.append((file_path, "Empty or silent"))
                        
                except Exception as e:
                    invalid_files.append((file_path, str(e)))
        
        print(f"\nValidation Results:")
        print(f"Valid files: {len(valid_files)}")
        print(f"Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            print(f"\nInvalid files:")
            for file_path, error in invalid_files[:10]:  # Show first 10
                print(f"  {file_path}: {error}")
        
        return valid_files, invalid_files
    
    def get_dataset_statistics(self):
        """Get statistics about the dataset"""
        print("Computing dataset statistics...")
        
        stats = {}
        
        for emotion in self.emotions:
            emotion_path = os.path.join(self.data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
            
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            
            durations = []
            sample_rates = []
            
            for file in tqdm(files[:100], desc=f"Analyzing {emotion}"):  # Sample first 100 files
                file_path = os.path.join(emotion_path, file)
                
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    duration = len(y) / sr
                    durations.append(duration)
                    sample_rates.append(sr)
                except:
                    continue
            
            stats[emotion] = {
                'file_count': len(files),
                'avg_duration': np.mean(durations) if durations else 0,
                'std_duration': np.std(durations) if durations else 0,
                'min_duration': np.min(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'avg_sample_rate': np.mean(sample_rates) if sample_rates else 0
            }
        
        return stats
    
    def print_statistics(self, stats):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        total_files = 0
        
        for emotion, stat in stats.items():
            print(f"\n{emotion.upper()}:")
            print(f"  Files: {stat['file_count']}")
            print(f"  Avg Duration: {stat['avg_duration']:.2f}s")
            print(f"  Duration Range: {stat['min_duration']:.2f}s - {stat['max_duration']:.2f}s")
            print(f"  Avg Sample Rate: {stat['avg_sample_rate']:.0f} Hz")
            
            total_files += stat['file_count']
        
        print(f"\nTOTAL FILES: {total_files}")

def main():
    """Main function for data preprocessing"""
    print("Audio Data Preprocessing")
    print("="*50)
    
    # Initialize components
    preprocessor = AudioPreprocessor()
    validator = DatasetValidator("organized_by_emotion")
    
    # Validate dataset
    valid_files, invalid_files = validator.validate_audio_files()
    
    # Get statistics
    stats = validator.get_dataset_statistics()
    validator.print_statistics(stats)
    
    # Example feature extraction
    if valid_files:
        print(f"\nExtracting features from sample file...")
        sample_file, sample_emotion = valid_files[0]
        
        y, sr = preprocessor.load_audio(sample_file)
        if y is not None:
            # Extract various features
            mel_spec = preprocessor.extract_mel_spectrogram(y)
            mfccs = preprocessor.extract_mfcc(y)
            spectral = preprocessor.extract_spectral_features(y)
            rhythm = preprocessor.extract_rhythm_features(y)
            pitch = preprocessor.extract_pitch_features(y)
            chroma = preprocessor.extract_chroma_features(y)
            
            print(f"Mel spectrogram shape: {mel_spec.shape}")
            print(f"MFCC shape: {mfccs.shape}")
            print(f"Spectral features: {len(spectral)} types")
            print(f"Rhythm features: {len(rhythm)} types")
            print(f"Pitch features: {len(pitch)} types")
            print(f"Chroma features: {len(chroma)} types")
    
    print("\nData preprocessing completed!")

if __name__ == "__main__":
    main()
