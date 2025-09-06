"""
Audio preprocessing utilities
"""

import numpy as np
import librosa


class AudioPreprocessor:
    """Audio preprocessing utilities for emotion recognition"""
    
    def __init__(self, sr=16000, duration=3.0):
        """
        Args:
            sr: Sample rate
            duration: Duration in seconds
        """
        self.sr = sr
        self.duration = duration
        self.target_length = int(sr * duration)
    
    def load_and_preprocess(self, audio_path):
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Normalize
            y = librosa.util.normalize(y)
            
            # Pad or trim to fixed length
            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
            else:
                y = y[:self.target_length]
            
            return y
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return np.zeros(self.target_length)
    
    def extract_mel_spectrogram(self, y, n_mels=64, fmax=8000, n_fft=1024, hop_length=256):
        """
        Extract mel spectrogram features
        
        Args:
            y: Audio signal
            n_mels: Number of mel bands
            fmax: Maximum frequency
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Log mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft, hop_length=hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def extract_mfcc(self, y, n_mfcc=13):
        """
        Extract MFCC features
        
        Args:
            y: Audio signal
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=n_mfcc)
        return mfcc
    
    def extract_chroma(self, y):
        """
        Extract chroma features
        
        Args:
            y: Audio signal
            
        Returns:
            Chroma features
        """
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        return chroma
    
    def extract_spectral_features(self, y):
        """
        Extract spectral features (centroid, bandwidth, rolloff)
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(y)
        
        return features
    
    def extract_comprehensive_features(self, audio_path):
        """
        Extract comprehensive feature set from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of extracted features
        """
        # Load audio
        y = self.load_and_preprocess(audio_path)
        
        features = {}
        
        # Time-domain features
        features['mel_spectrogram'] = self.extract_mel_spectrogram(y)
        features['mfcc'] = self.extract_mfcc(y)
        features['chroma'] = self.extract_chroma(y)
        
        # Spectral features
        spectral_features = self.extract_spectral_features(y)
        features.update(spectral_features)
        
        # Statistical features
        features['rms'] = librosa.feature.rms(y=y)
        features['tempo'] = librosa.beat.tempo(y=y, sr=self.sr)[0]
        
        return features

