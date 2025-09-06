#!/usr/bin/env python3
"""
Enhanced Emotion Recognition Models
Improved feature extraction and model architectures for higher accuracy.
"""

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedAudioFeatureExtractor:
    """Enhanced audio feature extractor with comprehensive feature set"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        
    def extract_all_features(self, audio_path):
        """Extract comprehensive features from audio file"""
        try:
            # Load audio with error handling
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Validate audio
            if len(y) == 0 or np.max(np.abs(y)) == 0:
                print(f"Warning: Empty or silent audio: {audio_path}")
                return self._get_default_features()
            
            if len(y) < sr * 0.1:  # Less than 0.1 seconds
                print(f"Warning: Audio too short: {audio_path}")
                return self._get_default_features()
            
            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            if len(y_trimmed) < sr * 0.05:
                print(f"Warning: Audio too short after trimming: {audio_path}")
                return self._get_default_features()
            
            features = []
            
            # 1. MFCCs with enhanced parameters
            try:
                mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
                mfcc_stats = self._extract_statistics(mfccs)
                features.extend(mfcc_stats)
            except Exception as e:
                print(f"MFCC extraction failed: {e}")
                features.extend([0.0] * 78)  # 13 * 6
            
            # 2. Enhanced pitch features
            try:
                pitch_features = self._extract_enhanced_pitch(y_trimmed, sr)
                features.extend(pitch_features)
            except Exception as e:
                print(f"Pitch extraction failed: {e}")
                features.extend([0.0] * 15)
            
            # 3. Spectral features
            try:
                spectral_features = self._extract_spectral_features(y_trimmed, sr)
                features.extend(spectral_features)
            except Exception as e:
                print(f"Spectral extraction failed: {e}")
                features.extend([0.0] * 30)
            
            # 4. Rhythm and tempo features
            try:
                rhythm_features = self._extract_rhythm_features(y_trimmed, sr)
                features.extend(rhythm_features)
            except Exception as e:
                print(f"Rhythm extraction failed: {e}")
                features.extend([0.0] * 10)
            
            # 5. Chroma features
            try:
                chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr, n_fft=2048, hop_length=512)
                chroma_stats = self._extract_statistics(chroma)
                features.extend(chroma_stats)
            except Exception as e:
                print(f"Chroma extraction failed: {e}")
                features.extend([0.0] * 72)  # 12 * 6
            
            # 6. Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(y_trimmed)
                zcr_stats = self._extract_statistics(zcr)
                features.extend(zcr_stats)
            except Exception as e:
                print(f"ZCR extraction failed: {e}")
                features.extend([0.0] * 6)
            
            # 7. Spectral rolloff
            try:
                rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
                rolloff_stats = self._extract_statistics(rolloff)
                features.extend(rolloff_stats)
            except Exception as e:
                print(f"Rolloff extraction failed: {e}")
                features.extend([0.0] * 6)
            
            # 8. Spectral centroid
            try:
                centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
                centroid_stats = self._extract_statistics(centroid)
                features.extend(centroid_stats)
            except Exception as e:
                print(f"Centroid extraction failed: {e}")
                features.extend([0.0] * 6)
            
            # 9. Spectral bandwidth
            try:
                bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
                bandwidth_stats = self._extract_statistics(bandwidth)
                features.extend(bandwidth_stats)
            except Exception as e:
                print(f"Bandwidth extraction failed: {e}")
                features.extend([0.0] * 6)
            
            # 10. Mel-frequency cepstral coefficients (additional)
            try:
                mel_spec = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
                mel_stats = self._extract_statistics(mel_spec)
                features.extend(mel_stats)
            except Exception as e:
                print(f"Mel spectrogram extraction failed: {e}")
                features.extend([0.0] * 768)  # 128 * 6
            
            # 11. Tonnetz features
            try:
                tonnetz = librosa.feature.tonnetz(y=y_trimmed, sr=sr)
                tonnetz_stats = self._extract_statistics(tonnetz)
                features.extend(tonnetz_stats)
            except Exception as e:
                print(f"Tonnetz extraction failed: {e}")
                features.extend([0.0] * 36)  # 6 * 6
            
            # Convert to numpy array and ensure finite values
            features = np.array([float(x) if np.isfinite(x) else 0.0 for x in features])
            
            return features
            
        except Exception as e:
            print(f"Critical error in feature extraction: {e}")
            return self._get_default_features()
    
    def _get_default_features(self):
        """Return default feature vector when extraction fails"""
        # Total expected features: 78 + 15 + 30 + 10 + 72 + 6 + 6 + 6 + 6 + 768 + 36 = 1033
        return np.zeros(1033)
    
    def _extract_statistics(self, feature_matrix):
        """Extract statistical features from a 2D feature matrix"""
        return np.concatenate([
            np.mean(feature_matrix, axis=1),
            np.std(feature_matrix, axis=1),
            np.median(feature_matrix, axis=1),
            np.min(feature_matrix, axis=1),
            np.max(feature_matrix, axis=1),
            np.var(feature_matrix, axis=1)
        ])
    
    def _extract_enhanced_pitch(self, audio, sr):
        """Extract enhanced pitch features"""
        try:
            # Fundamental frequency using YIN algorithm
            f0 = librosa.yin(audio, fmin=50, fmax=4000, sr=sr)
            f0 = f0[f0 > 0]  # Remove unvoiced segments
            
            if len(f0) == 0:
                return [0.0] * 15
            
            # Basic pitch statistics
            pitch_mean = np.mean(f0)
            pitch_std = np.std(f0)
            pitch_median = np.median(f0)
            pitch_min = np.min(f0)
            pitch_max = np.max(f0)
            pitch_var = np.var(f0)
            
            # Jitter (pitch period perturbation)
            if len(f0) > 1:
                jitter = np.mean(np.abs(np.diff(f0))) / pitch_mean if pitch_mean > 0 else 0
            else:
                jitter = 0
            
            # Shimmer (amplitude perturbation) - simplified
            shimmer = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            # Pitch range
            pitch_range = pitch_max - pitch_min
            
            # Voicing ratio
            voicing_ratio = len(f0) / len(audio) * sr if len(audio) > 0 else 0
            
            # Pitch slope (trend)
            if len(f0) > 2:
                x = np.arange(len(f0))
                pitch_slope = np.polyfit(x, f0, 1)[0]
            else:
                pitch_slope = 0
            
            # Pitch stability (inverse of variance)
            pitch_stability = 1.0 / (pitch_var + 1e-6)
            
            # Pitch entropy
            if len(f0) > 1:
                hist, _ = np.histogram(f0, bins=min(20, len(f0)//2))
                hist = hist / np.sum(hist)
                pitch_entropy = -np.sum(hist * np.log(hist + 1e-10))
            else:
                pitch_entropy = 0
            
            # Pitch skewness and kurtosis
            from scipy import stats
            pitch_skew = stats.skew(f0) if len(f0) > 2 else 0
            pitch_kurt = stats.kurtosis(f0) if len(f0) > 2 else 0
            
            return [
                pitch_mean, pitch_std, pitch_median, pitch_min, pitch_max, pitch_var,
                jitter, shimmer, pitch_range, voicing_ratio, pitch_slope, 
                pitch_stability, pitch_entropy, pitch_skew, pitch_kurt
            ]
            
        except Exception as e:
            print(f"Enhanced pitch extraction failed: {e}")
            return [0.0] * 15
    
    def _extract_spectral_features(self, audio, sr):
        """Extract spectral features"""
        try:
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            contrast_stats = self._extract_statistics(contrast)
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=audio)
            flatness_stats = self._extract_statistics(flatness)
            
            # Combine features
            return np.concatenate([contrast_stats, flatness_stats])
            
        except Exception as e:
            print(f"Spectral feature extraction failed: {e}")
            return [0.0] * 30
    
    def _extract_rhythm_features(self, audio, sr):
        """Extract rhythm and tempo features"""
        try:
            # Tempo and beats
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            # Beat strength
            beat_strength = np.mean(librosa.beat.beat_track(y=audio, sr=sr, units='time')[1])
            
            # Rhythm regularity
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                rhythm_regularity = 1.0 / (np.std(beat_intervals) + 1e-6)
            else:
                rhythm_regularity = 0
            
            # Tempo stability
            tempo_stability = 1.0 / (np.std([tempo]) + 1e-6)
            
            # Onset strength
            onset_strength = np.mean(librosa.onset.onset_strength(y=audio, sr=sr))
            
            # Onset rate
            onsets = librosa.onset.onset_detect(y=audio, sr=sr)
            onset_rate = len(onsets) / (len(audio) / sr) if len(audio) > 0 else 0
            
            # Rhythm complexity
            rhythm_complexity = np.std(beat_intervals) if len(beats) > 1 else 0
            
            # Tempo variation
            tempo_variation = np.std([tempo])
            
            # Beat density
            beat_density = len(beats) / (len(audio) / sr) if len(audio) > 0 else 0
            
            return [
                float(tempo), float(beat_strength), float(rhythm_regularity),
                float(tempo_stability), float(onset_strength), float(onset_rate),
                float(rhythm_complexity), float(tempo_variation), float(beat_density),
                float(len(beats))
            ]
            
        except Exception as e:
            print(f"Rhythm feature extraction failed: {e}")
            return [0.0] * 10

class EmotionDataset(Dataset):
    """Enhanced PyTorch Dataset for emotion recognition"""
    
    def __init__(self, file_paths, labels, transform=None, sr=22050, duration=3):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.sr = sr
        self.duration = duration
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Convert to log-mel spectrogram with enhanced parameters
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000, 
                n_fft=2048, hop_length=512
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to tensor
            log_mel_spec = torch.FloatTensor(log_mel_spec)
            
            # Apply transforms if provided
            if self.transform:
                log_mel_spec = self.transform(log_mel_spec)
            
            return log_mel_spec, label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(128, 130), label

class EnhancedEmotionCNN(nn.Module):
    """Enhanced CNN model with better architecture"""
    
    def __init__(self, num_classes=8, input_height=128, input_width=130):
        super(EnhancedEmotionCNN, self).__init__()
        
        # Enhanced architecture with more layers and better regularization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.1)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.2)
        
        # Calculate the size after convolutions
        self.fc_input_size = self._get_conv_output_size(input_height, input_width)
        
        # Enhanced fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.dropout_fc3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def _get_conv_output_size(self, height, width):
        """Calculate the output size after convolutions"""
        x = torch.zeros(1, 1, height, width)
        x = self.dropout1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu(self.bn4(self.conv4(x)))))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu(self.bn4(self.conv4(x)))))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout_fc3(x)
        x = self.fc4(x)
        
        return x

class EnhancedEmotionResNet(nn.Module):
    """Enhanced ResNet-inspired model"""
    
    def __init__(self, num_classes=8):
        super(EnhancedEmotionResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Enhanced residual blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced classifier
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        """Create a residual layer"""
        layers = []
        layers.append(ResidualBlock(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for ResNet"""
    
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class EnhancedEmotionRecognitionTrainer:
    """Enhanced trainer with improved models and training strategies"""
    
    def __init__(self):
        self.feature_extractor = EnhancedAudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_data(self, data_dir="organized_by_emotion", max_files_per_emotion=None):
        """Load and organize data from emotion directories"""
        file_paths = []
        labels = []
        
        emotion_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        emotion_dirs.sort()
        
        print(f"Found emotion directories: {emotion_dirs}")
        
        for emotion in emotion_dirs:
            emotion_path = os.path.join(data_dir, emotion)
            files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3', '.flac'))]
            
            if max_files_per_emotion:
                files = files[:max_files_per_emotion]
            
            for file in files:
                file_paths.append(os.path.join(emotion_path, file))
                labels.append(emotion)
        
        print(f"Loaded {len(file_paths)} files")
        print(f"Emotions: {set(labels)}")
        
        return file_paths, labels
    
    def train_classical_ml(self, file_paths, labels, test_size=0.2):
        """Train classical ML models with enhanced features"""
        print("Extracting features for classical ML...")
        
        features = []
        valid_labels = []
        failed_files = 0
        
        for i, (file_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths))):
            try:
                feature_vector = self.feature_extractor.extract_all_features(file_path)
                if feature_vector is not None and len(feature_vector) > 0:
                    features.append(feature_vector)
                    valid_labels.append(label)
                else:
                    failed_files += 1
            except Exception as e:
                print(f"Failed to extract features from {file_path}: {e}")
                failed_files += 1
        
        if len(features) == 0:
            print("ERROR: No valid features extracted!")
            return None
        
        print(f"Successfully extracted features from {len(features)} files")
        print(f"Failed to extract features from {failed_files} files")
        
        # Ensure all feature vectors have the same length
        expected_length = 1033  # Total expected features
        for i, feature_vector in enumerate(features):
            if len(feature_vector) != expected_length:
                print(f"Warning: Feature vector {i} has length {len(feature_vector)}, expected {expected_length}")
                # Pad or truncate to expected length
                if len(feature_vector) < expected_length:
                    features[i] = np.pad(feature_vector, (0, expected_length - len(feature_vector)), 'constant')
                else:
                    features[i] = feature_vector[:expected_length]
        
        # Convert to numpy arrays
        features = np.array(features)
        labels_encoded = self.label_encoder.fit_transform(valid_labels)
        
        # Check if we have enough samples for each class
        unique_labels, counts = np.unique(labels_encoded, return_counts=True)
        print(f"Class distribution: {dict(zip([self.label_encoder.classes_[i] for i in unique_labels], counts))}")
        
        # Split data (only if test_size > 0)
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels_encoded, test_size=test_size, random_state=42, stratify=labels_encoded
            )
        else:
            # No test split - use all data for training
            X_train, y_train = features, labels_encoded
            X_test, y_test = None, None
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = None
        
        # Enhanced models with better parameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=20, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', 
                C=10.0, 
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation for accuracy estimation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # If we have a test set, evaluate on it
            if X_test_scaled is not None and y_test is not None:
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"{name} Test Accuracy: {accuracy:.4f}")
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_score': cv_scores.mean(),
                    'predictions': y_pred,
                    'y_test': y_test
                }
            else:
                # No test set, use CV score as accuracy
                results[name] = {
                    'model': model,
                    'accuracy': cv_scores.mean(),
                    'cv_score': cv_scores.mean(),
                    'predictions': None,
                    'y_test': None
                }
        
        # Print detailed results
        print("\n" + "="*50)
        print("ENHANCED CLASSICAL ML RESULTS")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name}:")
            if result['y_test'] is not None:
                print(f"  Test Accuracy: {result['accuracy']:.4f}")
                print(f"  CV Score: {result['cv_score']:.4f}")
                
                # Classification report
                print(f"\n  Classification Report:")
                report = classification_report(
                    result['y_test'], result['predictions'], 
                    target_names=self.label_encoder.classes_
                )
                print(report)
            else:
                print(f"  CV Score (used as accuracy): {result['accuracy']:.4f}")
                print("  No test set available for detailed evaluation")
        
        return results
    
    def test_classical_ml(self, X_test, y_test, train_results):
        """Test classical ML models on test set"""
        print("Testing classical ML models on test set...")
        
        # Extract features for test set
        test_features = []
        test_labels = []
        failed_files = 0
        
        for i, (file_path, label) in enumerate(tqdm(zip(X_test, y_test), total=len(X_test))):
            try:
                feature_vector = self.feature_extractor.extract_all_features(file_path)
                if feature_vector is not None and len(feature_vector) > 0:
                    # Ensure consistent length
                    expected_length = 1033
                    if len(feature_vector) != expected_length:
                        if len(feature_vector) < expected_length:
                            feature_vector = np.pad(feature_vector, (0, expected_length - len(feature_vector)), 'constant')
                        else:
                            feature_vector = feature_vector[:expected_length]
                    
                    test_features.append(feature_vector)
                    test_labels.append(label)
                else:
                    failed_files += 1
            except Exception as e:
                print(f"Failed to extract features from {file_path}: {e}")
                failed_files += 1
        
        if len(test_features) == 0:
            print("ERROR: No valid test features extracted!")
            return None
        
        print(f"Successfully extracted test features from {len(test_features)} files")
        print(f"Failed to extract test features from {failed_files} files")
        
        # Convert to numpy arrays
        test_features = np.array(test_features)
        test_labels_encoded = self.label_encoder.transform(test_labels)
        
        # Scale features using the scaler from training
        test_features_scaled = self.scaler.transform(test_features)
        
        # Test each model
        test_results = {}
        
        for name, train_result in train_results.items():
            model = train_result['model']
            
            # Make predictions
            y_pred = model.predict(test_features_scaled)
            accuracy = accuracy_score(test_labels_encoded, y_pred)
            
            print(f"{name} Test Accuracy: {accuracy:.4f}")
            
            # Classification report
            print(f"\n{name} Test Classification Report:")
            report = classification_report(
                test_labels_encoded, y_pred, 
                target_names=self.label_encoder.classes_
            )
            print(report)
            
            test_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'y_test': test_labels_encoded
            }
        
        return test_results
    
    def train_deep_learning_with_validation(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        """Train deep learning models with validation"""
        print("Training deep learning models with validation...")
        
        # Create datasets
        train_dataset = EmotionDataset(X_train, y_train)
        val_dataset = EmotionDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Initialize models
        models = {
            'Enhanced CNN': EnhancedEmotionCNN(num_classes=len(set(y_train))).to(self.device),
            'Enhanced ResNet': EnhancedEmotionResNet(num_classes=len(set(y_train))).to(self.device)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Enhanced optimizer and scheduler
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop with early stopping
            best_val_accuracy = 0
            patience_counter = 0
            patience = 20
            train_losses = []
            val_accuracies = []
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        val_correct += pred.eq(target.view_as(pred)).sum().item()
                        val_total += target.size(0)
                
                val_accuracy = val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                
                train_losses.append(avg_train_loss)
                val_accuracies.append(val_accuracy)
                
                # Learning rate scheduling
                scheduler.step(val_accuracy)
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'models/best_{name.lower().replace(" ", "_")}.pth')
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
                
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            results[name] = {
                'model': model,
                'best_accuracy': best_val_accuracy,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }
            
            print(f"{name} Best Validation Accuracy: {best_val_accuracy:.4f}")
        
        return results
    
    def test_deep_learning(self, X_test, y_test, train_results, batch_size=64):
        """Test deep learning models"""
        print("Testing deep learning models...")
        
        test_dataset = EmotionDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        results = {}
        
        for name, train_result in train_results.items():
            model = train_result['model']
            model.eval()
            
            # Load best model
            try:
                model.load_state_dict(torch.load(f'models/best_{name.lower().replace(" ", "_")}.pth'))
            except:
                print(f"Could not load best model for {name}, using current model")
            
            test_correct = 0
            test_total = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    test_total += target.size(0)
                    
                    all_predictions.extend(pred.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy())
            
            test_accuracy = test_correct / test_total
            
            results[name] = {
                'model': model,
                'accuracy': test_accuracy,
                'predictions': all_predictions,
                'y_test': all_targets
            }
            
            print(f"{name} Test Accuracy: {test_accuracy:.4f}")
            
            # Classification report
            print(f"\n{name} Classification Report:")
            report = classification_report(
                all_targets, all_predictions, 
                target_names=self.label_encoder.classes_
            )
            print(report)
        
        return results
    
    def plot_results(self, classical_results, deep_results):
        """Plot training results and model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        if classical_results and deep_results:
            models = list(classical_results.keys()) + list(deep_results.keys())
            accuracies = []
            
            for name, result in classical_results.items():
                accuracies.append(result.get('accuracy', result.get('cv_score', 0)))
            
            for name, result in deep_results.items():
                accuracies.append(result.get('accuracy', result.get('best_accuracy', 0)))
            
            axes[0, 0].bar(models, accuracies)
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Training curves for deep learning
        if deep_results:
            for name, result in deep_results.items():
                if 'train_losses' in result and 'val_accuracies' in result:
                    axes[0, 1].plot(result['train_losses'], label=f'{name} Train Loss')
                    axes[1, 0].plot(result['val_accuracies'], label=f'{name} Val Accuracy')
            
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
        
        # Confusion matrix for best model
        if classical_results:
            best_model_name = max(classical_results.keys(), 
                                key=lambda x: classical_results[x].get('accuracy', classical_results[x].get('cv_score', 0)))
            best_result = classical_results[best_model_name]
            
            if 'predictions' in best_result and 'y_test' in best_result:
                cm = confusion_matrix(best_result['y_test'], best_result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.label_encoder.classes_,
                           yticklabels=self.label_encoder.classes_,
                           ax=axes[1, 1])
                axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
                axes[1, 1].set_xlabel('Predicted')
                axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Results plotted and saved as 'enhanced_model_comparison.png'")

if __name__ == "__main__":
    # Test the enhanced trainer
    trainer = EnhancedEmotionRecognitionTrainer()
    file_paths, labels = trainer.load_data(max_files_per_emotion=50)  # Test with small subset
    
    if len(file_paths) > 0:
        print("Testing enhanced feature extraction...")
        results = trainer.train_classical_ml(file_paths, labels, test_size=0.2)
        print("Enhanced training completed!")
    else:
        print("No data found for testing")
