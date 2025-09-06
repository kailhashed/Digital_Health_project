#!/usr/bin/env python3
"""
Naive Bayes Training Script for Emotion Recognition
Classical ML approach with feature extraction
"""

import os
import sys
import numpy as np
import librosa
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from data_loader import load_emotion_data, get_device

class NaiveBayesTrainer:
    """Naive Bayes trainer with feature extraction"""
    
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral']
        
        # Initialize model
        self.model = GaussianNB(var_smoothing=var_smoothing)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
    def extract_features(self, file_paths, labels):
        """Extract audio features for Naive Bayes"""
        print("🔄 Extracting audio features...")
        
        features = []
        valid_labels = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=22050, duration=3.0)
                y = librosa.util.normalize(y)
                
                # Extract features
                feature_vector = []
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                feature_vector.extend([
                    np.mean(spectral_centroids), np.std(spectral_centroids),
                    np.min(spectral_centroids), np.max(spectral_centroids)
                ])
                
                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                feature_vector.extend([
                    np.mean(spectral_rolloff), np.std(spectral_rolloff)
                ])
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                feature_vector.extend([
                    np.mean(zcr), np.std(zcr)
                ])
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for mfcc in mfccs:
                    feature_vector.extend([
                        np.mean(mfcc), np.std(mfcc)
                    ])
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                for c in chroma:
                    feature_vector.extend([
                        np.mean(c), np.std(c)
                    ])
                
                # Tonnetz features
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                for t in tonnetz:
                    feature_vector.extend([
                        np.mean(t), np.std(t)
                    ])
                
                # Rhythm features
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                feature_vector.append(tempo)
                
                # RMS energy
                rms = librosa.feature.rms(y=y)[0]
                feature_vector.extend([
                    np.mean(rms), np.std(rms)
                ])
                
                # Spectral bandwidth
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                feature_vector.extend([
                    np.mean(spectral_bandwidth), np.std(spectral_bandwidth)
                ])
                
                # Spectral contrast
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                for sc in spectral_contrast:
                    feature_vector.extend([
                        np.mean(sc), np.std(sc)
                    ])
                
                # Mel-frequency spectral coefficients
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                feature_vector.extend([
                    np.mean(mel_spec_db), np.std(mel_spec_db)
                ])
                
                features.append(feature_vector)
                valid_labels.append(labels[i])
                
                if (i + 1) % 1000 == 0:
                    print(f"   Processed {i + 1}/{len(file_paths)} files")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"✓ Extracted features from {len(features)} files")
        print(f"   Feature vector size: {len(features[0]) if features else 0}")
        
        return np.array(features), np.array(valid_labels)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Main training function"""
        print("🎭 Naive Bayes Emotion Recognition Training")
        print("=" * 50)
        
        # Scale features
        print("🔄 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        print("🚀 Training Naive Bayes model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"   Validation Accuracy: {val_acc:.4f}")
        
        return {'val_accuracy': val_acc}
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        print("🔍 Evaluating on test set...")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        test_pred = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"   Test Accuracy: {test_acc:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, test_pred, target_names=self.emotions, output_dict=True)
        
        return {
            'test_accuracy': test_acc,
            'predictions': test_pred,
            'targets': y_test,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path="results/naivebayes/confusion_matrix.png"):
        """Plot confusion matrix"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotions, yticklabels=self.emotions)
        plt.title('Naive Bayes Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def plot_class_probabilities(self, X_test, y_test, save_path="results/naivebayes/class_probabilities.png"):
        """Plot class probabilities"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        X_test_scaled = self.scaler.transform(X_test)
        class_probs = self.model.predict_proba(X_test_scaled)
        
        plt.figure(figsize=(12, 8))
        
        # Plot probability distributions for each class
        for i, emotion in enumerate(self.emotions):
            plt.subplot(2, 4, i+1)
            plt.hist(class_probs[:, i], bins=50, alpha=0.7, label=emotion)
            plt.title(f'{emotion} Probability Distribution')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Class probabilities saved to {save_path}")
    
    def save_model(self, path):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'emotions': self.emotions,
            'params': {
                'var_smoothing': self.var_smoothing
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

def main():
    """Main training function"""
    print("🎭 Naive Bayes Emotion Recognition Training")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data()
    
    # Extract features
    trainer = NaiveBayesTrainer(var_smoothing=1e-9)
    
    print("🔄 Extracting training features...")
    X_train_features, y_train_features = trainer.extract_features(X_train, y_train)
    
    print("🔄 Extracting validation features...")
    X_val_features, y_val_features = trainer.extract_features(X_val, y_val)
    
    print("🔄 Extracting test features...")
    X_test_features, y_test_features = trainer.extract_features(X_test, y_test)
    
    # Train model
    train_results = trainer.train(X_train_features, y_train_features, X_val_features, y_val_features)
    
    # Evaluate on test set
    test_results = trainer.evaluate(X_test_features, y_test_features)
    
    # Plot results
    trainer.plot_confusion_matrix(y_test_features, test_results['predictions'])
    trainer.plot_class_probabilities(X_test_features, y_test_features)
    
    # Save model and results
    trainer.save_model("models/naivebayes/best_naivebayes.pkl")
    
    results = {
        'model': 'Naive Bayes',
        'train_results': train_results,
        'test_results': test_results,
        'timestamp': datetime.now().isoformat(),
        'params': {
            'var_smoothing': trainer.var_smoothing
        }
    }
    
    os.makedirs("results/naivebayes", exist_ok=True)
    with open("results/naivebayes/naivebayes_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Naive Bayes training completed!")
    print(f"   Final test accuracy: {test_results['test_accuracy']:.4f}")
    print(f"   Results saved to results/naivebayes/")

if __name__ == "__main__":
    main()
