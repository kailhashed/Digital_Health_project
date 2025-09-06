#!/usr/bin/env python3
"""
XGBoost Training Script for Emotion Recognition
Gradient boosting with early stopping and comprehensive evaluation
"""

import os
import sys
import numpy as np
import librosa
import xgboost as xgb
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

class XGBoostTrainer:
    """XGBoost trainer with feature extraction and early stopping"""
    
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral']
        
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Training history
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'n_estimators': []
        }
        
    def extract_features(self, file_paths, labels):
        """Extract audio features for XGBoost"""
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
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val, patience=20):
        """Train with early stopping based on validation accuracy"""
        print(f"🚀 Starting XGBoost training with early stopping...")
        print(f"   Max estimators: {self.n_estimators}")
        print(f"   Early stopping patience: {patience}")
        
        best_val_acc = 0
        best_n_estimators = 0
        patience_counter = 0
        
        # Train incrementally
        for n_est in range(10, self.n_estimators + 1, 10):
            # Create model with current number of estimators
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            
            # Evaluate on validation set
            val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Store history
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['n_estimators'].append(n_est)
            
            print(f"   Estimators: {n_est:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_n_estimators = n_est
                patience_counter = 0
                # Update the main model
                self.model = model
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"   ⏹️ Early stopping triggered at {n_est} estimators")
                break
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        print(f"   Best number of estimators: {best_n_estimators}")
        
        return self.history
    
    def train(self, X_train, y_train, X_val, y_val):
        """Main training function"""
        print("🎭 XGBoost Emotion Recognition Training")
        print("=" * 50)
        
        # Scale features
        print("🔄 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train with early stopping
        history = self.train_with_early_stopping(X_train_scaled, y_train, X_val_scaled, y_val)
        
        return history
    
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
    
    def plot_training_history(self, save_path="results/xgboost/training_history.png"):
        """Plot training history"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['n_estimators'], self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['n_estimators'], self.history['val_acc'], label='Validation Accuracy')
        plt.title('XGBoost Training and Validation Accuracy')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Feature importance plot
        plt.subplot(1, 2, 2)
        feature_importance = self.model.feature_importances_
        top_features = np.argsort(feature_importance)[-20:]  # Top 20 features
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training history saved to {save_path}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path="results/xgboost/confusion_matrix.png"):
        """Plot confusion matrix"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotions, yticklabels=self.emotions)
        plt.title('XGBoost Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved to {save_path}")
    
    def save_model(self, path):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'history': self.history,
            'emotions': self.emotions,
            'params': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

def main():
    """Main training function"""
    print("🎭 XGBoost Emotion Recognition Training")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data()
    
    # Extract features
    trainer = XGBoostTrainer(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8)
    
    print("🔄 Extracting training features...")
    X_train_features, y_train_features = trainer.extract_features(X_train, y_train)
    
    print("🔄 Extracting validation features...")
    X_val_features, y_val_features = trainer.extract_features(X_val, y_val)
    
    print("🔄 Extracting test features...")
    X_test_features, y_test_features = trainer.extract_features(X_test, y_test)
    
    # Train model
    history = trainer.train(X_train_features, y_train_features, X_val_features, y_val_features)
    
    # Evaluate on test set
    test_results = trainer.evaluate(X_test_features, y_test_features)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(y_test_features, test_results['predictions'])
    
    # Save model and results
    trainer.save_model("models/xgboost/best_xgboost.pkl")
    
    results = {
        'model': 'XGBoost',
        'training_history': history,
        'test_results': test_results,
        'timestamp': datetime.now().isoformat(),
        'params': {
            'n_estimators': trainer.n_estimators,
            'learning_rate': trainer.learning_rate,
            'max_depth': trainer.max_depth,
            'subsample': trainer.subsample
        }
    }
    
    os.makedirs("results/xgboost", exist_ok=True)
    with open("results/xgboost/xgboost_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 XGBoost training completed!")
    print(f"   Final test accuracy: {test_results['test_accuracy']:.4f}")
    print(f"   Results saved to results/xgboost/")

if __name__ == "__main__":
    main()
