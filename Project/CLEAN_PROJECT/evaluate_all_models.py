#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Evaluates all 9 models and generates comparison metrics
"""

import os
import sys
import torch
import numpy as np
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_loader import load_emotion_data, create_data_loaders, get_device
from models.custom_models import EmotionDenseNet, EmotionResNet, EmotionLSTM, EmotionTransformer
from models.pretrained_models import FixedWav2Vec2Classifier, SimpleCNNAudioClassifier

# Import classical ML models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

class ModelEvaluator:
    """Comprehensive model evaluator"""
    
    def __init__(self):
        self.device = get_device()
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.results = {}
        
    def extract_classical_features(self, file_paths, labels):
        """Extract features for classical ML models"""
        print("🔄 Extracting features for classical ML models...")
        
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
                
                # RMS energy
                rms = librosa.feature.rms(y=y)[0]
                feature_vector.extend([
                    np.mean(rms), np.std(rms)
                ])
                
                features.append(feature_vector)
                valid_labels.append(labels[i])
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return np.array(features), np.array(valid_labels)
    
    def evaluate_deep_learning_model(self, model_name, model_class, model_path=None):
        """Evaluate a deep learning model"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        try:
            # Load data
            X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data()
            train_loader, val_loader, test_loader = create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test,
                batch_size=32, num_workers=0
            )
            
            # Initialize model
            model = model_class(num_classes=8).to(self.device)
            
            # Load trained weights if available
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"✓ Loaded trained weights from {model_path}")
            else:
                print(f"⚠️ No trained weights found at {model_path}, using random initialization")
            
            # Evaluate on test set
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_preds)
            
            # Get model parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            self.results[model_name] = {
                'type': 'Deep Learning',
                'accuracy': accuracy,
                'predictions': all_preds,
                'targets': all_targets,
                'parameters': total_params,
                'model_path': model_path
            }
            
            print(f"✓ {model_name} - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return False
    
    def evaluate_classical_model(self, model_name, model_class, model_path=None):
        """Evaluate a classical ML model"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        try:
            # Load data
            X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data()
            
            # Extract features
            X_train_features, y_train_features = self.extract_classical_features(X_train, y_train)
            X_test_features, y_test_features = self.extract_classical_features(X_test, y_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_features)
            X_test_scaled = scaler.transform(X_test_features)
            
            # Initialize model
            if model_name == 'XGBoost':
                model = model_class(n_estimators=200, random_state=42)
            else:
                model = model_class()
            
            # Load trained model if available
            if model_path and os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"✓ Loaded trained model from {model_path}")
            else:
                # Train model
                print(f"⚠️ No trained model found, training {model_name}...")
                model.fit(X_train_scaled, y_train_features)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_features, y_pred)
            
            self.results[model_name] = {
                'type': 'Classical ML',
                'accuracy': accuracy,
                'predictions': y_pred.tolist(),
                'targets': y_test_features.tolist(),
                'parameters': 'N/A',
                'model_path': model_path
            }
            
            print(f"✓ {model_name} - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return False
    
    def evaluate_all_models(self):
        """Evaluate all 9 models"""
        print("🚀 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        # Deep Learning Models
        deep_learning_models = [
            ('DenseNet', EmotionDenseNet, 'models/densenet/best_densenet.pth'),
            ('CRNN', None, 'models/crnn/best_crnn.pth'),  # CRNN is defined in train_crnn.py
            ('LSTM', EmotionLSTM, 'models/lstm/best_LSTM.pth'),
            ('ResNet', EmotionResNet, 'models/resnet/best_ResNet.pth'),
            ('Transformer', EmotionTransformer, 'models/transformer/best_Transformer.pth'),
            ('FixedWav2Vec2', FixedWav2Vec2Classifier, 'models/pretrained/best_FixedWav2Vec2.pth'),
            ('SimpleCNNAudio', SimpleCNNAudioClassifier, 'models/pretrained/best_SimpleCNNAudio.pth')
        ]
        
        # Classical ML Models
        classical_models = [
            ('AdaBoost', AdaBoostClassifier, 'models/adaboost/best_adaboost.pkl'),
            ('NaiveBayes', GaussianNB, 'models/naivebayes/best_naivebayes.pkl'),
            ('XGBoost', xgb.XGBClassifier, 'models/xgboost/best_xgboost.pkl')
        ]
        
        # Evaluate deep learning models
        for model_name, model_class, model_path in deep_learning_models:
            if model_class is not None:  # Skip CRNN for now as it's defined differently
                self.evaluate_deep_learning_model(model_name, model_class, model_path)
        
        # Evaluate classical ML models
        for model_name, model_class, model_path in classical_models:
            self.evaluate_classical_model(model_name, model_class, model_path)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n📊 GENERATING COMPARISON REPORT")
        print("=" * 60)
        
        # Sort models by accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Create comparison table
        print("\n🏆 MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Rank':<4} {'Model':<15} {'Type':<15} {'Accuracy':<10} {'Parameters':<12}")
        print("-" * 80)
        
        for rank, (model_name, result) in enumerate(sorted_models, 1):
            acc_pct = result['accuracy'] * 100
            params = result['parameters']
            if isinstance(params, int):
                params_str = f"{params:,}"
            else:
                params_str = str(params)
            
            print(f"{rank:<4} {model_name:<15} {result['type']:<15} {acc_pct:>7.2f}% {params_str:>12}")
        
        # Generate detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.results),
            'emotions': self.emotions,
            'results': self.results,
            'ranking': sorted_models
        }
        
        # Save results
        os.makedirs('results/comparison', exist_ok=True)
        
        # Save JSON report
        with open('results/comparison/comprehensive_evaluation.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save pickle for further analysis
        with open('results/comparison/comprehensive_evaluation.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        # Generate markdown report
        self.generate_markdown_report(sorted_models)
        
        print(f"\n✅ Results saved to results/comparison/")
        return sorted_models
    
    def generate_markdown_report(self, sorted_models):
        """Generate markdown comparison report"""
        report_path = 'results/comparison/comprehensive_evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# 🎭 Comprehensive Emotion Recognition Model Evaluation\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** 8 emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised)\n\n")
            f.write(f"**Features:** 8-channel audio features (mel-spectrogram, MFCC, chroma, spectral centroid, spectral rolloff, ZCR, RMS, tonnetz)\n\n")
            
            f.write("## 🏆 Model Performance Ranking\n\n")
            f.write("| Rank | Model | Type | Accuracy | Parameters |\n")
            f.write("|------|-------|------|----------|------------|\n")
            
            for rank, (model_name, result) in enumerate(sorted_models, 1):
                acc_pct = result['accuracy'] * 100
                params = result['parameters']
                if isinstance(params, int):
                    params_str = f"{params:,}"
                else:
                    params_str = str(params)
                
                f.write(f"| {rank} | {model_name} | {result['type']} | {acc_pct:.2f}% | {params_str} |\n")
            
            f.write("\n## 📊 Detailed Results\n\n")
            for model_name, result in sorted_models:
                f.write(f"### {model_name}\n\n")
                f.write(f"- **Type:** {result['type']}\n")
                f.write(f"- **Accuracy:** {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
                f.write(f"- **Parameters:** {result['parameters']}\n")
                f.write(f"- **Model Path:** {result['model_path']}\n\n")
            
            # Add insights
            if sorted_models:
                best_model, best_result = sorted_models[0]
                f.write("## 🎯 Key Insights\n\n")
                f.write(f"- **Best Performing Model:** {best_model} with {best_result['accuracy']*100:.2f}% accuracy\n")
                
                # Count by type
                deep_learning_count = sum(1 for _, result in self.results.items() if result['type'] == 'Deep Learning')
                classical_count = sum(1 for _, result in self.results.items() if result['type'] == 'Classical ML')
                
                f.write(f"- **Deep Learning Models:** {deep_learning_count} models evaluated\n")
                f.write(f"- **Classical ML Models:** {classical_count} models evaluated\n")
                
                # Performance by type
                dl_models = [(name, result) for name, result in self.results.items() if result['type'] == 'Deep Learning']
                cl_models = [(name, result) for name, result in self.results.items() if result['type'] == 'Classical ML']
                
                if dl_models:
                    dl_avg_acc = np.mean([result['accuracy'] for _, result in dl_models])
                    f.write(f"- **Average Deep Learning Accuracy:** {dl_avg_acc*100:.2f}%\n")
                
                if cl_models:
                    cl_avg_acc = np.mean([result['accuracy'] for _, result in cl_models])
                    f.write(f"- **Average Classical ML Accuracy:** {cl_avg_acc*100:.2f}%\n")
        
        print(f"📄 Markdown report saved to {report_path}")

def main():
    """Main evaluation function"""
    print("🎭 COMPREHENSIVE EMOTION RECOGNITION MODEL EVALUATION")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    evaluator.evaluate_all_models()
    
    # Generate comparison report
    sorted_models = evaluator.generate_comparison_report()
    
    # Final summary
    if sorted_models:
        best_model, best_result = sorted_models[0]
        print(f"\n🏆 CHAMPION: {best_model}")
        print(f"   Accuracy: {best_result['accuracy']*100:.2f}%")
        print(f"   Type: {best_result['type']}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 Results saved to results/comparison/")

if __name__ == "__main__":
    main()
