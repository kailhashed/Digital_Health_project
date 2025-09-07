#!/usr/bin/env python3
"""
Test Script for DenseNet Model - Epoch 38
Comprehensive evaluation of the trained DenseNet model
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')
from src.models.custom_models import EmotionDenseNet
from data_loader import load_emotion_data, create_data_loaders, get_device

class DenseNetTester:
    """DenseNet model tester for comprehensive evaluation"""
    
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.num_classes = 8
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # Initialize model
        self.model = EmotionDenseNet(
            num_classes=self.num_classes,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_init_features=64,
            dropout=0.1
        ).to(device)
        
        # Load trained weights
        self.load_model()
        
    def load_model(self):
        """Load the trained model weights"""
        print(f"🔄 Loading DenseNet model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Loaded model state dict")
                
                # Print training info if available
                if 'history' in checkpoint:
                    history = checkpoint['history']
                    if 'val_acc' in history and len(history['val_acc']) > 0:
                        best_val_acc = max(history['val_acc'])
                        print(f"   Best validation accuracy during training: {best_val_acc:.2f}%")
            else:
                self.model.load_state_dict(checkpoint)
                print("✓ Loaded model weights directly")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def test_model(self, test_loader):
        """Test the model on test dataset"""
        print("🔍 Testing DenseNet model on test dataset...")
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(test_loader)
        test_accuracy = accuracy_score(all_targets, all_predictions)
        
        print(f"📊 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets
        }
    
    def generate_detailed_report(self, results):
        """Generate detailed classification report"""
        print("\n📋 Detailed Classification Report:")
        print("=" * 60)
        
        # Classification report
        report = classification_report(
            results['targets'], 
            results['predictions'], 
            target_names=self.emotions,
            digits=4
        )
        print(report)
        
        # Per-class accuracy
        print("\n📈 Per-Class Accuracy:")
        print("-" * 30)
        cm = confusion_matrix(results['targets'], results['predictions'])
        for i, emotion in enumerate(self.emotions):
            class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"   {emotion:10s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        return report, cm
    
    def create_confusion_matrix_plot(self, cm, save_path="results/densenet/confusion_matrix_epoch38.png"):
        """Create and save confusion matrix visualization"""
        print(f"📊 Creating confusion matrix plot...")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.emotions, yticklabels=self.emotions)
        plt.title('DenseNet Epoch 38 - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def analyze_predictions(self, results):
        """Analyze prediction patterns and confidence"""
        print("\n🔍 Prediction Analysis:")
        print("-" * 30)
        
        probabilities = np.array(results['probabilities'])
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        
        # Calculate confidence statistics
        max_probs = np.max(probabilities, axis=1)
        mean_confidence = np.mean(max_probs)
        std_confidence = np.std(max_probs)
        
        print(f"   Mean prediction confidence: {mean_confidence:.4f}")
        print(f"   Std prediction confidence: {std_confidence:.4f}")
        print(f"   Min confidence: {np.min(max_probs):.4f}")
        print(f"   Max confidence: {np.max(max_probs):.4f}")
        
        # Analyze correct vs incorrect predictions
        correct_mask = predictions == targets
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        if len(correct_confidences) > 0:
            print(f"   Mean confidence (correct): {np.mean(correct_confidences):.4f}")
        if len(incorrect_confidences) > 0:
            print(f"   Mean confidence (incorrect): {np.mean(incorrect_confidences):.4f}")
        
        return {
            'mean_confidence': mean_confidence,
            'std_confidence': std_confidence,
            'correct_confidences': correct_confidences,
            'incorrect_confidences': incorrect_confidences
        }
    
    def save_results(self, results, report, cm, analysis):
        """Save all results to JSON file"""
        print("\n💾 Saving results...")
        
        # Prepare results dictionary
        results_dict = {
            'model_info': {
                'model_type': 'DenseNet',
                'epoch': 38,
                'model_path': self.model_path,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            },
            'test_metrics': {
                'test_loss': float(results['test_loss']),
                'test_accuracy': float(results['test_accuracy']),
                'test_accuracy_percent': float(results['test_accuracy'] * 100)
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'prediction_analysis': {
                'mean_confidence': float(analysis['mean_confidence']),
                'std_confidence': float(analysis['std_confidence']),
                'correct_predictions': int(np.sum(np.array(results['predictions']) == np.array(results['targets']))),
                'total_predictions': len(results['predictions'])
            },
            'emotions': self.emotions
        }
        
        # Save to file
        os.makedirs("results/densenet", exist_ok=True)
        results_file = "results/densenet/densenet_epoch38_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Results saved to {results_file}")
        return results_file

def main():
    """Main testing function"""
    try:
        print("🎭 DenseNet Epoch 38 Model Testing")
        print("=" * 50)
        
        # Get device
        device = get_device()
        
        # Model path
        model_path = "models/densenet/best_densenet_epoch_38.pth"
        
        # Load test data
        print("🔄 Loading test data...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_emotion_data("data/organized_by_emotion")
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            batch_size=32, num_workers=0
        )
        
        print(f"📊 Test dataset size: {len(X_test)} samples")
        
        # Initialize tester
        tester = DenseNetTester(model_path, device)
        
        # Test the model
        results = tester.test_model(test_loader)
        
        # Generate detailed report
        report, cm = tester.generate_detailed_report(results)
        
        # Analyze predictions
        analysis = tester.analyze_predictions(results)
        
        # Create confusion matrix plot
        tester.create_confusion_matrix_plot(cm)
        
        # Save all results
        results_file = tester.save_results(results, report, cm, analysis)
        
        print(f"\n🎉 DenseNet Epoch 38 testing completed!")
        print(f"   Final test accuracy: {results['test_accuracy']*100:.2f}%")
        print(f"   Results saved to: {results_file}")
        print(f"   Confusion matrix saved to: results/densenet/confusion_matrix_epoch38.png")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
