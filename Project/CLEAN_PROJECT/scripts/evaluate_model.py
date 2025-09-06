#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluate trained models on test datasets and generate metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path, data_dir, device='auto'):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to trained model
            data_dir: Path to organized emotion data
            device: Computation device
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.device = self._setup_device(device)
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.model = self._load_model()
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self):
        """Load trained model"""
        try:
            model_state = torch.load(self.model_path, map_location=self.device)
            
            # Determine model type
            if 'resnet' in str(self.model_path).lower():
                from custom_models import EmotionResNet
                model = EmotionResNet(num_classes=8)
            elif 'simplecnn' in str(self.model_path).lower():
                from custom_models import SimpleCNN
                model = SimpleCNN(num_classes=8)
            else:
                raise ValueError(f"Unknown model type: {self.model_path}")
            
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_test_data(self):
        """Load test dataset"""
        try:
            from dataset import EmotionDataset
            from utils import load_emotion_data, split_data
            
            # Load all data
            file_paths, labels = load_emotion_data(str(self.data_dir))
            
            # Split data (same random state for consistency)
            train_files, val_files, test_files, train_labels, val_labels, test_labels = split_data(
                file_paths, labels, test_size=0.1, val_size=0.1, random_state=42
            )
            
            # Create test dataset
            test_dataset = EmotionDataset(test_files, test_labels, 
                                        augmentation=False, training=False)
            
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            return test_loader, test_labels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load test data: {e}")
    
    def evaluate(self):
        """Comprehensive model evaluation"""
        print(f"Evaluating model: {self.model_path}")
        print(f"Device: {self.device}")
        
        # Load test data
        test_loader, true_labels = self._load_test_data()
        
        # Make predictions
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                
                outputs = self.model(batch_data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_true_labels.extend(batch_labels.numpy())
        
        # Calculate metrics
        results = self._calculate_metrics(all_true_labels, all_predictions, all_probabilities)
        
        # Generate visualizations
        self._create_visualizations(all_true_labels, all_predictions, all_probabilities)
        
        return results
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = np.mean(np.array(true_labels) == np.array(predictions))
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.emotions, 
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # ROC AUC (one-vs-rest)
        try:
            auc_scores = {}
            probabilities_array = np.array(probabilities)
            
            for i, emotion in enumerate(self.emotions):
                # Create binary labels for current emotion
                binary_true = (np.array(true_labels) == i).astype(int)
                emotion_probs = probabilities_array[:, i]
                
                if len(np.unique(binary_true)) == 2:  # Only if both classes present
                    auc = roc_auc_score(binary_true, emotion_probs)
                    auc_scores[emotion] = auc
            
            macro_auc = np.mean(list(auc_scores.values()))
            
        except Exception as e:
            print(f"Warning: Could not calculate AUC scores: {e}")
            auc_scores = {}
            macro_auc = 0.0
        
        # Per-emotion performance
        per_emotion_metrics = {}
        for emotion in self.emotions:
            emotion_idx = self.emotions.index(emotion)
            if emotion in class_report:
                per_emotion_metrics[emotion] = {
                    'precision': class_report[emotion]['precision'],
                    'recall': class_report[emotion]['recall'],
                    'f1_score': class_report[emotion]['f1-score'],
                    'support': class_report[emotion]['support'],
                    'auc': auc_scores.get(emotion, 0.0)
                }
        
        results = {
            'model_path': str(self.model_path),
            'test_accuracy': accuracy,
            'macro_f1': class_report['macro avg']['f1-score'],
            'weighted_f1': class_report['weighted avg']['f1-score'],
            'macro_auc': macro_auc,
            'per_emotion_metrics': per_emotion_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        return results
    
    def _create_visualizations(self, true_labels, predictions, probabilities):
        """Create evaluation visualizations"""
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        sns.heatmap(conf_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.emotions,
                   yticklabels=self.emotions)
        
        plt.title(f'Confusion Matrix - {self.model_path.stem}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save confusion matrix
        output_dir = self.model_path.parent
        conf_matrix_path = output_dir / 'confusion_matrix.png'
        plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {conf_matrix_path}")
        
        # Per-emotion performance bar chart
        emotions = []
        f1_scores = []
        
        # Calculate per-emotion metrics
        class_report = classification_report(
            true_labels, predictions, 
            target_names=self.emotions, 
            output_dict=True
        )
        
        for emotion in self.emotions:
            if emotion in class_report:
                emotions.append(emotion)
                f1_scores.append(class_report[emotion]['f1-score'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(emotions, f1_scores, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title(f'Per-Emotion F1 Scores - {self.model_path.stem}')
        plt.ylabel('F1 Score')
        plt.xlabel('Emotion')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save performance chart
        performance_chart_path = output_dir / 'per_emotion_performance.png'
        plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance chart saved to: {performance_chart_path}")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Evaluate trained emotion recognition model')
    parser.add_argument('--model', required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--data', required=True,
                       help='Path to organized emotion data directory')
    parser.add_argument('--output',
                       help='Path to save evaluation results (JSON)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Computation device')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model, args.data, args.device)
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Print results
        print("\\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {Path(args.model).stem}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"Macro F1-Score: {results['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
        print(f"Macro AUC: {results['macro_auc']:.4f}")
        
        print("\\nPer-Emotion Performance:")
        print("-" * 80)
        print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-" * 80)
        
        for emotion, metrics in results['per_emotion_metrics'].items():
            print(f"{emotion:<12} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
                  f"{metrics['f1_score']:<12.3f} {metrics['support']:<12}")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\nDetailed results saved to: {args.output}")
        
        # Save results to model directory
        model_dir = Path(args.model).parent
        results_path = model_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
