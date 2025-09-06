"""
Evaluation metrics for emotion recognition
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def calculate_accuracy(targets, predictions):
    """Calculate accuracy from predictions and targets"""
    if len(targets) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(targets, predictions) if t == p)
    return correct / len(targets)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, class_names=None):
        """
        Args:
            class_names: List of class names for reports
        """
        self.class_names = class_names
    
    def evaluate(self, y_true, y_pred):
        """
        Comprehensive evaluation of model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Basic accuracy
        results['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        results['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Per-class metrics
        results['per_class_accuracy'] = self._calculate_per_class_accuracy(y_true, y_pred)
        
        return results
    
    def _calculate_per_class_accuracy(self, y_true, y_pred):
        """Calculate per-class accuracy"""
        if self.class_names is None:
            unique_classes = sorted(list(set(y_true)))
        else:
            unique_classes = list(range(len(self.class_names)))
        
        per_class_acc = {}
        
        for class_idx in unique_classes:
            class_mask = np.array(y_true) == class_idx
            if np.sum(class_mask) > 0:
                class_pred = np.array(y_pred)[class_mask]
                class_true = np.array(y_true)[class_mask]
                accuracy = accuracy_score(class_true, class_pred)
                
                class_name = self.class_names[class_idx] if self.class_names else str(class_idx)
                per_class_acc[class_name] = accuracy
        
        return per_class_acc
    
    def print_evaluation_summary(self, results):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        # Per-class accuracy
        if 'per_class_accuracy' in results:
            print(f"\nPer-Class Accuracy:")
            for class_name, accuracy in results['per_class_accuracy'].items():
                print(f"  {class_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report summary
        if 'classification_report' in results:
            report = results['classification_report']
            if 'weighted avg' in report:
                weighted_avg = report['weighted avg']
                print(f"\nWeighted Average:")
                print(f"  Precision: {weighted_avg['precision']:.4f}")
                print(f"  Recall: {weighted_avg['recall']:.4f}")
                print(f"  F1-Score: {weighted_avg['f1-score']:.4f}")
    
    def save_results(self, results, filepath):
        """Save evaluation results to file"""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation results saved to: {filepath}")
    
    def generate_report_text(self, results, model_name="Model"):
        """Generate text report of evaluation results"""
        report_text = f"""
# {model_name} Evaluation Report

## Overall Performance
- **Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)

## Per-Class Performance
"""
        
        if 'per_class_accuracy' in results:
            for class_name, accuracy in results['per_class_accuracy'].items():
                report_text += f"- **{class_name}**: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
        
        if 'classification_report' in results:
            report_text += "\n## Detailed Classification Report\n\n"
            report = results['classification_report']
            
            # Header
            report_text += "| Class | Precision | Recall | F1-Score | Support |\n"
            report_text += "|-------|-----------|--------|----------|---------|\n"
            
            # Per-class metrics
            for class_name in self.class_names or []:
                if class_name in report:
                    metrics = report[class_name]
                    report_text += f"| {class_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {int(metrics['support'])} |\n"
            
            # Overall metrics
            if 'weighted avg' in report:
                metrics = report['weighted avg']
                report_text += f"| **Weighted Avg** | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {int(metrics['support'])} |\n"
        
        return report_text

