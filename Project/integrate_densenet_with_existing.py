#!/usr/bin/env python3
"""
Integration Script: DenseNet with Existing Model Framework
Integrates the optimized DenseNet into the existing comprehensive model comparison system.
"""

import os
import sys
import torch
import numpy as np
import json
import pickle
from datetime import datetime

# Add paths
sys.path.append('src/models')
from custom_models import EmotionDenseNet


class DenseNetIntegrator:
    """Integrates DenseNet with existing model framework"""
    
    def __init__(self):
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
    def load_existing_results(self):
        """Load existing model results for comparison"""
        results = {}
        
        # Load existing model results
        result_files = {
            'SimpleCNN': 'results/simplecnn_test_results.pkl',
            'ResNet': 'results/resnet_test_results.pkl',
            'LSTM': 'results/lstm/lstm_test_results.pkl',
            'Transformer': 'results/transformer/transformer_test_results.pkl'
        }
        
        for model_name, file_path in result_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        results[model_name] = pickle.load(f)
                    print(f"✓ Loaded {model_name} results")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
            else:
                print(f"✗ {model_name} results not found at {file_path}")
        
        return results
    
    def load_densenet_results(self):
        """Load the latest DenseNet results"""
        # Find the most recent DenseNet results
        densenet_dirs = [d for d in os.listdir('.') if d.startswith('results_optimized_densenet_')]
        
        if not densenet_dirs:
            print("No DenseNet results found")
            return None
        
        # Get the most recent directory
        latest_dir = sorted(densenet_dirs)[-1]
        
        # Load results
        results_file = os.path.join(latest_dir, 'full_evaluation_results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                densenet_results = pickle.load(f)
            print(f"✓ Loaded DenseNet results from {latest_dir}")
            return densenet_results
        else:
            print(f"✗ DenseNet results file not found in {latest_dir}")
            return None
    
    def create_unified_comparison(self, existing_results, densenet_results):
        """Create unified comparison including DenseNet"""
        
        if densenet_results is None:
            print("Cannot create comparison - DenseNet results not available")
            return None
        
        comparison = {
            'models': {},
            'comparison_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Add existing models
        for model_name, results in existing_results.items():
            if 'test_accuracy' in results:
                comparison['models'][model_name] = {
                    'accuracy': results['test_accuracy'],
                    'model_type': 'existing',
                    'parameters': results.get('model_parameters', 'Unknown')
                }
        
        # Add DenseNet
        comparison['models']['OptimizedDenseNet'] = {
            'accuracy': densenet_results['test_accuracy'],
            'model_type': 'densenet',
            'parameters': '7.6M',
            'false_positive_rate': np.mean([
                densenet_results['false_positive_analysis'][emotion]['false_positive_rate']
                for emotion in self.emotions
            ]),
            'classification_report': densenet_results['classification_report']
        }
        
        # Calculate comparison metrics
        accuracies = {name: data['accuracy'] for name, data in comparison['models'].items()}
        best_model = max(accuracies.items(), key=lambda x: x[1])
        
        comparison['comparison_metrics'] = {
            'best_model': best_model[0],
            'best_accuracy': best_model[1],
            'model_ranking': sorted(accuracies.items(), key=lambda x: x[1], reverse=True),
            'accuracy_improvements': {
                name: (acc - min(accuracies.values())) / min(accuracies.values()) * 100
                for name, acc in accuracies.items()
            }
        }
        
        return comparison
    
    def update_comprehensive_results(self, comparison):
        """Update the comprehensive model comparison results"""
        if comparison is None:
            return
        
        # Save updated comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = f"results/comprehensive_comparison_with_densenet_{timestamp}.json"
        os.makedirs('results', exist_ok=True)
        
        # Prepare JSON-serializable data
        json_data = comparison.copy()
        if 'OptimizedDenseNet' in json_data['models']:
            # Remove numpy arrays and complex objects
            if 'classification_report' in json_data['models']['OptimizedDenseNet']:
                del json_data['models']['OptimizedDenseNet']['classification_report']
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save full comparison with all data
        pickle_file = f"results/comprehensive_comparison_with_densenet_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(comparison, f)
        
        print(f"✓ Updated comprehensive comparison saved:")
        print(f"  JSON: {json_file}")
        print(f"  Pickle: {pickle_file}")
        
        return json_file, pickle_file
    
    def create_performance_summary(self, comparison):
        """Create a performance summary report"""
        if comparison is None:
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\nModel Performance Ranking:")
        for i, (model_name, accuracy) in enumerate(comparison['comparison_metrics']['model_ranking'], 1):
            improvement = comparison['comparison_metrics']['accuracy_improvements'][model_name]
            print(f"  {i}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%) [+{improvement:.1f}%]")
        
        print(f"\nBest Performing Model: {comparison['comparison_metrics']['best_model']}")
        print(f"Best Accuracy: {comparison['comparison_metrics']['best_accuracy']:.4f} ({comparison['comparison_metrics']['best_accuracy']*100:.2f}%)")
        
        # DenseNet specific metrics
        if 'OptimizedDenseNet' in comparison['models']:
            densenet_data = comparison['models']['OptimizedDenseNet']
            print(f"\nOptimized DenseNet Specific Metrics:")
            print(f"  Accuracy: {densenet_data['accuracy']:.4f} ({densenet_data['accuracy']*100:.2f}%)")
            print(f"  Average False Positive Rate: {densenet_data['false_positive_rate']:.3f}")
            print(f"  Parameters: {densenet_data['parameters']}")
            
            # Check if DenseNet is the best
            if comparison['comparison_metrics']['best_model'] == 'OptimizedDenseNet':
                print(f"  🏆 DenseNet achieved the BEST performance!")
            else:
                best_acc = comparison['comparison_metrics']['best_accuracy']
                densenet_acc = densenet_data['accuracy']
                diff = (best_acc - densenet_acc) * 100
                print(f"  📊 DenseNet is {diff:.2f}% behind the best model")
        
        print("="*80)
    
    def check_densenet_advantages(self, comparison):
        """Analyze DenseNet advantages over other models"""
        if comparison is None or 'OptimizedDenseNet' not in comparison['models']:
            return
        
        densenet_data = comparison['models']['OptimizedDenseNet']
        
        print(f"\nDenseNet Architecture Advantages:")
        print(f"  ✓ Dense connectivity for feature reuse")
        print(f"  ✓ Efficient parameter usage ({densenet_data['parameters']} parameters)")
        print(f"  ✓ Advanced training with focal loss and class balancing")
        print(f"  ✓ Low false positive rate: {densenet_data['false_positive_rate']:.3f}")
        print(f"  ✓ Optimized for emotion recognition tasks")
        
        # Compare with other models
        other_accuracies = [data['accuracy'] for name, data in comparison['models'].items() 
                           if name != 'OptimizedDenseNet']
        
        if other_accuracies:
            avg_other_acc = np.mean(other_accuracies)
            densenet_acc = densenet_data['accuracy']
            
            if densenet_acc > avg_other_acc:
                improvement = (densenet_acc - avg_other_acc) * 100
                print(f"  🎯 DenseNet outperforms average by {improvement:.2f}%")
            
            max_other_acc = max(other_accuracies)
            if densenet_acc > max_other_acc:
                improvement = (densenet_acc - max_other_acc) * 100
                print(f"  🏆 DenseNet outperforms best previous model by {improvement:.2f}%")


def main():
    """Main integration function"""
    print("DenseNet Integration with Existing Model Framework")
    print("="*70)
    
    integrator = DenseNetIntegrator()
    
    # Load existing results
    print("Loading existing model results...")
    existing_results = integrator.load_existing_results()
    
    # Load DenseNet results
    print("\nLoading DenseNet results...")
    densenet_results = integrator.load_densenet_results()
    
    # Create unified comparison
    print("\nCreating unified comparison...")
    comparison = integrator.create_unified_comparison(existing_results, densenet_results)
    
    # Update comprehensive results
    print("\nUpdating comprehensive results...")
    files = integrator.update_comprehensive_results(comparison)
    
    # Create performance summary
    integrator.create_performance_summary(comparison)
    
    # Analyze DenseNet advantages
    integrator.check_densenet_advantages(comparison)
    
    print(f"\nIntegration completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
