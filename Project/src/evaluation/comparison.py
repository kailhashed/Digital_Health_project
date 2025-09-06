"""
Model comparison utilities
"""

import os
import pickle
import numpy as np
from datetime import datetime


class ModelComparator:
    """Compare multiple emotion recognition models"""
    
    def __init__(self):
        self.results = {}
    
    def load_results(self, results_dir='results'):
        """Load all available model results"""
        all_results = {}
        
        # Custom Deep Learning Models
        try:
            with open(os.path.join(results_dir, 'all_deep_models_results.pkl'), 'rb') as f:
                custom_results = pickle.load(f)
            print(f"✓ Loaded {len(custom_results)} custom deep learning models")
            for name, result in custom_results.items():
                all_results[f"Custom-{name}"] = {
                    'type': 'Custom Deep Learning',
                    'val_acc': result['train']['best_val_acc'],
                    'test_acc': result['test']['accuracy'],
                    'trainable_params': 'All',
                    'total_params': 'Unknown'
                }
        except FileNotFoundError:
            print("⚠️  Custom deep learning results not found")
        
        # Pre-trained Models
        try:
            with open(os.path.join(results_dir, 'working_pretrained_results.pkl'), 'rb') as f:
                pretrained_results = pickle.load(f)
            print(f"✓ Loaded {len(pretrained_results)} pre-trained models")
            for name, result in pretrained_results.items():
                all_results[f"Pretrained-{name}"] = {
                    'type': 'Pre-trained Fine-tuned',
                    'val_acc': result['train']['best_val_acc'],
                    'test_acc': result['test']['accuracy'],
                    'trainable_params': result['model_info']['trainable_params'],
                    'total_params': result['model_info']['total_params']
                }
        except FileNotFoundError:
            print("⚠️  Pre-trained model results not found")
        
        # Classical ML Models
        try:
            with open(os.path.join(results_dir, 'enhanced_classical_ml_results.pkl'), 'rb') as f:
                classical_results = pickle.load(f)
            print(f"✓ Loaded {len(classical_results)} classical ML models")
            for name, result in classical_results.items():
                if isinstance(result, dict) and 'test_accuracy' in result:
                    all_results[f"Classical-{name}"] = {
                        'type': 'Classical ML',
                        'val_acc': result.get('val_accuracy', 0),
                        'test_acc': result['test_accuracy'],
                        'trainable_params': 'N/A',
                        'total_params': 'N/A'
                    }
        except FileNotFoundError:
            print("⚠️  Classical ML results not found")
        
        self.results = all_results
        return all_results
    
    def create_comparison_table(self):
        """Create a performance comparison table"""
        if not self.results:
            print("❌ No results available")
            return
        
        print("\n" + "="*100)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*100)
        
        # Sort by test accuracy
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<25} {'Type':<20} {'Val Acc':<10} {'Test Acc':<10} {'Performance':<12}")
        print("-" * 100)
        
        for rank, (model_name, result) in enumerate(sorted_results, 1):
            name_short = model_name.replace('Custom-', '').replace('Pretrained-', '').replace('Classical-', '')
            val_acc = result['val_acc']
            test_acc = result['test_acc']
            model_type = result['type']
            
            # Performance level
            if test_acc >= 0.7:
                perf_level = "Excellent"
            elif test_acc >= 0.5:
                perf_level = "Good"
            elif test_acc >= 0.3:
                perf_level = "Fair"
            elif test_acc >= 0.15:
                perf_level = "Poor"
            else:
                perf_level = "Failed"
            
            print(f"{rank:<4} {name_short:<25} {model_type:<20} {val_acc:<10.4f} {test_acc:<10.4f} {perf_level:<12}")
        
        return sorted_results
    
    def analyze_by_type(self):
        """Analyze performance by model type"""
        print("\n" + "="*80)
        print("📈 ANALYSIS BY MODEL TYPE")
        print("="*80)
        
        type_stats = {}
        
        for model_name, result in self.results.items():
            model_type = result['type']
            test_acc = result['test_acc']
            
            if model_type not in type_stats:
                type_stats[model_type] = []
            type_stats[model_type].append(test_acc)
        
        print(f"{'Type':<25} {'Count':<6} {'Best':<8} {'Avg':<8} {'Worst':<8} {'Std':<8}")
        print("-" * 70)
        
        for model_type, accuracies in type_stats.items():
            count = len(accuracies)
            best = max(accuracies)
            avg = np.mean(accuracies)
            worst = min(accuracies)
            std = np.std(accuracies)
            
            print(f"{model_type:<25} {count:<6} {best:<8.4f} {avg:<8.4f} {worst:<8.4f} {std:<8.4f}")
    
    def find_champions(self):
        """Find the best models overall and by category"""
        print("\n" + "="*80)
        print("🏆 CHAMPION ANALYSIS")
        print("="*80)
        
        if not self.results:
            return
        
        # Overall champion
        overall_best = max(self.results.items(), key=lambda x: x[1]['test_acc'])
        print(f"\n🥇 OVERALL CHAMPION: {overall_best[0]}")
        print(f"   Type: {overall_best[1]['type']}")
        print(f"   Test Accuracy: {overall_best[1]['test_acc']:.4f} ({overall_best[1]['test_acc']*100:.2f}%)")
        print(f"   Validation Accuracy: {overall_best[1]['val_acc']:.4f}")
        
        # Improvement over random
        baseline = 0.125  # 1/8 for 8 classes
        improvement = (overall_best[1]['test_acc'] - baseline) / baseline * 100
        print(f"   Improvement over random: +{improvement:.1f}%")
        
        # Best by type
        type_champions = {}
        for model_name, result in self.results.items():
            model_type = result['type']
            if model_type not in type_champions or result['test_acc'] > type_champions[model_type][1]['test_acc']:
                type_champions[model_type] = (model_name, result)
        
        print(f"\n🏆 CHAMPIONS BY TYPE:")
        for model_type, (name, result) in type_champions.items():
            print(f"   {model_type:<25}: {name} ({result['test_acc']:.4f})")
        
        return overall_best, type_champions
    
    def generate_insights(self):
        """Generate insights and recommendations"""
        print("\n" + "="*80)
        print("💡 INSIGHTS AND RECOMMENDATIONS")
        print("="*80)
        
        if not self.results:
            return
        
        accuracies = [r['test_acc'] for r in self.results.values()]
        
        print(f"\n📊 Statistical Summary:")
        print(f"   Models evaluated: {len(self.results)}")
        print(f"   Best accuracy: {max(accuracies):.4f} ({max(accuracies)*100:.2f}%)")
        print(f"   Average accuracy: {np.mean(accuracies):.4f} ({np.mean(accuracies)*100:.2f}%)")
        print(f"   Worst accuracy: {min(accuracies):.4f} ({min(accuracies)*100:.2f}%)")
        print(f"   Standard deviation: {np.std(accuracies):.4f}")
        
        # Performance distribution
        excellent = sum(1 for acc in accuracies if acc >= 0.7)
        good = sum(1 for acc in accuracies if 0.5 <= acc < 0.7)
        fair = sum(1 for acc in accuracies if 0.3 <= acc < 0.5)
        poor = sum(1 for acc in accuracies if 0.15 <= acc < 0.3)
        failed = sum(1 for acc in accuracies if acc < 0.15)
        
        print(f"\n🎯 Performance Distribution:")
        print(f"   Excellent (≥70%): {excellent} models")
        print(f"   Good (50-70%): {good} models")
        print(f"   Fair (30-50%): {fair} models")
        print(f"   Poor (15-30%): {poor} models")
        print(f"   Failed (<15%): {failed} models")
        
        # Recommendations
        print(f"\n🚀 Recommendations:")
        
        best_acc = max(accuracies)
        if best_acc >= 0.8:
            print("   • Excellent results achieved! Consider deployment.")
        elif best_acc >= 0.6:
            print("   • Good results. Consider ensemble methods or data augmentation.")
        elif best_acc >= 0.4:
            print("   • Moderate results. Try more advanced architectures or hypertuning.")
        else:
            print("   • Poor results. Review data quality and model architectures.")
        
        # Best approach analysis
        type_performance = {}
        for result in self.results.values():
            model_type = result['type']
            if model_type not in type_performance:
                type_performance[model_type] = []
            type_performance[model_type].append(result['test_acc'])
        
        if type_performance:
            best_type = max(type_performance.items(), key=lambda x: max(x[1]))
            print(f"   • Best performing approach: {best_type[0]} (max: {max(best_type[1]):.4f})")
    
    def save_comparison_report(self, filepath='results/comprehensive_comparison_report.md'):
        """Save a comprehensive comparison report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Comprehensive Emotion Recognition Model Comparison

**Generated:** {timestamp}
**Total Models Evaluated:** {len(self.results)}

## Executive Summary

This report presents a comprehensive comparison of all emotion recognition models evaluated in this project.

"""
        
        if self.results:
            # Get best model
            best_model = max(self.results.items(), key=lambda x: x[1]['test_acc'])
            
            report += f"""### 🏆 Best Performing Model
- **Model:** {best_model[0]}
- **Type:** {best_model[1]['type']}
- **Test Accuracy:** {best_model[1]['test_acc']:.4f} ({best_model[1]['test_acc']*100:.2f}%)
- **Validation Accuracy:** {best_model[1]['val_acc']:.4f}

"""
            
            # Performance table
            sorted_results = sorted(self.results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
            
            report += """## Performance Summary

| Rank | Model | Type | Val Acc | Test Acc | Performance |
|------|-------|------|---------|----------|-------------|
"""
            
            for rank, (model_name, result) in enumerate(sorted_results, 1):
                name_short = model_name.replace('Custom-', '').replace('Pretrained-', '').replace('Classical-', '')
                perf_level = "Excellent" if result['test_acc'] >= 0.7 else "Good" if result['test_acc'] >= 0.5 else "Fair" if result['test_acc'] >= 0.3 else "Poor"
                report += f"| {rank} | {name_short} | {result['type']} | {result['val_acc']:.4f} | {result['test_acc']:.4f} | {perf_level} |\n"
        
        report += f"""
## Key Findings

1. **Dataset**: 11,682+ audio files across 8 emotion classes
2. **Best Performance**: {max([r['test_acc'] for r in self.results.values()])*100:.2f}% accuracy
3. **Baseline**: Random performance = 12.5%

## Technical Implementation

- **Framework**: PyTorch for deep learning models
- **Audio Processing**: 16kHz sampling, 3-second clips
- **Feature Extraction**: Mel-spectrograms, raw audio, engineered features
- **Training**: Early stopping, gradient clipping, learning rate scheduling

---
*Report generated automatically by the emotion recognition evaluation system.*
"""
        
        # Save report
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📝 Comprehensive report saved: {filepath}")
    
    def run_full_comparison(self, results_dir='results'):
        """Run complete model comparison analysis"""
        print("🔍 Running comprehensive model comparison...")
        
        # Load results
        self.load_results(results_dir)
        
        if not self.results:
            print("❌ No results found to compare!")
            return
        
        print(f"\n✅ Total models found: {len(self.results)}")
        
        # Create comparison table
        sorted_results = self.create_comparison_table()
        
        # Analyze by type
        self.analyze_by_type()
        
        # Find champions
        champions = self.find_champions()
        
        # Generate insights
        self.generate_insights()
        
        # Save report
        self.save_comparison_report()
        
        print(f"\n🎉 Comprehensive comparison completed!")
        
        return {
            'results': self.results,
            'sorted_results': sorted_results,
            'champions': champions
        }

