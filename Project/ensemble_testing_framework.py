#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Ensemble Audio Emotion Analysis
Tests different parameter configurations and provides detailed performance analysis
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.models.ensemble_model import EnsembleAudioSentimentAnalyzer
from src.utils.config import Config

class EnsembleTestingFramework:
    """Comprehensive testing framework for ensemble model optimization"""
    
    def __init__(self, results_dir: str = "results_ensemble"):
        """
        Initialize testing framework
        Args:
            results_dir: Directory to store all results
        """
        self.results_dir = Path(results_dir)
        self.create_directory_structure()
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_session_dir = self.results_dir / f"test_session_{self.timestamp}"
        self.test_session_dir.mkdir(exist_ok=True)
        
        # Test configurations to evaluate
        self.test_configs = [
            # Balanced approaches
            {"vocal_weight": 0.5, "text_weight": 0.5, "whisper_size": "tiny", "conf_threshold": 0.7, "name": "balanced_tiny"},
            {"vocal_weight": 0.5, "text_weight": 0.5, "whisper_size": "base", "conf_threshold": 0.7, "name": "balanced_base"},
            
            # Moderate vocal bias (recommended)
            {"vocal_weight": 0.6, "text_weight": 0.4, "whisper_size": "tiny", "conf_threshold": 0.7, "name": "moderate_vocal_tiny"},
            {"vocal_weight": 0.65, "text_weight": 0.35, "whisper_size": "base", "conf_threshold": 0.7, "name": "moderate_vocal_base"},
            {"vocal_weight": 0.7, "text_weight": 0.3, "whisper_size": "base", "conf_threshold": 0.7, "name": "moderate_vocal_base_70"},
            
            # Strong vocal bias
            {"vocal_weight": 0.75, "text_weight": 0.25, "whisper_size": "base", "conf_threshold": 0.7, "name": "strong_vocal_base"},
            {"vocal_weight": 0.8, "text_weight": 0.2, "whisper_size": "base", "conf_threshold": 0.75, "name": "strong_vocal_high_thresh"},
            {"vocal_weight": 0.85, "text_weight": 0.15, "whisper_size": "small", "conf_threshold": 0.8, "name": "very_strong_vocal"},
            
            # Adaptive configurations
            {"vocal_weight": 0.65, "text_weight": 0.35, "whisper_size": "base", "conf_threshold": 0.6, "name": "adaptive_low_thresh"},
            {"vocal_weight": 0.65, "text_weight": 0.35, "whisper_size": "base", "conf_threshold": 0.8, "name": "adaptive_high_thresh"},
        ]
        
        print(f"🧪 Ensemble Testing Framework Initialized")
        print(f"📁 Results directory: {self.test_session_dir}")
        print(f"🔬 Test configurations: {len(self.test_configs)}")
    
    def create_directory_structure(self):
        """Create organized directory structure for results"""
        dirs = [
            self.results_dir,
            self.results_dir / "configurations",
            self.results_dir / "performance_metrics", 
            self.results_dir / "predictions",
            self.results_dir / "visualizations",
            self.results_dir / "reports",
            self.results_dir / "comparison_analysis"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, data_dir: str = "organized_by_emotion", 
                    max_files_per_emotion: int = None,
                    test_split: float = 0.3) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Load dataset for testing
        Args:
            data_dir: Directory containing emotion-organized audio files
            max_files_per_emotion: Limit files per emotion (for quick testing)
            test_split: Fraction of files to use for testing
        Returns:
            Tuple of (train_files, train_labels, test_files, test_labels)
        """
        print(f"📁 Loading dataset from: {data_dir}")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        all_files = []
        all_labels = []
        
        # Get emotion directories
        emotion_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        emotion_dirs = sorted(emotion_dirs, key=lambda x: x.name)
        
        print(f"📊 Found emotion categories: {[d.name for d in emotion_dirs]}")
        
        for emotion_dir in emotion_dirs:
            emotion = emotion_dir.name
            
            # Find audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                audio_files.extend(list(emotion_dir.glob(ext)))
            
            if max_files_per_emotion:
                audio_files = audio_files[:max_files_per_emotion]
            
            print(f"  {emotion}: {len(audio_files)} files")
            
            for audio_file in audio_files:
                all_files.append(str(audio_file))
                all_labels.append(emotion)
        
        print(f"📈 Total dataset: {len(all_files)} files across {len(emotion_dirs)} emotions")
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        
        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, 
            test_size=test_split, 
            random_state=42, 
            stratify=all_labels
        )
        
        print(f"📚 Training set: {len(train_files)} files")
        print(f"🧪 Test set: {len(test_files)} files")
        
        return train_files, train_labels, test_files, test_labels
    
    def test_configuration(self, config: Dict, test_files: List[str], 
                          test_labels: List[str], max_test_files: int = None) -> Dict:
        """
        Test a specific ensemble configuration
        Args:
            config: Configuration dictionary
            test_files: List of test file paths
            test_labels: List of ground truth labels
            max_test_files: Limit for quick testing
        Returns:
            Results dictionary
        """
        config_name = config['name']
        print(f"\n🔧 Testing configuration: {config_name}")
        print(f"   Vocal weight: {config['vocal_weight']}")
        print(f"   Text weight: {config['text_weight']}")
        print(f"   Whisper model: {config['whisper_size']}")
        print(f"   Confidence threshold: {config['conf_threshold']}")
        
        # Limit test files for quick testing if specified
        if max_test_files and len(test_files) > max_test_files:
            indices = np.random.choice(len(test_files), max_test_files, replace=False)
            test_files_subset = [test_files[i] for i in indices]
            test_labels_subset = [test_labels[i] for i in indices]
        else:
            test_files_subset = test_files
            test_labels_subset = test_labels
        
        # Create ensemble with configuration
        try:
            ensemble = EnsembleAudioSentimentAnalyzer(
                resnet_model_path="models/resnet/best_ResNet.pth",
                whisper_model_size=config['whisper_size'],
                vocal_weight=config['vocal_weight'],
                text_weight=config['text_weight'],
                confidence_threshold=config['conf_threshold']
            )
        except Exception as e:
            print(f"❌ Error creating ensemble: {e}")
            return {"error": str(e), "config": config}
        
        # Run predictions
        predictions = []
        detailed_results = []
        failed_files = []
        
        print(f"🔄 Running predictions on {len(test_files_subset)} files...")
        
        for i, (file_path, true_label) in enumerate(tqdm(zip(test_files_subset, test_labels_subset))):
            try:
                result = ensemble.predict_emotion(file_path, return_details=True)
                
                predictions.append(result['predicted_emotion'])
                detailed_results.append({
                    'file': Path(file_path).name,
                    'true_label': true_label,
                    'predicted_label': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'transcribed_text': result['transcribed_text'],
                    'vocal_prediction': result['component_predictions']['vocal'],
                    'text_prediction': result['component_predictions']['text'],
                    'weights_used': result['weights_used']
                })
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                failed_files.append(file_path)
                predictions.append('neutral')  # Default prediction
                
        # Calculate metrics
        accuracy = accuracy_score(test_labels_subset, predictions)
        f1_weighted = f1_score(test_labels_subset, predictions, average='weighted')
        f1_macro = f1_score(test_labels_subset, predictions, average='macro')
        precision_weighted = precision_score(test_labels_subset, predictions, average='weighted')
        recall_weighted = recall_score(test_labels_subset, predictions, average='weighted')
        
        # Classification report
        class_report = classification_report(
            test_labels_subset, predictions, 
            target_names=Config.EMOTION_CLASSES,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels_subset, predictions, labels=Config.EMOTION_CLASSES)
        
        results = {
            'config': config,
            'metrics': {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted
            },
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'detailed_predictions': detailed_results,
            'failed_files': failed_files,
            'test_files_count': len(test_files_subset),
            'failed_count': len(failed_files)
        }
        
        print(f"✅ Configuration {config_name} completed:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1 (weighted): {f1_weighted:.3f}")
        print(f"   F1 (macro): {f1_macro:.3f}")
        print(f"   Failed files: {len(failed_files)}")
        
        return results
    
    def run_comprehensive_testing(self, data_dir: str = "organized_by_emotion",
                                 max_files_per_emotion: int = 20,
                                 max_test_files: int = 100) -> Dict:
        """
        Run comprehensive testing on all configurations
        Args:
            data_dir: Dataset directory
            max_files_per_emotion: Limit per emotion for quick testing
            max_test_files: Total test files limit
        Returns:
            Complete results dictionary
        """
        print("🚀 Starting Comprehensive Ensemble Testing")
        print("="*70)
        
        # Load dataset
        train_files, train_labels, test_files, test_labels = self.load_dataset(
            data_dir, max_files_per_emotion
        )
        
        # Store dataset info
        dataset_info = {
            'total_files': len(train_files) + len(test_files),
            'train_files': len(train_files),
            'test_files': len(test_files),
            'emotions': list(set(test_labels)),
            'emotion_distribution': {emotion: test_labels.count(emotion) for emotion in set(test_labels)}
        }
        
        print(f"\n📊 Dataset Summary:")
        for emotion, count in dataset_info['emotion_distribution'].items():
            print(f"   {emotion}: {count} files")
        
        # Test all configurations
        all_results = {
            'dataset_info': dataset_info,
            'test_timestamp': self.timestamp,
            'configurations': {}
        }
        
        for config in self.test_configs:
            try:
                config_results = self.test_configuration(config, test_files, test_labels, max_test_files)
                all_results['configurations'][config['name']] = config_results
                
                # Save individual config results
                config_file = self.test_session_dir / f"config_{config['name']}.json"
                with open(config_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_results = self._make_json_serializable(config_results)
                    json.dump(serializable_results, f, indent=2)
                    
            except Exception as e:
                print(f"❌ Failed to test configuration {config['name']}: {e}")
                all_results['configurations'][config['name']] = {"error": str(e)}
        
        # Save complete results
        results_file = self.test_session_dir / "complete_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n✅ Comprehensive testing completed!")
        print(f"📁 Results saved to: {self.test_session_dir}")
        
        return all_results
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze and compare results across configurations
        Args:
            results: Complete results dictionary
        Returns:
            Analysis summary
        """
        print("\n📊 Analyzing Results...")
        
        config_names = []
        accuracies = []
        f1_scores = []
        vocal_weights = []
        text_weights = []
        whisper_models = []
        
        for config_name, config_results in results['configurations'].items():
            if 'error' in config_results:
                continue
                
            config_names.append(config_name)
            accuracies.append(config_results['metrics']['accuracy'])
            f1_scores.append(config_results['metrics']['f1_weighted'])
            vocal_weights.append(config_results['config']['vocal_weight'])
            text_weights.append(config_results['config']['text_weight'])
            whisper_models.append(config_results['config']['whisper_size'])
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Configuration': config_names,
            'Accuracy': accuracies,
            'F1_Weighted': f1_scores,
            'Vocal_Weight': vocal_weights,
            'Text_Weight': text_weights,
            'Whisper_Model': whisper_models
        })
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Find best configurations
        best_accuracy = comparison_df.iloc[0]
        best_f1 = comparison_df.loc[comparison_df['F1_Weighted'].idxmax()]
        
        analysis = {
            'comparison_dataframe': comparison_df,
            'best_accuracy_config': {
                'name': best_accuracy['Configuration'],
                'accuracy': best_accuracy['Accuracy'],
                'f1_score': best_accuracy['F1_Weighted'],
                'vocal_weight': best_accuracy['Vocal_Weight']
            },
            'best_f1_config': {
                'name': best_f1['Configuration'],
                'accuracy': best_f1['Accuracy'],
                'f1_score': best_f1['F1_Weighted'],
                'vocal_weight': best_f1['Vocal_Weight']
            },
            'performance_summary': {
                'mean_accuracy': comparison_df['Accuracy'].mean(),
                'std_accuracy': comparison_df['Accuracy'].std(),
                'mean_f1': comparison_df['F1_Weighted'].mean(),
                'std_f1': comparison_df['F1_Weighted'].std()
            }
        }
        
        print(f"🏆 Best Accuracy: {best_accuracy['Configuration']} ({best_accuracy['Accuracy']:.3f})")
        print(f"🎯 Best F1-Score: {best_f1['Configuration']} ({best_f1['F1_Weighted']:.3f})")
        
        return analysis
    
    def create_visualizations(self, results: Dict, analysis: Dict):
        """Create comprehensive visualizations"""
        print("\n📈 Creating Visualizations...")
        
        viz_dir = self.test_session_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        df = analysis['comparison_dataframe']
        
        # Accuracy comparison
        axes[0, 0].bar(range(len(df)), df['Accuracy'], color='skyblue')
        axes[0, 0].set_title('Accuracy by Configuration')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(df)))
        axes[0, 0].set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # F1 Score comparison
        axes[0, 1].bar(range(len(df)), df['F1_Weighted'], color='lightgreen')
        axes[0, 1].set_title('F1-Score by Configuration')
        axes[0, 1].set_ylabel('F1-Score (Weighted)')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # Vocal weight vs Performance
        axes[1, 0].scatter(df['Vocal_Weight'], df['Accuracy'], c='red', alpha=0.7, s=100)
        axes[1, 0].set_title('Vocal Weight vs Accuracy')
        axes[1, 0].set_xlabel('Vocal Weight')
        axes[1, 0].set_ylabel('Accuracy')
        
        # Whisper model comparison
        whisper_performance = df.groupby('Whisper_Model')['Accuracy'].mean()
        axes[1, 1].bar(whisper_performance.index, whisper_performance.values, color='orange')
        axes[1, 1].set_title('Average Accuracy by Whisper Model')
        axes[1, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix for Best Model
        best_config_name = analysis['best_accuracy_config']['name']
        best_results = results['configurations'][best_config_name]
        
        if 'confusion_matrix' in best_results:
            plt.figure(figsize=(10, 8))
            cm = np.array(best_results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=Config.EMOTION_CLASSES,
                       yticklabels=Config.EMOTION_CLASSES)
            plt.title(f'Confusion Matrix - Best Model ({best_config_name})')
            plt.ylabel('True Emotion')
            plt.xlabel('Predicted Emotion')
            plt.tight_layout()
            plt.savefig(viz_dir / f"confusion_matrix_best_{best_config_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Performance Distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(df['Accuracy'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Accuracy Scores')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.axvline(df['Accuracy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Accuracy"].mean():.3f}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(df['F1_Weighted'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Distribution of F1-Scores')
        plt.xlabel('F1-Score (Weighted)')
        plt.ylabel('Frequency')
        plt.axvline(df['F1_Weighted'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["F1_Weighted"].mean():.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {viz_dir}")
    
    def generate_report(self, results: Dict, analysis: Dict):
        """Generate comprehensive text report"""
        print("\n📝 Generating Report...")
        
        report_file = self.test_session_dir / "ensemble_testing_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Ensemble Audio Emotion Analysis - Testing Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Information
            f.write("## Dataset Information\n\n")
            dataset_info = results['dataset_info']
            f.write(f"- **Total Files:** {dataset_info['total_files']}\n")
            f.write(f"- **Training Files:** {dataset_info['train_files']}\n")
            f.write(f"- **Test Files:** {dataset_info['test_files']}\n")
            f.write(f"- **Emotion Classes:** {len(dataset_info['emotions'])}\n\n")
            
            f.write("### Emotion Distribution\n")
            for emotion, count in dataset_info['emotion_distribution'].items():
                f.write(f"- **{emotion}:** {count} files\n")
            f.write("\n")
            
            # Performance Summary
            f.write("## Performance Summary\n\n")
            perf = analysis['performance_summary']
            f.write(f"- **Mean Accuracy:** {perf['mean_accuracy']:.3f} ± {perf['std_accuracy']:.3f}\n")
            f.write(f"- **Mean F1-Score:** {perf['mean_f1']:.3f} ± {perf['std_f1']:.3f}\n\n")
            
            # Best Configurations
            f.write("## Best Configurations\n\n")
            best_acc = analysis['best_accuracy_config']
            best_f1 = analysis['best_f1_config']
            
            f.write("### Best Accuracy\n")
            f.write(f"- **Configuration:** {best_acc['name']}\n")
            f.write(f"- **Accuracy:** {best_acc['accuracy']:.3f}\n")
            f.write(f"- **F1-Score:** {best_acc['f1_score']:.3f}\n")
            f.write(f"- **Vocal Weight:** {best_acc['vocal_weight']}\n\n")
            
            f.write("### Best F1-Score\n")
            f.write(f"- **Configuration:** {best_f1['name']}\n")
            f.write(f"- **Accuracy:** {best_f1['accuracy']:.3f}\n")
            f.write(f"- **F1-Score:** {best_f1['f1_score']:.3f}\n")
            f.write(f"- **Vocal Weight:** {best_f1['vocal_weight']}\n\n")
            
            # Detailed Results Table
            f.write("## Detailed Results\n\n")
            f.write("| Configuration | Accuracy | F1-Score | Vocal Weight | Text Weight | Whisper Model |\n")
            f.write("|---------------|----------|----------|--------------|-------------|---------------|\n")
            
            df = analysis['comparison_dataframe']
            for _, row in df.iterrows():
                f.write(f"| {row['Configuration']} | {row['Accuracy']:.3f} | "
                       f"{row['F1_Weighted']:.3f} | {row['Vocal_Weight']:.2f} | "
                       f"{row['Text_Weight']:.2f} | {row['Whisper_Model']} |\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the testing results:\n\n")
            
            if best_acc['vocal_weight'] > 0.7:
                f.write("1. **High vocal bias** configurations perform best, suggesting vocal cues are more reliable than text for this dataset.\n")
            elif best_acc['vocal_weight'] < 0.6:
                f.write("1. **Balanced or text-biased** configurations perform best, suggesting clear emotional text in this dataset.\n")
            else:
                f.write("1. **Moderate vocal bias** configurations perform best, providing good balance between vocal and text cues.\n")
            
            f.write(f"2. **Recommended configuration:** {best_acc['name']} with {best_acc['accuracy']:.1%} accuracy.\n")
            f.write("3. **Use cases:** This configuration is suitable for detecting vocal-text emotion mismatches.\n")
            
        print(f"📋 Report saved to: {report_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

def main():
    """Main testing function"""
    print("🧪 ENSEMBLE MODEL TESTING & OPTIMIZATION")
    print("="*70)
    
    # Initialize testing framework
    framework = EnsembleTestingFramework()
    
    try:
        # Run comprehensive testing
        results = framework.run_comprehensive_testing(
            data_dir="organized_by_emotion",
            max_files_per_emotion=15,  # Limit for reasonable testing time
            max_test_files=80         # Total test files limit
        )
        
        # Analyze results
        analysis = framework.analyze_results(results)
        
        # Create visualizations
        framework.create_visualizations(results, analysis)
        
        # Generate report
        framework.generate_report(results, analysis)
        
        print("\n🎉 Testing completed successfully!")
        print(f"📁 All results saved to: {framework.test_session_dir}")
        
        # Print quick summary
        print(f"\n📊 QUICK SUMMARY:")
        best_config = analysis['best_accuracy_config']
        print(f"🏆 Best Configuration: {best_config['name']}")
        print(f"📈 Best Accuracy: {best_config['accuracy']:.3f}")
        print(f"🎯 Vocal Weight: {best_config['vocal_weight']}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
