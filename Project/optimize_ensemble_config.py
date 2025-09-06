#!/usr/bin/env python3
"""
Ensemble Configuration Optimizer
Systematically finds the best ensemble parameters through grid search
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.models.ensemble_model import EnsembleAudioSentimentAnalyzer
from src.utils.config import Config

class EnsembleOptimizer:
    """Optimize ensemble parameters through systematic search"""
    
    def __init__(self, results_dir: str = "optimization_results"):
        """Initialize optimizer"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Parameter search space
        self.param_grid = {
            'vocal_weights': [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            'whisper_models': ['tiny', 'base'],  # Focus on faster models for optimization
            'confidence_thresholds': [0.6, 0.7, 0.75, 0.8]
        }
        
        print(f"🔧 Ensemble Optimizer Initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🔍 Parameter grid size: {self.get_grid_size()} combinations")
    
    def get_grid_size(self) -> int:
        """Calculate total number of parameter combinations"""
        size = 1
        for values in self.param_grid.values():
            size *= len(values)
        return size
    
    def load_sample_dataset(self, data_dir: str = "organized_by_emotion", 
                           samples_per_emotion: int = 5) -> Tuple[List[str], List[str]]:
        """
        Load a small sample dataset for quick optimization
        Args:
            data_dir: Dataset directory
            samples_per_emotion: Number of samples per emotion
        Returns:
            Tuple of (file_paths, labels)
        """
        print(f"📁 Loading sample dataset: {samples_per_emotion} files per emotion")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        files = []
        labels = []
        
        emotion_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        emotion_dirs = sorted(emotion_dirs)
        
        for emotion_dir in emotion_dirs:
            emotion = emotion_dir.name
            
            # Find audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(list(emotion_dir.glob(ext)))
            
            # Sample files
            if len(audio_files) > samples_per_emotion:
                audio_files = np.random.choice(audio_files, samples_per_emotion, replace=False)
            
            for audio_file in audio_files:
                files.append(str(audio_file))
                labels.append(emotion)
        
        print(f"📊 Sample dataset: {len(files)} files, {len(set(labels))} emotions")
        return files, labels
    
    def evaluate_configuration(self, vocal_weight: float, whisper_model: str, 
                              confidence_threshold: float, test_files: List[str], 
                              test_labels: List[str]) -> Dict:
        """
        Evaluate a single configuration
        Args:
            vocal_weight: Vocal bias weight
            whisper_model: Whisper model size
            confidence_threshold: Confidence threshold for adaptive weighting
            test_files: Test file paths
            test_labels: True labels
        Returns:
            Evaluation results
        """
        text_weight = 1.0 - vocal_weight
        
        try:
            # Create ensemble
            ensemble = EnsembleAudioSentimentAnalyzer(
                resnet_model_path="models/resnet/best_ResNet.pth",
                whisper_model_size=whisper_model,
                vocal_weight=vocal_weight,
                text_weight=text_weight,
                confidence_threshold=confidence_threshold
            )
            
            # Make predictions
            predictions = []
            confidences = []
            vocal_confidences = []
            text_confidences = []
            failed_count = 0
            
            for file_path, true_label in zip(test_files, test_labels):
                try:
                    result = ensemble.predict_emotion(file_path, return_details=True)
                    predictions.append(result['predicted_emotion'])
                    confidences.append(result['confidence'])
                    
                    if 'component_predictions' in result:
                        vocal_confidences.append(result['component_predictions']['vocal']['confidence'])
                        text_confidences.append(result['component_predictions']['text']['confidence'])
                    
                except Exception:
                    predictions.append('neutral')  # Default on error
                    confidences.append(0.0)
                    vocal_confidences.append(0.0)
                    text_confidences.append(0.0)
                    failed_count += 1
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            f1_weighted = f1_score(test_labels, predictions, average='weighted')
            f1_macro = f1_score(test_labels, predictions, average='macro')
            
            # Additional metrics
            mean_confidence = np.mean(confidences)
            mean_vocal_confidence = np.mean(vocal_confidences) if vocal_confidences else 0.0
            mean_text_confidence = np.mean(text_confidences) if text_confidences else 0.0
            
            return {
                'vocal_weight': vocal_weight,
                'text_weight': text_weight,
                'whisper_model': whisper_model,
                'confidence_threshold': confidence_threshold,
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'mean_confidence': mean_confidence,
                'mean_vocal_confidence': mean_vocal_confidence,
                'mean_text_confidence': mean_text_confidence,
                'failed_count': failed_count,
                'total_files': len(test_files),
                'success_rate': (len(test_files) - failed_count) / len(test_files)
            }
            
        except Exception as e:
            return {
                'vocal_weight': vocal_weight,
                'text_weight': text_weight,
                'whisper_model': whisper_model,
                'confidence_threshold': confidence_threshold,
                'error': str(e),
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0
            }
    
    def grid_search(self, data_dir: str = "organized_by_emotion", 
                   samples_per_emotion: int = 5) -> pd.DataFrame:
        """
        Perform grid search optimization
        Args:
            data_dir: Dataset directory
            samples_per_emotion: Samples per emotion for testing
        Returns:
            Results DataFrame
        """
        print("🔍 Starting Grid Search Optimization")
        print("="*50)
        
        # Load sample dataset
        test_files, test_labels = self.load_sample_dataset(data_dir, samples_per_emotion)
        
        # Generate all parameter combinations
        combinations = list(product(
            self.param_grid['vocal_weights'],
            self.param_grid['whisper_models'],
            self.param_grid['confidence_thresholds']
        ))
        
        print(f"🧪 Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        for vocal_weight, whisper_model, conf_threshold in tqdm(combinations):
            result = self.evaluate_configuration(
                vocal_weight, whisper_model, conf_threshold, test_files, test_labels
            )
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by accuracy (primary) and F1-score (secondary)
        df = df.sort_values(['accuracy', 'f1_weighted'], ascending=False)
        
        # Save results
        results_file = self.results_dir / "grid_search_results.csv"
        df.to_csv(results_file, index=False)
        
        print(f"✅ Grid search completed!")
        print(f"📁 Results saved to: {results_file}")
        
        return df
    
    def find_optimal_configurations(self, df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Find optimal configurations from grid search results
        Args:
            df: Results DataFrame
            top_n: Number of top configurations to return
        Returns:
            Dictionary with optimal configurations
        """
        print(f"\n🏆 Finding Top {top_n} Configurations")
        print("="*40)
        
        # Filter out failed configurations
        valid_df = df[df['accuracy'] > 0].copy()
        
        if len(valid_df) == 0:
            print("❌ No valid configurations found!")
            return {}
        
        # Top configurations by accuracy
        top_accuracy = valid_df.head(top_n)
        
        # Top configurations by F1-score
        top_f1 = valid_df.nlargest(top_n, 'f1_weighted')
        
        # Best balanced configuration (high accuracy + high F1)
        valid_df['balanced_score'] = (valid_df['accuracy'] + valid_df['f1_weighted']) / 2
        top_balanced = valid_df.nlargest(top_n, 'balanced_score')
        
        # Analyze vocal weight trends
        vocal_weight_performance = valid_df.groupby('vocal_weight').agg({
            'accuracy': 'mean',
            'f1_weighted': 'mean'
        }).sort_values('accuracy', ascending=False)
        
        optimal_configs = {
            'top_accuracy': top_accuracy.to_dict('records'),
            'top_f1': top_f1.to_dict('records'),
            'top_balanced': top_balanced.to_dict('records'),
            'vocal_weight_analysis': vocal_weight_performance.to_dict('index'),
            'summary': {
                'best_vocal_weight': vocal_weight_performance.index[0],
                'best_accuracy': valid_df['accuracy'].max(),
                'best_f1': valid_df['f1_weighted'].max(),
                'mean_accuracy': valid_df['accuracy'].mean(),
                'std_accuracy': valid_df['accuracy'].std()
            }
        }
        
        # Print summary
        best_config = valid_df.iloc[0]
        print(f"🥇 Best Configuration:")
        print(f"   Vocal Weight: {best_config['vocal_weight']}")
        print(f"   Whisper Model: {best_config['whisper_model']}")
        print(f"   Confidence Threshold: {best_config['confidence_threshold']}")
        print(f"   Accuracy: {best_config['accuracy']:.3f}")
        print(f"   F1-Score: {best_config['f1_weighted']:.3f}")
        
        print(f"\n📊 Vocal Weight Analysis:")
        for vocal_weight, metrics in vocal_weight_performance.head(3).iterrows():
            print(f"   {vocal_weight}: Accuracy {metrics['accuracy']:.3f}, F1 {metrics['f1_weighted']:.3f}")
        
        return optimal_configs
    
    def create_optimization_report(self, df: pd.DataFrame, optimal_configs: Dict):
        """Create detailed optimization report"""
        print(f"\n📝 Creating Optimization Report...")
        
        report_file = self.results_dir / "optimization_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Ensemble Model Optimization Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            summary = optimal_configs['summary']
            f.write("## Optimization Summary\n\n")
            f.write(f"- **Configurations Tested:** {len(df)}\n")
            f.write(f"- **Best Accuracy:** {summary['best_accuracy']:.3f}\n")
            f.write(f"- **Best F1-Score:** {summary['best_f1']:.3f}\n")
            f.write(f"- **Optimal Vocal Weight:** {summary['best_vocal_weight']}\n")
            f.write(f"- **Mean Accuracy:** {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}\n\n")
            
            # Best configurations
            f.write("## Top Configurations\n\n")
            f.write("### By Accuracy\n")
            f.write("| Rank | Vocal Weight | Whisper | Conf. Threshold | Accuracy | F1-Score |\n")
            f.write("|------|--------------|---------|-----------------|----------|----------|\n")
            
            for i, config in enumerate(optimal_configs['top_accuracy'][:5], 1):
                f.write(f"| {i} | {config['vocal_weight']:.2f} | {config['whisper_model']} | "
                       f"{config['confidence_threshold']:.2f} | {config['accuracy']:.3f} | "
                       f"{config['f1_weighted']:.3f} |\n")
            
            f.write("\n### By F1-Score\n")
            f.write("| Rank | Vocal Weight | Whisper | Conf. Threshold | Accuracy | F1-Score |\n")
            f.write("|------|--------------|---------|-----------------|----------|----------|\n")
            
            for i, config in enumerate(optimal_configs['top_f1'][:5], 1):
                f.write(f"| {i} | {config['vocal_weight']:.2f} | {config['whisper_model']} | "
                       f"{config['confidence_threshold']:.2f} | {config['accuracy']:.3f} | "
                       f"{config['f1_weighted']:.3f} |\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            best_vocal_weight = summary['best_vocal_weight']
            
            if best_vocal_weight >= 0.75:
                f.write("1. **High vocal bias** (>=0.75) performs best - vocal cues are very reliable\n")
                f.write("2. Recommended for detecting emotional deception or sarcasm\n")
                f.write("3. Text analysis is less reliable for this dataset\n")
            elif best_vocal_weight >= 0.65:
                f.write("1. **Moderate vocal bias** (0.65-0.74) performs best - balanced approach\n")
                f.write("2. Recommended for general emotion recognition tasks\n")
                f.write("3. Good balance between vocal and text cues\n")
            else:
                f.write("1. **Balanced approach** (<0.65) performs best - text is reliable\n")
                f.write("2. Recommended when text clearly expresses emotions\n")
                f.write("3. Vocal and text cues are equally important\n")
            
            f.write(f"\n4. **Optimal configuration for deployment:**\n")
            best_config = optimal_configs['top_balanced'][0]
            f.write(f"   - Vocal Weight: {best_config['vocal_weight']}\n")
            f.write(f"   - Whisper Model: {best_config['whisper_model']}\n")
            f.write(f"   - Confidence Threshold: {best_config['confidence_threshold']}\n")
        
        print(f"📋 Optimization report saved to: {report_file}")

def main():
    """Main optimization function"""
    print("🔧 ENSEMBLE MODEL OPTIMIZATION")
    print("="*50)
    
    optimizer = EnsembleOptimizer()
    
    try:
        # Run grid search
        results_df = optimizer.grid_search(
            data_dir="organized_by_emotion",
            samples_per_emotion=3  # Small sample for quick optimization
        )
        
        # Find optimal configurations
        optimal_configs = optimizer.find_optimal_configurations(results_df)
        
        # Create report
        optimizer.create_optimization_report(results_df, optimal_configs)
        
        # Save optimal configs
        config_file = optimizer.results_dir / "optimal_configurations.json"
        with open(config_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_configs = {}
            for key, value in optimal_configs.items():
                if isinstance(value, dict):
                    json_configs[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                        for k, v in value.items()}
                else:
                    json_configs[key] = value
            json.dump(json_configs, f, indent=2, default=str)
        
        print(f"\n✅ Optimization completed successfully!")
        print(f"📁 Results directory: {optimizer.results_dir}")
        
        # Return best configuration for immediate use
        best_config = optimal_configs['top_balanced'][0]
        print(f"\n🚀 RECOMMENDED CONFIGURATION:")
        print(f"   vocal_weight={best_config['vocal_weight']}")
        print(f"   whisper_model='{best_config['whisper_model']}'")
        print(f"   confidence_threshold={best_config['confidence_threshold']}")
        print(f"   Expected accuracy: {best_config['accuracy']:.3f}")
        
        return best_config
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
