#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Ensemble vs ResNet
Tests both models on the complete dataset and generates detailed comparisons
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.models.ensemble_model import EnsembleAudioSentimentAnalyzer
    from src.models.custom_models import EmotionResNet
    from src.utils.config import Config
    import torch
    import torch.nn.functional as F
    import librosa
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the Project directory")
    sys.exit(1)


class ModelComparator:
    """Comprehensive model comparison framework"""
    
    def __init__(self, output_dir: str = None):
        """Initialize comparison framework"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            output_dir = f"comparison_results_{self.timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "predictions").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Initialize models
        self.ensemble_model = None
        self.resnet_model = None
        self.device = Config.DEVICE
        
        print(f"Model Comparator initialized. Output: {self.output_dir}")
    
    def load_models(self):
        """Load both ensemble and ResNet models"""
        print("\n🔧 Loading Models...")
        
        # Load Ensemble Model
        try:
            self.ensemble_model = EnsembleAudioSentimentAnalyzer(
                resnet_model_path="models/resnet/best_ResNet.pth",
                whisper_model_size="base",
                vocal_weight=0.9,  # High vocal bias as requested
                text_weight=0.1,
                confidence_threshold=0.7
            )
            print("✓ Ensemble model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load ensemble model: {e}")
            return False
        
        # Load standalone ResNet model
        try:
            self.resnet_model = EmotionResNet(num_classes=Config.NUM_CLASSES)
            
            if Path("models/resnet/best_ResNet.pth").exists():
                checkpoint = torch.load("models/resnet/best_ResNet.pth", map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.resnet_model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.resnet_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.resnet_model.load_state_dict(checkpoint)
                else:
                    self.resnet_model.load_state_dict(checkpoint)
                
                self.resnet_model = self.resnet_model.to(self.device)
                self.resnet_model.eval()
                print("✓ ResNet model loaded successfully")
            else:
                print("❌ ResNet model file not found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load ResNet model: {e}")
            return False
        
        return True
    
    def get_complete_dataset(self, data_dir: str = "organized_by_emotion") -> List[Tuple[str, str]]:
        """
        Get complete dataset for testing
        Args:
            data_dir: Directory containing emotion subdirectories
        Returns:
            List of (file_path, true_emotion) tuples
        """
        test_files = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Warning: Data directory {data_path} not found")
            return []
        
        emotions = [d.name for d in data_path.iterdir() if d.is_dir()]
        print(f"Found emotion directories: {emotions}")
        
        total_files = 0
        for emotion in emotions:
            emotion_dir = data_path / emotion
            audio_files = []
            
            # Find all audio files
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(list(emotion_dir.glob(ext)))
            
            for audio_file in audio_files:
                test_files.append((str(audio_file), emotion))
            
            print(f"  {emotion}: {len(audio_files)} files")
            total_files += len(audio_files)
        
        print(f"Total dataset: {total_files} files")
        return test_files
    
    def predict_resnet_only(self, audio_path: str) -> Dict:
        """
        Predict emotion using ResNet only
        Args:
            audio_path: Path to audio file
        Returns:
            Prediction dictionary
        """
        try:
            # Load and preprocess audio (same as in ensemble)
            y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, duration=Config.DURATION)
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=Config.N_MELS, 
                fmax=Config.FMAX,
                n_fft=Config.N_FFT, 
                hop_length=Config.HOP_LENGTH
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Convert to tensor and add batch dimension
            mel_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.resnet_model(mel_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            predicted_emotion = Config.EMOTION_CLASSES[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            return {
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob) for emotion, prob 
                    in zip(Config.EMOTION_CLASSES, probabilities)
                }
            }
            
        except Exception as e:
            print(f"ResNet prediction error for {audio_path}: {e}")
            return {
                'predicted_emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def compare_models(self, test_files: List[Tuple[str, str]]) -> Dict:
        """
        Compare both models on the test dataset
        Args:
            test_files: List of (file_path, true_emotion) tuples
        Returns:
            Comparison results dictionary
        """
        print(f"\n🔬 COMPARING MODELS ON COMPLETE DATASET")
        print("=" * 60)
        print(f"Dataset size: {len(test_files)} files")
        print("Models: Ensemble (ResNet+STT+BERT) vs ResNet Standalone")
        print("=" * 60)
        
        # Results storage
        results = {
            'ensemble': {
                'predictions': [],
                'confidences': [],
                'processing_times': [],
                'vocal_predictions': [],
                'text_predictions': [],
                'transcriptions': []
            },
            'resnet': {
                'predictions': [],
                'confidences': [],
                'processing_times': []
            },
            'common': {
                'true_labels': [],
                'file_paths': []
            }
        }
        
        start_time = time.time()
        
        for i, (file_path, true_emotion) in enumerate(test_files):
            print(f"\nProcessing [{i+1}/{len(test_files)}]: {Path(file_path).name}")
            
            # Store common info
            results['common']['true_labels'].append(true_emotion)
            results['common']['file_paths'].append(file_path)
            
            # Test Ensemble Model
            try:
                ens_start = time.time()
                ensemble_result = self.ensemble_model.predict_emotion(file_path, return_details=True)
                ens_time = time.time() - ens_start
                
                results['ensemble']['predictions'].append(ensemble_result['predicted_emotion'])
                results['ensemble']['confidences'].append(ensemble_result['confidence'])
                results['ensemble']['processing_times'].append(ens_time)
                results['ensemble']['transcriptions'].append(ensemble_result['transcribed_text'])
                
                # Extract component predictions if available
                if 'component_predictions' in ensemble_result:
                    vocal_pred = ensemble_result['component_predictions']['vocal']['emotion']
                    text_pred = ensemble_result['component_predictions']['text']['emotion']
                    results['ensemble']['vocal_predictions'].append(vocal_pred)
                    results['ensemble']['text_predictions'].append(text_pred)
                else:
                    results['ensemble']['vocal_predictions'].append('unknown')
                    results['ensemble']['text_predictions'].append('unknown')
                
                print(f"  Ensemble: {ensemble_result['predicted_emotion']} ({ensemble_result['confidence']:.3f}) - {ens_time:.2f}s")
                
            except Exception as e:
                print(f"  Ensemble Error: {e}")
                results['ensemble']['predictions'].append('neutral')
                results['ensemble']['confidences'].append(0.0)
                results['ensemble']['processing_times'].append(0.0)
                results['ensemble']['transcriptions'].append('')
                results['ensemble']['vocal_predictions'].append('unknown')
                results['ensemble']['text_predictions'].append('unknown')
            
            # Test ResNet Model
            try:
                resnet_start = time.time()
                resnet_result = self.predict_resnet_only(file_path)
                resnet_time = time.time() - resnet_start
                
                results['resnet']['predictions'].append(resnet_result['predicted_emotion'])
                results['resnet']['confidences'].append(resnet_result['confidence'])
                results['resnet']['processing_times'].append(resnet_time)
                
                print(f"  ResNet:   {resnet_result['predicted_emotion']} ({resnet_result['confidence']:.3f}) - {resnet_time:.2f}s")
                
            except Exception as e:
                print(f"  ResNet Error: {e}")
                results['resnet']['predictions'].append('neutral')
                results['resnet']['confidences'].append(0.0)
                results['resnet']['processing_times'].append(0.0)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(test_files) - i - 1)
                print(f"\nProgress: {i+1}/{len(test_files)} | Avg: {avg_time:.2f}s/file | ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start_time
        print(f"\n✅ Comparison completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per file: {total_time/len(test_files):.2f}s")
        
        # Save raw results
        results_file = self.output_dir / "predictions" / "comparison_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for model_key, model_data in results.items():
                serializable_results[model_key] = {}
                for data_key, data_values in model_data.items():
                    serializable_results[model_key][data_key] = [
                        float(v) if isinstance(v, np.floating) else v for v in data_values
                    ]
            json.dump(serializable_results, f, indent=2)
        
        print(f"Raw results saved to: {results_file}")
        return results
    
    def calculate_comparison_metrics(self, results: Dict) -> Dict:
        """
        Calculate comparative metrics for both models
        Args:
            results: Comparison results dictionary
        Returns:
            Metrics dictionary
        """
        print(f"\n📊 CALCULATING COMPARISON METRICS")
        print("=" * 50)
        
        true_labels = results['common']['true_labels']
        
        # Calculate metrics for each model
        metrics = {}
        
        for model_name in ['ensemble', 'resnet']:
            predictions = results[model_name]['predictions']
            confidences = results[model_name]['confidences']
            processing_times = results[model_name]['processing_times']
            
            # Overall metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, average='weighted'
            )
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = \
                precision_recall_fscore_support(true_labels, predictions, average=None, labels=Config.EMOTION_CLASSES)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions, labels=Config.EMOTION_CLASSES)
            
            # Classification report
            class_report = classification_report(
                true_labels, predictions, 
                labels=Config.EMOTION_CLASSES, 
                output_dict=True
            )
            
            metrics[model_name] = {
                'overall': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'avg_confidence': np.mean(confidences),
                    'avg_processing_time': np.mean(processing_times),
                    'total_samples': len(true_labels)
                },
                'per_class': {
                    'emotions': Config.EMOTION_CLASSES,
                    'precision': precision_per_class.tolist(),
                    'recall': recall_per_class.tolist(),
                    'f1_score': f1_per_class.tolist(),
                    'support': support_per_class.tolist()
                },
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report
            }
            
            # Print key metrics
            print(f"\n{model_name.upper()} Model:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  Avg Confidence: {np.mean(confidences):.3f}")
            print(f"  Avg Processing Time: {np.mean(processing_times):.3f}s")
        
        # Ensemble-specific analysis
        if results['ensemble']['transcriptions']:
            transcriptions = results['ensemble']['transcriptions']
            successful_transcriptions = sum(1 for t in transcriptions if t.strip())
            transcription_rate = successful_transcriptions / len(transcriptions)
            
            metrics['ensemble']['stt_analysis'] = {
                'transcription_rate': transcription_rate,
                'successful_transcriptions': successful_transcriptions
            }
            
            print(f"\nSTT Analysis:")
            print(f"  Transcription Success Rate: {transcription_rate:.3f}")
        
        # Head-to-head comparison
        ensemble_acc = metrics['ensemble']['overall']['accuracy']
        resnet_acc = metrics['resnet']['overall']['accuracy']
        accuracy_diff = ensemble_acc - resnet_acc
        
        print(f"\n🏆 HEAD-TO-HEAD COMPARISON:")
        print(f"  Ensemble Accuracy: {ensemble_acc:.3f}")
        print(f"  ResNet Accuracy:   {resnet_acc:.3f}")
        print(f"  Difference:        {accuracy_diff:+.3f}")
        
        if accuracy_diff > 0.01:
            print(f"  Winner: ENSEMBLE (+{accuracy_diff:.3f})")
        elif accuracy_diff < -0.01:
            print(f"  Winner: RESNET (+{-accuracy_diff:.3f})")
        else:
            print(f"  Result: TIE (difference < 0.01)")
        
        # Save metrics
        metrics_file = self.output_dir / "metrics" / "comparison_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        
        return metrics
    
    def create_comparison_visualizations(self, results: Dict, metrics: Dict):
        """
        Create comprehensive comparison visualizations
        Args:
            results: Comparison results dictionary
            metrics: Metrics dictionary
        """
        print(f"\n📈 CREATING COMPARISON VISUALIZATIONS")
        print("=" * 50)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Accuracy Comparison
        plt.figure(figsize=(12, 8))
        
        # Overall comparison
        plt.subplot(2, 2, 1)
        model_names = ['Ensemble\n(ResNet+STT+BERT)', 'ResNet\nStandalone']
        accuracies = [
            metrics['ensemble']['overall']['accuracy'],
            metrics['resnet']['overall']['accuracy']
        ]
        colors = ['purple', 'orange']
        
        bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8)
        plt.title('Overall Accuracy Comparison', fontweight='bold', fontsize=14)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        plt.subplot(2, 2, 2)
        f1_scores = [
            metrics['ensemble']['overall']['f1_score'],
            metrics['resnet']['overall']['f1_score']
        ]
        
        bars = plt.bar(model_names, f1_scores, color=colors, alpha=0.8)
        plt.title('F1-Score Comparison', fontweight='bold', fontsize=14)
        plt.ylabel('F1-Score')
        plt.ylim(0, 1)
        
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confidence comparison
        plt.subplot(2, 2, 3)
        confidences = [
            metrics['ensemble']['overall']['avg_confidence'],
            metrics['resnet']['overall']['avg_confidence']
        ]
        
        bars = plt.bar(model_names, confidences, color=colors, alpha=0.8)
        plt.title('Average Confidence Comparison', fontweight='bold', fontsize=14)
        plt.ylabel('Average Confidence')
        plt.ylim(0, 1)
        
        for bar, conf in zip(bars, confidences):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Processing time comparison
        plt.subplot(2, 2, 4)
        proc_times = [
            metrics['ensemble']['overall']['avg_processing_time'],
            metrics['resnet']['overall']['avg_processing_time']
        ]
        
        bars = plt.bar(model_names, proc_times, color=colors, alpha=0.8)
        plt.title('Processing Time Comparison', fontweight='bold', fontsize=14)
        plt.ylabel('Time (seconds)')
        
        for bar, time_val in zip(bars, proc_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        comparison_file = self.output_dir / "visualizations" / "model_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison saved to: {comparison_file}")
        
        # 2. Per-class Performance Comparison
        plt.figure(figsize=(16, 10))
        
        emotions = Config.EMOTION_CLASSES
        x = np.arange(len(emotions))
        width = 0.35
        
        # F1-scores per class
        plt.subplot(2, 2, 1)
        ensemble_f1 = metrics['ensemble']['per_class']['f1_score']
        resnet_f1 = metrics['resnet']['per_class']['f1_score']
        
        plt.bar(x - width/2, ensemble_f1, width, label='Ensemble', alpha=0.8, color='purple')
        plt.bar(x + width/2, resnet_f1, width, label='ResNet', alpha=0.8, color='orange')
        
        plt.title('F1-Score per Emotion', fontweight='bold')
        plt.ylabel('F1-Score')
        plt.xlabel('Emotion')
        plt.xticks(x, emotions, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        
        # Precision per class
        plt.subplot(2, 2, 2)
        ensemble_prec = metrics['ensemble']['per_class']['precision']
        resnet_prec = metrics['resnet']['per_class']['precision']
        
        plt.bar(x - width/2, ensemble_prec, width, label='Ensemble', alpha=0.8, color='purple')
        plt.bar(x + width/2, resnet_prec, width, label='ResNet', alpha=0.8, color='orange')
        
        plt.title('Precision per Emotion', fontweight='bold')
        plt.ylabel('Precision')
        plt.xlabel('Emotion')
        plt.xticks(x, emotions, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        
        # Recall per class
        plt.subplot(2, 2, 3)
        ensemble_rec = metrics['ensemble']['per_class']['recall']
        resnet_rec = metrics['resnet']['per_class']['recall']
        
        plt.bar(x - width/2, ensemble_rec, width, label='Ensemble', alpha=0.8, color='purple')
        plt.bar(x + width/2, resnet_rec, width, label='ResNet', alpha=0.8, color='orange')
        
        plt.title('Recall per Emotion', fontweight='bold')
        plt.ylabel('Recall')
        plt.xlabel('Emotion')
        plt.xticks(x, emotions, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        
        # Support (sample count)
        plt.subplot(2, 2, 4)
        support = metrics['ensemble']['per_class']['support']  # Same for both models
        
        plt.bar(emotions, support, alpha=0.8, color='lightblue')
        plt.title('Sample Count per Emotion', fontweight='bold')
        plt.ylabel('Number of Samples')
        plt.xlabel('Emotion')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        per_class_file = self.output_dir / "visualizations" / "per_class_comparison.png"
        plt.savefig(per_class_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class comparison saved to: {per_class_file}")
        
        # 3. Confusion Matrices Comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ensemble confusion matrix
        ensemble_cm = np.array(metrics['ensemble']['confusion_matrix'])
        sns.heatmap(
            ensemble_cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=emotions,
            yticklabels=emotions,
            ax=axes[0]
        )
        axes[0].set_title('Ensemble Model - Confusion Matrix', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Predicted Emotion')
        axes[0].set_ylabel('True Emotion')
        
        # ResNet confusion matrix
        resnet_cm = np.array(metrics['resnet']['confusion_matrix'])
        sns.heatmap(
            resnet_cm, 
            annot=True, 
            fmt='d', 
            cmap='Oranges',
            xticklabels=emotions,
            yticklabels=emotions,
            ax=axes[1]
        )
        axes[1].set_title('ResNet Model - Confusion Matrix', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Predicted Emotion')
        axes[1].set_ylabel('True Emotion')
        
        plt.tight_layout()
        confusion_file = self.output_dir / "visualizations" / "confusion_matrices.png"
        plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to: {confusion_file}")
    
    def generate_comparison_report(self, results: Dict, metrics: Dict):
        """
        Generate comprehensive comparison report
        Args:
            results: Comparison results dictionary
            metrics: Metrics dictionary
        """
        print(f"\n📝 GENERATING COMPARISON REPORT")
        print("=" * 40)
        
        report_file = self.output_dir / "reports" / "model_comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Model Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            ensemble_acc = metrics['ensemble']['overall']['accuracy']
            resnet_acc = metrics['resnet']['overall']['accuracy']
            accuracy_diff = ensemble_acc - resnet_acc
            
            f.write(f"**Dataset Size:** {metrics['ensemble']['overall']['total_samples']} audio files\n")
            f.write(f"**Models Compared:** Ensemble (ResNet+STT+BERT) vs ResNet Standalone\n\n")
            
            if accuracy_diff > 0.01:
                winner = "Ensemble"
                advantage = accuracy_diff
            elif accuracy_diff < -0.01:
                winner = "ResNet"
                advantage = -accuracy_diff
            else:
                winner = "Tie"
                advantage = abs(accuracy_diff)
            
            f.write(f"**Winner:** {winner}")
            if winner != "Tie":
                f.write(f" (advantage: +{advantage:.3f})")
            f.write("\n\n")
            
            # Detailed Results
            f.write("## Detailed Performance Comparison\n\n")
            f.write("| Metric | Ensemble | ResNet | Difference |\n")
            f.write("|--------|----------|--------|-----------|\n")
            
            # Overall metrics
            ens_metrics = metrics['ensemble']['overall']
            res_metrics = metrics['resnet']['overall']
            
            f.write(f"| Accuracy | {ens_metrics['accuracy']:.3f} | {res_metrics['accuracy']:.3f} | {ens_metrics['accuracy'] - res_metrics['accuracy']:+.3f} |\n")
            f.write(f"| Precision | {ens_metrics['precision']:.3f} | {res_metrics['precision']:.3f} | {ens_metrics['precision'] - res_metrics['precision']:+.3f} |\n")
            f.write(f"| Recall | {ens_metrics['recall']:.3f} | {res_metrics['recall']:.3f} | {ens_metrics['recall'] - res_metrics['recall']:+.3f} |\n")
            f.write(f"| F1-Score | {ens_metrics['f1_score']:.3f} | {res_metrics['f1_score']:.3f} | {ens_metrics['f1_score'] - res_metrics['f1_score']:+.3f} |\n")
            f.write(f"| Avg Confidence | {ens_metrics['avg_confidence']:.3f} | {res_metrics['avg_confidence']:.3f} | {ens_metrics['avg_confidence'] - res_metrics['avg_confidence']:+.3f} |\n")
            f.write(f"| Processing Time | {ens_metrics['avg_processing_time']:.3f}s | {res_metrics['avg_processing_time']:.3f}s | {ens_metrics['avg_processing_time'] - res_metrics['avg_processing_time']:+.3f}s |\n")
            
            f.write("\n")
            
            # Per-class analysis
            f.write("## Per-Class Performance Analysis\n\n")
            f.write("### F1-Score Comparison by Emotion\n\n")
            f.write("| Emotion | Ensemble | ResNet | Difference | Winner |\n")
            f.write("|---------|----------|--------|-----------|---------|\n")
            
            ensemble_wins = 0
            resnet_wins = 0
            ties = 0
            
            for i, emotion in enumerate(Config.EMOTION_CLASSES):
                ens_f1 = metrics['ensemble']['per_class']['f1_score'][i]
                res_f1 = metrics['resnet']['per_class']['f1_score'][i]
                diff = ens_f1 - res_f1
                
                if diff > 0.01:
                    winner = "Ensemble"
                    ensemble_wins += 1
                elif diff < -0.01:
                    winner = "ResNet"
                    resnet_wins += 1
                else:
                    winner = "Tie"
                    ties += 1
                
                f.write(f"| {emotion} | {ens_f1:.3f} | {res_f1:.3f} | {diff:+.3f} | {winner} |\n")
            
            f.write(f"\n**Per-Class Summary:** Ensemble wins: {ensemble_wins}, ResNet wins: {resnet_wins}, Ties: {ties}\n\n")
            
            # STT Analysis
            if 'stt_analysis' in metrics['ensemble']:
                stt = metrics['ensemble']['stt_analysis']
                f.write("## Speech-to-Text Analysis\n\n")
                f.write(f"- **Transcription Success Rate:** {stt['transcription_rate']:.3f}\n")
                f.write(f"- **Successful Transcriptions:** {stt['successful_transcriptions']}\n")
                f.write(f"- **Total Files:** {len(results['ensemble']['transcriptions'])}\n\n")
            
            # Key Insights
            f.write("## Key Insights\n\n")
            
            if accuracy_diff > 0.02:
                f.write(f"1. **Ensemble Advantage:** The ensemble model shows significant improvement (+{accuracy_diff:.3f}) over standalone ResNet\n")
            elif accuracy_diff < -0.02:
                f.write(f"1. **ResNet Efficiency:** Standalone ResNet outperforms the ensemble (+{-accuracy_diff:.3f}), suggesting the additional components may not be beneficial\n")
            else:
                f.write(f"1. **Comparable Performance:** Both models show similar accuracy (difference: {accuracy_diff:+.3f})\n")
            
            # Processing time insight
            time_diff = ens_metrics['avg_processing_time'] - res_metrics['avg_processing_time']
            if time_diff > 0.1:
                f.write(f"2. **Processing Time:** Ensemble is slower (+{time_diff:.3f}s per file) due to STT and text processing\n")
            else:
                f.write(f"2. **Processing Time:** Similar processing times between models\n")
            
            # Confidence insight
            conf_diff = ens_metrics['avg_confidence'] - res_metrics['avg_confidence']
            if conf_diff > 0.05:
                f.write(f"3. **Confidence:** Ensemble shows higher confidence in predictions (+{conf_diff:.3f})\n")
            elif conf_diff < -0.05:
                f.write(f"3. **Confidence:** ResNet shows higher confidence in predictions (+{-conf_diff:.3f})\n")
            else:
                f.write(f"3. **Confidence:** Similar confidence levels between models\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if accuracy_diff > 0.01:
                f.write("1. **Use Ensemble Model** for maximum accuracy in production\n")
                f.write("2. **Consider computational cost** - ensemble requires more resources\n")
            elif accuracy_diff < -0.01:
                f.write("1. **Use ResNet Model** for optimal efficiency and accuracy\n")
                f.write("2. **Ensemble overhead** doesn't justify the complexity\n")
            else:
                f.write("1. **Choice depends on requirements** - similar performance\n")
                f.write("2. **Use ResNet** for speed, **Ensemble** for interpretability\n")
            
            f.write("3. **Consider application context** - real-time vs batch processing\n")
            f.write("4. **Monitor STT quality** if using ensemble in production\n\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("- `metrics/comparison_metrics.json` - Detailed numerical metrics\n")
            f.write("- `visualizations/model_comparison.png` - Overall performance comparison\n")
            f.write("- `visualizations/per_class_comparison.png` - Per-emotion analysis\n")
            f.write("- `visualizations/confusion_matrices.png` - Confusion matrix comparison\n")
            f.write("- `predictions/comparison_results.json` - Raw prediction results\n\n")
        
        print(f"Comparison report saved to: {report_file}")


def main():
    """Main comparison function"""
    print("🔥 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print("Comparing: Ensemble (ResNet+STT+BERT) vs ResNet Standalone")
    print("Dataset: Complete emotion dataset (all files)")
    print("=" * 80)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load models
    if not comparator.load_models():
        print("❌ Failed to load models. Exiting.")
        return
    
    # Get complete dataset
    test_files = comparator.get_complete_dataset()
    
    if not test_files:
        print("❌ No test files found. Please check the data directory.")
        return
    
    print(f"📊 Testing on {len(test_files)} files from complete dataset")
    
    # Compare models
    results = comparator.compare_models(test_files)
    
    # Calculate metrics
    metrics = comparator.calculate_comparison_metrics(results)
    
    # Create visualizations
    comparator.create_comparison_visualizations(results, metrics)
    
    # Generate report
    comparator.generate_comparison_report(results, metrics)
    
    print(f"\n🎉 COMPARISON COMPLETE!")
    print(f"Results saved to: {comparator.output_dir}")
    
    # Final summary
    ensemble_acc = metrics['ensemble']['overall']['accuracy']
    resnet_acc = metrics['resnet']['overall']['accuracy']
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"  Ensemble Accuracy: {ensemble_acc:.3f}")
    print(f"  ResNet Accuracy:   {resnet_acc:.3f}")
    print(f"  Difference:        {ensemble_acc - resnet_acc:+.3f}")
    
    if ensemble_acc > resnet_acc + 0.01:
        print(f"  🏆 WINNER: ENSEMBLE MODEL")
    elif resnet_acc > ensemble_acc + 0.01:
        print(f"  🏆 WINNER: RESNET MODEL")
    else:
        print(f"  🤝 RESULT: TIE (very close performance)")


if __name__ == "__main__":
    main()
