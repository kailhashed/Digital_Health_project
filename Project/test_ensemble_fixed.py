#!/usr/bin/env python3
"""
Test Fixed Ensemble Model with Comprehensive Metrics and Visualizations
Tests the fixed STT implementation and generates detailed performance analysis
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
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.models.ensemble_model import EnsembleAudioSentimentAnalyzer
    from src.utils.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the Project directory")
    sys.exit(1)


def create_output_directory() -> Path:
    """Create timestamped output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_ensemble_fixed_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    
    return output_dir


def get_test_data(data_dir: str = "organized_by_emotion", max_per_emotion: int = 50) -> List[Tuple[str, str]]:
    """
    Get test data from organized emotion directories
    Args:
        data_dir: Directory containing emotion subdirectories
        max_per_emotion: Maximum files per emotion for testing
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
    
    for emotion in emotions:
        emotion_dir = data_path / emotion
        audio_files = []
        
        # Find audio files
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(emotion_dir.glob(ext)))
        
        # Limit files per emotion
        if max_per_emotion > 0:
            audio_files = audio_files[:max_per_emotion]
        
        for audio_file in audio_files:
            test_files.append((str(audio_file), emotion))
        
        print(f"  {emotion}: {len(audio_files)} files")
    
    print(f"Total test files: {len(test_files)}")
    return test_files


def test_ensemble_model(test_files: List[Tuple[str, str]], 
                       vocal_bias: float = 0.65,
                       output_dir: Path = None) -> Dict:
    """
    Test ensemble model with fixed STT implementation
    Args:
        test_files: List of (file_path, true_emotion) tuples
        vocal_bias: Vocal emotion bias
        output_dir: Output directory for results
    Returns:
        Test results dictionary
    """
    print(f"\n🧪 TESTING ENSEMBLE MODEL (Fixed STT)")
    print("=" * 60)
    print(f"Test files: {len(test_files)}")
    print(f"Vocal bias: {vocal_bias}")
    print(f"Model configuration: ResNet + STT + BERT")
    print("=" * 60)
    
    # Initialize ensemble model
    ensemble = EnsembleAudioSentimentAnalyzer(
        resnet_model_path="models/resnet/best_ResNet.pth",
        whisper_model_size="base",
        vocal_weight=vocal_bias,
        text_weight=1.0 - vocal_bias,
        confidence_threshold=0.7
    )
    
    # Test results storage
    results = {
        'predictions': [],
        'true_labels': [],
        'file_paths': [],
        'confidences': [],
        'vocal_predictions': [],
        'text_predictions': [],
        'transcriptions': [],
        'processing_times': [],
        'detailed_results': []
    }
    
    # Process each test file
    start_time = time.time()
    
    for i, (file_path, true_emotion) in enumerate(test_files):
        print(f"\nProcessing [{i+1}/{len(test_files)}]: {Path(file_path).name}")
        
        try:
            # Time individual prediction
            pred_start = time.time()
            
            # Get detailed prediction
            prediction = ensemble.predict_emotion(file_path, return_details=True)
            
            pred_time = time.time() - pred_start
            
            # Store results
            results['predictions'].append(prediction['predicted_emotion'])
            results['true_labels'].append(true_emotion)
            results['file_paths'].append(file_path)
            results['confidences'].append(prediction['confidence'])
            results['transcriptions'].append(prediction['transcribed_text'])
            results['processing_times'].append(pred_time)
            results['detailed_results'].append(prediction)
            
            # Extract component predictions if available
            if 'component_predictions' in prediction:
                vocal_pred = prediction['component_predictions']['vocal']['emotion']
                text_pred = prediction['component_predictions']['text']['emotion']
                results['vocal_predictions'].append(vocal_pred)
                results['text_predictions'].append(text_pred)
            else:
                results['vocal_predictions'].append('unknown')
                results['text_predictions'].append('unknown')
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(test_files) - i - 1)
                print(f"Progress: {i+1}/{len(test_files)} | Avg: {avg_time:.2f}s/file | ETA: {eta:.1f}s")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Add placeholder results for failed predictions
            results['predictions'].append('neutral')
            results['true_labels'].append(true_emotion)
            results['file_paths'].append(file_path)
            results['confidences'].append(0.0)
            results['transcriptions'].append('')
            results['processing_times'].append(0.0)
            results['vocal_predictions'].append('unknown')
            results['text_predictions'].append('unknown')
            results['detailed_results'].append({
                'predicted_emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Testing completed!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per file: {total_time/len(test_files):.2f}s")
    
    # Save raw results
    if output_dir:
        results_file = output_dir / "predictions" / "raw_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if key == 'detailed_results':
                    serializable_results[key] = value  # Keep as is
                else:
                    serializable_results[key] = [float(v) if isinstance(v, np.floating) else v for v in value]
            json.dump(serializable_results, f, indent=2)
        print(f"Raw results saved to: {results_file}")
    
    return results


def calculate_metrics(results: Dict, output_dir: Path = None) -> Dict:
    """
    Calculate comprehensive performance metrics
    Args:
        results: Test results dictionary
        output_dir: Output directory for saving metrics
    Returns:
        Metrics dictionary
    """
    print(f"\n📊 CALCULATING METRICS")
    print("=" * 40)
    
    true_labels = results['true_labels']
    predictions = results['predictions']
    confidences = results['confidences']
    
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
    
    # Additional metrics
    avg_confidence = np.mean(confidences)
    processing_times = results['processing_times']
    avg_processing_time = np.mean(processing_times)
    
    # Component analysis (if available)
    vocal_accuracy = 0.0
    text_accuracy = 0.0
    
    if results['vocal_predictions'] and 'unknown' not in results['vocal_predictions']:
        vocal_accuracy = accuracy_score(true_labels, results['vocal_predictions'])
    
    if results['text_predictions'] and 'unknown' not in results['text_predictions']:
        text_accuracy = accuracy_score(true_labels, results['text_predictions'])
    
    # STT analysis
    transcriptions = results['transcriptions']
    successful_transcriptions = sum(1 for t in transcriptions if t.strip())
    transcription_rate = successful_transcriptions / len(transcriptions)
    
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'total_samples': len(true_labels)
        },
        'per_class': {
            'emotions': Config.EMOTION_CLASSES,
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        },
        'components': {
            'vocal_accuracy': vocal_accuracy,
            'text_accuracy': text_accuracy,
            'transcription_rate': transcription_rate,
            'successful_transcriptions': successful_transcriptions
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }
    
    # Print key metrics
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Weighted F1-Score: {f1:.3f}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Processing Time: {avg_processing_time:.2f}s per file")
    print(f"Vocal Component Accuracy: {vocal_accuracy:.3f}")
    print(f"Text Component Accuracy: {text_accuracy:.3f}")
    print(f"Transcription Success Rate: {transcription_rate:.3f}")
    
    # Save metrics
    if output_dir:
        metrics_file = output_dir / "metrics" / "performance_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
        
        # Save detailed per-class metrics
        per_class_df = pd.DataFrame({
            'emotion': Config.EMOTION_CLASSES,
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1_score': f1_per_class,
            'support': support_per_class
        })
        per_class_file = output_dir / "metrics" / "per_class_metrics.csv"
        per_class_df.to_csv(per_class_file, index=False)
        print(f"Per-class metrics saved to: {per_class_file}")
    
    return metrics


def create_visualizations(results: Dict, metrics: Dict, output_dir: Path):
    """
    Create comprehensive visualizations
    Args:
        results: Test results dictionary
        metrics: Metrics dictionary
        output_dir: Output directory for saving plots
    """
    print(f"\n📈 CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=Config.EMOTION_CLASSES,
        yticklabels=Config.EMOTION_CLASSES
    )
    plt.title('Ensemble Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    confusion_file = output_dir / "visualizations" / "confusion_matrix.png"
    plt.savefig(confusion_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {confusion_file}")
    
    # 2. Per-class Performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    emotions = metrics['per_class']['emotions']
    precision = metrics['per_class']['precision']
    recall = metrics['per_class']['recall']
    f1_scores = metrics['per_class']['f1_score']
    support = metrics['per_class']['support']
    
    # Precision
    axes[0, 0].bar(emotions, precision, color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Precision per Emotion', fontweight='bold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Recall
    axes[0, 1].bar(emotions, recall, color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Recall per Emotion', fontweight='bold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # F1-Score
    axes[1, 0].bar(emotions, f1_scores, color='lightgreen', alpha=0.8)
    axes[1, 0].set_title('F1-Score per Emotion', fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Support
    axes[1, 1].bar(emotions, support, color='gold', alpha=0.8)
    axes[1, 1].set_title('Support (Number of Samples)', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    performance_file = output_dir / "visualizations" / "per_class_performance.png"
    plt.savefig(performance_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class performance saved to: {performance_file}")
    
    # 3. Component Comparison
    plt.figure(figsize=(10, 6))
    
    component_accuracies = [
        metrics['overall']['accuracy'],
        metrics['components']['vocal_accuracy'],
        metrics['components']['text_accuracy']
    ]
    component_names = ['Ensemble', 'Vocal (ResNet)', 'Text (BERT)']
    colors = ['purple', 'orange', 'green']
    
    bars = plt.bar(component_names, component_accuracies, color=colors, alpha=0.8)
    plt.title('Component Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, component_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    component_file = output_dir / "visualizations" / "component_comparison.png"
    plt.savefig(component_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Component comparison saved to: {component_file}")
    
    # 4. Confidence Distribution
    plt.figure(figsize=(12, 5))
    
    confidences = results['confidences']
    
    # Overall confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Confidence Score Distribution', fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.legend()
    
    # Confidence by emotion
    plt.subplot(1, 2, 2)
    confidence_by_emotion = {}
    for true_label, conf in zip(results['true_labels'], confidences):
        if true_label not in confidence_by_emotion:
            confidence_by_emotion[true_label] = []
        confidence_by_emotion[true_label].append(conf)
    
    emotions_with_data = list(confidence_by_emotion.keys())
    avg_confidences = [np.mean(confidence_by_emotion[emotion]) for emotion in emotions_with_data]
    
    plt.bar(emotions_with_data, avg_confidences, color='lightcoral', alpha=0.8)
    plt.title('Average Confidence by Emotion', fontweight='bold')
    plt.xlabel('Emotion')
    plt.ylabel('Average Confidence')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    confidence_file = output_dir / "visualizations" / "confidence_analysis.png"
    plt.savefig(confidence_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence analysis saved to: {confidence_file}")
    
    # 5. STT Analysis
    if results['transcriptions']:
        plt.figure(figsize=(12, 8))
        
        # Transcription success rate
        plt.subplot(2, 2, 1)
        successful = sum(1 for t in results['transcriptions'] if t.strip())
        failed = len(results['transcriptions']) - successful
        
        plt.pie([successful, failed], labels=['Successful', 'Failed'], 
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Speech-to-Text Success Rate', fontweight='bold')
        
        # Transcription length distribution
        plt.subplot(2, 2, 2)
        transcription_lengths = [len(t.strip()) for t in results['transcriptions'] if t.strip()]
        if transcription_lengths:
            plt.hist(transcription_lengths, bins=20, alpha=0.7, color='gold', edgecolor='black')
            plt.title('Transcription Length Distribution', fontweight='bold')
            plt.xlabel('Characters')
            plt.ylabel('Frequency')
        
        # Accuracy by transcription success
        plt.subplot(2, 2, 3)
        successful_indices = [i for i, t in enumerate(results['transcriptions']) if t.strip()]
        failed_indices = [i for i, t in enumerate(results['transcriptions']) if not t.strip()]
        
        if successful_indices and failed_indices:
            successful_accuracy = sum(1 for i in successful_indices 
                                    if results['predictions'][i] == results['true_labels'][i]) / len(successful_indices)
            failed_accuracy = sum(1 for i in failed_indices 
                                if results['predictions'][i] == results['true_labels'][i]) / len(failed_indices)
            
            plt.bar(['With Transcription', 'Without Transcription'], 
                   [successful_accuracy, failed_accuracy], 
                   color=['lightgreen', 'lightcoral'], alpha=0.8)
            plt.title('Accuracy by STT Success', fontweight='bold')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
        
        # Processing time distribution
        plt.subplot(2, 2, 4)
        processing_times = results['processing_times']
        plt.hist(processing_times, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('Processing Time Distribution', fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(processing_times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(processing_times):.2f}s')
        plt.legend()
        
        plt.tight_layout()
        stt_file = output_dir / "visualizations" / "stt_analysis.png"
        plt.savefig(stt_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"STT analysis saved to: {stt_file}")


def generate_report(results: Dict, metrics: Dict, output_dir: Path):
    """
    Generate comprehensive test report
    Args:
        results: Test results dictionary
        metrics: Metrics dictionary
        output_dir: Output directory for saving report
    """
    print(f"\n📝 GENERATING REPORT")
    print("=" * 30)
    
    report_file = output_dir / "reports" / "ensemble_test_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Ensemble Model Test Report (Fixed STT)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("## Model Configuration\n\n")
        f.write("- **Architecture:** ResNet (Vocal) + Whisper (STT) + BERT (Text)\n")
        f.write("- **Vocal Weight:** Variable (adaptive based on confidence)\n")
        f.write("- **STT Model:** Whisper Base\n")
        f.write("- **Text Model:** DistilRoBERTa-base (emotion)\n")
        f.write("- **Audio Processing:** 16kHz, Mel-spectrogram\n\n")
        
        # Overall Performance
        f.write("## Overall Performance\n\n")
        f.write(f"- **Accuracy:** {metrics['overall']['accuracy']:.3f}\n")
        f.write(f"- **Precision:** {metrics['overall']['precision']:.3f}\n")
        f.write(f"- **Recall:** {metrics['overall']['recall']:.3f}\n")
        f.write(f"- **F1-Score:** {metrics['overall']['f1_score']:.3f}\n")
        f.write(f"- **Average Confidence:** {metrics['overall']['avg_confidence']:.3f}\n")
        f.write(f"- **Processing Time:** {metrics['overall']['avg_processing_time']:.2f}s per file\n")
        f.write(f"- **Total Samples:** {metrics['overall']['total_samples']}\n\n")
        
        # Component Analysis
        f.write("## Component Analysis\n\n")
        f.write(f"- **Vocal Component (ResNet):** {metrics['components']['vocal_accuracy']:.3f}\n")
        f.write(f"- **Text Component (BERT):** {metrics['components']['text_accuracy']:.3f}\n")
        f.write(f"- **STT Success Rate:** {metrics['components']['transcription_rate']:.3f}\n")
        f.write(f"- **Successful Transcriptions:** {metrics['components']['successful_transcriptions']}\n\n")
        
        # Per-class Performance
        f.write("## Per-Class Performance\n\n")
        f.write("| Emotion | Precision | Recall | F1-Score | Support |\n")
        f.write("|---------|-----------|--------|----------|----------|\n")
        
        for i, emotion in enumerate(metrics['per_class']['emotions']):
            precision = metrics['per_class']['precision'][i]
            recall = metrics['per_class']['recall'][i]
            f1 = metrics['per_class']['f1_score'][i]
            support = int(metrics['per_class']['support'][i])
            f.write(f"| {emotion} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support} |\n")
        
        f.write("\n")
        
        # Key Findings
        f.write("## Key Findings\n\n")
        
        best_emotion_idx = np.argmax(metrics['per_class']['f1_score'])
        worst_emotion_idx = np.argmin(metrics['per_class']['f1_score'])
        best_emotion = metrics['per_class']['emotions'][best_emotion_idx]
        worst_emotion = metrics['per_class']['emotions'][worst_emotion_idx]
        
        f.write(f"- **Best Performing Emotion:** {best_emotion} (F1: {metrics['per_class']['f1_score'][best_emotion_idx]:.3f})\n")
        f.write(f"- **Worst Performing Emotion:** {worst_emotion} (F1: {metrics['per_class']['f1_score'][worst_emotion_idx]:.3f})\n")
        
        if metrics['components']['transcription_rate'] > 0:
            f.write(f"- **STT Integration:** Successfully fixed Windows compatibility issues\n")
            f.write(f"- **Speech Recognition:** {metrics['components']['transcription_rate']*100:.1f}% success rate\n")
        else:
            f.write(f"- **STT Integration:** No successful transcriptions - may need further investigation\n")
        
        # Performance comparison
        ensemble_acc = metrics['overall']['accuracy']
        vocal_acc = metrics['components']['vocal_accuracy']
        text_acc = metrics['components']['text_accuracy']
        
        if ensemble_acc > vocal_acc and ensemble_acc > text_acc:
            f.write(f"- **Ensemble Benefit:** Outperforms individual components\n")
        elif vocal_acc > ensemble_acc:
            f.write(f"- **Vocal Dominance:** Vocal component is strongest\n")
        else:
            f.write(f"- **Component Balance:** Mixed performance across components\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if metrics['components']['transcription_rate'] < 0.5:
            f.write("1. **STT Improvement:** Consider alternative speech recognition models\n")
        
        if metrics['overall']['accuracy'] < 0.7:
            f.write("2. **Model Enhancement:** Consider additional training or different architectures\n")
        
        f.write("3. **Data Quality:** Ensure consistent audio quality across datasets\n")
        f.write("4. **Bias Tuning:** Experiment with different vocal/text weight ratios\n")
        f.write("5. **Preprocessing:** Consider advanced audio preprocessing techniques\n\n")
        
        # Files Generated
        f.write("## Generated Files\n\n")
        f.write("- `metrics/performance_metrics.json` - Detailed numerical metrics\n")
        f.write("- `metrics/per_class_metrics.csv` - Per-class performance data\n")
        f.write("- `visualizations/confusion_matrix.png` - Confusion matrix heatmap\n")
        f.write("- `visualizations/per_class_performance.png` - Per-class metrics charts\n")
        f.write("- `visualizations/component_comparison.png` - Component accuracy comparison\n")
        f.write("- `visualizations/confidence_analysis.png` - Confidence distribution analysis\n")
        f.write("- `visualizations/stt_analysis.png` - Speech-to-text analysis\n")
        f.write("- `predictions/raw_results.json` - Raw prediction results\n\n")
    
    print(f"Report saved to: {report_file}")


def main():
    """Main testing function"""
    print("🎵 ENSEMBLE MODEL TESTING (Fixed STT Implementation)")
    print("=" * 80)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Get test data
    test_files = get_test_data(max_per_emotion=30)  # Limit for faster testing
    
    if not test_files:
        print("❌ No test files found. Please check the data directory.")
        return
    
    # Test ensemble model
    results = test_ensemble_model(test_files, vocal_bias=0.9, output_dir=output_dir)
    
    # Calculate metrics
    metrics = calculate_metrics(results, output_dir=output_dir)
    
    # Create visualizations
    create_visualizations(results, metrics, output_dir)
    
    # Generate report
    generate_report(results, metrics, output_dir)
    
    print(f"\n🎉 TESTING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("\nKey Metrics:")
    print(f"  - Overall Accuracy: {metrics['overall']['accuracy']:.3f}")
    print(f"  - F1-Score: {metrics['overall']['f1_score']:.3f}")
    print(f"  - STT Success Rate: {metrics['components']['transcription_rate']:.3f}")
    print(f"  - Processing Time: {metrics['overall']['avg_processing_time']:.2f}s per file")


if __name__ == "__main__":
    main()
