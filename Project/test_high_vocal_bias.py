#!/usr/bin/env python3
"""
High Vocal Bias Testing (0.9) - Complete Dataset Analysis
Tests ensemble with 90% vocal bias on the entire dataset to maximize vocal emotion detection
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
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

class HighVocalBiasTest:
    """Test ensemble with 0.9 vocal bias on complete dataset"""
    
    def __init__(self):
        """Initialize high vocal bias test"""
        self.vocal_bias = 0.9
        self.text_bias = 0.1
        
        # Create results directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results_high_vocal_bias_{self.timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "detailed_results").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "analysis").mkdir(exist_ok=True)
        (self.results_dir / "predictions").mkdir(exist_ok=True)
        
        print("🎯 HIGH VOCAL BIAS ENSEMBLE TEST (0.9)")
        print("="*50)
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🎚️ Vocal bias: {self.vocal_bias} (90%)")
        print(f"📝 Text bias: {self.text_bias} (10%)")
    
    def load_complete_dataset(self, data_dir: str = "organized_by_emotion"):
        """
        Load the complete dataset
        Args:
            data_dir: Dataset directory
        Returns:
            Tuple of (file_paths, labels, emotion_distribution)
        """
        print(f"\n📁 Loading complete dataset from: {data_dir}")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        all_files = []
        all_labels = []
        emotion_distribution = {}
        
        # Get all emotion directories
        emotion_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        emotion_dirs = sorted(emotion_dirs, key=lambda x: x.name)
        
        print(f"📊 Found emotion categories: {[d.name for d in emotion_dirs]}")
        
        for emotion_dir in emotion_dirs:
            emotion = emotion_dir.name
            
            # Find all audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                audio_files.extend(list(emotion_dir.glob(ext)))
            
            print(f"  {emotion}: {len(audio_files)} files")
            emotion_distribution[emotion] = len(audio_files)
            
            for audio_file in audio_files:
                all_files.append(str(audio_file))
                all_labels.append(emotion)
        
        print(f"📈 Total dataset: {len(all_files)} files across {len(emotion_dirs)} emotions")
        
        return all_files, all_labels, emotion_distribution
    
    def create_ensemble_model(self):
        """Create ensemble model with high vocal bias"""
        print(f"\n🔧 Creating ensemble with {self.vocal_bias} vocal bias...")
        
        try:
            ensemble = EnsembleAudioSentimentAnalyzer(
                resnet_model_path="models/resnet/best_ResNet.pth",
                whisper_model_size="tiny",  # Fast processing for large dataset
                vocal_weight=self.vocal_bias,
                text_weight=self.text_bias,
                confidence_threshold=0.7  # Standard threshold
            )
            
            print("✅ Ensemble model created successfully")
            return ensemble
            
        except Exception as e:
            print(f"❌ Error creating ensemble: {e}")
            raise
    
    def run_complete_dataset_test(self, max_files_per_emotion: int = None):
        """
        Run test on complete dataset
        Args:
            max_files_per_emotion: Limit files per emotion (None = all files)
        """
        print(f"\n🚀 Starting complete dataset test...")
        
        # Load dataset
        all_files, all_labels, emotion_distribution = self.load_complete_dataset()
        
        # Limit files if specified
        if max_files_per_emotion:
            print(f"🔢 Limiting to {max_files_per_emotion} files per emotion...")
            limited_files = []
            limited_labels = []
            
            for emotion in set(all_labels):
                emotion_files = [f for f, l in zip(all_files, all_labels) if l == emotion]
                emotion_files = emotion_files[:max_files_per_emotion]
                limited_files.extend(emotion_files)
                limited_labels.extend([emotion] * len(emotion_files))
            
            all_files = limited_files
            all_labels = limited_labels
            
            print(f"📊 Limited dataset: {len(all_files)} files")
        
        # Create ensemble model
        ensemble = self.create_ensemble_model()
        
        # Run predictions
        print(f"🔄 Running predictions on {len(all_files)} files...")
        
        predictions = []
        detailed_results = []
        failed_files = []
        vocal_high_confidence_count = 0
        vocal_text_disagreements = 0
        
        for i, (file_path, true_label) in enumerate(tqdm(zip(all_files, all_labels), 
                                                          total=len(all_files), 
                                                          desc="Processing files")):
            try:
                result = ensemble.predict_emotion(file_path, return_details=True)
                
                predictions.append(result['predicted_emotion'])
                
                # Extract component predictions
                vocal_pred = result['component_predictions']['vocal']
                text_pred = result['component_predictions']['text']
                
                # Track high vocal confidence cases
                if vocal_pred['confidence'] > 0.8:
                    vocal_high_confidence_count += 1
                
                # Track vocal-text disagreements
                if vocal_pred['emotion'] != text_pred['emotion'] and text_pred['emotion'] != 'neutral':
                    vocal_text_disagreements += 1
                
                detailed_result = {
                    'file': Path(file_path).name,
                    'file_path': file_path,
                    'true_emotion': true_label,
                    'predicted_emotion': result['predicted_emotion'],
                    'ensemble_confidence': result['confidence'],
                    'transcribed_text': result['transcribed_text'],
                    'vocal_emotion': vocal_pred['emotion'],
                    'vocal_confidence': vocal_pred['confidence'],
                    'text_emotion': text_pred['emotion'],
                    'text_confidence': text_pred['confidence'],
                    'weights_used': result['weights_used'],
                    'vocal_text_agree': vocal_pred['emotion'] == text_pred['emotion'],
                    'ensemble_correct': result['predicted_emotion'] == true_label,
                    'vocal_correct': vocal_pred['emotion'] == true_label,
                    'text_correct': text_pred['emotion'] == true_label
                }
                
                detailed_results.append(detailed_result)
                
            except Exception as e:
                print(f"\n❌ Error processing {file_path}: {e}")
                failed_files.append(file_path)
                predictions.append('neutral')  # Default prediction
                
                # Add failed case to results
                detailed_results.append({
                    'file': Path(file_path).name,
                    'file_path': file_path,
                    'true_emotion': true_label,
                    'predicted_emotion': 'neutral',
                    'ensemble_confidence': 0.0,
                    'error': str(e)
                })
        
        # Calculate overall metrics
        print(f"\n📊 Calculating performance metrics...")
        
        valid_predictions = [p for p in predictions if p != 'neutral' or 'error' not in detailed_results[predictions.index(p)]]
        valid_labels = [l for i, l in enumerate(all_labels) if predictions[i] != 'neutral' or 'error' not in detailed_results[i]]
        
        accuracy = accuracy_score(valid_labels, valid_predictions)
        f1_weighted = f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(valid_labels, valid_predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
        
        # Classification report
        class_report = classification_report(
            valid_labels, valid_predictions,
            target_names=Config.EMOTION_CLASSES,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(valid_labels, valid_predictions, labels=Config.EMOTION_CLASSES)
        
        # Compile results
        results = {
            'test_configuration': {
                'vocal_bias': self.vocal_bias,
                'text_bias': self.text_bias,
                'whisper_model': 'tiny',
                'confidence_threshold': 0.7,
                'total_files': len(all_files),
                'failed_files': len(failed_files),
                'success_rate': (len(all_files) - len(failed_files)) / len(all_files)
            },
            'performance_metrics': {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted
            },
            'vocal_analysis': {
                'high_confidence_cases': vocal_high_confidence_count,
                'high_confidence_rate': vocal_high_confidence_count / len(detailed_results),
                'vocal_text_disagreements': vocal_text_disagreements,
                'disagreement_rate': vocal_text_disagreements / len(detailed_results)
            },
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'detailed_predictions': detailed_results,
            'emotion_distribution': emotion_distribution,
            'failed_files': failed_files
        }
        
        # Save results
        results_file = self.results_dir / "complete_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed predictions
        predictions_df = pd.DataFrame(detailed_results)
        predictions_file = self.results_dir / "predictions" / "detailed_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        
        print(f"✅ Testing completed!")
        print(f"📈 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"🎯 F1-Score (weighted): {f1_weighted:.3f}")
        print(f"🔊 High vocal confidence cases: {vocal_high_confidence_count} ({vocal_high_confidence_count/len(detailed_results)*100:.1f}%)")
        print(f"⚡ Vocal-text disagreements: {vocal_text_disagreements} ({vocal_text_disagreements/len(detailed_results)*100:.1f}%)")
        print(f"❌ Failed files: {len(failed_files)}")
        
        return results
    
    def analyze_per_emotion_performance(self, results):
        """Analyze performance for each emotion"""
        print(f"\n📊 Analyzing per-emotion performance...")
        
        df = pd.DataFrame(results['detailed_predictions'])
        
        # Filter out failed cases
        valid_df = df[~df.get('error', pd.Series([False]*len(df)))]
        
        emotion_analysis = []
        
        for emotion in Config.EMOTION_CLASSES:
            emotion_data = valid_df[valid_df['true_emotion'] == emotion]
            
            if len(emotion_data) == 0:
                continue
            
            # Calculate metrics for this emotion
            total_cases = len(emotion_data)
            correct_cases = len(emotion_data[emotion_data['ensemble_correct'] == True])
            accuracy = correct_cases / total_cases if total_cases > 0 else 0
            
            # Vocal vs text performance
            vocal_correct = len(emotion_data[emotion_data['vocal_correct'] == True])
            text_correct = len(emotion_data[emotion_data['text_correct'] == True])
            vocal_accuracy = vocal_correct / total_cases if total_cases > 0 else 0
            text_accuracy = text_correct / total_cases if total_cases > 0 else 0
            
            # Average confidences
            avg_ensemble_conf = emotion_data['ensemble_confidence'].mean()
            avg_vocal_conf = emotion_data['vocal_confidence'].mean()
            avg_text_conf = emotion_data['text_confidence'].mean()
            
            # Vocal-text agreement
            agreements = len(emotion_data[emotion_data['vocal_text_agree'] == True])
            agreement_rate = agreements / total_cases if total_cases > 0 else 0
            
            emotion_analysis.append({
                'emotion': emotion,
                'total_cases': total_cases,
                'ensemble_accuracy': accuracy,
                'vocal_accuracy': vocal_accuracy,
                'text_accuracy': text_accuracy,
                'ensemble_confidence': avg_ensemble_conf,
                'vocal_confidence': avg_vocal_conf,
                'text_confidence': avg_text_conf,
                'vocal_text_agreement': agreement_rate,
                'vocal_advantage': vocal_accuracy - text_accuracy
            })
        
        emotion_df = pd.DataFrame(emotion_analysis)
        emotion_df = emotion_df.sort_values('ensemble_accuracy', ascending=False)
        
        # Save analysis
        analysis_file = self.results_dir / "analysis" / "per_emotion_analysis.csv"
        emotion_df.to_csv(analysis_file, index=False)
        
        print(f"📈 Per-emotion analysis:")
        print(emotion_df[['emotion', 'ensemble_accuracy', 'vocal_accuracy', 'text_accuracy', 'vocal_advantage']].to_string(index=False, float_format='%.3f'))
        
        return emotion_df
    
    def create_visualizations(self, results, emotion_analysis):
        """Create comprehensive visualizations"""
        print(f"\n📊 Creating visualizations...")
        
        viz_dir = self.results_dir / "visualizations"
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=Config.EMOTION_CLASSES,
                   yticklabels=Config.EMOTION_CLASSES)
        plt.title(f'Confusion Matrix - High Vocal Bias (0.9)\nOverall Accuracy: {results["performance_metrics"]["accuracy"]:.3f}')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.savefig(viz_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-emotion performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        x = range(len(emotion_analysis))
        axes[0, 0].bar(x, emotion_analysis['ensemble_accuracy'], alpha=0.7, label='Ensemble', color='blue')
        axes[0, 0].bar([i+0.3 for i in x], emotion_analysis['vocal_accuracy'], alpha=0.7, label='Vocal Only', color='red')
        axes[0, 0].bar([i+0.6 for i in x], emotion_analysis['text_accuracy'], alpha=0.7, label='Text Only', color='green')
        axes[0, 0].set_title('Accuracy by Emotion')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks([i+0.3 for i in x])
        axes[0, 0].set_xticklabels(emotion_analysis['emotion'], rotation=45)
        axes[0, 0].legend()
        
        # Vocal advantage
        axes[0, 1].bar(x, emotion_analysis['vocal_advantage'], 
                      color=['green' if v > 0 else 'red' for v in emotion_analysis['vocal_advantage']])
        axes[0, 1].set_title('Vocal Advantage (Vocal - Text Accuracy)')
        axes[0, 1].set_ylabel('Accuracy Difference')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(emotion_analysis['emotion'], rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Confidence levels
        axes[1, 0].bar(x, emotion_analysis['vocal_confidence'], alpha=0.7, label='Vocal Confidence', color='red')
        axes[1, 0].bar([i+0.3 for i in x], emotion_analysis['text_confidence'], alpha=0.7, label='Text Confidence', color='green')
        axes[1, 0].set_title('Average Confidence by Emotion')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_xticks([i+0.15 for i in x])
        axes[1, 0].set_xticklabels(emotion_analysis['emotion'], rotation=45)
        axes[1, 0].legend()
        
        # Sample counts
        axes[1, 1].bar(x, emotion_analysis['total_cases'], alpha=0.7, color='purple')
        axes[1, 1].set_title('Dataset Distribution')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(emotion_analysis['emotion'], rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Vocal vs Text Performance
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(emotion_analysis['vocal_accuracy'], emotion_analysis['text_accuracy'], 
                   s=emotion_analysis['total_cases']*2, alpha=0.6, c=range(len(emotion_analysis)), cmap='viridis')
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal Performance')
        plt.xlabel('Vocal Accuracy')
        plt.ylabel('Text Accuracy')
        plt.title('Vocal vs Text Accuracy')
        plt.legend()
        
        # Add emotion labels
        for i, row in emotion_analysis.iterrows():
            plt.annotate(row['emotion'], (row['vocal_accuracy'], row['text_accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.subplot(1, 2, 2)
        disagreement_rates = 1 - emotion_analysis['vocal_text_agreement']
        plt.bar(range(len(emotion_analysis)), disagreement_rates, alpha=0.7, color='orange')
        plt.title('Vocal-Text Disagreement Rate')
        plt.ylabel('Disagreement Rate')
        plt.xticks(range(len(emotion_analysis)), emotion_analysis['emotion'], rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "vocal_vs_text_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {viz_dir}")
    
    def generate_comprehensive_report(self, results, emotion_analysis):
        """Generate comprehensive report"""
        print(f"\n📝 Generating comprehensive report...")
        
        report_file = self.results_dir / "comprehensive_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# High Vocal Bias Ensemble Test Report (0.9)\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test Configuration
            f.write("## Test Configuration\n\n")
            config = results['test_configuration']
            f.write(f"- **Vocal Bias:** {config['vocal_bias']} (90%)\n")
            f.write(f"- **Text Bias:** {config['text_bias']} (10%)\n")
            f.write(f"- **Whisper Model:** {config['whisper_model']}\n")
            f.write(f"- **Confidence Threshold:** {config['confidence_threshold']}\n")
            f.write(f"- **Total Files Tested:** {config['total_files']}\n")
            f.write(f"- **Success Rate:** {config['success_rate']:.3f}\n\n")
            
            # Overall Performance
            f.write("## Overall Performance\n\n")
            metrics = results['performance_metrics']
            f.write(f"- **Accuracy:** {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)\n")
            f.write(f"- **F1-Score (Weighted):** {metrics['f1_weighted']:.3f}\n")
            f.write(f"- **F1-Score (Macro):** {metrics['f1_macro']:.3f}\n")
            f.write(f"- **Precision (Weighted):** {metrics['precision_weighted']:.3f}\n")
            f.write(f"- **Recall (Weighted):** {metrics['recall_weighted']:.3f}\n\n")
            
            # Vocal Analysis
            f.write("## Vocal Bias Analysis\n\n")
            vocal = results['vocal_analysis']
            f.write(f"- **High Vocal Confidence Cases:** {vocal['high_confidence_cases']} ({vocal['high_confidence_rate']*100:.1f}%)\n")
            f.write(f"- **Vocal-Text Disagreements:** {vocal['vocal_text_disagreements']} ({vocal['disagreement_rate']*100:.1f}%)\n\n")
            
            f.write("### Key Insights:\n")
            f.write(f"1. **High vocal confidence:** {vocal['high_confidence_rate']*100:.1f}% of cases had vocal confidence > 0.8\n")
            f.write(f"2. **Component disagreement:** {vocal['disagreement_rate']*100:.1f}% disagreement rate between vocal and text\n")
            f.write(f"3. **Vocal dominance:** 90% vocal bias strongly prioritizes vocal emotional cues\n\n")
            
            # Per-Emotion Performance
            f.write("## Per-Emotion Performance\n\n")
            f.write("| Emotion | Accuracy | Vocal Acc. | Text Acc. | Vocal Advantage | Samples |\n")
            f.write("|---------|----------|------------|-----------|-----------------|----------|\n")
            
            for _, row in emotion_analysis.iterrows():
                f.write(f"| {row['emotion']} | {row['ensemble_accuracy']:.3f} | "
                       f"{row['vocal_accuracy']:.3f} | {row['text_accuracy']:.3f} | "
                       f"{row['vocal_advantage']:+.3f} | {row['total_cases']} |\n")
            
            f.write("\n")
            
            # Best and Worst Performing Emotions
            best_emotions = emotion_analysis.nlargest(3, 'ensemble_accuracy')['emotion'].tolist()
            worst_emotions = emotion_analysis.nsmallest(3, 'ensemble_accuracy')['emotion'].tolist()
            most_vocal_advantage = emotion_analysis.nlargest(3, 'vocal_advantage')['emotion'].tolist()
            
            f.write("### Performance Highlights\n\n")
            f.write(f"**Best Performing Emotions:** {', '.join(best_emotions)}\n")
            f.write(f"**Most Challenging Emotions:** {', '.join(worst_emotions)}\n")
            f.write(f"**Highest Vocal Advantage:** {', '.join(most_vocal_advantage)}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### When to Use High Vocal Bias (0.9)\n\n")
            f.write("1. **Detecting Emotional Deception:** When someone's words don't match their vocal tone\n")
            f.write("2. **Sarcasm Detection:** High vocal bias captures vocal patterns that contradict text\n")
            f.write("3. **Mental Health Applications:** Vocal cues often reveal true emotional state\n")
            f.write("4. **Security/Interrogation:** Detecting stress or deception through vocal analysis\n\n")
            
            f.write("### Configuration Optimization\n\n")
            avg_accuracy = metrics['accuracy']
            if avg_accuracy > 0.75:
                f.write("✅ **Excellent Performance** - High vocal bias works very well for this dataset\n")
                f.write("- Consider keeping 0.9 vocal bias for production\n")
                f.write("- Focus on emotions with high vocal advantage\n")
            elif avg_accuracy > 0.65:
                f.write("✅ **Good Performance** - High vocal bias is effective\n")
                f.write("- 0.9 vocal bias is suitable for vocal-priority applications\n")
                f.write("- Consider 0.8 vocal bias for more balanced approach\n")
            else:
                f.write("⚠️ **Moderate Performance** - Consider adjusting vocal bias\n")
                f.write("- Try reducing vocal bias to 0.7-0.8\n")
                f.write("- Investigate text component reliability\n")
            
            f.write(f"\n### Next Steps\n\n")
            f.write("1. **Production Deployment:** Use this configuration for vocal-priority applications\n")
            f.write("2. **Edge Case Analysis:** Investigate failed predictions for improvement opportunities\n")
            f.write("3. **Real-time Testing:** Validate performance on live audio streams\n")
            f.write("4. **Domain Adaptation:** Fine-tune for specific use case domains\n")
        
        print(f"📋 Comprehensive report saved to: {report_file}")

def main():
    """Main testing function"""
    print("🎯 HIGH VOCAL BIAS ENSEMBLE TEST - COMPLETE DATASET")
    print("="*60)
    
    tester = HighVocalBiasTest()
    
    try:
        # Run complete dataset test (use all files)
        results = tester.run_complete_dataset_test(max_files_per_emotion=None)
        
        # Analyze per-emotion performance
        emotion_analysis = tester.analyze_per_emotion_performance(results)
        
        # Create visualizations
        tester.create_visualizations(results, emotion_analysis)
        
        # Generate comprehensive report
        tester.generate_comprehensive_report(results, emotion_analysis)
        
        print(f"\n🎉 COMPLETE TESTING FINISHED!")
        print(f"📁 All results saved to: {tester.results_dir}")
        
        # Final summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"🎯 Overall Accuracy: {results['performance_metrics']['accuracy']:.3f}")
        print(f"🎯 F1-Score: {results['performance_metrics']['f1_weighted']:.3f}")
        print(f"🔊 High Vocal Confidence: {results['vocal_analysis']['high_confidence_rate']*100:.1f}%")
        print(f"⚡ Vocal-Text Disagreements: {results['vocal_analysis']['disagreement_rate']*100:.1f}%")
        print(f"📈 Files Processed: {results['test_configuration']['total_files']}")
        print(f"✅ Success Rate: {results['test_configuration']['success_rate']*100:.1f}%")
        
        # Show best performing emotions
        best_emotions = emotion_analysis.nlargest(3, 'ensemble_accuracy')
        print(f"\n🏆 Best Performing Emotions:")
        for _, row in best_emotions.iterrows():
            print(f"   {row['emotion']}: {row['ensemble_accuracy']:.3f} accuracy")
        
        print(f"\n🚀 High vocal bias (0.9) testing complete!")
        print(f"📊 Results demonstrate strong vocal emotion prioritization")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
