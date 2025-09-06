#!/usr/bin/env python3
"""
Example Usage of Ensemble Audio Emotion Analysis
Demonstrates various use cases and configurations
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.models.ensemble_model import create_ensemble_model, EnsembleAudioSentimentAnalyzer
from src.utils.config import Config

def example_1_basic_usage():
    """Example 1: Basic emotion analysis with default settings"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage with Default Settings")
    print("=" * 60)
    
    # Create ensemble model with moderate vocal bias (recommended)
    ensemble = create_ensemble_model(vocal_bias=0.65)
    
    # Find a sample audio file
    sample_audio = find_sample_audio()
    if not sample_audio:
        print("❌ No sample audio found")
        return
    
    print(f"Analyzing: {sample_audio.name}")
    result = ensemble.predict_emotion(str(sample_audio))
    
    print(f"\n📊 Results:")
    print(f"Predicted Emotion: {result['predicted_emotion']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Transcription: '{result['transcribed_text']}'")

def example_2_high_vocal_bias():
    """Example 2: High vocal bias for detecting vocal-text mismatch"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: High Vocal Bias (Detect Sarcasm/Deception)")
    print("=" * 60)
    
    # Create ensemble with high vocal bias
    ensemble = create_ensemble_model(vocal_bias=0.85)
    
    sample_audio = find_sample_audio()
    if not sample_audio:
        return
    
    print(f"Analyzing with 85% vocal bias: {sample_audio.name}")
    result = ensemble.predict_emotion(str(sample_audio), return_details=True)
    
    print(f"\n📊 Detailed Results:")
    print(f"Final Prediction: {result['predicted_emotion']} ({result['confidence']:.3f})")
    print(f"Transcription: '{result['transcribed_text']}'")
    
    if 'component_predictions' in result:
        vocal = result['component_predictions']['vocal']
        text = result['component_predictions']['text']
        print(f"\nComponent Analysis:")
        print(f"  Vocal Emotion: {vocal['emotion']} (confidence: {vocal['confidence']:.3f})")
        print(f"  Text Emotion:  {text['emotion']} (confidence: {text['confidence']:.3f})")
        
        # Check for vocal-text mismatch
        if vocal['emotion'] != text['emotion'] and text['emotion'] != 'neutral':
            print(f"\n⚠️  VOCAL-TEXT MISMATCH DETECTED!")
            print(f"   Person said emotion: {text['emotion']}")
            print(f"   But vocal tone suggests: {vocal['emotion']}")
            print(f"   → High vocal bias correctly prioritizes vocal cues")

def example_3_balanced_approach():
    """Example 3: Balanced approach for clear emotional speech"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Balanced Approach (Equal Weight)")
    print("=" * 60)
    
    # Create ensemble with equal weights
    ensemble = create_ensemble_model(vocal_bias=0.5)
    
    sample_audio = find_sample_audio()
    if not sample_audio:
        return
    
    print(f"Analyzing with equal weights: {sample_audio.name}")
    result = ensemble.predict_emotion(str(sample_audio), return_details=True)
    
    print(f"\n📊 Balanced Analysis:")
    print(f"Prediction: {result['predicted_emotion']} ({result['confidence']:.3f})")
    
    if 'component_predictions' in result:
        vocal = result['component_predictions']['vocal']
        text = result['component_predictions']['text']
        
        print(f"Vocal: {vocal['emotion']} ({vocal['confidence']:.3f})")
        print(f"Text:  {text['emotion']} ({text['confidence']:.3f})")
        
        # Analysis of agreement
        if vocal['emotion'] == text['emotion']:
            print(f"✅ Vocal and text emotions agree - high confidence result")
        else:
            print(f"⚖️  Vocal and text emotions differ - balanced weighting applied")

def example_4_batch_processing():
    """Example 4: Batch processing multiple files"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 60)
    
    # Find multiple sample files
    sample_files = find_multiple_audio_files(max_files=3)
    if len(sample_files) < 2:
        print("❌ Not enough sample files for batch demo")
        return
    
    print(f"Processing {len(sample_files)} audio files...")
    
    # Create ensemble
    ensemble = create_ensemble_model(vocal_bias=0.7)
    
    # Process batch
    file_paths = [str(f) for f in sample_files]
    results = ensemble.predict_batch(file_paths)
    
    print(f"\n📊 Batch Results:")
    print("-" * 50)
    
    for i, (file_path, result) in enumerate(zip(sample_files, results), 1):
        print(f"{i}. {Path(file_path).name[:30]}")
        print(f"   Emotion: {result['predicted_emotion']} ({result['confidence']:.3f})")
        if result['transcribed_text']:
            print(f"   Text: '{result['transcribed_text'][:50]}...'")

def example_5_custom_configuration():
    """Example 5: Custom ensemble configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 60)
    
    # Create custom ensemble
    ensemble = EnsembleAudioSentimentAnalyzer(
        resnet_model_path="models/resnet/best_ResNet.pth",
        whisper_model_size="small",  # Better accuracy than base
        vocal_weight=0.75,           # Strong vocal bias
        text_weight=0.25,
        confidence_threshold=0.8     # Higher threshold for adaptive weighting
    )
    
    sample_audio = find_sample_audio()
    if not sample_audio:
        return
    
    print(f"Custom analysis: {sample_audio.name}")
    print("Configuration:")
    print("- Whisper: small model (better accuracy)")
    print("- Vocal bias: 75%")
    print("- Confidence threshold: 0.8")
    
    result = ensemble.predict_emotion(str(sample_audio), return_details=True)
    
    print(f"\n📊 Custom Results:")
    print(f"Prediction: {result['predicted_emotion']} ({result['confidence']:.3f})")
    
    if 'weights_used' in result:
        weights = result['weights_used']
        print(f"Weights used: Vocal={weights['vocal_weight']:.2f}, Text={weights['text_weight']:.2f}")

def example_6_adaptive_weighting_demo():
    """Example 6: Demonstrate adaptive weighting"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Adaptive Weighting Demonstration")
    print("=" * 60)
    
    # Test with different confidence scenarios
    ensemble = create_ensemble_model(vocal_bias=0.65)
    
    sample_audio = find_sample_audio()
    if not sample_audio:
        return
    
    print("Testing adaptive weighting logic...")
    print("(Weights increase when vocal confidence > threshold)")
    
    result = ensemble.predict_emotion(str(sample_audio), return_details=True)
    
    if 'component_predictions' in result and 'weights_used' in result:
        vocal_conf = result['component_predictions']['vocal']['confidence']
        weights = result['weights_used']
        
        print(f"\nVocal confidence: {vocal_conf:.3f}")
        print(f"Confidence threshold: {ensemble.confidence_threshold}")
        print(f"Weights used: Vocal={weights['vocal_weight']:.2f}, Text={weights['text_weight']:.2f}")
        
        if vocal_conf >= ensemble.confidence_threshold:
            print("✅ High vocal confidence → Increased vocal weight")
        else:
            print("⚖️  Standard confidence → Default weights")

def find_sample_audio():
    """Find a sample audio file"""
    data_dirs = ["organized_by_emotion", "Dataset"]
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                audio_files = list(data_path.rglob(ext))
                if audio_files:
                    return audio_files[0]
    return None

def find_multiple_audio_files(max_files=5):
    """Find multiple sample audio files"""
    files = []
    data_dirs = ["organized_by_emotion", "Dataset"]
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                audio_files = list(data_path.rglob(ext))
                files.extend(audio_files[:max_files])
                if len(files) >= max_files:
                    return files[:max_files]
    return files

def main():
    """Run all examples"""
    print("🎵 ENSEMBLE AUDIO EMOTION ANALYSIS - EXAMPLES")
    print("="*70)
    print("This script demonstrates various use cases and configurations")
    print("of the ensemble audio emotion analysis system.")
    print("="*70)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_high_vocal_bias()
        example_3_balanced_approach()
        example_4_batch_processing()
        example_5_custom_configuration()
        example_6_adaptive_weighting_demo()
        
        print("\n" + "="*70)
        print("🎯 KEY TAKEAWAYS")
        print("="*70)
        print("1. Use vocal_bias=0.65-0.7 for general purpose analysis")
        print("2. Use vocal_bias=0.8+ to detect sarcasm/vocal-text mismatch")
        print("3. Use vocal_bias=0.5 when text is clear and emotionally rich")
        print("4. The system automatically adapts weights for high-confidence vocal predictions")
        print("5. Batch processing is efficient for analyzing multiple files")
        print("6. Custom configurations allow fine-tuning for specific use cases")
        
        print(f"\n🔧 CONFIGURATION RECOMMENDATIONS")
        print("="*70)
        print("Mental Health/Therapy: vocal_bias=0.8, small whisper model")
        print("Customer Service: vocal_bias=0.6, base whisper model")  
        print("Social Media Analysis: vocal_bias=0.5, tiny whisper model")
        print("Security/Deception Detection: vocal_bias=0.9, medium whisper model")
        
        print(f"\n✅ Examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Ensure you have:")
        print("1. Installed requirements: pip install -r requirements_ensemble.txt")
        print("2. Audio files in organized_by_emotion directory")
        print("3. Trained ResNet model at models/resnet/best_ResNet.pth")

if __name__ == "__main__":
    main()
