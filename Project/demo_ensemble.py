#!/usr/bin/env python3
"""
Demo script for Ensemble Audio Emotion Analysis
Shows how to use the ensemble model with different configurations
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.models.ensemble_model import create_ensemble_model
    from src.utils.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the Project directory and have installed requirements")
    sys.exit(1)


def demo_ensemble_analysis():
    """Demonstrate ensemble audio emotion analysis"""
    
    print("🎵 ENSEMBLE AUDIO EMOTION ANALYSIS DEMO 🎵")
    print("="*70)
    
    # Find a sample audio file from the dataset
    sample_audio = None
    data_dir = Path("organized_by_emotion")
    
    if data_dir.exists():
        # Look for a sample audio file
        for emotion_dir in data_dir.iterdir():
            if emotion_dir.is_dir():
                audio_files = list(emotion_dir.glob("*.wav"))
                if audio_files:
                    sample_audio = audio_files[0]
                    print(f"📁 Found sample audio: {sample_audio}")
                    print(f"   Expected emotion: {emotion_dir.name}")
                    break
    
    if sample_audio is None:
        print("❌ No sample audio found in organized_by_emotion directory")
        print("Please ensure you have audio files in the dataset directory")
        return
    
    print("\n" + "="*70)
    print("DEMONSTRATION: Different Vocal Bias Settings")
    print("="*70)
    
    # Test different vocal bias settings
    bias_settings = [
        (0.5, "Equal weight between vocal and text"),
        (0.65, "Moderate vocal bias (recommended)"),
        (0.8, "Strong vocal bias"),
        (0.9, "Very strong vocal bias")
    ]
    
    results = []
    
    for vocal_bias, description in bias_settings:
        print(f"\n🔧 Testing with {description}")
        print(f"   Vocal weight: {vocal_bias:.1f}, Text weight: {1-vocal_bias:.1f}")
        print("-" * 50)
        
        try:
            # Create ensemble model with specific bias
            ensemble = create_ensemble_model(vocal_bias=vocal_bias)
            
            # Analyze the sample audio
            result = ensemble.predict_emotion(str(sample_audio), return_details=True)
            
            results.append({
                'bias': vocal_bias,
                'description': description,
                'result': result
            })
            
        except Exception as e:
            print(f"❌ Error with vocal bias {vocal_bias}: {e}")
            continue
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON OF RESULTS")
    print("="*70)
    
    print(f"{'Vocal Bias':<12} {'Predicted':<12} {'Confidence':<12} {'Vocal Pred':<12} {'Text Pred':<12}")
    print("-" * 60)
    
    for result_data in results:
        result = result_data['result']
        vocal_pred = result['component_predictions']['vocal']['emotion']
        text_pred = result['component_predictions']['text']['emotion']
        
        print(f"{result_data['bias']:<12.1f} "
              f"{result['predicted_emotion']:<12} "
              f"{result['confidence']:<12.3f} "
              f"{vocal_pred:<12} "
              f"{text_pred:<12}")
    
    # Show transcription
    if results:
        transcription = results[0]['result']['transcribed_text']
        print(f"\n📝 Transcribed Text: '{transcription}'")
    
    print("\n" + "="*70)
    print("ANALYSIS INSIGHTS")
    print("="*70)
    
    if len(results) >= 2:
        # Compare different bias settings
        moderate_result = next((r for r in results if r['bias'] == 0.65), None)
        strong_result = next((r for r in results if r['bias'] == 0.8), None)
        
        if moderate_result and strong_result:
            mod_emotion = moderate_result['result']['predicted_emotion']
            strong_emotion = strong_result['result']['predicted_emotion']
            
            if mod_emotion == strong_emotion:
                print("✅ Consistent predictions across different vocal bias settings")
                print(f"   Both moderate (0.65) and strong (0.8) bias predict: {mod_emotion}")
            else:
                print("⚠️  Different predictions with different vocal bias:")
                print(f"   Moderate bias (0.65): {mod_emotion}")
                print(f"   Strong bias (0.8): {strong_emotion}")
                print("   Consider the context - is vocal or text emotion more reliable?")
    
    # Show component analysis
    if results:
        result = results[0]['result']
        vocal_conf = result['component_predictions']['vocal']['confidence']
        text_conf = result['component_predictions']['text']['confidence']
        
        print(f"\n🎯 Component Confidence Analysis:")
        print(f"   Vocal (ResNet) confidence: {vocal_conf:.3f}")
        print(f"   Text (BERT) confidence: {text_conf:.3f}")
        
        if vocal_conf > text_conf:
            print("   → Vocal component is more confident")
            print("   → Recommended: Use higher vocal bias (0.7-0.8)")
        elif text_conf > vocal_conf:
            print("   → Text component is more confident") 
            print("   → Recommended: Use lower vocal bias (0.5-0.6)")
        else:
            print("   → Similar confidence levels")
            print("   → Recommended: Use moderate vocal bias (0.65)")
    
    print("\n" + "="*70)
    print("VOCAL BIAS RECOMMENDATIONS")
    print("="*70)
    print("• 0.5-0.6: When text is clear and emotionally expressive")
    print("• 0.65-0.7: General purpose (recommended default)")
    print("• 0.75-0.85: When vocal cues are strong or text is ambiguous")
    print("• 0.9+: When focusing primarily on vocal emotion patterns")
    print("\nThe model is designed to detect cases where someone says one thing")
    print("but their vocal tone suggests a different emotion!")
    
    print("\n✅ Demo completed!")


def demo_batch_processing():
    """Demonstrate batch processing of multiple audio files"""
    
    print("\n" + "="*70)
    print("BONUS: BATCH PROCESSING DEMO")
    print("="*70)
    
    data_dir = Path("organized_by_emotion")
    if not data_dir.exists():
        print("❌ No organized_by_emotion directory found")
        return
    
    # Find a few sample files from different emotions
    sample_files = []
    for emotion_dir in data_dir.iterdir():
        if emotion_dir.is_dir():
            audio_files = list(emotion_dir.glob("*.wav"))[:2]  # Take 2 files per emotion
            for audio_file in audio_files:
                sample_files.append((audio_file, emotion_dir.name))
        
        if len(sample_files) >= 6:  # Limit to 6 files for demo
            break
    
    if len(sample_files) < 3:
        print("❌ Not enough sample files found for batch demo")
        return
    
    print(f"📁 Found {len(sample_files)} sample files for batch processing")
    
    # Create ensemble model
    ensemble = create_ensemble_model(vocal_bias=0.65)
    
    # Process batch
    audio_paths = [str(path) for path, _ in sample_files]
    batch_results = ensemble.predict_batch(audio_paths)
    
    # Display results
    print(f"\n📊 Batch Processing Results:")
    print("-" * 60)
    print(f"{'File':<25} {'Expected':<12} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 60)
    
    correct_predictions = 0
    for (audio_path, expected_emotion), result in zip(sample_files, batch_results):
        file_name = Path(audio_path).name[:24]  # Truncate long names
        predicted = result['predicted_emotion']
        confidence = result['confidence']
        
        is_correct = predicted == expected_emotion
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{file_name:<25} {expected_emotion:<12} {predicted:<12} {confidence:<12.3f} {status}")
    
    accuracy = (correct_predictions / len(sample_files)) * 100
    print(f"\n📈 Batch Accuracy: {correct_predictions}/{len(sample_files)} ({accuracy:.1f}%)")


if __name__ == "__main__":
    try:
        demo_ensemble_analysis()
        demo_batch_processing()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Please ensure you have:")
        print("1. Installed requirements: pip install -r requirements_ensemble.txt")
        print("2. Downloaded audio dataset in organized_by_emotion directory")
        print("3. Trained ResNet model at models/resnet/best_ResNet.pth")
