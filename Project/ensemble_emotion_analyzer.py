#!/usr/bin/env python3
"""
Ensemble Audio Emotion Analyzer
Complete pipeline for audio sentiment analysis using STT + BERT + ResNet ensemble
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

try:
    from src.models.ensemble_model import EnsembleAudioSentimentAnalyzer, create_ensemble_model
    from src.utils.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the Project directory")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Ensemble Audio Emotion Analysis')
    parser.add_argument('audio_path', type=str, 
                       help='Path to audio file or directory containing audio files')
    parser.add_argument('--vocal-bias', type=float, default=0.65,
                       help='Vocal emotion bias (0.5=equal, >0.5=vocal bias, default=0.65)')
    parser.add_argument('--resnet-model', type=str, 
                       default='models/resnet/best_ResNet.pth',
                       help='Path to trained ResNet model')
    parser.add_argument('--whisper-model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size for STT')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--detailed', action='store_true',
                       help='Include detailed component predictions')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of audio files')
    
    args = parser.parse_args()
    
    print("🎵 ENSEMBLE AUDIO EMOTION ANALYZER 🎵")
    print("="*60)
    print(f"Audio Input: {args.audio_path}")
    print(f"Vocal Bias: {args.vocal_bias} (Text: {1-args.vocal_bias})")
    print(f"ResNet Model: {args.resnet_model}")
    print(f"Whisper Model: {args.whisper_model}")
    print("="*60)
    
    # Initialize ensemble model
    try:
        ensemble = EnsembleAudioSentimentAnalyzer(
            resnet_model_path=args.resnet_model,
            whisper_model_size=args.whisper_model,
            vocal_weight=args.vocal_bias,
            text_weight=1.0 - args.vocal_bias
        )
    except Exception as e:
        print(f"Error initializing ensemble model: {e}")
        return
    
    # Process audio
    results = []
    
    if args.batch or Path(args.audio_path).is_dir():
        # Process directory
        audio_dir = Path(args.audio_path)
        audio_files = []
        
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(audio_dir.glob(ext))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            try:
                result = ensemble.predict_emotion(str(audio_file), return_details=args.detailed)
                result['file_path'] = str(audio_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    'file_path': str(audio_file),
                    'predicted_emotion': 'neutral',
                    'confidence': 0.0,
                    'error': str(e)
                })
    
    else:
        # Process single file
        audio_file = Path(args.audio_path)
        if not audio_file.exists():
            print(f"Audio file not found: {audio_file}")
            return
        
        try:
            result = ensemble.predict_emotion(str(audio_file), return_details=args.detailed)
            result['file_path'] = str(audio_file)
            results.append(result)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return
    
    # Display results summary
    print(f"\n📊 SUMMARY RESULTS")
    print("="*60)
    
    if len(results) == 1:
        # Single file results
        result = results[0]
        print(f"File: {Path(result['file_path']).name}")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if 'transcribed_text' in result:
            print(f"Transcribed Text: '{result['transcribed_text']}'")
        
        if args.detailed and 'component_predictions' in result:
            print(f"\nComponent Predictions:")
            vocal = result['component_predictions']['vocal']
            text = result['component_predictions']['text']
            print(f"  Vocal (ResNet): {vocal['emotion']} ({vocal['confidence']:.3f})")
            print(f"  Text (BERT):   {text['emotion']} ({text['confidence']:.3f})")
            
            weights = result['weights_used']
            print(f"  Weights Used: Vocal={weights['vocal_weight']:.2f}, Text={weights['text_weight']:.2f}")
    
    else:
        # Batch results summary
        emotion_counts = {}
        total_confidence = 0
        
        for result in results:
            emotion = result.get('predicted_emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += result.get('confidence', 0)
        
        print(f"Processed {len(results)} files")
        print(f"Average Confidence: {total_confidence/len(results):.3f}")
        print(f"\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"  {emotion}: {count} files ({percentage:.1f}%)")
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    print("\n✅ Analysis complete!")


def quick_analyze(audio_path: str, vocal_bias: float = 0.65) -> Dict:
    """
    Quick analysis function for programmatic use
    Args:
        audio_path: Path to audio file
        vocal_bias: Vocal emotion bias (default 0.65)
    Returns:
        Prediction results dictionary
    """
    ensemble = create_ensemble_model(vocal_bias=vocal_bias)
    return ensemble.predict_emotion(audio_path)


if __name__ == "__main__":
    main()
