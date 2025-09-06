#!/usr/bin/env python3
"""
Script for emotion prediction using trained models
Uses the organized codebase structure
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

from models import EmotionTransformer, EmotionLSTM, EmotionResNet
from models import FixedWav2Vec2Classifier, SimpleCNNAudioClassifier
from data import AudioPreprocessor, SimpleLabelEncoder
from utils import Config, setup_logger


class EmotionPredictor:
    """Emotion prediction using trained models"""
    
    def __init__(self, model_path, model_type='custom'):
        """
        Args:
            model_path: Path to trained model
            model_type: Type of model ('custom' or 'pretrained')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = Config.DEVICE
        self.preprocessor = AudioPreprocessor()
        
        # Load model
        self.model, self.model_info = self._load_model()
        self.model.eval()
        
        # Setup label encoder
        self.label_encoder = SimpleLabelEncoder()
        self.label_encoder.classes_ = np.array(Config.EMOTION_CLASSES)
        self.label_encoder.class_to_idx = {label: idx for idx, label in enumerate(Config.EMOTION_CLASSES)}
        
        print(f"✓ Model loaded: {self.model_info.get('model_class', 'Unknown')}")
        print(f"✓ Classes: {Config.EMOTION_CLASSES}")
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model_class_name = checkpoint.get('model_class', '')
            
            # Determine model class
            model_classes = {
                'EmotionTransformer': EmotionTransformer,
                'EmotionLSTM': EmotionLSTM,
                'EmotionResNet': EmotionResNet,
                'FixedWav2Vec2Classifier': FixedWav2Vec2Classifier,
                'SimpleCNNAudioClassifier': SimpleCNNAudioClassifier
            }
            
            if model_class_name not in model_classes:
                raise ValueError(f"Unknown model class: {model_class_name}")
            
            # Initialize model
            model_class = model_classes[model_class_name]
            model = model_class(num_classes=Config.NUM_CLASSES)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            return model, checkpoint
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict_audio_file(self, audio_path):
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio
            if 'CNN' in self.model_info.get('model_class', '') or 'Wav2Vec2' in self.model_info.get('model_class', ''):
                # Raw audio for CNN and Wav2Vec2 models
                audio = self.preprocessor.load_and_preprocess(audio_path)
                input_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            else:
                # Spectrogram for other models
                audio = self.preprocessor.load_and_preprocess(audio_path)
                mel_spec = self.preprocessor.extract_mel_spectrogram(audio)
                input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_idx].item()
            
            # Get predicted emotion
            predicted_emotion = self.label_encoder.classes_[predicted_idx]
            
            # Get all probabilities
            all_probs = {
                emotion: probabilities[0, idx].item()
                for idx, emotion in enumerate(self.label_encoder.classes_)
            }
            
            return {
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'audio_file': audio_path
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, audio_paths):
        """
        Predict emotions for multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(audio_paths)} audio files...")
        
        for audio_path in audio_paths:
            try:
                result = self.predict_audio_file(audio_path)
                results.append(result)
                print(f"✓ {os.path.basename(audio_path)}: {result['predicted_emotion']} ({result['confidence']:.3f})")
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
                continue
        
        return results


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description="Emotion Recognition Prediction")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--audio', help='Single audio file to predict')
    parser.add_argument('--audio_dir', help='Directory containing audio files')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--model_type', choices=['custom', 'pretrained'], default='custom', 
                       help='Type of model')
    
    args = parser.parse_args()
    
    # Setup
    Config.create_directories()
    logger = setup_logger("emotion_prediction")
    
    print("🎭 EMOTION RECOGNITION PREDICTION")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Device: {Config.DEVICE}")
    
    try:
        # Initialize predictor
        predictor = EmotionPredictor(args.model, args.model_type)
        
        # Collect audio files
        audio_files = []
        
        if args.audio:
            if os.path.exists(args.audio):
                audio_files.append(args.audio)
            else:
                print(f"❌ Audio file not found: {args.audio}")
                return False
        
        if args.audio_dir:
            if os.path.exists(args.audio_dir):
                for file in os.listdir(args.audio_dir):
                    if file.lower().endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(args.audio_dir, file))
            else:
                print(f"❌ Audio directory not found: {args.audio_dir}")
                return False
        
        if not audio_files:
            print("❌ No audio files specified")
            return False
        
        print(f"\n🎵 Processing {len(audio_files)} audio files...")
        
        # Make predictions
        results = predictor.predict_batch(audio_files)
        
        # Display results
        print(f"\n📊 PREDICTION RESULTS")
        print("="*50)
        
        emotion_counts = {}
        for result in results:
            emotion = result['predicted_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print(f"\n🎯 {os.path.basename(result['audio_file'])}")
            print(f"   Predicted: {emotion} ({result['confidence']:.3f})")
            
            # Show top 3 emotions
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            print("   Top 3:")
            for emo, prob in sorted_probs:
                print(f"     {emo}: {prob:.3f}")
        
        # Summary
        print(f"\n📈 SUMMARY")
        print("="*30)
        total_files = len(results)
        print(f"Files processed: {total_files}")
        
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / total_files) * 100
            print(f"{emotion}: {count} ({percentage:.1f}%)")
        
        # Save results if requested
        if args.output:
            import json
            
            # Convert numpy types to Python types for JSON
            json_results = []
            for result in results:
                json_result = {
                    'audio_file': result['audio_file'],
                    'predicted_emotion': result['predicted_emotion'],
                    'confidence': float(result['confidence']),
                    'all_probabilities': {k: float(v) for k, v in result['all_probabilities'].items()}
                }
                json_results.append(json_result)
            
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n💾 Results saved to: {args.output}")
        
        logger.info(f"Emotion prediction completed for {len(results)} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        logger.error(f"Prediction failed: {e}")
        return False


if __name__ == "__main__":
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    success = main()
    if success:
        print("\n✨ Prediction completed successfully!")
    else:
        print("\n💥 Prediction failed!")

