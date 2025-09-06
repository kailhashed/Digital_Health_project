#!/usr/bin/env python3
"""
Emotion Prediction Script
Load trained models and predict emotions from audio files.
"""

import torch
import librosa
import numpy as np
import json
import argparse
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class EmotionPredictor:
    """Emotion prediction using trained models"""
    
    def __init__(self, model_path, device='auto'):
        """
        Initialize the emotion predictor
        
        Args:
            model_path: Path to trained model (.pth file)
            device: 'auto', 'cuda', or 'cpu'
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.model = self._load_model()
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self):
        """Load the trained model"""
        try:
            model_state = torch.load(self.model_path, map_location=self.device)
            
            # Determine model type from path
            if 'resnet' in str(self.model_path).lower():
                from custom_models import EmotionResNet
                model = EmotionResNet(num_classes=8)
            elif 'simplecnn' in str(self.model_path).lower():
                from custom_models import SimpleCNN
                model = SimpleCNN(num_classes=8)
            elif 'wav2vec2' in str(self.model_path).lower():
                from pretrained_models import FixedWav2Vec2Classifier
                model = FixedWav2Vec2Classifier(num_classes=8)
            elif 'simplecnnaudio' in str(self.model_path).lower():
                from pretrained_models import SimpleCNNAudio
                model = SimpleCNNAudio(num_classes=8)
            else:
                raise ValueError(f"Unknown model type for path: {self.model_path}")
            
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio file for model input"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, duration=3.0)
            
            # Pad or trim to 3 seconds
            target_length = 16000 * 3
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # For CNN models (ResNet, SimpleCNN), create mel-spectrogram
            if 'resnet' in str(self.model_path).lower() or 'simplecnn' in str(self.model_path).lower():
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, sr=16000, n_mels=64, hop_length=512, n_fft=1024
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize
                mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
                
                # Add batch and channel dimensions
                input_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
                
            # For audio-based models (SimpleCNNAudio, Wav2Vec2), use raw audio
            else:
                input_tensor = torch.FloatTensor(audio).unsqueeze(0)
            
            return input_tensor.to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess audio {audio_path}: {e}")
    
    def predict_file(self, audio_path):
        """
        Predict emotion from a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Prediction results with emotion and confidence
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Preprocess audio
            input_tensor = self._preprocess_audio(audio_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_emotion = self.emotions[predicted_class]
            
            return {
                'file': str(audio_path),
                'emotion': predicted_emotion,
                'confidence': round(confidence, 3),
                'all_probabilities': {
                    emotion: round(prob, 3) 
                    for emotion, prob in zip(self.emotions, probabilities[0].tolist())
                }
            }
            
        except Exception as e:
            return {
                'file': str(audio_path),
                'error': str(e),
                'emotion': None,
                'confidence': 0.0
            }
    
    def predict_directory(self, audio_dir, extensions=('.wav', '.mp3', '.flac', '.m4a')):
        """
        Predict emotions for all audio files in a directory
        
        Args:
            audio_dir: Path to directory containing audio files
            extensions: Tuple of valid audio file extensions
            
        Returns:
            dict: Results for all files
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            raise FileNotFoundError(f"Directory not found: {audio_dir}")
        
        results = {}
        audio_files = []
        
        # Find all audio files
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return results
        
        print(f"Processing {len(audio_files)} audio files...")
        
        for audio_file in audio_files:
            result = self.predict_file(audio_file)
            results[audio_file.name] = result
            
            if 'error' in result:
                print(f"Error processing {audio_file.name}: {result['error']}")
            else:
                print(f"{audio_file.name}: {result['emotion']} (confidence: {result['confidence']})")
        
        return results

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Predict emotions from audio files')
    parser.add_argument('--model', required=True, 
                       help='Path to trained model (.pth file)')
    parser.add_argument('--audio', 
                       help='Path to single audio file')
    parser.add_argument('--audio_dir', 
                       help='Path to directory containing audio files')
    parser.add_argument('--output', 
                       help='Path to save results (JSON format)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Computation device')
    
    args = parser.parse_args()
    
    if not args.audio and not args.audio_dir:
        parser.error("Either --audio or --audio_dir must be specified")
    
    try:
        # Initialize predictor
        print(f"Loading model from {args.model}...")
        predictor = EmotionPredictor(args.model, args.device)
        print(f"Model loaded on {predictor.device}")
        
        results = {}
        
        # Predict single file
        if args.audio:
            print(f"Predicting emotion for {args.audio}...")
            result = predictor.predict_file(args.audio)
            results[Path(args.audio).name] = result
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Predicted emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
        
        # Predict directory
        if args.audio_dir:
            print(f"Predicting emotions for files in {args.audio_dir}...")
            dir_results = predictor.predict_directory(args.audio_dir)
            results.update(dir_results)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Print summary
        successful_predictions = sum(1 for r in results.values() if 'error' not in r)
        total_files = len(results)
        print(f"\\nSummary: {successful_predictions}/{total_files} files processed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())