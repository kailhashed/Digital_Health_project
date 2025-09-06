"""
Ensemble Audio Sentiment Analysis Model
Combines Speech-to-Text, BERT, and ResNet for comprehensive emotion detection
with bias towards vocal emotion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")

# STT and NLP imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available for BERT component")

# Local imports
from .custom_models import EmotionResNet
from ..utils.config import Config


class SpeechToTextModule:
    """Speech-to-Text component using Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize STT module
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        
        if WHISPER_AVAILABLE:
            try:
                print(f"Loading Whisper {model_size} model...")
                self.model = whisper.load_model(model_size)
                print(f"✓ Whisper {model_size} model loaded successfully")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                self.model = None
        else:
            print("Whisper not available - STT component disabled")
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        Args:
            audio_path: Path to audio file
        Returns:
            Transcribed text
        """
        if self.model is None:
            return ""
        
        try:
            # Convert path to absolute path
            audio_path = str(Path(audio_path).resolve())
            
            # Check if file exists
            if not Path(audio_path).exists():
                print(f"STT Error: Audio file not found: {audio_path}")
                return ""
            
            # Use librosa to load audio first (bypasses ffmpeg issues on Windows)
            try:
                import librosa
                # Load audio with librosa at 16kHz (Whisper's expected sample rate)
                audio_data, sr = librosa.load(audio_path, sr=16000)
                
                # Ensure audio is not empty
                if len(audio_data) == 0:
                    print(f"STT Error: Empty audio file: {audio_path}")
                    return ""
                
                # Transcribe using the audio array (no ffmpeg needed)
                result = self.model.transcribe(audio_data)
                text = result["text"].strip()
                print(f"STT Transcription: '{text}'")
                return text
                
            except Exception as librosa_error:
                print(f"STT Librosa Error: {librosa_error}")
                
                # Fallback: try direct file transcription
                try:
                    result = self.model.transcribe(audio_path)
                    text = result["text"].strip()
                    print(f"STT Transcription (direct): '{text}'")
                    return text
                except Exception as whisper_error:
                    print(f"STT Whisper Error: {whisper_error}")
                    return ""
                    
        except Exception as e:
            print(f"STT Critical Error: {e}")
            return ""


class BERTEmotionClassifier:
    """BERT-based text emotion classification"""
    
    def __init__(self):
        """Initialize BERT emotion classifier"""
        self.model = None
        self.tokenizer = None
        self.emotion_mapping = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a pre-trained emotion classification model
                model_name = "j-hartmann/emotion-english-distilroberta-base"
                print(f"Loading BERT emotion model: {model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                
                # Model's emotion labels - map to our 8 classes
                bert_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
                our_emotions = Config.EMOTION_CLASSES  # ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
                
                # Create mapping from BERT emotions to our emotions
                self.emotion_mapping = {
                    'anger': 'angry',
                    'disgust': 'disgust', 
                    'fear': 'fearful',
                    'joy': 'happy',
                    'neutral': 'neutral',
                    'sadness': 'sad',
                    'surprise': 'surprised'
                }
                
                print("✓ BERT emotion classifier loaded successfully")
                
            except Exception as e:
                print(f"Error loading BERT model: {e}")
                self.model = None
        else:
            print("Transformers not available - BERT component disabled")
    
    def predict_emotion_probabilities(self, text: str) -> np.ndarray:
        """
        Predict emotion probabilities from text
        Args:
            text: Input text
        Returns:
            Probability distribution over our 8 emotion classes
        """
        if self.model is None or not text.strip():
            # Return uniform distribution if no model or empty text
            return np.ones(len(Config.EMOTION_CLASSES)) / len(Config.EMOTION_CLASSES)
        
        try:
            # Tokenize and predict
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1).numpy()[0]
            
            # Map BERT emotions to our emotion classes
            our_probs = np.zeros(len(Config.EMOTION_CLASSES))
            bert_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            
            for i, bert_emotion in enumerate(bert_emotions):
                if bert_emotion in self.emotion_mapping:
                    our_emotion = self.emotion_mapping[bert_emotion]
                    if our_emotion in Config.EMOTION_CLASSES:
                        our_idx = Config.EMOTION_CLASSES.index(our_emotion)
                        our_probs[our_idx] = probabilities[i]
            
            # Handle 'calm' - not directly mapped, use low arousal from neutral
            calm_idx = Config.EMOTION_CLASSES.index('calm')
            neutral_idx = Config.EMOTION_CLASSES.index('neutral')
            our_probs[calm_idx] = our_probs[neutral_idx] * 0.3  # Some probability for calm
            
            # Normalize to sum to 1
            our_probs = our_probs / (our_probs.sum() + 1e-8)
            
            predicted_emotion = Config.EMOTION_CLASSES[np.argmax(our_probs)]
            confidence = np.max(our_probs)
            print(f"BERT Prediction: {predicted_emotion} (confidence: {confidence:.3f})")
            
            return our_probs
            
        except Exception as e:
            print(f"BERT prediction error: {e}")
            # Return uniform distribution on error
            return np.ones(len(Config.EMOTION_CLASSES)) / len(Config.EMOTION_CLASSES)


class VocalEmotionClassifier:
    """ResNet-based vocal emotion classification"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize vocal emotion classifier
        Args:
            model_path: Path to trained ResNet model
        """
        self.device = Config.DEVICE
        self.model = None
        
        # Default model path
        if model_path is None:
            model_path = "models/resnet/best_ResNet.pth"
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load trained ResNet model"""
        try:
            # Initialize ResNet model
            self.model = EmotionResNet(num_classes=Config.NUM_CLASSES)
            
            # Load trained weights
            if Path(self.model_path).exists():
                print(f"Loading ResNet model from: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                print("✓ ResNet vocal emotion model loaded successfully")
                
            else:
                print(f"Warning: ResNet model not found at {self.model_path}")
                print("Using randomly initialized model")
                self.model = self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            print(f"Error loading ResNet model: {e}")
            # Initialize with random weights as fallback
            self.model = EmotionResNet(num_classes=Config.NUM_CLASSES).to(self.device)
            self.model.eval()
    
    def predict_emotion_probabilities(self, audio_path: str) -> np.ndarray:
        """
        Predict emotion probabilities from audio
        Args:
            audio_path: Path to audio file
        Returns:
            Probability distribution over emotion classes
        """
        if self.model is None:
            return np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES
        
        try:
            # Load and preprocess audio
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
                outputs = self.model(mel_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            predicted_emotion = Config.EMOTION_CLASSES[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            print(f"ResNet Prediction: {predicted_emotion} (confidence: {confidence:.3f})")
            
            return probabilities
            
        except Exception as e:
            print(f"Vocal emotion prediction error: {e}")
            return np.ones(Config.NUM_CLASSES) / Config.NUM_CLASSES


class EnsembleAudioSentimentAnalyzer:
    """
    Ensemble model combining STT, BERT, and ResNet
    with bias towards vocal emotion classification
    """
    
    def __init__(self, 
                 resnet_model_path: str = None,
                 whisper_model_size: str = "base",
                 vocal_weight: float = 0.6,
                 text_weight: float = 0.4,
                 confidence_threshold: float = 0.7):
        """
        Initialize ensemble model
        Args:
            resnet_model_path: Path to trained ResNet model
            whisper_model_size: Whisper model size for STT
            vocal_weight: Weight for vocal emotion component (bias towards vocal)
            text_weight: Weight for text emotion component
            confidence_threshold: Threshold for high-confidence vocal predictions
        """
        print("Initializing Ensemble Audio Sentiment Analyzer...")
        print(f"Configuration: vocal_weight={vocal_weight}, text_weight={text_weight}")
        
        self.vocal_weight = vocal_weight
        self.text_weight = text_weight
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.stt_module = SpeechToTextModule(whisper_model_size)
        self.bert_classifier = BERTEmotionClassifier()
        self.vocal_classifier = VocalEmotionClassifier(resnet_model_path)
        
        # Ensure weights sum to 1
        total_weight = self.vocal_weight + self.text_weight
        self.vocal_weight /= total_weight
        self.text_weight /= total_weight
        
        print(f"✓ Ensemble model initialized successfully")
        print(f"Final weights: vocal={self.vocal_weight:.2f}, text={self.text_weight:.2f}")
    
    def predict_emotion(self, audio_path: str, return_details: bool = False) -> Dict:
        """
        Predict emotion from audio file using ensemble approach
        Args:
            audio_path: Path to audio file
            return_details: Whether to return detailed component predictions
        Returns:
            Dictionary with prediction results
        """
        print(f"\n🎵 Analyzing audio: {Path(audio_path).name}")
        print("-" * 50)
        
        # Step 1: Get vocal emotion prediction (ResNet)
        print("1. Analyzing vocal characteristics...")
        vocal_probs = self.vocal_classifier.predict_emotion_probabilities(audio_path)
        vocal_confidence = np.max(vocal_probs)
        vocal_emotion = Config.EMOTION_CLASSES[np.argmax(vocal_probs)]
        
        # Step 2: Get text transcription (STT)
        print("\n2. Transcribing speech to text...")
        transcribed_text = self.stt_module.transcribe(audio_path)
        
        # Step 3: Get text emotion prediction (BERT)
        print("\n3. Analyzing text sentiment...")
        if transcribed_text.strip():
            text_probs = self.bert_classifier.predict_emotion_probabilities(transcribed_text)
            text_confidence = np.max(text_probs)
            text_emotion = Config.EMOTION_CLASSES[np.argmax(text_probs)]
        else:
            print("No text transcribed - using neutral text emotion")
            text_probs = np.zeros(Config.NUM_CLASSES)
            text_probs[Config.EMOTION_CLASSES.index('neutral')] = 1.0
            text_confidence = 1.0
            text_emotion = 'neutral'
        
        # Step 4: Ensemble prediction with vocal bias
        print("\n4. Combining predictions with vocal bias...")
        
        # Apply vocal bias logic
        if vocal_confidence >= self.confidence_threshold:
            # High confidence vocal prediction - increase vocal weight
            adaptive_vocal_weight = min(0.85, self.vocal_weight + 0.2)
            adaptive_text_weight = 1.0 - adaptive_vocal_weight
            print(f"High vocal confidence detected - increasing vocal weight to {adaptive_vocal_weight:.2f}")
        else:
            adaptive_vocal_weight = self.vocal_weight
            adaptive_text_weight = self.text_weight
        
        # Weighted ensemble
        ensemble_probs = (adaptive_vocal_weight * vocal_probs + 
                         adaptive_text_weight * text_probs)
        
        # Final prediction
        final_emotion = Config.EMOTION_CLASSES[np.argmax(ensemble_probs)]
        final_confidence = np.max(ensemble_probs)
        
        # Results
        results = {
            'predicted_emotion': final_emotion,
            'confidence': float(final_confidence),
            'transcribed_text': transcribed_text,
            'probabilities': {
                emotion: float(prob) for emotion, prob 
                in zip(Config.EMOTION_CLASSES, ensemble_probs)
            }
        }
        
        if return_details:
            results['component_predictions'] = {
                'vocal': {
                    'emotion': vocal_emotion,
                    'confidence': float(vocal_confidence),
                    'probabilities': vocal_probs.tolist()
                },
                'text': {
                    'emotion': text_emotion,
                    'confidence': float(text_confidence),
                    'probabilities': text_probs.tolist()
                }
            }
            results['weights_used'] = {
                'vocal_weight': adaptive_vocal_weight,
                'text_weight': adaptive_text_weight
            }
        
        # Print results
        print(f"\n📊 ENSEMBLE RESULTS")
        print(f"Predicted Emotion: {final_emotion}")
        print(f"Confidence: {final_confidence:.3f}")
        print(f"Vocal: {vocal_emotion} ({vocal_confidence:.3f})")
        print(f"Text: {text_emotion} ({text_confidence:.3f})")
        print(f"Transcription: '{transcribed_text}'")
        
        return results
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Predict emotions for multiple audio files
        Args:
            audio_paths: List of audio file paths
        Returns:
            List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_emotion(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'predicted_emotion': 'neutral',
                    'confidence': 0.0,
                    'transcribed_text': '',
                    'probabilities': {emotion: 0.0 for emotion in Config.EMOTION_CLASSES},
                    'error': str(e)
                })
        return results
    
    def update_weights(self, vocal_weight: float, text_weight: float):
        """
        Update ensemble weights
        Args:
            vocal_weight: New vocal weight
            text_weight: New text weight
        """
        total = vocal_weight + text_weight
        self.vocal_weight = vocal_weight / total
        self.text_weight = text_weight / total
        print(f"Updated weights: vocal={self.vocal_weight:.2f}, text={self.text_weight:.2f}")


# Factory function for easy instantiation
def create_ensemble_model(resnet_model_path: str = "Project/models/resnet/best_ResNet.pth",
                         vocal_bias: float = 0.6) -> EnsembleAudioSentimentAnalyzer:
    """
    Create an ensemble model with specified configuration
    Args:
        resnet_model_path: Path to trained ResNet model
        vocal_bias: Bias towards vocal emotion (0.5 = equal, >0.5 = vocal bias)
    Returns:
        Configured ensemble model
    """
    text_weight = 1.0 - vocal_bias
    return EnsembleAudioSentimentAnalyzer(
        resnet_model_path=resnet_model_path,
        vocal_weight=vocal_bias,
        text_weight=text_weight
    )


if __name__ == "__main__":
    # Example usage
    ensemble = create_ensemble_model(vocal_bias=0.7)  # 70% vocal bias
    
    # Test with an audio file
    test_audio = "path/to/test/audio.wav"
    if Path(test_audio).exists():
        result = ensemble.predict_emotion(test_audio, return_details=True)
        print("\nDetailed Results:")
        print(result)
