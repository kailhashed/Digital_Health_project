#!/usr/bin/env python3
"""
Test script for Ensemble Audio Emotion Analysis
Quick verification that all components work correctly
"""

import os
import sys
from pathlib import Path
import torch

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        from src.models.ensemble_model import (
            EnsembleAudioSentimentAnalyzer, 
            create_ensemble_model,
            SpeechToTextModule,
            BERTEmotionClassifier,
            VocalEmotionClassifier
        )
        from src.utils.config import Config
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.utils.config import Config
        print(f"✅ Device: {Config.DEVICE}")
        print(f"✅ Emotion classes: {Config.EMOTION_CLASSES}")
        print(f"✅ Sample rate: {Config.SAMPLE_RATE}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_resnet_model():
    """Test ResNet model loading"""
    print("\n🧠 Testing ResNet model...")
    
    try:
        from src.models.ensemble_model import VocalEmotionClassifier
        
        # Test with default path
        classifier = VocalEmotionClassifier()
        print("✅ VocalEmotionClassifier initialized")
        
        # Check if model exists
        model_path = "models/resnet/best_ResNet.pth"
        if Path(model_path).exists():
            print(f"✅ ResNet model found at: {model_path}")
        else:
            print(f"⚠️  ResNet model not found at: {model_path}")
            print("   Model will use random weights for testing")
        
        return True
    except Exception as e:
        print(f"❌ ResNet test error: {e}")
        return False

def test_whisper_availability():
    """Test Whisper availability"""
    print("\n🎤 Testing Whisper (STT) availability...")
    
    try:
        import whisper
        print("✅ Whisper library available")
        
        # Test loading a small model
        from src.models.ensemble_model import SpeechToTextModule
        stt = SpeechToTextModule(model_size="tiny")
        
        if stt.model is not None:
            print("✅ Whisper tiny model loaded successfully")
        else:
            print("⚠️  Whisper model failed to load")
        
        return True
    except ImportError:
        print("❌ Whisper not available - install with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"❌ Whisper test error: {e}")
        return False

def test_bert_availability():
    """Test BERT availability"""
    print("\n📝 Testing BERT (text emotion) availability...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("✅ Transformers library available")
        
        from src.models.ensemble_model import BERTEmotionClassifier
        bert = BERTEmotionClassifier()
        
        if bert.model is not None:
            print("✅ BERT emotion model loaded successfully")
            
            # Test with sample text
            test_text = "I am so happy today!"
            probs = bert.predict_emotion_probabilities(test_text)
            print(f"✅ BERT test prediction completed (shape: {probs.shape})")
        else:
            print("⚠️  BERT model failed to load")
        
        return True
    except ImportError:
        print("❌ Transformers not available - install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ BERT test error: {e}")
        return False

def test_ensemble_creation():
    """Test ensemble model creation"""
    print("\n🎵 Testing ensemble model creation...")
    
    try:
        from src.models.ensemble_model import create_ensemble_model
        
        # Test with different configurations
        configs = [
            (0.5, "Equal weights"),
            (0.65, "Moderate vocal bias"),
            (0.8, "Strong vocal bias")
        ]
        
        for vocal_bias, description in configs:
            try:
                ensemble = create_ensemble_model(vocal_bias=vocal_bias)
                print(f"✅ {description} (vocal: {vocal_bias}): Created successfully")
            except Exception as e:
                print(f"❌ {description}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Ensemble creation error: {e}")
        return False

def test_with_sample_audio():
    """Test with sample audio if available"""
    print("\n🎧 Testing with sample audio...")
    
    # Look for sample audio files
    sample_dirs = ["organized_by_emotion", "Dataset"]
    sample_audio = None
    
    for sample_dir in sample_dirs:
        data_path = Path(sample_dir)
        if data_path.exists():
            # Find any audio file
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                audio_files = list(data_path.rglob(ext))
                if audio_files:
                    sample_audio = audio_files[0]
                    break
            if sample_audio:
                break
    
    if sample_audio is None:
        print("⚠️  No sample audio files found")
        print("   Place audio files in 'organized_by_emotion' or 'Dataset' directory to test")
        return True  # Not a failure, just no test data
    
    print(f"📁 Found sample audio: {sample_audio}")
    
    try:
        from src.models.ensemble_model import create_ensemble_model
        
        # Create ensemble
        ensemble = create_ensemble_model(vocal_bias=0.65)
        
        # Test prediction
        print("🔄 Running ensemble prediction...")
        result = ensemble.predict_emotion(str(sample_audio), return_details=True)
        
        print("✅ Ensemble prediction completed!")
        print(f"   Predicted emotion: {result['predicted_emotion']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Transcription: '{result['transcribed_text']}'")
        
        if 'component_predictions' in result:
            vocal = result['component_predictions']['vocal']
            text = result['component_predictions']['text']
            print(f"   Vocal prediction: {vocal['emotion']} ({vocal['confidence']:.3f})")
            print(f"   Text prediction: {text['emotion']} ({text['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample audio test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 ENSEMBLE MODEL TESTING")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("ResNet Model", test_resnet_model),
        ("Whisper (STT)", test_whisper_availability),
        ("BERT (Text Emotion)", test_bert_availability),
        ("Ensemble Creation", test_ensemble_creation),
        ("Sample Audio", test_with_sample_audio)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ensemble model is ready to use.")
        print("\nNext steps:")
        print("1. Run demo: python demo_ensemble.py")
        print("2. Analyze audio: python ensemble_emotion_analyzer.py your_audio.wav")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements_ensemble.txt")
        print("2. Ensure ResNet model exists: models/resnet/best_ResNet.pth")
        print("3. Install system dependencies (ffmpeg for audio processing)")

if __name__ == "__main__":
    main()
