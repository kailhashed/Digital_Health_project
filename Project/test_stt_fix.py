#!/usr/bin/env python3
"""
Test STT fix for Windows ffmpeg issues
"""

import os
import sys
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

def test_stt_fix():
    """Test if the STT fix works"""
    print("🧪 Testing STT Fix")
    print("="*30)
    
    # Find a sample audio file
    sample_audio = None
    data_dirs = ["organized_by_emotion", "Dataset"]
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            for ext in ["*.wav", "*.mp3", "*.flac"]:
                audio_files = list(data_path.rglob(ext))
                if audio_files:
                    sample_audio = audio_files[0]
                    break
            if sample_audio:
                break
    
    if sample_audio is None:
        print("❌ No sample audio found")
        return False
    
    print(f"📁 Testing with: {sample_audio}")
    
    try:
        from src.models.ensemble_model import SpeechToTextModule
        
        # Test STT module
        stt = SpeechToTextModule(model_size="tiny")
        
        if stt.model is None:
            print("❌ STT model not loaded")
            return False
        
        # Test transcription
        print("🔄 Testing transcription...")
        text = stt.transcribe(str(sample_audio))
        
        if text:
            print(f"✅ Transcription successful: '{text}'")
            return True
        else:
            print("⚠️ Transcription returned empty text")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stt_fix()
    if success:
        print("\n✅ STT fix successful! Ready to run ensemble test.")
    else:
        print("\n❌ STT fix failed. Check error messages above.")
