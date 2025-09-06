"""
Configuration management for emotion recognition
"""

import os
import torch


class Config:
    """Configuration class for emotion recognition project"""
    
    # Data settings
    DATA_DIR = "organized_by_emotion"
    SAMPLE_RATE = 16000
    DURATION = 3.0
    
    # Model settings
    NUM_CLASSES = 8
    EMOTION_CLASSES = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Training settings
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8
    
    # Pre-trained model settings
    PRETRAINED_BATCH_SIZE = 8
    PRETRAINED_EPOCHS = 15
    PRETRAINED_LEARNING_RATE = 0.0001
    PRETRAINED_PATIENCE = 5
    
    # Data split settings
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Device settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Directory settings
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"
    
    # Audio feature settings
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 256
    FMAX = 8000
    
    # Transformer settings
    D_MODEL = 128
    N_HEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    
    # LSTM settings
    HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    BIDIRECTIONAL = True
    
    # ResNet settings
    RESNET_CHANNELS = [16, 32, 64]
    RESNET_BLOCKS = [2, 2, 2]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_save_path(cls, model_name, model_type='custom'):
        """Get save path for model"""
        if model_type == 'pretrained':
            model_dir = f"pretrained_{model_name.lower()}"
        else:
            model_dir = model_name.lower()
        
        return os.path.join(cls.MODELS_DIR, model_dir, f"best_{model_name}.pth")
    
    @classmethod
    def get_results_path(cls, filename):
        """Get path for results file"""
        return os.path.join(cls.RESULTS_DIR, filename)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("EMOTION RECOGNITION CONFIGURATION")
        print("="*60)
        
        print(f"\n📊 Data Settings:")
        print(f"   Data Directory: {cls.DATA_DIR}")
        print(f"   Sample Rate: {cls.SAMPLE_RATE} Hz")
        print(f"   Duration: {cls.DURATION} seconds")
        print(f"   Number of Classes: {cls.NUM_CLASSES}")
        
        print(f"\n🧠 Model Settings:")
        print(f"   Device: {cls.DEVICE}")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Learning Rate: {cls.LEARNING_RATE}")
        print(f"   Epochs: {cls.EPOCHS}")
        
        print(f"\n📁 Directory Settings:")
        print(f"   Models: {cls.MODELS_DIR}")
        print(f"   Results: {cls.RESULTS_DIR}")
        print(f"   Logs: {cls.LOGS_DIR}")
        
        print(f"\n🎵 Audio Features:")
        print(f"   Mel Bands: {cls.N_MELS}")
        print(f"   FFT Size: {cls.N_FFT}")
        print(f"   Hop Length: {cls.HOP_LENGTH}")
        
        print("="*60)
    
    @classmethod
    def update_config(cls, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
                print(f"Updated {key.upper()}: {value}")
            else:
                print(f"Warning: Unknown config parameter: {key}")


# Create default configuration instance
config = Config()

