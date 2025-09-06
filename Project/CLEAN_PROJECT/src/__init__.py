"""
Emotion Recognition Models Package
"""

from .custom_models import (
    EmotionTransformer,
    EmotionLSTM, 
    EmotionResNet
)

from .pretrained_models import (
    FixedWav2Vec2Classifier,
    SimpleCNNAudioClassifier
)

__all__ = [
    'EmotionTransformer',
    'EmotionLSTM',
    'EmotionResNet',
    'FixedWav2Vec2Classifier',
    'SimpleCNNAudioClassifier'
]

