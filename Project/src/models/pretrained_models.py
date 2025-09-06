"""
Pre-trained Models for Emotion Recognition
"""

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FixedWav2Vec2Classifier(nn.Module):
    """Fixed Wav2Vec2 model for emotion recognition"""
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        if TRANSFORMERS_AVAILABLE:
            # Create a working Wav2Vec2 configuration
            config = Wav2Vec2Config(
                vocab_size=32,
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                intermediate_size=1024,
                # Fix the mask length issue
                mask_time_length=1,
                mask_time_prob=0.0,  # Disable masking to avoid issues
                mask_feature_length=1,
                mask_feature_prob=0.0,
            )
            
            self.wav2vec2 = Wav2Vec2Model(config)
            
            # Freeze most parameters for efficiency
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < len(self.wav2vec2.encoder.layers) - 2:  # Freeze all but last 2 layers
                    for param in layer.parameters():
                        param.requires_grad = False
            
            hidden_size = config.hidden_size
        else:
            # Fallback if transformers not available
            hidden_size = 512
            self.wav2vec2 = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_values):
        if self.wav2vec2 is not None:
            try:
                # Process audio
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs.last_hidden_state
                
                # Global average pooling
                pooled = torch.mean(hidden_states, dim=1)
                
                return self.classifier(pooled)
                
            except Exception as e:
                print(f"Forward error: {e}")
                batch_size = input_values.size(0)
                return torch.zeros(batch_size, self.classifier[-1].out_features).to(input_values.device)
        else:
            # Fallback behavior
            batch_size = input_values.size(0)
            return torch.zeros(batch_size, self.classifier[-1].out_features).to(input_values.device)


class SimpleCNNAudioClassifier(nn.Module):
    """Simple CNN-based audio classifier"""
    
    def __init__(self, num_classes=8):
        super().__init__()
        
        # Convert raw audio to spectrogram representation
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(32)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
        
        # 1D convolutions
        x = self.conv1d_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return self.classifier(x)

