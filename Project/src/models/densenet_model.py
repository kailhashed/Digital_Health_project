"""
DenseNet Model Implementation for Emotion Recognition
Integrated into the structured codebase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DenseBlock(nn.Module):
    """Dense Block for DenseNet"""
    
    def __init__(self, in_channels, growth_rate, num_layers, dropout=0.1):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, 4 * growth_rate, 
                         kernel_size=1, bias=False),
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * growth_rate, growth_rate, 
                         kernel_size=3, padding=1, bias=False),
                nn.Dropout2d(dropout)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    """Transition layer between dense blocks"""
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(TransitionLayer, self).__init__()
        
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Dropout2d(dropout),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)


class DenseNetEmotionModel(nn.Module):
    """DenseNet model for emotion recognition - Integrated version"""
    
    def __init__(self, num_classes=8, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, dropout=0.1):
        super(DenseNetEmotionModel, self).__init__()
        
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.dropout = dropout
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks and transition layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_features, growth_rate, num_layers, dropout)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # Add transition layer (except after last dense block)
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2, dropout)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        features = self.features(x)
        out = self.avgpool(features)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        return out
    
    def get_model_info(self):
        """Get model information for logging and saving"""
        return {
            'model_type': 'DenseNet',
            'architecture': 'DenseNetEmotionModel',
            'num_classes': self.num_classes,
            'growth_rate': self.growth_rate,
            'block_config': self.block_config,
            'num_init_features': self.num_init_features,
            'dropout': self.dropout,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_densenet_model(config=None):
    """Factory function to create DenseNet model"""
    if config is None:
        config = {
            'num_classes': 8,
            'growth_rate': 32,
            'block_config': (6, 12, 24, 16),
            'num_init_features': 64,
            'dropout': 0.1
        }
    
    return DenseNetEmotionModel(**config)


def create_densenet_variants():
    """Create different DenseNet model variants"""
    variants = {
        'densenet_small': {
            'num_classes': 8,
            'growth_rate': 16,
            'block_config': (4, 8, 12, 8),
            'num_init_features': 32,
            'dropout': 0.1
        },
        'densenet_medium': {
            'num_classes': 8,
            'growth_rate': 32,
            'block_config': (6, 12, 24, 16),
            'num_init_features': 64,
            'dropout': 0.1
        },
        'densenet_large': {
            'num_classes': 8,
            'growth_rate': 48,
            'block_config': (8, 16, 32, 24),
            'num_init_features': 96,
            'dropout': 0.1
        }
    }
    
    models = {}
    for name, config in variants.items():
        models[name] = create_densenet_model(config)
    
    return models


if __name__ == "__main__":
    # Test model creation
    print("Testing DenseNet model creation...")
    
    model = create_densenet_model()
    info = model.get_model_info()
    
    print(f"Model Type: {info['model_type']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Parameters: {info['total_parameters']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 128, 130)  # Batch of 2
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ DenseNet model test passed!")
