"""
Data utilities for emotion recognition
"""

import random
import numpy as np


class SimpleLabelEncoder:
    """Simple label encoder for emotion classes"""
    
    def __init__(self):
        self.classes_ = None
        self.class_to_idx = {}
        
    def fit_transform(self, labels):
        """Fit encoder and transform labels to indices"""
        unique_labels = sorted(list(set(labels)))
        self.classes_ = np.array(unique_labels)
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return [self.class_to_idx[label] for label in labels]
    
    def transform(self, labels):
        """Transform labels to indices"""
        return [self.class_to_idx[label] for label in labels]
    
    def inverse_transform(self, indices):
        """Transform indices back to labels"""
        return [self.classes_[idx] for idx in indices]


def train_test_split(files, labels, test_ratio=0.1, val_ratio=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        files: List of file paths
        labels: List of labels
        test_ratio: Ratio for test set
        val_ratio: Ratio for validation set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    random.seed(random_state)
    
    # Combine and shuffle
    combined = list(zip(files, labels))
    random.shuffle(combined)
    
    total = len(combined)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    
    train_data = combined[:total - test_size - val_size]
    val_data = combined[total - test_size - val_size:total - test_size]
    test_data = combined[total - test_size:]
    
    X_train, y_train = zip(*train_data) if train_data else ([], [])
    X_val, y_val = zip(*val_data) if val_data else ([], [])
    X_test, y_test = zip(*test_data) if test_data else ([], [])
    
    return list(X_train), list(y_train), list(X_val), list(y_val), list(X_test), list(y_test)


def load_emotion_data(data_dir="organized_by_emotion"):
    """
    Load emotion data from organized directory structure
    
    Args:
        data_dir: Directory containing emotion subdirectories
        
    Returns:
        Tuple of (file_paths, labels)
    """
    import os
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    file_paths = []
    labels = []
    
    emotion_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    emotion_dirs.sort()
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(data_dir, emotion)
        files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for file in files:
            file_paths.append(os.path.join(emotion_path, file))
            labels.append(emotion)
    
    return file_paths, labels

