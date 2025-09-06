#!/usr/bin/env python3
"""
Comprehensive Evaluation of DenseNet Epoch 52 Model on Actual Dataset
Tests the best performance model with real data and generates detailed results.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add path for models
sys.path.append('src/models')
from custom_models import EmotionDenseNet


class EmotionDataset(Dataset):
    """Dataset class for emotion recognition evaluation"""
    
    def __init__(self, file_paths, labels, sr=22050, n_mels=128, duration=3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load and preprocess audio
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            y = librosa.util.normalize(y)
            
            # Pad or trim to exact duration
            target_length = int(self.sr * self.duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            return torch.FloatTensor(log_mel_spec), label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.n_mels, 130), label


def load_data(data_path="organized_by_emotion"):
    """Load data from organized dataset"""
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    file_paths = []
    labels = []
    
    print("Loading dataset for evaluation...")
    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion} directory not found")
            continue
        
        files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        for file in files:
            file_path = os.path.join(emotion_path, file)
            file_paths.append(file_path)
            labels.append(emotion_idx)
        
        print(f"  {emotion}: {len(files)} files")
    
    print(f"Total files loaded: {len(file_paths)}")
    return file_paths, labels


def split_data_consistent(file_paths, labels):
    """Split data using the same 80-10-10 split as training"""
    # Use same random state as training for consistency
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split (consistent with training):")
    print(f"Training set: {len(X_train)} files ({len(X_train)/len(file_paths)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} files ({len(X_val)/len(file_paths)*100:.1f}%)")
    print(f"Test set: {len(X_test)} files ({len(X_test)/len(file_paths)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_epoch_52_model():
    """Load the epoch 52 best performance model"""
    model_path = "models/densenet_current_best.pth"
    
    if not os.path.exists(model_path):
        model_path = "models/densenet_epoch_52_best.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Epoch 52 model not found at expected locations")
    
    print(f"Loading epoch 52 model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model_config = checkpoint['model_config']
    model = EmotionDenseNet(**model_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def evaluate_model_on_split(model, data_loader, split_name, device):
    """Evaluate model on a specific data split"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nEvaluating on {split_name} set...")
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc=f'Evaluating {split_name}'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            total_loss += loss.item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    avg_loss = total_loss / len(data_loader)
    
    print(f"{split_name} Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Loss: {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }


def generate_detailed_report(results, emotions):
    """Generate detailed classification report and confusion matrix"""
    print(f"\n{'='*60}")
    print("DETAILED EVALUATION RESULTS")
    print(f"{'='*60}")
    
    for split_name, result in results.items():
        print(f"\n{split_name.upper()} SET RESULTS:")
        print(f"{'='*40}")
        
        # Overall metrics
        print(f"Overall Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"Overall Loss: {result['loss']:.4f}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        report = classification_report(
            result['targets'], result['predictions'], 
            target_names=emotions, output_dict=True
        )
        
        # Print formatted report
        print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for emotion in emotions:
            metrics = report[emotion]
            print(f"{emotion:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}")
        
        # Macro and weighted averages
        print("-" * 60)
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        print(f"{'Macro Avg':<12} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} "
              f"{macro_avg['f1-score']:<10.3f}")
        print(f"{'Weighted Avg':<12} {weighted_avg['precision']:<10.3f} {weighted_avg['recall']:<10.3f} "
              f"{weighted_avg['f1-score']:<10.3f}")
        
        # Store detailed report
        result['classification_report'] = report
        result['confusion_matrix'] = confusion_matrix(result['targets'], result['predictions'])


def plot_confusion_matrices(results, emotions, save_dir):
    """Plot confusion matrices for all splits"""
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (split_name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=emotions, yticklabels=emotions, ax=axes[idx])
        axes[idx].set_title(f'{split_name.title()} Set Confusion Matrix')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved: {plot_path}")
    
    plt.show()


def plot_per_class_performance(results, emotions, save_dir):
    """Plot per-class performance metrics"""
    splits = list(results.keys())
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for metric_idx, metric in enumerate(metrics):
        data = []
        for emotion in emotions:
            emotion_data = []
            for split in splits:
                report = results[split]['classification_report']
                emotion_data.append(report[emotion][metric])
            data.append(emotion_data)
        
        # Create grouped bar chart
        x = np.arange(len(emotions))
        width = 0.35 if len(splits) == 2 else 0.25
        
        for i, split in enumerate(splits):
            values = [data[j][i] for j in range(len(emotions))]
            axes[metric_idx].bar(x + i*width, values, width, label=split.title())
        
        axes[metric_idx].set_xlabel('Emotions')
        axes[metric_idx].set_ylabel(metric.title())
        axes[metric_idx].set_title(f'Per-Class {metric.title()}')
        axes[metric_idx].set_xticks(x + width/2)
        axes[metric_idx].set_xticklabels(emotions, rotation=45)
        axes[metric_idx].legend()
        axes[metric_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'per_class_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Per-class performance plot saved: {plot_path}")
    
    plt.show()


def save_results(results, checkpoint, save_dir):
    """Save evaluation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results for saving
    save_data = {
        'model_info': {
            'epoch': checkpoint['current_epoch'],
            'best_val_accuracy': checkpoint['best_val_accuracy'],
            'model_config': checkpoint['model_config'],
            'evaluation_timestamp': timestamp
        },
        'evaluation_results': {}
    }
    
    # Process results for each split
    for split_name, result in results.items():
        save_data['evaluation_results'][split_name] = {
            'accuracy': result['accuracy'],
            'loss': result['loss'],
            'classification_report': result['classification_report'],
            'confusion_matrix': result['confusion_matrix'].tolist()
        }
    
    # Save as JSON
    json_path = os.path.join(save_dir, f'densenet_epoch_52_evaluation_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Save full results as pickle
    pickle_path = os.path.join(save_dir, f'densenet_epoch_52_full_results_{timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'model_info': save_data['model_info'],
            'results': results,
            'checkpoint': checkpoint
        }, f)
    
    print(f"Results saved:")
    print(f"  JSON: {json_path}")
    print(f"  Pickle: {pickle_path}")
    
    return json_path, pickle_path


def main():
    """Main evaluation function"""
    print("DenseNet Epoch 52 - Comprehensive Dataset Evaluation")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_densenet_epoch_52_evaluation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Load model
    try:
        model, checkpoint = load_epoch_52_model()
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Best validation accuracy: {checkpoint['best_val_accuracy']:.2f}%")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and split data
    try:
        file_paths, labels = load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_consistent(file_paths, labels)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create datasets and loaders
    datasets = {
        'validation': EmotionDataset(X_val, y_val),
        'test': EmotionDataset(X_test, y_test)
    }
    
    data_loaders = {
        name: DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        for name, dataset in datasets.items()
    }
    
    # Evaluate model on all splits
    results = {}
    for split_name, data_loader in data_loaders.items():
        results[split_name] = evaluate_model_on_split(model, data_loader, split_name, device)
    
    # Generate detailed reports
    generate_detailed_report(results, emotions)
    
    # Create visualizations
    plot_confusion_matrices(results, emotions, results_dir)
    plot_per_class_performance(results, emotions, results_dir)
    
    # Save results
    json_path, pickle_path = save_results(results, checkpoint, results_dir)
    
    # Final summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    for split_name, result in results.items():
        print(f"{split_name.upper()} SET:")
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  Loss: {result['loss']:.4f}")
    
    print(f"\nModel Information:")
    print(f"  Epoch: {checkpoint['current_epoch']}")
    print(f"  Training Validation Accuracy: {checkpoint['best_val_accuracy']:.2f}%")
    print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved in: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
