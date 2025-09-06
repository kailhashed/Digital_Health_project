#!/usr/bin/env python3
"""
Optimized DenseNet Training for High Accuracy and Low False Positives
Focus on achieving the best possible performance with minimal misclassifications.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
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

# Add src to path
sys.path.append('src/models')
from custom_models import EmotionDenseNet


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and reducing false positives"""
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing to reduce overconfidence and improve generalization"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class OptimizedEmotionDataset(Dataset):
    """Enhanced dataset with advanced preprocessing for better accuracy"""
    
    def __init__(self, file_paths, labels, sr=22050, n_mels=128, duration=3.0, 
                 augment=False, normalize_per_sample=True):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.augment = augment
        self.normalize_per_sample = normalize_per_sample
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio with higher quality
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            # Advanced preprocessing
            y = self._preprocess_audio(y)
            
            # Extract enhanced mel-spectrogram
            mel_spec = self._extract_enhanced_features(y, sr)
            
            # Data augmentation during training
            if self.augment:
                mel_spec = self._augment_spectrogram(mel_spec)
            
            return torch.FloatTensor(mel_spec), label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.n_mels, 130), label
    
    def _preprocess_audio(self, y):
        """Advanced audio preprocessing"""
        # Remove silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Ensure minimum length
        if len(y_trimmed) < self.sr * 0.1:
            y_trimmed = y
        
        # Normalize
        y_trimmed = librosa.util.normalize(y_trimmed)
        
        # Pad or trim to exact duration
        target_length = int(self.sr * self.duration)
        if len(y_trimmed) < target_length:
            y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)), mode='constant')
        else:
            y_trimmed = y_trimmed[:target_length]
        
        return y_trimmed
    
    def _extract_enhanced_features(self, y, sr):
        """Extract enhanced mel-spectrogram features"""
        # Higher quality mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, 
            n_fft=2048, hop_length=256,  # Higher resolution
            fmin=20, fmax=8000,  # Focus on speech range
            window='hann'
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
        
        # Normalize per sample if enabled
        if self.normalize_per_sample:
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
        
        return log_mel_spec
    
    def _augment_spectrogram(self, mel_spec):
        """Light spectrogram augmentation"""
        if np.random.random() < 0.3:
            # Time masking
            time_mask_width = min(10, mel_spec.shape[1] // 10)
            time_mask_start = np.random.randint(0, max(1, mel_spec.shape[1] - time_mask_width))
            mel_spec[:, time_mask_start:time_mask_start + time_mask_width] *= 0.1
        
        if np.random.random() < 0.3:
            # Frequency masking
            freq_mask_width = min(8, mel_spec.shape[0] // 10)
            freq_mask_start = np.random.randint(0, max(1, mel_spec.shape[0] - freq_mask_width))
            mel_spec[freq_mask_start:freq_mask_start + freq_mask_width, :] *= 0.1
        
        return mel_spec


class OptimizedDenseNetTrainer:
    """Advanced trainer focused on high accuracy and low false positives"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        # Training configuration optimized for accuracy
        self.config = {
            'model': {
                'num_classes': 8,
                'growth_rate': 32,
                'block_config': (6, 12, 24, 16),
                'num_init_features': 64,
                'dropout': 0.3  # Higher dropout for better generalization
            },
            'training': {
                'epochs': 150,
                'batch_size': 32,
                'learning_rate': 0.0005,  # Lower initial LR
                'weight_decay': 1e-4,
                'early_stopping_patience': 15,
                'lr_scheduler_patience': 8,
                'lr_scheduler_factor': 0.5,
                'min_lr': 1e-7,
                'warmup_epochs': 5
            },
            'loss': {
                'type': 'focal',  # focal, ce, label_smoothing, weighted_ce
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                'label_smoothing': 0.1
            },
            'data': {
                'augment_training': True,
                'normalize_per_sample': True,
                'use_class_weights': True,
                'balanced_sampling': True
            }
        }
        
        print(f"Optimized DenseNet Trainer initialized on {self.device}")
    
    def load_data(self, data_path="organized_by_emotion"):
        """Load and analyze dataset"""
        file_paths = []
        labels = []
        
        print("Loading dataset...")
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_path = os.path.join(data_path, emotion)
            if not os.path.exists(emotion_path):
                continue
            
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            for file in files:
                file_path = os.path.join(emotion_path, file)
                file_paths.append(file_path)
                labels.append(emotion_idx)
            
            print(f"  {emotion}: {len(files)} files")
        
        print(f"Total files: {len(file_paths)}")
        
        # Analyze class distribution
        self._analyze_class_distribution(labels)
        
        return file_paths, labels
    
    def _analyze_class_distribution(self, labels):
        """Analyze and print class distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        
        print("\nClass distribution:")
        for emotion_idx, count in zip(unique, counts):
            emotion = self.emotions[emotion_idx]
            percentage = count / len(labels) * 100
            print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
        
        # Calculate class weights for balanced training
        self.class_weights = compute_class_weight('balanced', classes=unique, y=labels)
        print(f"\nCalculated class weights: {dict(zip(self.emotions, self.class_weights))}")
    
    def split_data_stratified(self, file_paths, labels):
        """Advanced stratified splitting with validation"""
        # Convert to numpy for easier handling
        file_paths = np.array(file_paths)
        labels = np.array(labels)
        
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Second split: 80% train, 20% val (of the remaining 80%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"\nStratified data split:")
        print(f"Training: {len(X_train)} samples ({len(X_train)/len(file_paths)*100:.1f}%)")
        print(f"Validation: {len(X_val)} samples ({len(X_val)/len(file_paths)*100:.1f}%)")
        print(f"Test: {len(X_test)} samples ({len(X_test)/len(file_paths)*100:.1f}%)")
        
        # Verify stratification worked
        for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            unique, counts = np.unique(split_labels, return_counts=True)
            print(f"{split_name} distribution:", {self.emotions[i]: c for i, c in zip(unique, counts)})
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_optimized_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Create optimized data loaders with advanced sampling"""
        
        # Create datasets
        train_dataset = OptimizedEmotionDataset(
            X_train, y_train, 
            augment=self.config['data']['augment_training'],
            normalize_per_sample=self.config['data']['normalize_per_sample']
        )
        val_dataset = OptimizedEmotionDataset(X_val, y_val, augment=False)
        test_dataset = OptimizedEmotionDataset(X_test, y_test, augment=False)
        
        # Create balanced sampler for training
        if self.config['data']['balanced_sampling']:
            class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_train])
            samples_weight = torch.from_numpy(samples_weight).double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            
            train_loader = DataLoader(
                train_dataset, batch_size=self.config['training']['batch_size'],
                sampler=sampler, num_workers=0
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.config['training']['batch_size'],
                shuffle=True, num_workers=0
            )
        
        val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], 
                               shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], 
                                shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader
    
    def create_optimized_model(self):
        """Create optimized DenseNet model"""
        model = EmotionDenseNet(**self.config['model']).to(self.device)
        
        # Advanced weight initialization
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
        
        print(f"Created optimized DenseNet model:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Memory footprint: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} MB")
        
        return model
    
    def create_optimized_loss_function(self):
        """Create advanced loss function"""
        loss_type = self.config['loss']['type']
        
        if loss_type == 'focal':
            class_weights = torch.FloatTensor(self.class_weights).to(self.device)
            criterion = FocalLoss(
                alpha=self.config['loss']['focal_alpha'],
                gamma=self.config['loss']['focal_gamma'],
                weight=class_weights
            )
        elif loss_type == 'label_smoothing':
            criterion = LabelSmoothingLoss(
                num_classes=8,
                smoothing=self.config['loss']['label_smoothing']
            )
        elif loss_type == 'weighted_ce':
            class_weights = torch.FloatTensor(self.class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        print(f"Using {loss_type} loss function")
        return criterion
    
    def create_optimized_optimizer(self, model):
        """Create advanced optimizer with warmup"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=self.config['training']['lr_scheduler_patience'],
            factor=self.config['training']['lr_scheduler_factor'],
            min_lr=self.config['training']['min_lr'],
            verbose=True
        )
        
        return optimizer, scheduler
    
    def train_with_optimization(self, train_loader, val_loader, save_dir):
        """Advanced training loop with all optimizations"""
        model = self.create_optimized_model()
        criterion = self.create_optimized_loss_function()
        optimizer, scheduler = self.create_optimized_optimizer(model)
        
        # Training history
        history = {
            'train_losses': [], 'train_accuracies': [], 'train_precisions': [],
            'val_losses': [], 'val_accuracies': [], 'val_precisions': [],
            'learning_rates': [], 'false_positive_rates': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        print(f"\nStarting optimized training for {self.config['training']['epochs']} epochs")
        print(f"Early stopping patience: {self.config['training']['early_stopping_patience']}")
        
        for epoch in range(self.config['training']['epochs']):
            # Warmup learning rate
            if epoch < self.config['training']['warmup_epochs']:
                warmup_lr = self.config['training']['learning_rate'] * (epoch + 1) / self.config['training']['warmup_epochs']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Training phase
            train_metrics = self._train_epoch_advanced(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_metrics = self._validate_epoch_advanced(model, val_loader, criterion)
            
            # Update learning rate (after warmup)
            if epoch >= self.config['training']['warmup_epochs']:
                scheduler.step(val_metrics['accuracy'])
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save metrics
            history['train_losses'].append(train_metrics['loss'])
            history['train_accuracies'].append(train_metrics['accuracy'])
            history['train_precisions'].append(train_metrics['precision'])
            history['val_losses'].append(val_metrics['loss'])
            history['val_accuracies'].append(val_metrics['accuracy'])
            history['val_precisions'].append(val_metrics['precision'])
            history['learning_rates'].append(current_lr)
            history['false_positive_rates'].append(val_metrics['false_positive_rate'])
            
            # Check for best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(model, optimizer, epoch + 1, val_metrics['accuracy'], 
                                    save_dir, 'best')
            else:
                patience_counter += 1
            
            # Print progress
            print(f"  Train: Loss {train_metrics['loss']:.4f}, Acc {train_metrics['accuracy']:.3f}, Prec {train_metrics['precision']:.3f}")
            print(f"  Val:   Loss {val_metrics['loss']:.4f}, Acc {val_metrics['accuracy']:.3f}, Prec {val_metrics['precision']:.3f}")
            print(f"  False Positive Rate: {val_metrics['false_positive_rate']:.3f}")
            print(f"  Best Val Acc: {best_val_acc:.3f}, Patience: {patience_counter}/{self.config['training']['early_stopping_patience']}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history
    
    def _train_epoch_advanced(self, model, train_loader, criterion, optimizer):
        """Advanced training epoch with detailed metrics"""
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress
            current_acc = accuracy_score(all_targets, all_predictions)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.3f}'})
        
        # Calculate detailed metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, _, _, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
        avg_loss = total_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def _validate_epoch_advanced(self, model, val_loader, criterion):
        """Advanced validation with false positive analysis"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation', leave=False):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate detailed metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, _, _, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
        avg_loss = total_loss / len(val_loader)
        
        # Calculate false positive rate
        cm = confusion_matrix(all_targets, all_predictions)
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        false_positive_rate = np.mean(fp / (fp + tn + 1e-8))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'false_positive_rate': false_positive_rate,
            'predictions': all_predictions,
            'targets': all_targets,
            'confusion_matrix': cm
        }
    
    def _save_checkpoint(self, model, optimizer, epoch, accuracy, save_dir, name):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(save_dir, f'optimized_densenet_{name}.pth')
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def comprehensive_evaluation(self, model, test_loader, save_dir):
        """Comprehensive evaluation focused on accuracy and false positives"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        print("\nPerforming comprehensive evaluation...")
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions, 
                                     target_names=self.emotions, output_dict=True)
        
        # Detailed false positive analysis
        fp_analysis = self._analyze_false_positives(cm, all_targets, all_predictions)
        
        results = {
            'test_accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'false_positive_analysis': fp_analysis,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        # Save results
        self._save_evaluation_results(results, save_dir)
        
        return results
    
    def _analyze_false_positives(self, cm, targets, predictions):
        """Detailed false positive analysis"""
        fp_analysis = {}
        
        for i, emotion in enumerate(self.emotions):
            # False positives for this class
            fp = cm[:, i].sum() - cm[i, i]
            # True negatives
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            # False positive rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Most confused classes
            confused_with = {}
            for j, other_emotion in enumerate(self.emotions):
                if i != j and cm[j, i] > 0:
                    confused_with[other_emotion] = int(cm[j, i])
            
            fp_analysis[emotion] = {
                'false_positives': int(fp),
                'false_positive_rate': float(fpr),
                'confused_with': confused_with
            }
        
        return fp_analysis
    
    def _save_evaluation_results(self, results, save_dir):
        """Save comprehensive evaluation results"""
        # Create visualizations
        self._plot_detailed_results(results, save_dir)
        
        # Save JSON summary
        json_results = {
            'test_accuracy': results['test_accuracy'],
            'classification_report': results['classification_report'],
            'false_positive_analysis': results['false_positive_analysis'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save full results
        with open(os.path.join(save_dir, 'full_evaluation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Evaluation results saved to {save_dir}")
    
    def _plot_detailed_results(self, results, save_dir):
        """Create detailed visualization plots"""
        # 1. Enhanced confusion matrix
        plt.figure(figsize=(12, 10))
        cm_normalized = results['confusion_matrix'].astype('float') / results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.emotions, yticklabels=self.emotions,
                   cbar_kws={'label': 'Normalized Count'})
        plt.title('Normalized Confusion Matrix - Optimized DenseNet')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. False positive analysis
        fp_rates = [results['false_positive_analysis'][emotion]['false_positive_rate'] 
                   for emotion in self.emotions]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.emotions, fp_rates, color='red', alpha=0.7)
        plt.title('False Positive Rates by Emotion Class')
        plt.xlabel('Emotion')
        plt.ylabel('False Positive Rate')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, fp_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'false_positive_rates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-class performance
        precisions = [results['classification_report'][emotion]['precision'] for emotion in self.emotions]
        recalls = [results['classification_report'][emotion]['recall'] for emotion in self.emotions]
        f1_scores = [results['classification_report'][emotion]['f1-score'] for emotion in self.emotions]
        
        x = np.arange(len(self.emotions))
        width = 0.25
        
        plt.figure(figsize=(14, 8))
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Emotion Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, self.emotions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function"""
    print("Optimized DenseNet Training for High Accuracy and Low False Positives")
    print("=" * 80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_optimized_densenet_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = OptimizedDenseNetTrainer()
    
    # Load and split data
    file_paths, labels = trainer.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data_stratified(file_paths, labels)
    
    # Create optimized data loaders
    train_loader, val_loader, test_loader = trainer.create_optimized_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print(f"\nStarting optimized training...")
    print(f"Results will be saved to: {save_dir}")
    
    # Train model
    model, history = trainer.train_with_optimization(train_loader, val_loader, save_dir)
    
    # Comprehensive evaluation
    results = trainer.comprehensive_evaluation(model, test_loader, save_dir)
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS - OPTIMIZED DENSENET")
    print(f"{'='*80}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"Best Validation Accuracy: {max(history['val_accuracies']):.4f}")
    
    print(f"\nFalse Positive Analysis:")
    for emotion in trainer.emotions:
        fp_info = results['false_positive_analysis'][emotion]
        print(f"  {emotion}: FPR = {fp_info['false_positive_rate']:.3f}, FP = {fp_info['false_positives']}")
    
    avg_fpr = np.mean([results['false_positive_analysis'][emotion]['false_positive_rate'] 
                      for emotion in trainer.emotions])
    print(f"\nAverage False Positive Rate: {avg_fpr:.3f}")
    
    print(f"\nResults saved in: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
