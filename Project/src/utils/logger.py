"""
Logging utilities for emotion recognition
"""

import os
import logging
from datetime import datetime


def setup_logger(name="emotion_recognition", log_dir="logs", level=logging.INFO):
    """
    Set up logger for the project
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


class TrainingLogger:
    """Logger specifically for training progress"""
    
    def __init__(self, model_name, log_dir="logs"):
        self.model_name = model_name
        self.logger = setup_logger(f"training_{model_name}", log_dir)
        self.epoch_metrics = []
    
    def log_epoch(self, epoch, train_loss, val_acc, is_best=False):
        """Log epoch results"""
        message = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}"
        if is_best:
            message += " [BEST]"
        
        self.logger.info(message)
        
        self.epoch_metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'is_best': is_best
        })
    
    def log_training_start(self, total_epochs, batch_size, learning_rate):
        """Log training start information"""
        self.logger.info(f"Starting training for {self.model_name}")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Learning rate: {learning_rate}")
    
    def log_training_end(self, best_val_acc, total_epochs):
        """Log training completion"""
        self.logger.info(f"Training completed for {self.model_name}")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        self.logger.info(f"Total epochs trained: {total_epochs}")
    
    def log_test_results(self, test_acc, num_samples):
        """Log test results"""
        self.logger.info(f"Test results for {self.model_name}")
        self.logger.info(f"Test accuracy: {test_acc:.4f}")
        self.logger.info(f"Test samples: {num_samples}")
    
    def save_metrics(self, filepath=None):
        """Save training metrics to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/{self.model_name}_metrics_{timestamp}.json"
        
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to: {filepath}")


# Default logger instance
logger = setup_logger()

