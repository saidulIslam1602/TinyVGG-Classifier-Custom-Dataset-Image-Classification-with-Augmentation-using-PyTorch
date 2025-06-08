"""
Training Module for TinyVGG Image Classifier

This module provides a comprehensive training framework with support for:
- Advanced optimization techniques
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Metrics tracking
- Wandb integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_weights(self, model: nn.Module):
        if self.best_weights:
            model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.best_weights.items()})


class MetricsTracker:
    """Track and compute various training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        # Convert to CPU numpy arrays
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Store for later analysis
        self.predictions.extend(pred_np.argmax(axis=1))
        self.targets.extend(target_np)
        
        # Calculate accuracy
        accuracy = (pred_np.argmax(axis=1) == target_np).mean()
        
        self.metrics['loss'].append(loss)
        self.metrics['accuracy'].append(accuracy)
    
    def get_average_metrics(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.metrics.items()}
    
    def get_classification_report(self, class_names: List[str]) -> str:
        return classification_report(
            self.targets, 
            self.predictions, 
            target_names=class_names,
            zero_division=0
        )
    
    def get_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.targets, self.predictions)


class TinyVGGTrainer:
    """
    Comprehensive trainer for TinyVGG model with modern ML practices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        experiment_name: str = "tinyvgg_experiment",
        save_dir: str = "experiments",
        use_wandb: bool = False,
        use_tensorboard: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            class_names: List of class names
            device: Device to train on
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            experiment_name: Name for the experiment
            save_dir: Directory to save results
            use_wandb: Whether to use Weights & Biases logging
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.experiment_name = experiment_name
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Setup criterion
        if criterion is None:
            # Use class weights if available
            if hasattr(train_loader.dataset, 'get_class_weights'):
                class_weights = train_loader.dataset.get_class_weights().to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        else:
            self.scheduler = scheduler
        
        # Setup directories
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project="tinyvgg-food-classification",
                name=experiment_name,
                config={
                    "model": model.__class__.__name__,
                    "optimizer": optimizer.__class__.__name__ if optimizer else "AdamW",
                    "scheduler": scheduler.__class__.__name__ if scheduler else "CosineAnnealingWarmRestarts",
                    "batch_size": train_loader.batch_size,
                    "num_classes": len(class_names),
                    "device": str(self.device)
                }
            )
            wandb.watch(self.model)
        
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=self.save_dir / "tensorboard")
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"  Scheduler: {self.scheduler.__class__.__name__}")
        logger.info(f"  Save directory: {self.save_dir}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            metrics_tracker.update(output, target, loss.item())
            
            # Log batch progress
            if batch_idx % 50 == 0:
                logger.debug(f"Train Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return metrics_tracker.get_average_metrics()
    
    def validate_epoch(self) -> Tuple[Dict[str, float], MetricsTracker]:
        """Validate for one epoch."""
        self.model.eval()
        metrics_tracker = MetricsTracker()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                metrics_tracker.update(output, target, loss.item())
        
        return metrics_tracker.get_average_metrics(), metrics_tracker
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'class_names': self.class_names
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    
    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best_only: bool = True,
        save_frequency: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            save_frequency: How often to save checkpoints
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics, val_tracker = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Log to external services
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics['loss'],
                    "train_accuracy": train_metrics['accuracy'],
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                    "learning_rate": current_lr
                })
            
            if self.use_tensorboard:
                self.tb_writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
                self.tb_writer.add_scalar("Loss/Validation", val_metrics['loss'], epoch)
                self.tb_writer.add_scalar("Accuracy/Train", train_metrics['accuracy'], epoch)
                self.tb_writer.add_scalar("Accuracy/Validation", val_metrics['accuracy'], epoch)
                self.tb_writer.add_scalar("Learning_Rate", current_lr, epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            if not save_best_only or is_best or (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Early stopping check
            if early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                early_stopping.restore_weights(self.model)
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Save final results
        self.save_results(val_tracker)
        
        # Close loggers
        if self.use_tensorboard:
            self.tb_writer.close()
        
        return self.history
    
    def save_results(self, final_val_tracker: MetricsTracker):
        """Save training results and visualizations."""
        # Save training history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save classification report
        report = final_val_tracker.get_classification_report(self.class_names)
        report_path = self.save_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Create and save plots
        self.plot_training_history()
        self.plot_confusion_matrix(final_val_tracker.get_confusion_matrix())
        
        logger.info(f"Results saved to {self.save_dir}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        ax1.plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_accuracy'], label='Train Accuracy', alpha=0.8)
        ax2.plot(self.history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(self.history['learning_rate'], alpha=0.8)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        ax4.axis('off')
        final_train_acc = self.history['train_accuracy'][-1]
        final_val_acc = self.history['val_accuracy'][-1]
        best_val_acc = max(self.history['val_accuracy'])
        
        stats_text = f"""
        Final Training Accuracy: {final_train_acc:.4f}
        Final Validation Accuracy: {final_val_acc:.4f}
        Best Validation Accuracy: {best_val_acc:.4f}
        Total Epochs: {len(self.history['train_loss'])}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.save_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would typically be run from the main training script
    logger.info("Trainer module loaded successfully") 