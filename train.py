#!/usr/bin/env python3
"""
TinyVGG Image Classifier Training Script

This script provides a comprehensive training pipeline for the TinyVGG model
with support for configuration management, logging, and experiment tracking.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --experiment-name my_experiment
"""

import argparse
import logging
import yaml
import torch
import random
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.tiny_vgg import create_tiny_vgg
from src.data.dataset import create_data_loaders, download_sample_data
from src.training.trainer import TinyVGGTrainer
from src.utils.visualization import plot_dataset_distribution


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: dict) -> torch.nn.Module:
    """Create model from configuration."""
    model_config = config['model']
    
    model = create_tiny_vgg(
        num_classes=model_config['num_classes'],
        input_channels=model_config['input_channels'],
        model_size=model_config['size']
    )
    
    return model


def create_optimizer_from_config(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    training_config = config['training']
    
    optimizer_name = training_config['optimizer']
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    
    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler_from_config(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler from configuration."""
    training_config = config['training']
    scheduler_name = training_config['scheduler']
    
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    elif scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train TinyVGG Image Classifier")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override experiment name from config"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download sample dataset if not present"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        help="Override device from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.device:
        config['deployment']['device'] = args.device
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.wandb:
        config['logging']['use_wandb'] = True
    if args.tensorboard:
        config['logging']['use_tensorboard'] = True
    
    # Setup logging
    setup_logging(
        log_level=config['logging']['log_level'],
        log_file=config['logging'].get('log_file')
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting TinyVGG training...")
    logger.info(f"Configuration: {config}")
    
    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    
    # Download data if requested
    if args.download_data:
        logger.info("Downloading sample dataset...")
        data_path = download_sample_data(config['data']['data_dir'])
        logger.info(f"Data downloaded to: {data_path}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, test_loader, class_names = create_data_loaders(
        train_dir=config['data']['train_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Update class names in config
    config['data']['class_names'] = class_names
    config['model']['num_classes'] = len(class_names)
    
    # Visualize dataset distribution
    class_counts = {}
    for class_name in class_names:
        train_class_dir = Path(config['data']['train_dir']) / class_name
        if train_class_dir.exists():
            class_counts[class_name] = len(list(train_class_dir.glob('*')))
    
    # Create model
    logger.info("Creating model...")
    model = create_model_from_config(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer_from_config(model, config)
    scheduler = create_scheduler_from_config(optimizer, config)
    
    # Setup device
    if config['deployment']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['deployment']['device'])
    
    logger.info(f"Using device: {device}")
    
    # Create trainer
    trainer = TinyVGGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=class_names,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=config['experiment']['name'],
        save_dir=config['experiment']['save_dir'],
        use_wandb=config['logging']['use_wandb'],
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    history = trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        save_best_only=config['training']['checkpoint']['save_best_only'],
        save_frequency=config['training']['checkpoint']['save_frequency']
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    logger.info(f"Training completed in {training_duration}")
    logger.info(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    
    # Save final configuration with results
    experiment_dir = Path(config['experiment']['save_dir']) / config['experiment']['name']
    config_save_path = experiment_dir / "final_config.yaml"
    
    # Add training results to config
    config['results'] = {
        'training_duration': str(training_duration),
        'best_val_accuracy': max(history['val_accuracy']),
        'final_train_accuracy': history['train_accuracy'][-1],
        'final_val_accuracy': history['val_accuracy'][-1],
        'total_epochs': len(history['train_loss']),
        'model_parameters': sum(p.numel() for p in model.parameters())
    }
    
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Final configuration saved to: {config_save_path}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 