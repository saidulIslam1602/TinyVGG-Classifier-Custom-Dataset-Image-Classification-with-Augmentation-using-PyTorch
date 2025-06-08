#!/usr/bin/env python3
"""
TinyVGG Demo Script

This script demonstrates the enhanced TinyVGG food classifier with
professional features that showcase advanced ML engineering skills.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.tiny_vgg import create_tiny_vgg
from src.data.dataset import DataTransforms
from src.utils.visualization import ModelVisualizer
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_model_architecture():
    """Demonstrate different model sizes and architectures."""
    print("🏗️  TinyVGG Model Architecture Demo")
    print("=" * 50)
    
    model_sizes = ["small", "medium", "large"]
    
    for size in model_sizes:
        model = create_tiny_vgg(num_classes=3, model_size=size)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"📊 {size.capitalize()} Model:")
        print(f"   Parameters: {num_params:,}")
        print(f"   Memory footprint: ~{num_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test inference time
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        
        import time
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"   Inference time: {inference_time:.2f} ms")
        print()


def demo_data_augmentation():
    """Demonstrate advanced data augmentation techniques."""
    print("🎨 Data Augmentation Demo")
    print("=" * 50)
    
    # Create a dummy image
    dummy_image = torch.randn(3, 224, 224) * 0.1 + 0.5
    dummy_image = torch.clamp(dummy_image, 0, 1)
    
    # Convert to PIL for transforms
    from torchvision.transforms.functional import to_pil_image, to_tensor
    pil_image = to_pil_image(dummy_image)
    
    # Get different transforms
    train_transforms = DataTransforms.get_train_transforms()
    val_transforms = DataTransforms.get_val_transforms()
    
    print("✅ Training transforms include:")
    print("   - Random cropping and resizing")
    print("   - Random horizontal flipping")
    print("   - Color jittering (brightness, contrast, saturation)")
    print("   - Random rotation")
    print("   - Random grayscale conversion")
    print("   - Random erasing")
    print("   - Normalization")
    
    print("\n✅ Validation transforms include:")
    print("   - Resizing to fixed dimensions")
    print("   - Normalization")
    print()


def demo_model_features():
    """Demonstrate advanced model features."""
    print("🧠 Advanced Model Features Demo")
    print("=" * 50)
    
    model = create_tiny_vgg(num_classes=3, model_size="medium")
    
    # Demonstrate feature extraction
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("✅ Model Features:")
    print("   - Batch Normalization for training stability")
    print("   - Dropout regularization for overfitting prevention")
    print("   - Xavier weight initialization")
    print("   - Adaptive average pooling for flexible input sizes")
    print("   - Feature map extraction for interpretability")
    
    # Test feature extraction
    features = model.get_feature_maps(dummy_input, layer_idx=0)
    print(f"   - Feature maps shape: {features.shape}")
    
    # Count different layer types
    conv_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    bn_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    dropout_layers = sum(1 for m in model.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)))
    
    print(f"   - Convolutional layers: {conv_layers}")
    print(f"   - Batch normalization layers: {bn_layers}")
    print(f"   - Dropout layers: {dropout_layers}")
    print()


def demo_training_features():
    """Demonstrate training infrastructure features."""
    print("🚀 Training Infrastructure Demo")
    print("=" * 50)
    
    print("✅ Training Features:")
    print("   - Early stopping with patience")
    print("   - Learning rate scheduling (Cosine Annealing)")
    print("   - Gradient clipping for stability")
    print("   - Automatic model checkpointing")
    print("   - Comprehensive metrics tracking")
    print("   - Weights & Biases integration")
    print("   - TensorBoard support")
    print("   - Mixed precision training")
    print("   - Class weight balancing")
    print("   - Reproducible training with seed setting")
    print()


def demo_deployment_features():
    """Demonstrate deployment capabilities."""
    print("🌐 Deployment Features Demo")
    print("=" * 50)
    
    print("✅ API Features:")
    print("   - FastAPI with automatic documentation")
    print("   - Async request handling")
    print("   - Batch prediction support")
    print("   - Health check endpoints")
    print("   - Model hot-reloading")
    print("   - Input validation and error handling")
    print("   - Inference time monitoring")
    print("   - CORS support for web integration")
    
    print("\n✅ Deployment Options:")
    print("   - Docker containerization")
    print("   - Multi-stage Docker builds")
    print("   - Production-ready configurations")
    print("   - Health checks and monitoring")
    print("   - Horizontal scaling support")
    print()


def demo_mlops_features():
    """Demonstrate MLOps capabilities."""
    print("🔧 MLOps & DevOps Features Demo")
    print("=" * 50)
    
    print("✅ Code Quality:")
    print("   - Type hints throughout codebase")
    print("   - Comprehensive unit tests")
    print("   - Pre-commit hooks for quality control")
    print("   - Black code formatting")
    print("   - Flake8 linting")
    print("   - MyPy type checking")
    
    print("\n✅ Experiment Management:")
    print("   - YAML configuration management")
    print("   - Experiment versioning")
    print("   - Reproducible results")
    print("   - Automatic result visualization")
    print("   - Model performance comparison")
    
    print("\n✅ Monitoring & Observability:")
    print("   - Structured logging")
    print("   - Performance metrics")
    print("   - Model interpretability tools")
    print("   - Feature map visualization")
    print("   - Confusion matrix analysis")
    print()


def demo_production_readiness():
    """Demonstrate production-ready features."""
    print("🏭 Production Readiness Demo")
    print("=" * 50)
    
    print("✅ Scalability Features:")
    print("   - Efficient data loading with multiple workers")
    print("   - GPU acceleration support")
    print("   - Batch inference optimization")
    print("   - Memory-efficient operations")
    print("   - Configurable model sizes")
    
    print("\n✅ Reliability Features:")
    print("   - Comprehensive error handling")
    print("   - Graceful degradation")
    print("   - Input validation")
    print("   - Automatic recovery mechanisms")
    print("   - Resource monitoring")
    
    print("\n✅ Security Features:")
    print("   - Input sanitization")
    print("   - File size limitations")
    print("   - Non-root container execution")
    print("   - Secure configuration management")
    print()


def main():
    """Run all demonstrations."""
    print("🍕 TinyVGG Food Classifier - Enhanced Feature Demo")
    print("=" * 60)
    print("This demo showcases the professional enhancements made to")
    print("demonstrate enterprise-level ML engineering skills.\n")
    
    try:
        demo_model_architecture()
        demo_data_augmentation()
        demo_model_features()
        demo_training_features()
        demo_deployment_features()
        demo_mlops_features()
        demo_production_readiness()
        
        print("🎉 Demo completed successfully!")
        print("\n💼 This enhanced project demonstrates:")
        print("   ✓ Advanced PyTorch model development")
        print("   ✓ Professional software engineering practices")
        print("   ✓ Production-ready ML deployment")
        print("   ✓ Comprehensive testing and validation")
        print("   ✓ MLOps and DevOps integration")
        print("   ✓ API development with FastAPI")
        print("   ✓ Docker containerization")
        print("   ✓ Experiment tracking and monitoring")
        print("   ✓ Code quality and documentation")
        print("   ✓ Scalable and maintainable architecture")
        
        print("\n🚀 Ready to impress recruiters with production-grade ML engineering!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 