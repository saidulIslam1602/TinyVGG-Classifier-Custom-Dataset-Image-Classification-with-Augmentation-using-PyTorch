"""
Unit tests for TinyVGG model components.

This module contains comprehensive tests for the TinyVGG model architecture,
including model creation, forward pass, parameter counting, and edge cases.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.tiny_vgg import TinyVGG, TinyVGGBlock, create_tiny_vgg


class TestTinyVGGBlock:
    """Test cases for TinyVGGBlock component."""
    
    def test_block_creation(self):
        """Test that TinyVGGBlock can be created with valid parameters."""
        block = TinyVGGBlock(in_channels=3, out_channels=64)
        assert isinstance(block, nn.Module)
        assert block.conv.in_channels == 3
        assert block.conv.out_channels == 64
    
    def test_block_forward_pass(self):
        """Test forward pass through TinyVGGBlock."""
        block = TinyVGGBlock(in_channels=3, out_channels=64, use_pooling=True)
        x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        
        output = block(x)
        
        # With pooling, spatial dimensions should be halved
        expected_shape = (2, 64, 112, 112)
        assert output.shape == expected_shape
    
    def test_block_without_pooling(self):
        """Test TinyVGGBlock without pooling."""
        block = TinyVGGBlock(in_channels=3, out_channels=64, use_pooling=False)
        x = torch.randn(2, 3, 224, 224)
        
        output = block(x)
        
        # Without pooling, spatial dimensions should remain the same
        expected_shape = (2, 64, 224, 224)
        assert output.shape == expected_shape
    
    def test_block_dropout(self):
        """Test that dropout is applied during training."""
        block = TinyVGGBlock(in_channels=3, out_channels=64, dropout_rate=0.5)
        block.train()  # Set to training mode
        
        x = torch.randn(2, 3, 224, 224)
        output1 = block(x)
        output2 = block(x)
        
        # Outputs should be different due to dropout randomness
        assert not torch.allclose(output1, output2)
    
    def test_block_evaluation_mode(self):
        """Test that dropout is not applied during evaluation."""
        block = TinyVGGBlock(in_channels=3, out_channels=64, dropout_rate=0.5)
        block.eval()  # Set to evaluation mode
        
        x = torch.randn(2, 3, 224, 224)
        
        # Set manual seed for reproducibility
        torch.manual_seed(42)
        output1 = block(x)
        
        torch.manual_seed(42)
        output2 = block(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestTinyVGG:
    """Test cases for the complete TinyVGG model."""
    
    def test_model_creation_default(self):
        """Test model creation with default parameters."""
        model = TinyVGG()
        assert isinstance(model, nn.Module)
        assert model.input_channels == 3
        assert model.output_classes == 3
        assert model.hidden_units == [64, 128, 256]
    
    def test_model_creation_custom(self):
        """Test model creation with custom parameters."""
        hidden_units = [32, 64, 128]
        model = TinyVGG(
            input_channels=1,
            hidden_units=hidden_units,
            output_classes=10,
            dropout_rate=0.1
        )
        
        assert model.input_channels == 1
        assert model.output_classes == 10
        assert model.hidden_units == hidden_units
    
    def test_model_forward_pass(self):
        """Test forward pass through the complete model."""
        model = TinyVGG(num_classes=5)
        x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
        
        output = model(x)
        
        expected_shape = (4, 5)  # Batch size x num_classes
        assert output.shape == expected_shape
    
    def test_model_different_input_sizes(self):
        """Test model with different input sizes."""
        model = TinyVGG()
        
        # Test various input sizes
        input_sizes = [(1, 3, 224, 224), (2, 3, 256, 256), (1, 3, 128, 128)]
        
        for batch_size, channels, height, width in input_sizes:
            x = torch.randn(batch_size, channels, height, width)
            output = model(x)
            assert output.shape == (batch_size, 3)
    
    def test_model_parameter_count(self):
        """Test that model has expected number of parameters."""
        model = TinyVGG()
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should have a reasonable number of parameters (not too small, not too large)
        assert 100_000 < num_params < 10_000_000
    
    def test_model_trainable_parameters(self):
        """Test that all parameters are trainable by default."""
        model = TinyVGG()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        assert trainable_params == total_params
    
    def test_model_feature_extraction(self):
        """Test feature map extraction."""
        model = TinyVGG()
        x = torch.randn(1, 3, 224, 224)
        
        # Test that we can extract feature maps
        features = model.get_feature_maps(x)
        assert features is not None
        assert len(features.shape) == 4  # Should be 4D tensor
    
    def test_model_different_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 5, 10, 100]:
            model = TinyVGG(output_classes=num_classes)
            x = torch.randn(2, 3, 224, 224)
            output = model(x)
            assert output.shape == (2, num_classes)
    
    def test_model_weight_initialization(self):
        """Test that weights are properly initialized."""
        model = TinyVGG()
        
        # Check that conv weights are not all zeros
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))
    
    def test_model_batch_norm_layers(self):
        """Test that batch normalization layers are present."""
        model = TinyVGG()
        
        bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        assert len(bn_layers) > 0  # Should have batch norm layers
    
    def test_model_dropout_layers(self):
        """Test that dropout layers are present."""
        model = TinyVGG(dropout_rate=0.2)
        
        dropout_layers = [module for module in model.modules() 
                         if isinstance(module, (nn.Dropout, nn.Dropout2d))]
        assert len(dropout_layers) > 0  # Should have dropout layers


class TestCreateTinyVGG:
    """Test cases for the factory function."""
    
    def test_create_small_model(self):
        """Test creation of small model variant."""
        model = create_tiny_vgg(num_classes=5, model_size="small")
        
        # Small model should have fewer parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params < 500_000
    
    def test_create_medium_model(self):
        """Test creation of medium model variant."""
        model = create_tiny_vgg(num_classes=5, model_size="medium")
        
        # Medium model should have moderate number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert 500_000 < num_params < 2_000_000
    
    def test_create_large_model(self):
        """Test creation of large model variant."""
        model = create_tiny_vgg(num_classes=5, model_size="large")
        
        # Large model should have more parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 2_000_000
    
    def test_invalid_model_size(self):
        """Test that invalid model size falls back to medium."""
        model = create_tiny_vgg(num_classes=5, model_size="invalid")
        
        # Should create a medium model by default
        num_params = sum(p.numel() for p in model.parameters())
        assert 500_000 < num_params < 2_000_000
    
    def test_different_input_channels(self):
        """Test model creation with different input channels."""
        for channels in [1, 3, 4]:
            model = create_tiny_vgg(num_classes=3, input_channels=channels)
            x = torch.randn(1, channels, 224, 224)
            output = model(x)
            assert output.shape == (1, 3)


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_input(self):
        """Test model with very small input size."""
        model = TinyVGG()
        x = torch.randn(1, 3, 32, 32)  # Very small input
        
        # Should not crash, even with small input
        output = model(x)
        assert output.shape == (1, 3)
    
    def test_single_pixel_input(self):
        """Test model with single pixel input."""
        model = TinyVGG()
        x = torch.randn(1, 3, 1, 1)  # Single pixel
        
        # Should handle gracefully due to adaptive pooling
        output = model(x)
        assert output.shape == (1, 3)
    
    def test_zero_input(self):
        """Test model with zero input."""
        model = TinyVGG()
        x = torch.zeros(1, 3, 224, 224)
        
        output = model(x)
        assert output.shape == (1, 3)
        assert not torch.isnan(output).any()
    
    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = TinyVGG()
        x = torch.randn(64, 3, 224, 224)  # Large batch
        
        output = model(x)
        assert output.shape == (64, 3)
    
    def test_model_eval_train_modes(self):
        """Test switching between eval and train modes."""
        model = TinyVGG(dropout_rate=0.5)
        x = torch.randn(2, 3, 224, 224)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to dropout
        assert output_train.shape == output_eval.shape
        assert output_train.shape == (2, 3)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"]) 