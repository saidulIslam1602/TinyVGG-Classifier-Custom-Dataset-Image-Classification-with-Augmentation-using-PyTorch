"""
Visualization Utilities for TinyVGG Model

This module provides utilities for visualizing model predictions, 
feature maps, and other interpretability tools.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
import seaborn as sns
from pathlib import Path
import cv2


class ModelVisualizer:
    """Utility class for visualizing model predictions and features."""
    
    def __init__(self, model: torch.nn.Module, class_names: List[str], device: torch.device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model.eval()
    
    def predict_single_image(self, image: torch.Tensor, top_k: int = 3) -> Dict:
        """
        Make prediction on a single image and return top-k results.
        
        Args:
            image: Input image tensor
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            image = image.to(self.device)
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
            
            predictions = []
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                predictions.append({
                    'class': self.class_names[class_idx],
                    'probability': prob,
                    'confidence': prob * 100
                })
        
        return {
            'predictions': predictions,
            'raw_outputs': outputs.cpu(),
            'probabilities': probabilities.cpu()
        }
    
    def visualize_prediction(
        self, 
        image: torch.Tensor, 
        true_label: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize model prediction for a single image.
        
        Args:
            image: Input image tensor
            true_label: True label for the image (optional)
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Get prediction
        result = self.predict_single_image(image, top_k=len(self.class_names))
        
        # Prepare image for display
        img_display = self._tensor_to_display_image(image)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(img_display)
        ax1.axis('off')
        title = f"Predicted: {result['predictions'][0]['class']}"
        if true_label:
            title += f"\nTrue: {true_label}"
            # Color code based on correctness
            color = 'green' if result['predictions'][0]['class'] == true_label else 'red'
            ax1.set_title(title, color=color, fontsize=12, fontweight='bold')
        else:
            ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # Create probability bar chart
        classes = [pred['class'] for pred in result['predictions']]
        probs = [pred['probability'] for pred in result['predictions']]
        
        bars = ax2.barh(classes, probs, color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probabilities')
        ax2.set_xlim(0, 1)
        
        # Add probability text on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_feature_maps(
        self, 
        image: torch.Tensor, 
        layer_idx: int = 0,
        max_channels: int = 16,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize feature maps from a specific layer.
        
        Args:
            image: Input image tensor
            layer_idx: Index of the layer to visualize
            max_channels: Maximum number of channels to display
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Get feature maps
        with torch.no_grad():
            feature_maps = self.model.get_feature_maps(image, layer_idx)
        
        feature_maps = feature_maps.squeeze(0).cpu()  # Remove batch dimension
        num_channels = min(feature_maps.shape[0], max_channels)
        
        # Create subplot grid
        cols = 4
        rows = (num_channels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_channels):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            feature_map = feature_maps[i].numpy()
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Channel {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_channels, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        fig.suptitle(f'Feature Maps - Layer {layer_idx}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_grad_cam(
        self, 
        image: torch.Tensor, 
        target_class: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, plt.Figure]:
        """
        Generate Grad-CAM visualization.
        
        Args:
            image: Input image tensor
            target_class: Target class for Grad-CAM (if None, uses predicted class)
            save_path: Path to save the visualization
            
        Returns:
            Tuple of (grad_cam_heatmap, matplotlib_figure)
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(image)
        
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = outputs[0, target_class]
        class_score.backward()
        
        # Get feature maps and gradients from the last convolutional layer
        # This is a simplified version - in practice, you'd want to hook into specific layers
        gradients = image.grad
        
        # Generate heatmap (simplified approach)
        heatmap = gradients.abs().mean(dim=1).squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img_display = self._tensor_to_display_image(image.squeeze())
        ax1.imshow(img_display)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        ax2.imshow(heatmap_resized, cmap='jet')
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(img_display)
        ax3.imshow(heatmap_resized, cmap='jet', alpha=0.4)
        ax3.set_title(f'Overlay - Class: {self.class_names[target_class]}')
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return heatmap_resized, fig
    
    def visualize_batch_predictions(
        self, 
        images: torch.Tensor, 
        true_labels: Optional[List[str]] = None,
        num_images: int = 8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize predictions for a batch of images.
        
        Args:
            images: Batch of image tensors
            true_labels: True labels for the images
            num_images: Number of images to display
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        num_images = min(num_images, images.shape[0])
        cols = 4
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_images):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            # Get prediction
            result = self.predict_single_image(images[i])
            pred_class = result['predictions'][0]['class']
            pred_prob = result['predictions'][0]['probability']
            
            # Display image
            img_display = self._tensor_to_display_image(images[i])
            ax.imshow(img_display)
            
            # Create title
            title = f"Pred: {pred_class}\nConf: {pred_prob:.3f}"
            if true_labels:
                true_class = true_labels[i]
                title += f"\nTrue: {true_class}"
                color = 'green' if pred_class == true_class else 'red'
                ax.set_title(title, color=color, fontsize=10)
            else:
                ax.set_title(title, fontsize=10)
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_images, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _tensor_to_display_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable numpy array."""
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        
        # Convert to numpy and transpose
        img_np = tensor.detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # Clip values to [0, 1]
        img_np = np.clip(img_np, 0, 1)
        
        return img_np


def plot_dataset_distribution(class_counts: Dict[str, int], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Bar plot
    bars = ax1.bar(classes, counts, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Dataset Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(counts),
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90,
           colors=plt.cm.Set3(np.linspace(0, 1, len(classes))))
    ax2.set_title('Class Proportion')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_comparison_plot(
    models_history: Dict[str, Dict], 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison plots for multiple model training histories.
    
    Args:
        models_history: Dictionary mapping model names to their training histories
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_history)))
    
    for (model_name, history), color in zip(models_history.items(), colors):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training loss
        ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', 
                label=f'{model_name} (Train)', alpha=0.8)
        
        # Validation loss
        ax2.plot(epochs, history['val_loss'], color=color, linestyle='--', 
                label=f'{model_name} (Val)', alpha=0.8)
        
        # Training accuracy
        ax3.plot(epochs, history['train_accuracy'], color=color, linestyle='-', 
                label=f'{model_name} (Train)', alpha=0.8)
        
        # Validation accuracy
        ax4.plot(epochs, history['val_accuracy'], color=color, linestyle='--', 
                label=f'{model_name} (Val)', alpha=0.8)
    
    # Customize plots
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Training Accuracy Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Validation Accuracy Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully") 