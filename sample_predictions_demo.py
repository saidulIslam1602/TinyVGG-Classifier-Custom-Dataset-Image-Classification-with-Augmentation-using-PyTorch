#!/usr/bin/env python3
"""
Sample Predictions Demo for TinyVGG Food Classifier

This script loads the trained model and generates sample predictions
with comprehensive visualizations to showcase model performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import sys
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.tiny_vgg import create_tiny_vgg
from src.data.dataset import create_data_loaders, DataTransforms
from src.utils.visualization import ModelVisualizer

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PredictionDemo:
    """Generate sample predictions and visualizations."""
    
    def __init__(self, experiment_path: str):
        self.experiment_path = Path(experiment_path)
        self.results_dir = self.experiment_path / "sample_predictions"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load configuration and model
        self.load_model_and_config()
        
        # Set up data
        self.setup_data()
        
    def load_model_and_config(self):
        """Load the trained model and configuration."""
        # Load configuration (handle numpy objects gracefully)
        try:
            with open(self.experiment_path / "final_config.yaml", 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            # If there are numpy objects, load with unsafe loader or create minimal config
            try:
                with open(self.experiment_path / "final_config.yaml", 'r') as f:
                    self.config = yaml.unsafe_load(f)
            except:
                # Create minimal config
                self.config = {
                    'model': {'size': 'medium', 'input_channels': 3, 'num_classes': 3},
                    'data': {'data_dir': 'data', 'batch_size': 8}
                }
        
        # Create model
        self.model = create_tiny_vgg(
            num_classes=self.config['model']['num_classes'],
            input_channels=self.config['model']['input_channels'],
            model_size=self.config['model']['size']
        )
        
        # Load trained weights
        model_path = self.experiment_path / "best_model.pth"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model weights: {e}")
                print("⚠️ Using randomly initialized weights for demo purposes")
        else:
            print("⚠️ No trained model found, using randomly initialized weights")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.class_names = ['pizza', 'steak', 'sushi']
        
    def setup_data(self):
        """Set up data loaders for sampling."""
        try:
            # Try to load data
            train_loader, val_loader, class_names = create_data_loaders(
                data_dir=self.config['data']['data_dir'],
                batch_size=8,  # Small batch for visualization
                num_workers=0
            )
            self.val_loader = val_loader
            self.has_data = True
            print("✅ Data loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load data: {e}")
            print("Will generate synthetic examples instead")
            self.has_data = False
            
    def create_synthetic_samples(self):
        """Create synthetic food-like images for demonstration."""
        samples = []
        labels = []
        
        # Create different patterns for each class
        for class_idx, class_name in enumerate(self.class_names):
            for _ in range(3):  # 3 samples per class
                # Create base image
                img = torch.randn(3, 224, 224)
                
                # Add class-specific patterns
                if class_name == 'pizza':
                    # Pizza: circular with random toppings
                    center = (112, 112)
                    y, x = torch.meshgrid(torch.arange(224), torch.arange(224))
                    mask = ((x - center[0])**2 + (y - center[1])**2) < 80**2
                    img[0][mask] += 0.8  # Red toppings
                    img[1][mask] += 0.6  # Orange cheese
                    img[2][mask] += 0.2  # Brown crust
                    
                elif class_name == 'steak':
                    # Steak: oval with grill marks
                    img += torch.tensor([0.6, 0.3, 0.2]).view(3, 1, 1)  # Brown color
                    # Add grill marks
                    for i in range(0, 224, 20):
                        img[:, i:i+3, :] += 0.3
                        
                elif class_name == 'sushi':
                    # Sushi: rectangular with layers
                    img[0, 50:150, 50:174] += 0.8  # Red fish
                    img[1, 150:200, 50:174] += 0.9  # White rice
                    img[2, 150:200, 50:174] += 0.9
                    img[2, 40:60, 40:184] += 0.2  # Dark nori
                
                # Normalize
                img = torch.clamp(img, 0, 1)
                samples.append(img)
                labels.append(class_name)
                
        return samples, labels
        
    def get_sample_batch(self):
        """Get a batch of samples for visualization."""
        if self.has_data:
            try:
                # Get a batch from validation set
                data_iter = iter(self.val_loader)
                images, labels = next(data_iter)
                
                # Convert labels to class names
                label_names = [self.class_names[label.item()] for label in labels]
                
                return images, label_names
            except:
                pass
        
        # Fallback to synthetic samples
        return self.create_synthetic_samples()
        
    def create_prediction_grid(self):
        """Create a grid showing predictions for multiple samples."""
        # Get samples
        images, true_labels = self.get_sample_batch()
        
        # Limit to 9 images for a 3x3 grid
        num_images = min(9, len(images))
        images = images[:num_images] if torch.is_tensor(images) else images[:num_images]
        true_labels = true_labels[:num_images]
        
        # Create visualizer
        visualizer = ModelVisualizer(self.model, self.class_names, self.device)
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('TinyVGG Food Classifier - Sample Predictions', fontsize=16, fontweight='bold')
        
        for i in range(num_images):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Get image
            img = images[i] if torch.is_tensor(images[i]) else images[i]
            
            # Get prediction
            result = visualizer.predict_single_image(img, top_k=3)
            
            # Display image
            if torch.is_tensor(img):
                img_display = img.permute(1, 2, 0).cpu().numpy()
                img_display = np.clip(img_display, 0, 1)
            else:
                img_display = img
                
            ax.imshow(img_display)
            
            # Create title with prediction info
            pred_class = result['predictions'][0]['class']
            pred_conf = result['predictions'][0]['confidence']
            true_class = true_labels[i]
            
            # Color based on correctness
            color = 'green' if pred_class == true_class else 'red'
            
            title = f"Pred: {pred_class} ({pred_conf:.1f}%)\nTrue: {true_class}"
            ax.set_title(title, color=color, fontweight='bold', fontsize=10)
            ax.axis('off')
            
            # Add confidence bar
            for j, pred in enumerate(result['predictions'][:3]):
                bar_y = 0.95 - j * 0.08
                bar_width = pred['confidence'] / 100 * 0.3
                ax.add_patch(plt.Rectangle((0.02, bar_y), bar_width, 0.05, 
                                         color=plt.cm.viridis(pred['confidence']/100),
                                         alpha=0.8, transform=ax.transAxes))
                ax.text(0.35, bar_y + 0.025, f"{pred['class']}: {pred['confidence']:.1f}%",
                       transform=ax.transAxes, fontsize=8, va='center')
        
        # Hide unused subplots
        for i in range(num_images, 9):
            row, col = i // 3, i % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prediction_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_confidence_analysis(self):
        """Analyze prediction confidence patterns."""
        # Get multiple batches for analysis
        all_confidences = {'correct': [], 'incorrect': []}
        all_predictions = []
        
        # Get samples
        images, true_labels = self.get_sample_batch()
        visualizer = ModelVisualizer(self.model, self.class_names, self.device)
        
        # Collect predictions
        for img, true_label in zip(images, true_labels):
            result = visualizer.predict_single_image(img, top_k=len(self.class_names))
            pred_class = result['predictions'][0]['class']
            pred_conf = result['predictions'][0]['confidence']
            
            if pred_class == true_label:
                all_confidences['correct'].append(pred_conf)
            else:
                all_confidences['incorrect'].append(pred_conf)
                
            all_predictions.append({
                'true': true_label,
                'pred': pred_class,
                'confidence': pred_conf,
                'correct': pred_class == true_label
            })
        
        # Create confidence analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Confidence distribution
        if all_confidences['correct']:
            ax1.hist(all_confidences['correct'], bins=10, alpha=0.7, color='green', 
                    label=f'Correct ({len(all_confidences["correct"])})')
        if all_confidences['incorrect']:
            ax1.hist(all_confidences['incorrect'], bins=10, alpha=0.7, color='red',
                    label=f'Incorrect ({len(all_confidences["incorrect"])})')
        
        ax1.set_xlabel('Confidence (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Per-class confidence
        class_confidences = {cls: [] for cls in self.class_names}
        for pred in all_predictions:
            class_confidences[pred['true']].append(pred['confidence'])
        
        for i, (cls, confs) in enumerate(class_confidences.items()):
            if confs:
                ax2.boxplot(confs, positions=[i], labels=[cls])
        
        ax2.set_ylabel('Confidence (%)')
        ax2.set_title('Confidence by True Class')
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion matrix with confidence
        conf_matrix = np.zeros((len(self.class_names), len(self.class_names)))
        conf_values = np.zeros((len(self.class_names), len(self.class_names)))
        
        for pred in all_predictions:
            true_idx = self.class_names.index(pred['true'])
            pred_idx = self.class_names.index(pred['pred'])
            conf_matrix[true_idx, pred_idx] += 1
            conf_values[true_idx, pred_idx] += pred['confidence']
        
        # Average confidence for each cell
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_conf = np.divide(conf_values, conf_matrix, 
                               out=np.zeros_like(conf_values), where=(conf_matrix!=0))
        
        im = ax3.imshow(avg_conf, cmap='RdYlGn', vmin=0, vmax=100)
        ax3.set_xticks(range(len(self.class_names)))
        ax3.set_yticks(range(len(self.class_names)))
        ax3.set_xticklabels(self.class_names)
        ax3.set_yticklabels(self.class_names)
        ax3.set_xlabel('Predicted Class')
        ax3.set_ylabel('True Class')
        ax3.set_title('Average Confidence by Class Pair')
        
        # Add text annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if conf_matrix[i, j] > 0:
                    text = f'{avg_conf[i, j]:.1f}%\n({int(conf_matrix[i, j])})'
                    ax3.text(j, i, text, ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='Average Confidence (%)')
        
        # 4. Calibration analysis
        confidence_bins = np.arange(0, 101, 10)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            lower, upper = confidence_bins[i], confidence_bins[i + 1]
            mask = [(p['confidence'] >= lower) and (p['confidence'] < upper) for p in all_predictions]
            bin_preds = [p for p, m in zip(all_predictions, mask) if m]
            
            if bin_preds:
                accuracy = sum(p['correct'] for p in bin_preds) / len(bin_preds)
                avg_conf = np.mean([p['confidence'] for p in bin_preds]) / 100
                bin_accuracies.append(accuracy)
                bin_confidences.append(avg_conf)
                bin_counts.append(len(bin_preds))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((lower + upper) / 200)
                bin_counts.append(0)
        
        # Plot calibration
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax4.scatter(bin_confidences, bin_accuracies, s=[c*20 + 10 for c in bin_counts], 
                   alpha=0.7, label='Model Calibration')
        
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            if count > 0:
                ax4.annotate(f'{count}', (conf, acc), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Mean Predicted Probability')
        ax4.set_ylabel('Fraction of Positives')
        ax4.set_title('Reliability Diagram')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_feature_analysis(self):
        """Analyze learned features and feature maps."""
        # Get a sample
        images, true_labels = self.get_sample_batch()
        sample_img = images[0] if torch.is_tensor(images[0]) else images[0]
        sample_label = true_labels[0]
        
        # Get feature maps
        if hasattr(self.model, 'get_feature_maps'):
            try:
                # Get feature maps from different layers
                feature_maps = []
                layer_names = []
                
                for i, layer in enumerate(self.model.features):
                    if isinstance(layer, torch.nn.Conv2d):
                        features = self.model.get_feature_maps(sample_img.unsqueeze(0).to(self.device), layer_idx=i)
                        feature_maps.append(features.cpu())
                        layer_names.append(f'Conv Layer {i+1}')
                        
                if feature_maps:
                    self.visualize_feature_maps(sample_img, sample_label, feature_maps, layer_names)
            except Exception as e:
                print(f"Could not extract feature maps: {e}")
        
        # Create filter visualization
        self.visualize_filters()
        
    def visualize_feature_maps(self, sample_img, sample_label, feature_maps, layer_names):
        """Visualize feature maps from different layers."""
        num_layers = min(3, len(feature_maps))  # Show up to 3 layers
        
        fig = plt.figure(figsize=(20, 5 * num_layers))
        
        for layer_idx in range(num_layers):
            features = feature_maps[layer_idx][0]  # Remove batch dimension
            layer_name = layer_names[layer_idx]
            
            # Show original image in first row
            if layer_idx == 0:
                ax_orig = plt.subplot(num_layers, 9, 1)
                if torch.is_tensor(sample_img):
                    img_display = sample_img.permute(1, 2, 0).cpu().numpy()
                    img_display = np.clip(img_display, 0, 1)
                else:
                    img_display = sample_img
                ax_orig.imshow(img_display)
                ax_orig.set_title(f'Input\n({sample_label})')
                ax_orig.axis('off')
            
            # Show feature maps (first 8 channels)
            num_channels = min(8, features.shape[0])
            for ch in range(num_channels):
                ax = plt.subplot(num_layers, 9, layer_idx * 9 + ch + 2)
                feature = features[ch].detach().numpy()
                
                # Normalize for visualization
                feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
                
                ax.imshow(feature, cmap='viridis')
                ax.set_title(f'{layer_name}\nCh {ch+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_filters(self):
        """Visualize learned convolutional filters."""
        # Get first convolutional layer
        first_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is None:
            return
            
        # Get weights
        weights = first_conv.weight.data.cpu()
        num_filters = min(16, weights.shape[0])  # Show up to 16 filters
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Learned Convolutional Filters (First Layer)', fontsize=14, fontweight='bold')
        
        for i in range(num_filters):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            # Get filter
            filter_weights = weights[i]
            
            # If 3 channels, show as RGB
            if filter_weights.shape[0] == 3:
                # Normalize to [0, 1]
                filter_norm = filter_weights.permute(1, 2, 0)
                filter_norm = (filter_norm - filter_norm.min()) / (filter_norm.max() - filter_norm.min() + 1e-8)
                ax.imshow(filter_norm)
            else:
                # Show first channel in grayscale
                filter_viz = filter_weights[0]
                ax.imshow(filter_viz, cmap='RdBu_r')
            
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, 16):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'learned_filters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_demos(self):
        """Generate all demonstration visualizations."""
        print("🎨 Generating Sample Prediction Demonstrations...")
        print("=" * 60)
        
        print("📸 Creating prediction grid...")
        self.create_prediction_grid()
        
        print("📊 Analyzing prediction confidence...")
        self.create_confidence_analysis()
        
        print("🧠 Analyzing learned features...")
        self.create_feature_analysis()
        
        print("✅ All demonstrations generated successfully!")
        print(f"📁 Results saved to: {self.results_dir}")
        
        # List all generated files
        generated_files = list(self.results_dir.glob("*"))
        print("\n📄 Generated Files:")
        for file in generated_files:
            print(f"   • {file.name}")


def main():
    """Main function to generate sample prediction demonstrations."""
    experiment_path = "experiments/demo_run"
    
    if not Path(experiment_path).exists():
        print(f"❌ Experiment path not found: {experiment_path}")
        print("Please run training first or specify correct experiment path.")
        return
    
    # Create demo generator
    demo = PredictionDemo(experiment_path)
    demo.generate_all_demos()
    
    print("\n🚀 Sample prediction demonstrations completed!")
    print("These demonstrations showcase:")
    print("• Model predictions on sample images")
    print("• Confidence analysis and calibration")
    print("• Learned feature visualization")
    print("• Filter analysis and interpretability")


if __name__ == "__main__":
    main()