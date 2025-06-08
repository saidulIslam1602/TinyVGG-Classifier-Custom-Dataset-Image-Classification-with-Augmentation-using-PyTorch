"""
FastAPI Application for TinyVGG Image Classification

This module provides a REST API for image classification using the trained TinyVGG model.
Includes endpoints for single image prediction, batch prediction, and model health checks.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import logging
from typing import List, Dict, Optional, Any
import yaml
from pathlib import Path
import time
import numpy as np
from datetime import datetime

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.tiny_vgg import create_tiny_vgg
from data.dataset import DataTransforms


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="TinyVGG Image Classifier API",
    description="A production-ready API for food image classification using TinyVGG model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and related components
model = None
device = None
class_names = None
transforms = None
model_loaded_at = None


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.class_names = None
        self.transforms = None
        self.model_info = {}
        
    def load_model(self, model_path: str) -> bool:
        """Load the trained model."""
        try:
            # Determine device
            if config['deployment']['device'] == 'auto':
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config['deployment']['device'])
            
            logger.info(f"Loading model on device: {self.device}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration
            model_config = checkpoint.get('model_config', config['model'])
            self.class_names = checkpoint.get('class_names', config['data']['class_names'])
            
            # Create model
            self.model = create_tiny_vgg(
                num_classes=len(self.class_names),
                input_channels=model_config.get('input_channels', 3),
                model_size=model_config.get('size', 'medium')
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transforms = DataTransforms.get_inference_transforms(
                image_size=config['data']['image_size']
            )
            
            # Store model info
            self.model_info = {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size': model_config.get('size', 'medium'),
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'device': str(self.device),
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Model loaded successfully: {self.model_info}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """Make prediction on a single image."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            start_time = time.time()
            
            # Preprocess image
            image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)), dim=1)
                
                predictions = []
                for i in range(top_probs.shape[1]):
                    class_idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    predictions.append({
                        'class': self.class_names[class_idx],
                        'probability': float(prob),
                        'confidence': float(prob * 100)
                    })
            
            inference_time = time.time() - start_time
            
            return {
                'predictions': predictions,
                'inference_time_ms': round(inference_time * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = config['inference']['model_path']
    if Path(model_path).exists():
        success = model_manager.load_model(model_path)
        if not success:
            logger.error("Failed to load model on startup")
    else:
        logger.warning(f"Model file not found: {model_path}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "TinyVGG Image Classifier API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model_manager.model is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "device": str(model_manager.device) if model_manager.device else "unknown"
    }


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get model information."""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_manager.model_info


@app.post("/predict", tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = 3
):
    """
    Predict class for uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
        top_k: Number of top predictions to return (default: 3)
    """
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    # Check file size
    max_size = config['api']['max_file_size']
    if hasattr(file, 'size') and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size} bytes"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Make prediction
        result = model_manager.predict(image, top_k=top_k)
        
        # Add metadata
        result['filename'] = file.filename
        result['file_size'] = len(contents)
        result['image_size'] = image.size
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = 3
):
    """
    Predict classes for multiple images.
    
    Args:
        files: List of image files
        top_k: Number of top predictions to return for each image
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            # Validate file
            if not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'File must be an image'
                })
                continue
            
            # Read and process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Make prediction
            result = model_manager.predict(image, top_k=top_k)
            result['filename'] = file.filename
            result['file_size'] = len(contents)
            result['image_size'] = image.size
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'results': results,
        'processed_count': len(results),
        'timestamp': datetime.now().isoformat()
    }


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from disk."""
    model_path = config['inference']['model_path']
    
    if not Path(model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}"
        )
    
    success = model_manager.load_model(model_path)
    
    if success:
        return {
            "message": "Model reloaded successfully",
            "model_info": model_manager.model_info
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to reload model"
        )


@app.get("/config", tags=["Configuration"])
async def get_config():
    """Get API configuration (filtered for security)."""
    # Return only safe configuration parameters
    safe_config = {
        'model': config['model'],
        'data': {
            'class_names': config['data']['class_names'],
            'image_size': config['data']['image_size']
        },
        'api': {
            'max_file_size': config['api']['max_file_size'],
            'allowed_extensions': config['api']['allowed_extensions']
        }
    }
    return safe_config


@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    """Handle file too large errors."""
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['reload']
    ) 