"""EfficientNet-based damage classification model."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import numpy as np
from loguru import logger

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    logger.warning("efficientnet_pytorch not available, using torchvision ResNet")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from ..utils.config_loader import get_config


class DamageClassifier:
    """Damage type classification model."""
    
    def __init__(
        self,
        num_classes: int = 5,
        architecture: str = 'efficientnet-b2',
        pretrained: bool = True,
        dropout: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize damage classifier.
        
        Args:
            num_classes: Number of damage classes
            architecture: Model architecture
            pretrained: Use pretrained weights
            dropout: Dropout rate
            device: Device to use
        """
        config = get_config()
        
        if device is None:
            device = config.device
        
        self.device = device
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Create model
        if TIMM_AVAILABLE:
            # Use timm library (most flexible)
            self.model = timm.create_model(
                'efficientnet_b2',
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout
            )
            logger.info(f"Created EfficientNet-B2 from timm with {num_classes} classes")
        
        elif EFFICIENTNET_AVAILABLE and architecture.startswith('efficientnet'):
            # Use efficientnet_pytorch
            if pretrained:
                self.model = EfficientNet.from_pretrained(
                    architecture.replace('-', '_'),
                    num_classes=num_classes
                )
            else:
                self.model = EfficientNet.from_name(
                    architecture.replace('-', '_'),
                    num_classes=num_classes
                )
            logger.info(f"Created {architecture} with {num_classes} classes")
        
        else:
            # Fallback to ResNet
            import torchvision.models as models
            self.model = models.resnet50(pretrained=pretrained)
            
            # Replace final layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
            logger.info(f"Created ResNet-50 with {num_classes} classes")
        
        self.model.to(self.device)
        
        # Get class names
        self.class_names = config.damage_classes
    
    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> tuple[float, float]:
        """
        Single training step.
        
        Args:
            images: Input images [B, C, H, W]
            labels: Ground truth labels [B]
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        optimizer.zero_grad()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean().item()
        
        return loss.item(), accuracy
    
    def predict(
        self,
        image: torch.Tensor,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Predict damage class.
        
        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W]
            return_probabilities: Return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        result = {
            'class_id': predicted_class,
            'class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"class_{predicted_class}",
            'confidence': confidence_score,
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities.squeeze().cpu().numpy().tolist()
            result['all_classes'] = {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(min(len(self.class_names), probabilities.shape[1]))
            }
        
        return result
    
    def predict_batch(
        self,
        images: torch.Tensor,
        return_probabilities: bool = False
    ) -> list[Dict[str, Any]]:
        """
        Predict damage classes for batch of images.
        
        Args:
            images: Input images [B, C, H, W]
            return_probabilities: Return class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, dim=1)
        
        results = []
        for i in range(len(images)):
            predicted_class = predicted[i].item()
            confidence_score = confidences[i].item()
            
            result = {
                'class_id': predicted_class,
                'class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"class_{predicted_class}",
                'confidence': confidence_score,
            }
            
            if return_probabilities:
                result['probabilities'] = probabilities[i].cpu().numpy().tolist()
                result['all_classes'] = {
                    self.class_names[j]: probabilities[i, j].item()
                    for j in range(min(len(self.class_names), probabilities.shape[1]))
                }
            
            results.append(result)
        
        return results
    
    def save(self, save_path: str):
        """Save model weights."""
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, model_path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Model loaded from {model_path}")
        self.model.to(self.device)
