"""Grad-CAM for model explainability."""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, List
from loguru import logger


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN explanations."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = 'cpu'
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute activations on
            device: Device to use
        """
        self.model = model
        self.device = device
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        if target_layer is not None:
            target_layer.register_forward_hook(self._save_activation)
            target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_image: Input tensor [1, C, H, W] or [C, H, W]
            target_class: Target class index (if None, uses prediction)
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Ensure batch dimension
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        input_image = input_image.to(self.device)
        input_image.requires_grad = True
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return np.zeros((input_image.shape[2], input_image.shape[3]))
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU to keep only positive contributions
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        
        return cam
    
    def visualize(
        self,
        input_image: np.ndarray,
        cam: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            input_image: Original image [H, W, C] (RGB)
            cam: CAM heatmap [H, W]
           colormap: OpenCV colormap
            alpha: Blending alpha
            
        Returns:
            Visualization image
        """
        # Apply colormap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        visualization = (heatmap * alpha + input_image * (1 - alpha)).astype(np.uint8)
        
        return visualization


def apply_gradcam_to_classifier(
    classifier,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Apply Grad-CAM to a damage classifier.
    
    Args:
        classifier: DamageClassifier instance
        image: Preprocessed image tensor
        original_image: Original image for visualization
        target_class: Target class (optional)
        colormap: Colormap name
        
    Returns:
        Grad-CAM visualization
    """
    # Find target layer (usually last conv layer)
    target_layer = None
    
    # Try to find last convolutional or feature layer
    for name, module in classifier.model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Sequential)):
            target_layer = module
    
    if target_layer is None:
        logger.warning("Could not find suitable target layer for Grad-CAM")
        return original_image
    
    # Create Grad-CAM
    gradcam = GradCAM(
        model=classifier.model,
        target_layer=target_layer,
        device=classifier.device
    )
    
    # Generate CAM
    cam = gradcam.generate_cam(image, target_class)
    
    # Visualize
    colormap_cv = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_JET)
    visualization = gradcam.visualize(original_image, cam, colormap_cv)
    
    return visualization
