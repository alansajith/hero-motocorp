"""U-Net model for fine damage segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np
from loguru import logger

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    logger.warning("segmentation_models_pytorch not available, using basic U-Net")

from ..utils.config_loader import get_config


class UNet(nn.Module):
    """Basic U-Net architecture for binary segmentation."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._double_conv(in_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        return self.out(dec1)


class DamageSegmentationModel:
    """Wrapper for U-Net segmentation model."""
    
    def __init__(
        self,
        architecture: str = 'unet',
        encoder: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        device: Optional[str] = None
    ):
        """
        Initialize segmentation model.
        
        Args:
            architecture: Model architecture ('unet', 'unetplusplus')
            encoder: Encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            classes: Number of output classes
            device: Device to use
        """
        config = get_config()
        
        if device is None:
            device = config.device
        
        self.device = device
        self.classes = classes
        
        # Create model
        if SMP_AVAILABLE:
            if architecture == 'unetplusplus':
                self.model = smp.UnetPlusPlus(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=classes,
                )
            else:
                self.model = smp.Unet(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=classes,
                )
            logger.info(f"Created {architecture} with {encoder} encoder")
        else:
            self.model = UNet(in_channels=in_channels, out_channels=classes)
            logger.info("Created basic U-Net")
        
        self.model.to(self.device)
    
    def train_step(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Single training step.
        
        Args:
            image: Input image tensor [B, C, H, W]
            mask: Ground truth mask tensor [B, 1, H, W]
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Loss value
        """
        self.model.train()
        optimizer.zero_grad()
        
        image = image.to(self.device)
        mask = mask.to(self.device)
        
        # Forward pass
        pred = self.model(image)
        loss = criterion(pred, mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """
        Predict segmentation mask.
        
        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W]
            
        Returns:
            Binary mask [H, W]
        """
        self.model.eval()
        
        # Handle single image
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            pred = self.model(image)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
        
        mask = pred.squeeze().cpu().numpy()
        return mask
    
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


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice
