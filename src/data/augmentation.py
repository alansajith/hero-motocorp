"""Data augmentation pipeline using Albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, Any
from loguru import logger

from ..utils.config_loader import get_config


def get_train_transforms(config: Dict[str, Any] = None) -> A.Compose:
    """
    Get training data augmentation pipeline.
    
    Args:
        config: Augmentation configuration dict
        
    Returns:
        Albumentations composition
    """
    if config is None:
        cfg = get_config()
        config = cfg.get('augmentation.train', {})
    
    transforms = [
        # Geometric transforms
        A.HorizontalFlip(p=config.get('horizontal_flip', 0.5)),
        A.VerticalFlip(p=config.get('vertical_flip', 0.0)),
        A.Rotate(limit=config.get('rotation_limit', 15), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=config.get('rotation_limit', 15),
            p=0.5
        ),
        
        # Color and brightness
        A.RandomBrightnessContrast(
            brightness_limit=config.get('brightness_limit', 0.2),
            contrast_limit=config.get('contrast_limit', 0.2),
            p=0.5
        ),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        
        # Blur and noise
        A.OneOf([
            A.MotionBlur(blur_limit=config.get('blur_limit', 3), p=1.0),
            A.MedianBlur(blur_limit=config.get('blur_limit', 3), p=1.0),
            A.GaussianBlur(blur_limit=config.get('blur_limit', 3), p=1.0),
        ], p=0.3),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=config.get('noise_prob', 0.3)),
        
        # Occlusion simulation
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=config.get('occlusion_prob', 0.2)
        ),
        
        # Weather effects
        A.OneOf([
            A.RandomRain(p=1.0),
            A.RandomFog(p=1.0),
            A.RandomSunFlare(p=1.0),
        ], p=0.1),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
    
    logger.info(f"Created training augmentation pipeline with {len(transforms)} transforms")
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
    )


def get_val_transforms() -> A.Compose:
    """
    Get validation/test data augmentation pipeline (no augmentation, just normalization).
    
    Returns:
        Albumentations composition
    """
    transforms = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        )
    )


def get_segmentation_transforms(image_size: int = 256, is_train: bool = True) -> A.Compose:
    """
    Get augmentation pipeline for segmentation tasks.
    
    Args:
        image_size: Target image size
        is_train: Whether this is for training
        
    Returns:
        Albumentations composition
    """
    if is_train:
        cfg = get_config()
        config = cfg.get('augmentation.train', {})
        
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=config.get('horizontal_flip', 0.5)),
            A.Rotate(limit=config.get('rotation_limit', 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.get('brightness_limit', 0.2),
                contrast_limit=config.get('contrast_limit', 0.2),
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=config.get('noise_prob', 0.3)),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms, additional_targets={'mask': 'mask'})


def get_classification_transforms(image_size: int = 224, is_train: bool = True) -> A.Compose:
    """
    Get augmentation pipeline for classification tasks.
    
    Args:
        image_size: Target image size
        is_train: Whether this is for training
        
    Returns:
        Albumentations composition
    """
    if is_train:
        cfg = get_config()
        config = cfg.get('augmentation.train', {})
        
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=config.get('horizontal_flip', 0.5)),
            A.Rotate(limit=config.get('rotation_limit', 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.get('brightness_limit', 0.2),
                contrast_limit=config.get('contrast_limit', 0.2),
                p=0.5
            ),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=config.get('noise_prob', 0.3)),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms)
