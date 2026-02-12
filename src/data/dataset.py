"""PyTorch Dataset classes for vehicle damage detection."""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from loguru import logger

from .augmentation import get_train_transforms, get_val_transforms
from ..utils.config_loader import get_config


class DamageDetectionDataset(Dataset):
    """
    Dataset for vehicle damage detection with bounding boxes and segmentation masks.
    
    Supports YOLO and COCO annotation formats.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 640,
        transforms: Optional[Any] = None,
        annotation_format: str = 'yolo'
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing images and annotations
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size
            transforms: Albumentations transforms
            annotation_format: 'yolo' or 'coco'
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.image_size = image_size
        self.annotation_format = annotation_format
        
        # Get transforms
        if transforms is None:
            if split == 'train':
                self.transforms = get_train_transforms()
            else:
                self.transforms = get_val_transforms()
        else:
            self.transforms = transforms
        
        # Load image paths
        self.image_dir = self.data_dir / 'images'
        self.label_dir = self.data_dir / 'labels'
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')) + 
                                 list(self.image_dir.glob('*.png')) +
                                 list(self.image_dir.glob('*.jpeg')))
        
        logger.info(f"Loaded {len(self.image_paths)} images for {split} split")
        
        # Load class names
        config = get_config()
        self.class_names = config.damage_classes
        self.num_classes = len(self.class_names)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        if self.annotation_format == 'yolo':
            boxes, labels, masks = self._load_yolo_annotations(image_path, image.shape)
        else:
            boxes, labels, masks = self._load_coco_annotations(image_path, image.shape)
        
        # Apply transforms
        if self.transforms is not None and len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            # No boxes, just transform image
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            image = transform(image=image)['image']
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long),
            'image_path': str(image_path),
        }
    
    def _load_yolo_annotations(
        self, 
        image_path: Path, 
        image_shape: Tuple[int, int, int]
    ) -> Tuple[List, List, Optional[List]]:
        """Load YOLO format annotations."""
        label_path = self.label_dir / f"{image_path.stem}.txt"
        
        boxes = []
        labels = []
        masks = None
        
        if not label_path.exists():
            return boxes, labels, masks
        
        h, w = image_shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                # YOLO format: class_id x_center y_center width height (normalized)
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert to Pascal VOC format (x1, y1, x2, y2)
                x1 = (x_center - width / 2) * w
                y1 = (y_center - height / 2) * h
                x2 = (x_center + width / 2) * w
                y2 = (y_center + height / 2) * h
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        
        return boxes, labels, masks
    
    def _load_coco_annotations(
        self,
        image_path: Path,
        image_shape: Tuple[int, int, int]
    ) -> Tuple[List, List, Optional[List]]:
        """Load COCO format annotations."""
        # Placeholder for COCO format support
        # Implement based on your COCO annotation structure
        return [], [], None


class SegmentationDataset(Dataset):
    """Dataset for fine damage segmentation using U-Net."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: int = 256,
        is_train: bool = True
    ):
        """
        Initialize segmentation dataset.
        
        Args:
            images_dir: Directory with cropped damage images
            masks_dir: Directory with segmentation masks
            image_size: Target image size
            is_train: Whether this is training data
        """
        from .augmentation import get_segmentation_transforms
        
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.is_train = is_train
        
        self.image_paths = sorted(list(self.images_dir.glob('*.jpg')) + 
                                 list(self.images_dir.glob('*.png')))
        
        self.transforms = get_segmentation_transforms(image_size, is_train)
        
        logger.info(f"Loaded {len(self.image_paths)} images for segmentation")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Apply transforms
        transformed = self.transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert mask to float and add channel dimension
        mask = mask.unsqueeze(0).float() / 255.0
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(image_path),
        }


class ClassificationDataset(Dataset):
    """Dataset for damage type classification."""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        is_train: bool = True
    ):
        """
        Initialize classification dataset.
        
        Args:
            data_dir: Directory with subdirectories for each class
            image_size: Target image size
            is_train: Whether this is training data
        """
        from .augmentation import get_classification_transforms
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.is_train = is_train
        
        # Get class names from subdirectories
        config = get_config()
        self.class_names = config.damage_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load image paths and labels
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        self.transforms = get_classification_transforms(image_size, is_train)
        
        logger.info(f"Loaded {len(self.samples)} images for classification")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        image_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transforms(image=image)
        image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': image_path,
        }
