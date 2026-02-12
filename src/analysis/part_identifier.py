"""Vehicle part identification using YOLO."""

import numpy as np
from typing import Optional, Dict, Any, List
from loguru import logger
from ultralytics import YOLO
from pathlib import Path

from ..utils.config_loader import get_config


class VehiclePartIdentifier:
    """Identifies vehicle parts in images using YOLO."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize part identifier.
        
        Args:
            model_path: Path to trained model. If None, uses pretrained model.
            conf_threshold: Confidence threshold
            device: Device to use
        """
        config = get_config()
        
        if device is None:
            device = config.device
        
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load model
        if model_path is None or not Path(model_path).exists():
            # Use pretrained model (or YOLOv8s as placeholder)
            model_name = config.get('part_identification.model_name', 'yolov8s.pt')
            logger.info(f"Loading pretrained YOLO model for part identification: {model_name}")
            self.model = YOLO(model_name)
        else:
            logger.info(f"Loading part identification model from: {model_path}")
            self.model = YOLO(model_path)
        
        # Vehicle part names
        self.part_names = config.vehicle_parts
        
        logger.info(f"Initialized vehicle part identifier with {len(self.part_names)} parts")
    
    def identify(
        self,
        image: np.ndarray,
        damage_box: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Identify vehicle part in image.
        
        Args:
            image: Input image
            damage_box: Damage bounding box [x1, y1, x2, y2] (optional)
            
        Returns:
            Dictionary with part identification results
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return {
                'part_detected': False,
                'part_name': 'unknown',
                'confidence': 0.0
            }
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        
        # If damage box provided, find overlapping part
        if damage_box is not None and len(boxes) > 0:
            best_iou = 0
            best_idx = 0
            
            for i, part_box in enumerate(boxes):
                iou = self._calculate_iou(damage_box, part_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou > 0.1:  # Some overlap
                part_id = labels[best_idx]
                part_name = self.part_names[part_id] if part_id < len(self.part_names) else f"part_{part_id}"
                confidence = float(scores[best_idx])
                
                return {
                    'part_detected': True,
                    'part_name': part_name,
                    'confidence': confidence,
                    'box': boxes[best_idx].tolist()
                }
        
        # Otherwise return highest confidence detection
        if len(scores) > 0:
            best_idx = np.argmax(scores)
            part_id = labels[best_idx]
            part_name = self.part_names[part_id] if part_id < len(self.part_names) else f"part_{part_id}"
            
            return {
                'part_detected': True,
                'part_name': part_name,
                'confidence': float(scores[best_idx]),
                'box': boxes[best_idx].tolist()
            }
        
        return {
            'part_detected': False,
            'part_name': 'unknown',
            'confidence': 0.0
        }
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
