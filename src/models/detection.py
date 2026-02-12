"""YOLOv8 damage detection model wrapper."""

import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from loguru import logger

from ..utils.config_loader import get_config


class DamageDetector:
    """YOLOv8 Segmentation model for damage detection and localization."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize damage detector.
        
        Args:
            model_path: Path to trained model weights. If None, uses pretrained model.
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use ('mps', 'cuda', 'cpu'). If None, auto-detect.
        """
        config = get_config()
        
        if device is None:
            device = config.device
        
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        if model_path is None or not Path(model_path).exists():
            # Use pretrained model
            model_name = config.get('detection.model_name', 'yolov8m-seg.pt')
            logger.info(f"Loading pretrained YOLOv8 model: {model_name}")
            self.model = YOLO(model_name)
        else:
            logger.info(f"Loading trained model from: {model_path}")
            self.model = YOLO(model_path)
        
        logger.info(f"Using device: {self.device}")
        
        # Get class names
        self.class_names = config.damage_classes
        self.num_classes = len(self.class_names)
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        save_dir: str = 'models/detection'
    ) -> Dict[str, Any]:
        """
        Train the YOLOv8 model.
        
        Args:
            data_yaml: Path to data.yaml configuration file
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size
            save_dir: Directory to save trained model
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting YOLOv8 training for {epochs} epochs")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=self.device,
            project=save_dir,
            name='damage_detection',
            patience=20,
            save=True,
            plots=True,
        )
        
        logger.info("Training completed")
        return results
    
    def predict(
        self,
        image: np.ndarray,
        return_masks: bool = True
    ) -> Dict[str, Any]:
        """
        Predict damages in an image.
        
        Args:
            image: Input image (BGR or RGB numpy array)
            return_masks: Whether to return segmentation masks
            
        Returns:
            Dictionary with detections:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - scores: List of confidence scores
                - labels: List of class labels
                - masks: List of segmentation masks (if return_masks=True)
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = {
            'boxes': [],
            'scores': [],
            'labels': [],
            'masks': [],
            'class_names': []
        }
        
        if len(results) == 0:
            return detections
        
        result = results[0]
        
        # Extract boxes and scores
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(int)
            
            detections['boxes'] = boxes.tolist()
            detections['scores'] = scores.tolist()
            detections['labels'] = labels.tolist()
            detections['class_names'] = [self.class_names[label] if label < len(self.class_names) 
                                        else f"class_{label}" for label in labels]
        
        # Extract masks
        if return_masks and result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # [N, H, W]
            detections['masks'] = masks.tolist()
        
        return detections
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        return_masks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict damages in multiple images.
        
        Args:
            images: List of input images
            return_masks: Whether to return segmentation masks
            
        Returns:
            List of detection dictionaries
        """
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        all_detections = []
        
        for result in results:
            detections = {
                'boxes': [],
                'scores': [],
                'labels': [],
                'masks': [],
                'class_names': []
            }
            
            # Extract boxes and scores
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)
                
                detections['boxes'] = boxes.tolist()
                detections['scores'] = scores.tolist()
                detections['labels'] = labels.tolist()
                detections['class_names'] = [self.class_names[label] if label < len(self.class_names)
                                            else f"class_{label}" for label in labels]
            
            # Extract masks
            if return_masks and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                detections['masks'] = masks.tolist()
            
            all_detections.append(detections)
        
        return all_detections
    
    def save(self, save_path: str):
        """Save model weights."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, model_path: str):
        """Load model weights."""
        self.model = YOLO(model_path)
        logger.info(f"Model loaded from {model_path}")
