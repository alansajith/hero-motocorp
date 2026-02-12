"""End-to-end inference pipeline for vehicle damage detection."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from loguru import logger

from ..models.detection import DamageDetector
from ..models.segmentation import DamageSegmentationModel
from ..models.classification import DamageClassifier
from ..analysis.severity_estimator import SeverityEstimator
from ..analysis.cosmetic_classifier import CosmeticFunctionalClassifier
from ..analysis.part_identifier import VehiclePartIdentifier
from ..utils.config_loader import get_config


class DamageDetectionPipeline:
    """End-to-end pipeline for vehicle damage detection and assessment."""
    
    def __init__(
        self,
        detection_model_path: Optional[str] = None,
        segmentation_model_path: Optional[str] = None,
        classification_model_path: Optional[str] = None,
        part_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the complete damage detection pipeline.
        
        Args:
            detection_model_path: Path to YOLOv8 detection model
            segmentation_model_path: Path to U-Net segmentation model
            classification_model_path: Path to classification model
            part_model_path: Path to part identification model
            device: Device to use ('mps', 'cuda', 'cpu')
        """
        config = get_config()
        
        if device is None:
            device = config.device
        
        self.device = device
        self.config = config
        
        logger.info(f"Initializing damage detection pipeline on device: {device}")
        
        # Initialize models
        logger.info("Loading damage detector...")
        self.detector = DamageDetector(
            model_path=detection_model_path,
            conf_threshold=config.get('detection.conf_threshold', 0.25),
            device=device
        )
        
        # Segmentation model (optional for refined masks)
        self.use_segmentation = segmentation_model_path is not None
        if self.use_segmentation:
            logger.info("Loading segmentation model...")
            self.segmentation_model = DamageSegmentationModel(
                architecture=config.get('segmentation.architecture', 'unet'),
                encoder=config.get('segmentation.encoder', 'resnet34'),
                device=device
            )
            if Path(segmentation_model_path).exists():
                self.segmentation_model.load(segmentation_model_path)
        else:
            self.segmentation_model = None
            logger.info("Segmentation model not provided, using YOLO masks")
        
        # Classification model (optional)
        self.use_classification = classification_model_path is not None
        if self.use_classification:
            logger.info("Loading classification model...")
            self.classifier = DamageClassifier(
                num_classes=config.num_damage_classes,
                architecture=config.get('classification.architecture', 'efficientnet-b2'),
                device=device
            )
            if Path(classification_model_path).exists():
                self.classifier.load(classification_model_path)
        else:
            self.classifier = None
            logger.info("Classification model not provided, using YOLO classes")
        
        # Part identification (optional)
        self.use_part_identification = config.get('part_identification.enabled', True)
        if self.use_part_identification:
            logger.info("Loading part identifier...")
            self.part_identifier = VehiclePartIdentifier(
                model_path=part_model_path,
                device=device
            )
        else:
            self.part_identifier = None
        
        # Analysis modules
        logger.info("Initializing analysis modules...")
        self.severity_estimator = SeverityEstimator()
        self.cosmetic_classifier = CosmeticFunctionalClassifier()
        
        logger.info("Pipeline initialization complete")
    
    def process_image(
        self,
        image: Union[str, np.ndarray],
        return_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image: Input image (path or numpy array)
            return_visualization: Whether to return visualization
            
        Returns:
            Dictionary with complete damage assessment
        """
        # Load image if path
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_path = None
        
        logger.info(f"Processing image of shape {image.shape}")
        
        # Stage 1: Damage Detection
        logger.debug("Running damage detection...")
        detections = self.detector.predict(image, return_masks=True)
        
        if len(detections['boxes']) == 0:
            logger.info("No damage detected")
            return {
                'damage_detected': False,
                'damages': [],
                'image_path': image_path,
                'image_shape': list(image.shape)
            }
        
        logger.info(f"Detected {len(detections['boxes'])} damage region(s)")
        
        # Process each detected damage
        damages = []
        
        for i in range(len(detections['boxes'])):
            box = detections['boxes'][i]
            score = detections['scores'][i]
            label = detections['labels'][i]
            damage_class = detections['class_names'][i]
            
            logger.debug(f"Processing damage {i+1}: {damage_class} (confidence: {score:.2f})")
            
            # Extract damage region
            x1, y1, x2, y2 = map(int, box)
            damage_region = image[y1:y2, x1:x2]
            
            # Get mask
            if detections['masks'] and i < len(detections['masks']):
                mask = np.array(detections['masks'][i])
                # Resize mask to match detection box
                mask_resized = cv2.resize(mask, (x2-x1, y2-y1))
            else:
                mask_resized = np.ones((y2-y1, x2-x1))
            
            # Stage 2: Fine Segmentation (optional)
           
 if self.use_segmentation and self.segmentation_model is not None:
                logger.debug("Running fine segmentation...")
                # Convert to tensor
                from ..data.augmentation import get_segmentation_transforms
                transform = get_segmentation_transforms(
                    image_size=self.config.get('segmentation.input_size', 256),
                    is_train=False
                )
                damage_tensor = transform(image=damage_region)['image']
                refined_mask = self.segmentation_model.predict(damage_tensor)
                # Resize back to original
                refined_mask = cv2.resize(refined_mask, (x2-x1, y2-y1))
            else:
                refined_mask = mask_resized
            
            # Stage 3: Classification (optional)
            if self.use_classification and self.classifier is not None:
                logger.debug("Running damage classification...")
                from ..data.augmentation import get_classification_transforms
                transform = get_classification_transforms(
                    image_size=self.config.get('classification.input_size', 224),
                    is_train=False
                )
                damage_tensor = transform(image=damage_region)['image']
                classification_result = self.classifier.predict(damage_tensor, return_probabilities=True)
                damage_class = classification_result['class_name']
                classification_confidence = classification_result['confidence']
            else:
                classification_confidence = score
            
            # Stage 4: Vehicle Part Identification
            vehicle_part = None
            part_confidence = 0.0
            
            if self.use_part_identification and self.part_identifier is not None:
                logger.debug("Identifying vehicle part...")
                part_result = self.part_identifier.identify(image, damage_box=box)
                if part_result['part_detected']:
                    vehicle_part = part_result['part_name']
                    part_confidence = part_result['confidence']
            
            # Stage 5: Severity Estimation
            logger.debug("Estimating severity...")
            severity_result = self.severity_estimator.estimate(
                mask=refined_mask,
                damage_type=damage_class,
                vehicle_part=vehicle_part
            )
            
            # Stage 6: Cosmetic vs Functional Classification
            logger.debug("Classifying cosmetic vs functional...")
            cosmetic_result = self.cosmetic_classifier.classify(
                damage_type=damage_class,
                severity_level=severity_result['severity_level'],
                vehicle_part=vehicle_part
            )
            
            # Compile results for this damage
            damage_info = {
                'damage_id': i,
                'bounding_box': box,
                'detection_confidence': float(score),
                'damage_type': damage_class,
                'classification_confidence': float(classification_confidence),
                'vehicle_part': vehicle_part,
                'part_confidence': float(part_confidence),
                'severity': severity_result['severity_level'],
                'severity_score': severity_result['severity_score'],
                'damage_area_pixels': severity_result['damage_area'],
                'is_cosmetic': cosmetic_result['is_cosmetic'],
                'is_functional': cosmetic_result['is_functional'],
                'cosmetic_confidence': cosmetic_result['confidence'],
                'explanation': cosmetic_result['explanation'],
            }
            
            damages.append(damage_info)
        
        # Compile final results
        result = {
            'damage_detected': True,
            'num_damages': len(damages),
            'damages': damages,
            'image_path': image_path,
            'image_shape': list(image.shape),
            'overall_assessment': self._generate_overall_assessment(damages)
        }
        
        logger.info(f"Processing complete: {len(damages)} damage(s) found")
        
        return result
    
    def _generate_overall_assessment(self, damages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall assessment summary."""
        if not damages:
            return {
                'max_severity': 'none',
                'has_functional_damage': False,
                'total_damages': 0
            }
        
        # Find maximum severity
        severity_order = {'low': 1, 'medium': 2, 'high': 3}
        max_severity = max(damages, key=lambda d: severity_order.get(d['severity'], 0))['severity']
        
        # Check if any functional damage
        has_functional = any(d['is_functional'] for d in damages)
        
        # Count by type
        damage_types = {}
        for d in damages:
            dtype = d['damage_type']
            damage_types[dtype] = damage_types.get(dtype, 0) + 1
        
        return {
            'max_severity': max_severity,
            'has_functional_damage': has_functional,
            'total_damages': len(damages),
            'damage_types_count': damage_types,
            'requires_immediate_attention': max_severity == 'high' or has_functional
        }
    
    def process_batch(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images.
        
        Args:
            images: List of images (paths or arrays)
            
        Returns:
            List of results for each image
        """
        results = []
        
        for idx, image in enumerate(images):
            logger.info(f"Processing image {idx + 1}/{len(images)}")
            result = self.process_image(image)
            results.append(result)
        
        return results
