"""Severity estimation for vehicle damage."""

import numpy as np
from typing import Dict, Any, Optional, List
from loguru import logger

from ..utils.config_loader import get_config


class SeverityEstimator:
    """Estimates damage severity based on area, location, and type."""
    
    def __init__(self):
        """Initialize severity estimator with configuration."""
        config = get_config()
        
        self.area_thresholds = config.get('severity.area_thresholds', {
            'low': 0.05,
            'medium': 0.15,
            'high': 0.15
        })
        
        self.length_thresholds = config.get('severity.length_thresholds', {
            'low': 100,
            'medium': 300,
            'high': 300
        })
        
        self.critical_parts = config.get('severity.critical_parts', [])
        self.critical_multiplier = config.get('severity.critical_multiplier', 1.5)
        
        self.severity_levels = config.severity_levels
        
        logger.info("Initialized severity estimator")
    
    def estimate(
        self,
        mask: np.ndarray,
        damage_type: str,
        vehicle_part: Optional[str] = None,
        reference_area: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Estimate damage severity.
        
        Args:
            mask: Binary segmentation mask of damage
            damage_type: Type of damage (scratch, dent, crack, etc.)
            vehicle_part: Vehicle part affected (optional)
            reference_area: Total area of the vehicle part for normalization (optional)
            
        Returns:
            Dictionary with severity information
        """
        # Calculate damage metrics
        damage_area = np.sum(mask > 0)
        
        # Calculate damage length (for cracks/scratches)
        damage_length = self._calculate_length(mask)
        
        # Normalize area if reference provided
        if reference_area is not None and reference_area > 0:
            area_ratio = damage_area / reference_area
        else:
            area_ratio = damage_area / (mask.shape[0] * mask.shape[1])
        
        # Base severity score (0-1)
        if damage_type in ['scratch', 'damage']:
            # Use length-based thresholds
            if damage_length < self.length_thresholds['low']:
                base_score = 0.2
                severity_level = 'low'
            elif damage_length < self.length_thresholds['medium']:
                base_score = 0.5
                severity_level = 'medium'
            else:
                base_score = 0.8
                severity_level = 'high'
        else:
            # Use area-based thresholds
            if area_ratio < self.area_thresholds['low']:
                base_score = 0.2
                severity_level = 'low'
            elif area_ratio < self.area_thresholds['medium']:
                base_score = 0.5
                severity_level = 'medium'
            else:
                base_score = 0.8
                severity_level = 'high'
        
        # Apply location weighting
        if vehicle_part and vehicle_part in self.critical_parts:
            base_score *= self.critical_multiplier
            # Clamp to 1.0
            base_score = min(base_score, 1.0)
            
            # May increase severity level
            if base_score >= 0.6:
                severity_level = 'high'
            elif base_score >= 0.3:
                severity_level = 'medium'
        
        # Map to discrete severity level
        severity_score = base_score
        
        result = {
            'severity_level': severity_level,
            'severity_score': float(severity_score),
            'damage_area': int(damage_area),
            'damage_length': float(damage_length),
            'area_ratio': float(area_ratio),
            'is_critical_part': vehicle_part in self.critical_parts if vehicle_part else False
        }
        
        logger.debug(f"Estimated severity: {severity_level} (score: {severity_score:.2f})")
        
        return result
    
    def _calculate_length(self, mask: np.ndarray) -> float:
        """
        Calculate length of damage (for linear damages like cracks/scratches).
        
        Uses skeleton length as approximation.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Approximate length in pixels
        """
        try:
            from skimage.morphology import skeletonize
            from scipy import ndimage
            
            # Convert to binary
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Skeletonize
            skeleton = skeletonize(binary_mask)
            
            # Count skeleton pixels as length approximation
            length = np.sum(skeleton)
            
            return float(length)
        
        except ImportError:
            # Fallback: use bounding box diagonal
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                return 0.0
            
            height = y_coords.max() - y_coords.min()
            width = x_coords.max() - x_coords.min()
            
            # Diagonal as length approximation
            length = np.sqrt(height**2 + width**2)
            
            return float(length)
    
    def estimate_batch(
        self,
        masks: List[np.ndarray],
        damage_types: List[str],
        vehicle_parts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Estimate severity for multiple damages.
        
        Args:
            masks: List of segmentation masks
            damage_types: List of damage types
            vehicle_parts: List of vehicle parts (optional)
            
        Returns:
            List of severity estimation results
        """
        if vehicle_parts is None:
            vehicle_parts = [None] * len(masks)
        
        results = []
        for mask, damage_type, vehicle_part in zip(masks, damage_types, vehicle_parts):
            result = self.estimate(mask, damage_type, vehicle_part)
            results.append(result)
        
        return results
