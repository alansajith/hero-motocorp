"""Visualization overlays for damage detection results."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

from ..utils.config_loader import get_config


class DamageVisualizer:
    """Creates visual overlays for damage detection results."""
    
    def __init__(self):
        """Initialize visualizer with configuration."""
        config = get_config()
        
        self.show_masks = config.get('explainability.visualization.show_masks', True)
        self.show_boxes = config.get('explainability.visualization.show_boxes', True)
        self.show_labels = config.get('explainability.visualization.show_labels', True)
        self.show_confidence = config.get('explainability.visualization.show_confidence', True)
        self.overlay_alpha = config.get('explainability.visualization.overlay_alpha', 0.5)
        self.line_thickness = config.get('explainability.visualization.line_thickness', 2)
        self.font_scale = config.get('explainability.visualization.font_scale', 0.6)
        
        # Color scheme
        self.severity_colors = {
            'low': (0, 255, 0),      # Green
            'medium': (255, 165, 0),  # Orange
            'high': (255, 0, 0),      # Red
        }
        
        self.damage_type_colors = {
            'scratch': (255, 255, 0),     # Yellow
            'dent': (255, 128, 0),        # Orange
            'crack': (255, 0, 0),         # Red
            'paint_damage': (128, 0, 255), # Purple
            'broken_part': (255, 0, 255),  # Magenta
        }
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        show_severity: bool = True
    ) -> np.ndarray:
        """
        Visualize damage detections on image.
        
        Args:
            image: Original image (RGB)
            detections: Detection results from pipeline
            show_severity: Color boxes by severity
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        if not detections.get('damage_detected', False):
            return vis_image
        
        damages = detections.get('damages', [])
        
        for damage in damages:
            # Get damage info
            box = damage['bounding_box']
            damage_type = damage['damage_type']
            severity = damage['severity']
            confidence = damage['detection_confidence']
            
            # Determine color
            if show_severity:
                color = self.severity_colors.get(severity, (0, 255, 255))
            else:
                color = self.damage_type_colors.get(damage_type, (0, 255, 255))
            
            # Draw bounding box
            if self.show_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Draw label
            if self.show_labels or self.show_confidence:
                label_parts = []
                
                if self.show_labels:
                    label_parts.append(damage_type.replace('_', ' ').title())
                
                if self.show_confidence:
                    label_parts.append(f"{confidence:.2f}")
                
                if show_severity:
                    label_parts.append(severity.upper())
                
                label = ' | '.join(label_parts)
                
                # Draw label background
                x1, y1 = map(int, box[:2])
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width + 5, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return vis_image
    
    def create_side_by_side(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
        labels: Tuple[str, str] = ('Original', 'Detected')
    ) -> np.ndarray:
        """
        Create side-by-side comparison.
        
        Args:
            original: Original image
            annotated: Annotated image
            labels: Labels for each image
            
        Returns:
            Side-by-side image
        """
        # Ensure same height
        h1, w1 = original.shape[:2]
        h2, w2 = annotated.shape[:2]
        
        max_height = max(h1, h2)
        
        if h1 < max_height:
            original = cv2.resize(original, (int(w1 * max_height / h1), max_height))
        if h2 < max_height:
            annotated = cv2.resize(annotated, (int(w2 * max_height / h2), max_height))
        
        # Concatenate
        result = np.hstack([original, annotated])
        
        # Add labels
        cv2.putText(
            result,
            labels[0],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        cv2.putText(
            result,
            labels[1],
            (w1 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        return result
    
    def draw_severity_indicator(
        self,
        image: np.ndarray,
        overall_assessment: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw overall severity indicator on image.
        
        Args:
            image: Input image
            overall_assessment: Overall assessment from pipeline
            
        Returns:
            Image with severity indicator
        """
        vis_image = image.copy()
        
        severity = overall_assessment.get('max_severity', 'none')
        total_damages = overall_assessment.get('total_damages', 0)
        has_functional = overall_assessment.get('has_functional_damage', False)
        
        # Create info box
        info_lines = [
            f"Total Damages: {total_damages}",
            f"Max Severity: {severity.upper()}",
            f"Functional: {'YES' if has_functional else 'NO'}"
        ]
        
        # Calculate box size
        max_width = 0
        total_height = 10
        
        for line in info_lines:
            (w, h), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            max_width = max(max_width, w)
            total_height += h + baseline + 8
        
        # Draw background
        x, y = 10, 10
        cv2.rectangle(
            vis_image,
            (x, y),
            (x + max_width + 20, y + total_height),
            (0, 0, 0),
            -1
        )
        
        # Draw border (colored by severity)
        color = self.severity_colors.get(severity, (128, 128, 128))
        cv2.rectangle(
            vis_image,
            (x, y),
            (x + max_width + 20, y + total_height),
            color,
            3
        )
        
        # Draw text
        y_offset = y + 25
        for line in info_lines:
            cv2.putText(
                vis_image,
                line,
                (x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y_offset += 30
        
        return vis_image
