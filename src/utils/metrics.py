"""Evaluation metrics for damage detection, segmentation, and classification."""

import numpy as np
import torch
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for segmentation.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient for segmentation.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dice coefficient
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2 * intersection / total


def calculate_pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate pixel-wise accuracy for segmentation.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Pixel accuracy
    """
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    
    return correct / total


class SegmentationMetrics:
    """Segmentation metrics tracker."""
    
    def __init__(self):
        self.ious = []
        self.dices = []
        self.pixel_accs = []
    
    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """Update metrics with a new prediction."""
        self.ious.append(calculate_iou(pred_mask, gt_mask))
        self.dices.append(calculate_dice(pred_mask, gt_mask))
        self.pixel_accs.append(calculate_pixel_accuracy(pred_mask, gt_mask))
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        return {
            'mean_iou': np.mean(self.ious) if self.ious else 0.0,
            'mean_dice': np.mean(self.dices) if self.dices else 0.0,
            'mean_pixel_accuracy': np.mean(self.pixel_accs) if self.pixel_accs else 0.0,
        }
    
    def reset(self):
        """Reset all metrics."""
        self.ious = []
        self.dices = []
        self.pixel_accs = []


class ClassificationMetrics:
    """Classification metrics tracker."""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.y_true = []
        self.y_pred = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Update metrics with new predictions."""
        self.y_pred.extend(predictions.flatten().tolist())
        self.y_true.extend(targets.flatten().tolist())
    
    def compute(self) -> Dict[str, any]:
        """Compute classification metrics."""
        if not self.y_true or not self.y_pred:
            return {}
        
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(
                self.y_true, self.y_pred, average=None, zero_division=0
            )
        
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                results[f'{class_name}_precision'] = precision_per_class[i]
                results[f'{class_name}_recall'] = recall_per_class[i]
                results[f'{class_name}_f1'] = f1_per_class[i]
                results[f'{class_name}_support'] = support[i]
        
        return results
    
    def reset(self):
        """Reset all metrics."""
        self.y_true = []
        self.y_pred = []


class DetectionMetrics:
    """Detection metrics tracker (simplified mAP calculation)."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.predictions = []
        self.ground_truths = []
    
    def update(self, pred_boxes: List[List[float]], pred_scores: List[float],
               gt_boxes: List[List[float]]):
        """
        Update metrics with new detections.
        
        Args:
            pred_boxes: List of predicted boxes [x1, y1, x2, y2]
            pred_scores: List of confidence scores
            gt_boxes: List of ground truth boxes [x1, y1, x2, y2]
        """
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores
        })
        self.ground_truths.append({
            'boxes': gt_boxes
        })
    
    def compute(self) -> Dict[str, float]:
        """Compute detection metrics (simplified mAP)."""
        if not self.predictions or not self.ground_truths:
            return {'mAP': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Simplified metrics: average precision and recall
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_boxes = np.array(pred['boxes'])
            gt_boxes = np.array(gt['boxes'])
            
            total_gt += len(gt_boxes)
            
            if len(pred_boxes) == 0:
                continue
            
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            
            # Calculate IoU between all predictions and ground truths
            matched_gt = set()
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    iou = self._box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        
        # Simplified mAP (average of precision and recall)
        mAP = (precision + recall) / 2 if (precision + recall) > 0 else 0.0
        
        return {
            'mAP': mAP,
            'precision': precision,
            'recall': recall,
        }
    
    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.ground_truths = []


class SeverityMetrics:
    """Severity estimation metrics tracker."""
    
    def __init__(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Update metrics with new predictions."""
        self.predictions.extend(predictions.flatten().tolist())
        self.targets.extend(targets.flatten().tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute severity estimation metrics."""
        if not self.predictions or not self.targets:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - targets))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Accuracy (for discrete severity levels 0, 1, 2)
        accuracy = np.mean(predictions == targets)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
        }
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
