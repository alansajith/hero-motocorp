"""Cosmetic vs functional damage classification."""

from typing import Dict, Any, List
from loguru import logger

from ..utils.config_loader import get_config


class CosmeticFunctionalClassifier:
    """Classifies damage as cosmetic or functional based on type, severity, and location."""
    
    def __init__(self):
        """Initialize classifier with configuration."""
        config = get_config()
        
        self.cosmetic_types = config.get('cosmetic_functional.cosmetic_types', [
            'scratch'
        ])
        
        self.functional_types = config.get('cosmetic_functional.functional_types', [
            'damage'
        ])
        
        self.ambiguous_types = config.get('cosmetic_functional.ambiguous_types', [
            'dent'
        ])
        
        self.critical_parts = config.get('cosmetic_functional.critical_parts', [])
        
        logger.info("Initialized cosmetic/functional classifier")
    
    def classify(
        self,
        damage_type: str,
        severity_level: str,
        vehicle_part: str = None
    ) -> Dict[str, Any]:
        """
        Classify damage as cosmetic or functional.
        
        Args:
            damage_type: Type of damage
            severity_level: Severity level ('low', 'medium', 'high')
            vehicle_part: Vehicle part affected (optional)
            
        Returns:
            Dictionary with classification results
        """
        is_cosmetic = self._determine_cosmetic(damage_type, severity_level, vehicle_part)
        
        # Generate explanation
        explanation = self._generate_explanation(
            damage_type, severity_level, vehicle_part, is_cosmetic
        )
        
        result = {
            'is_cosmetic': is_cosmetic,
            'is_functional': not is_cosmetic,
            'confidence': self._calculate_confidence(damage_type, severity_level, vehicle_part),
            'explanation': explanation
        }
        
        logger.debug(f"Classified as {'cosmetic' if is_cosmetic else 'functional'}: {explanation}")
        
        return result
    
    def _determine_cosmetic(
        self,
        damage_type: str,
        severity_level: str,
        vehicle_part: str = None
    ) -> bool:
        """Determine if damage is cosmetic."""
        # Check critical parts first
        if vehicle_part and vehicle_part in self.critical_parts:
            # Damage on critical parts is always functional
            return False
        
        # Check damage type
        if damage_type in self.cosmetic_types:
            # Surface-level damages are cosmetic
            return True
        
        if damage_type in self.functional_types:
            # Structural damages are functional
            return False
        
        if damage_type in self.ambiguous_types:
            # Ambiguous damages depend on severity
            # Low severity dents are cosmetic, medium/high are functional
            return severity_level == 'low'
        
        # Default to functional for unknown types (conservative)
        return False
    
    def _calculate_confidence(
        self,
        damage_type: str,
        severity_level: str,
        vehicle_part: str = None
    ) -> float:
        """
        Calculate confidence score for classification.
        
        Returns:
            Confidence score between 0 and 1
        """
        # High confidence for clear-cut cases
        if damage_type in self.cosmetic_types:
            base_confidence = 0.9
        elif damage_type in self.functional_types:
            base_confidence = 0.9
        else:
            # Lower confidence for ambiguous types
            base_confidence = 0.6
        
        # Reduce confidence for borderline severities
        if damage_type in self.ambiguous_types:
            if severity_level == 'medium':
                base_confidence *= 0.8
        
        # High confidence for critical parts
        if vehicle_part and vehicle_part in self.critical_parts:
            base_confidence = max(base_confidence, 0.85)
        
        return base_confidence
    
    def _generate_explanation(
        self,
        damage_type: str,
        severity_level: str,
        vehicle_part: str,
        is_cosmetic: bool
    ) -> str:
        """Generate human-readable explanation for classification."""
        if vehicle_part and vehicle_part in self.critical_parts:
            return f"{damage_type.replace('_', ' ').title()} on {vehicle_part} (critical component) - affects functionality"
        
        if damage_type in self.cosmetic_types:
            return f"{damage_type.replace('_', ' ').title()} - surface-level cosmetic issue"
        
        if damage_type in self.functional_types:
            return f"{damage_type.replace('_', ' ').title()} - structural damage affecting functionality"
        
        if damage_type in self.ambiguous_types:
            if is_cosmetic:
                return f"{severity_level.title()} severity {damage_type} - primarily cosmetic"
            else:
                return f"{severity_level.title()} severity {damage_type} - may affect functionality"
        
        return f"Assessment based on {damage_type} type and {severity_level} severity"
    
    def classify_batch(
        self,
        damage_types: List[str],
        severity_levels: List[str],
        vehicle_parts: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple damages.
        
        Args:
            damage_types: List of damage types
            severity_levels: List of severity levels
            vehicle_parts: List of vehicle parts (optional)
            
        Returns:
            List of classification results
        """
        if vehicle_parts is None:
            vehicle_parts = [None] * len(damage_types)
        
        results = []
        for damage_type, severity, part in zip(damage_types, severity_levels, vehicle_parts):
            result = self.classify(damage_type, severity, part)
            results.append(result)
        
        return results
