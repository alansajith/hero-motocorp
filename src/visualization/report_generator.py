"""Report generator for damage assessments."""

import cv2
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from .overlay import DamageVisualizer


class ReportGenerator:
    """Generates comprehensive damage assessment reports."""
    
    def __init__(self):
        """Initialize report generator."""
        self.visualizer = DamageVisualizer()
    
    def generate_report(
        self,
        image: Any,
        results: Dict[str, Any],
        output_dir: str,
        image_name: str = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive report with visualizations and JSON.
        
        Args:
            image: Original image (numpy array or path)
            results: Detection results from pipeline
            output_dir: Directory to save report
            image_name: Name for output files
            
        Returns:
            Dictionary with paths to generated files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load image if needed
        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            if image_name is None:
                image_name = Path(image).stem
        else:
            original_image = image
            if image_name is None:
                image_name = f"damage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create visualizations
        annotated_image = self.visualizer.visualize_detections(
            original_image, results, show_severity=True
        )
        
        if results.get('damage_detected', False):
            annotated_image = self.visualizer.draw_severity_indicator(
                annotated_image,
                results.get('overall_assessment', {})
            )
        
        # Side-by-side comparison
        comparison = self.visualizer.create_side_by_side(
            original_image, annotated_image
        )
        
        # Save images
        annotated_path = output_path / f"{image_name}_annotated.jpg"
        comparison_path = output_path / f"{image_name}_comparison.jpg"
        
        cv2.imwrite(
            str(annotated_path),
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            str(comparison_path),
            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        )
        
        # Save JSON report
        json_path = output_path / f"{image_name}_report.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'image_name': image_name,
            'results': results,
            'files': {
                'annotated_image': str(annotated_path),
                'comparison_image': str(comparison_path),
                'json_report': str(json_path)
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Report generated: {json_path}")
        
        return report_data['files']
    
    def generate_text_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable text summary.
        
        Args:
            results: Detection results
            
        Returns:
            Text summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("VEHICLE DAMAGE ASSESSMENT REPORT")
        summary.append("=" * 60)
        summary.append("")
        
        if not results.get('damage_detected', False):
            summary.append("✓ NO DAMAGE DETECTED")
            summary.append("")
            return '\n'.join(summary)
        
        overall = results.get('overall_assessment', {})
        damages = results.get('damages', [])
        
        summary.append(f"Total Damages Found: {overall.get('total_damages', 0)}")
        summary.append(f"Maximum Severity: {overall.get('max_severity', 'N/A').upper()}")
        summary.append(f"Functional Damage: {'YES' if overall.get('has_functional_damage') else 'NO'}")
        summary.append(f"Requires Immediate Attention: {'YES' if overall.get('requires_immediate_attention') else 'NO'}")
        summary.append("")
        summary.append("-" * 60)
        summary.append("DAMAGE DETAILS")
        summary.append("-" * 60)
        summary.append("")
        
        for i, damage in enumerate(damages, 1):
            summary.append(f"Damage #{i}:")
            summary.append(f"  Type: {damage['damage_type'].replace('_', ' ').title()}")
            summary.append(f"  Severity: {damage['severity'].upper()}")
            summary.append(f"  Location: {damage.get('vehicle_part', 'Unknown')}")
            summary.append(f"  Classification: {'Cosmetic' if damage['is_cosmetic'] else 'Functional'}")
            summary.append(f"  Confidence: {damage['detection_confidence']:.2%}")
            summary.append(f"  Assessment: {damage.get('explanation', 'N/A')}")
            summary.append("")
        
        summary.append("=" * 60)
        
        return '\n'.join(summary)
