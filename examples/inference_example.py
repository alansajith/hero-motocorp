"""Inference example for vehicle damage detection."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import argparse
from loguru import logger

from src.pipeline.inference import DamageDetectionPipeline
from src.visualization.report_generator import ReportGenerator
from src.utils.config_loader import get_config


def main():
    """Run inference example."""
    parser = argparse.ArgumentParser(description="Vehicle damage detection inference")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--detection-model', type=str, default=None, help='Detection model path')
    parser.add_argument('--segmentation-model', type=str, default=None, help='Segmentation model path')
    parser.add_argument('--classification-model', type=str, default=None, help='Classification model path')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    logger.info("=" * 80)
    logger.info("VEHICLE DAMAGE DETECTION - INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Input image: {args.image}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 80)
    
    # Initialize pipeline
    logger.info("\nInitializing pipeline...")
    pipeline = DamageDetectionPipeline(
        detection_model_path=args.detection_model,
        segmentation_model_path=args.segmentation_model,
        classification_model_path=args.classification_model,
    )
    
    # Load image
    logger.info(f"\nLoading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Failed to load image: {args.image}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Image shape: {image.shape}")
    
    # Process
    logger.info("\nProcessing...")
    results = pipeline.process_image(image)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    
    if not results['damage_detected']:
        logger.info("✓ NO DAMAGE DETECTED")
    else:
        overall = results['overall_assessment']
        logger.info(f"Total Damages: {overall['total_damages']}")
        logger.info(f"Maximum Severity: {overall['max_severity'].upper()}")
        logger.info(f"Functional Damage: {'YES' if overall['has_functional_damage'] else 'NO'}")
        logger.info(f"Requires Attention: {'YES' if overall['requires_immediate_attention'] else 'NO'}")
        
        logger.info("\nDamage Details:")
        logger.info("-" * 80)
        
        for i, damage in enumerate(results['damages'], 1):
            logger.info(f"\nDamage #{i}:")
            logger.info(f"  Type: {damage['damage_type']}")
            logger.info(f"  Severity: {damage['severity'].upper()}")
            logger.info(f"  Part: {damage.get('vehicle_part', 'Unknown')}")
            logger.info(f"  Classification: {'Cosmetic' if damage['is_cosmetic'] else 'Functional'}")
            logger.info(f"  Confidence: {damage['detection_confidence']:.2%}")
            logger.info(f"  Explanation: {damage['explanation']}")
    
    # Generate report
    if not args.no_report:
        logger.info("\n" + "=" * 80)
        logger.info("Generating report...")
        
        report_gen = ReportGenerator()
        files = report_gen.generate_report(
            image=args.image,
            results=results,
            output_dir=args.output
        )
        
        logger.info(f"Report files saved:")
        for name, path in files.items():
            logger.info(f"  {name}: {path}")
        
        # Print text summary
        logger.info("\n" + report_gen.generate_text_summary(results))
    
    logger.info("\n" + "=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
