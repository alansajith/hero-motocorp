"""Training script for YOLOv8 damage detection model."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
from loguru import logger

from src.models.detection import DamageDetector
from src.utils.config_loader import get_config


def main():
    """Train YOLOv8 detection model."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 damage detection model")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--save-dir', type=str, default='models/detection', help='Save directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    logger.info("=" * 60)
    logger.info("YOLOv8 Damage Detection Training")
    logger.info("=" * 60)
    logger.info(f"Data config: {args.data}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 60)
    
    # Initialize detector
    detector = DamageDetector(device=config.device)
    
    # Train
    logger.info("Starting training...")
    results = detector.train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        save_dir=args.save_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Model saved to {args.save_dir}")


if __name__ == '__main__':
    main()
