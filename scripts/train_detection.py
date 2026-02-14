"""Training script for YOLOv8 damage detection model.

Usage:
  python scripts/train_detection.py --data data.yaml --epochs 150 --batch 16 --lr 0.005
  python scripts/train_detection.py --data data.yaml --epochs 150 --batch 16 --lr 0.005 --model yolov8l.pt
  python scripts/train_detection.py --data data.yaml --resume runs/train/damage_detection/weights/last.pt
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger

from src.models.detection import DamageDetector
from src.utils.config_loader import get_config


def main():
    """Train YOLOv8 detection model."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 damage detection model")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--model', type=str, default=None, help='YOLO model variant (e.g. yolov8l.pt)')
    parser.add_argument('--save-dir', type=str, default='models/detection', help='Save directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (0 to disable)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    logger.info("=" * 60)
    logger.info("YOLOv8 Damage Detection Training")
    logger.info("=" * 60)
    logger.info(f"Data config: {args.data}")
    logger.info(f"Model: {args.model or config.get('detection.model_name', 'yolov8l.pt')}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Resume: {args.resume or 'None'}")
    logger.info("=" * 60)

    # Initialize detector
    model_path = args.resume if args.resume else None
    model_name = args.model  # Can be None, will use config default
    detector = DamageDetector(
        model_path=model_path,
        device=config.device,
        model_name_override=model_name,
    )

    # Train
    logger.info("Starting training...")
    results = detector.train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        save_dir=args.save_dir,
        lr=args.lr,
        patience=args.patience,
        workers=args.workers,
        resume=args.resume is not None,
    )

    logger.info("Training completed!")
    logger.info(f"Model saved to {args.save_dir}")

    # Print final metrics
    if results:
        logger.info("=" * 60)
        logger.info("Final Training Metrics")
        logger.info("=" * 60)
        try:
            metrics = results.results_dict
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        except Exception:
            logger.info("  (Metrics available in training output directory)")


if __name__ == '__main__':
    main()
