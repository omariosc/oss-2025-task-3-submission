#!/usr/bin/env python3
"""
Quick Training Test - Train YOLO-Pose with fewer epochs for rapid testing
"""

import sys
from pathlib import Path
import logging
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_train():
    """Quick training with 5 epochs to test the pipeline"""

    logger.info("="*70)
    logger.info("QUICK YOLO-POSE TRAINING (5 EPOCHS)")
    logger.info("="*70)

    # Dataset configuration
    data_yaml = "/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/keypoint_dataset/data.yaml"

    # Load base model
    logger.info("Loading base YOLO-Pose model...")
    model = YOLO('yolov8n-pose.pt')

    # Quick training parameters
    training_params = {
        'data': data_yaml,
        'epochs': 5,  # Quick test
        'imgsz': 640,
        'batch': 16,  # Larger batch for speed
        'device': 'cpu',
        'patience': 3,
        'save': True,
        'project': 'quick_keypoint_test',
        'name': 'test_run',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'box': 0.05,
        'pose': 12.0,
        'kobj': 1.0,
        'val': True,
        'verbose': False,
        'plots': False
    }

    # Train
    logger.info("Starting quick training...")
    results = model.train(**training_params)

    # Get model path
    model_path = Path('quick_keypoint_test/test_run/weights/last.pt')

    if model_path.exists():
        logger.info(f"✅ Training complete! Model saved at: {model_path}")

        # Test on validation immediately
        logger.info("\nTesting on validation set...")
        trained_model = YOLO(str(model_path))

        # Run validation
        metrics = trained_model.val(data=data_yaml)

        if metrics:
            logger.info("\nQuick Test Metrics:")
            logger.info(f"  Box mAP50: {metrics.box.map50:.3f}")
            logger.info(f"  Box mAP50-95: {metrics.box.map:.3f}")

        return str(model_path)

    return None

if __name__ == "__main__":
    model_path = quick_train()
    if model_path:
        print(f"\n✅ Quick test complete! Model at: {model_path}")
    else:
        print("\n❌ Quick test failed!")