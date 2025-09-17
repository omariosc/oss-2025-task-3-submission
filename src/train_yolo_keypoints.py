#!/usr/bin/env python3
"""
Train YOLO-Pose for Keypoint Detection
Using the created keypoint dataset from MOT annotations
"""

import sys
from pathlib import Path
import logging
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_keypoint_model():
    """Train YOLO-Pose model for keypoint detection"""

    logger.info("="*70)
    logger.info("TRAINING YOLO-POSE FOR KEYPOINT DETECTION")
    logger.info("="*70)

    # Dataset configuration
    data_yaml = "/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/keypoint_dataset/data.yaml"

    # Check if dataset exists
    if not Path(data_yaml).exists():
        logger.error(f"Dataset config not found at {data_yaml}")
        return None

    # Load base model - use yolov8n-pose for faster training
    logger.info("Loading base YOLO-Pose model...")
    model = YOLO('yolov8n-pose.pt')  # Nano version for speed

    # Training parameters optimized for quick results
    training_params = {
        'data': data_yaml,
        'epochs': 30,  # Quick training
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',  # MPS has known issues with Pose models
        'patience': 5,  # Early stopping
        'save': True,
        'project': 'keypoint_training',
        'name': 'surgical_keypoints',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,  # Box loss gain
        'pose': 12.0,  # Pose loss gain (keypoints)
        'kobj': 1.0,   # Keypoint obj loss gain
        'label_smoothing': 0.0,
        'nms': True,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True
    }

    logger.info(f"Training with parameters:")
    for key, value in training_params.items():
        logger.info(f"  {key}: {value}")

    # Train the model
    try:
        logger.info("\nStarting training...")
        results = model.train(**training_params)

        # Save the best model
        best_model_path = Path('keypoint_training/surgical_keypoints/weights/best.pt')
        if best_model_path.exists():
            logger.info(f"✅ Training complete! Best model saved at: {best_model_path}")
            return str(best_model_path)
        else:
            logger.warning("Training completed but best model not found")
            return None

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


def validate_trained_model(model_path):
    """Quick validation of the trained model"""

    if not model_path or not Path(model_path).exists():
        logger.error("Model path not valid")
        return

    logger.info("\n" + "="*70)
    logger.info("VALIDATING TRAINED MODEL")
    logger.info("="*70)

    # Load trained model
    model = YOLO(model_path)

    # Run validation
    data_yaml = "/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/keypoint_dataset/data.yaml"

    logger.info("Running validation on test set...")
    metrics = model.val(data=data_yaml)

    if metrics:
        logger.info("\nValidation Metrics:")
        logger.info(f"  Box mAP50: {metrics.box.map50:.3f}")
        logger.info(f"  Box mAP50-95: {metrics.box.map:.3f}")
        if hasattr(metrics, 'pose'):
            logger.info(f"  Pose mAP50: {metrics.pose.map50:.3f}")
            logger.info(f"  Pose mAP50-95: {metrics.pose.map:.3f}")

    return metrics


def test_on_sample_frame(model_path):
    """Test the trained model on a sample frame"""

    if not model_path or not Path(model_path).exists():
        return

    logger.info("\n" + "="*70)
    logger.info("TESTING ON SAMPLE FRAME")
    logger.info("="*70)

    # Load model
    model = YOLO(model_path)

    # Test on a validation frame
    test_image = "/Users/scsoc/Desktop/synpase/endovis2025/task_3/data/val/frames/E66F/E66F_frame_0.png"

    if Path(test_image).exists():
        logger.info(f"Testing on: {test_image}")

        # Run inference
        results = model(test_image)

        if results and len(results) > 0:
            result = results[0]

            # Check if we have keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.xy  # Get keypoint coordinates

                logger.info(f"\nDetection Results:")
                logger.info(f"  Number of objects: {len(keypoints)}")

                for i, obj_kps in enumerate(keypoints):
                    valid_kps = obj_kps[obj_kps[:, 0] > 0]  # Filter valid keypoints
                    logger.info(f"  Object {i+1}: {len(valid_kps)} keypoints detected")

                # Calculate total keypoints
                total_keypoints = sum(len(kps[kps[:, 0] > 0]) for kps in keypoints)
                logger.info(f"\nTotal keypoints detected: {total_keypoints}")

                if total_keypoints >= 20:
                    logger.info("✅ Good keypoint coverage!")
                else:
                    logger.info("⚠️  Need more keypoints (target: ~22)")

            else:
                logger.warning("No keypoints detected")
    else:
        logger.warning(f"Test image not found: {test_image}")


def main():
    """Main training pipeline"""

    # Train the model
    model_path = train_keypoint_model()

    if model_path:
        # Validate
        validate_trained_model(model_path)

        # Test on sample
        test_on_sample_frame(model_path)

        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Trained model saved at: {model_path}")
        logger.info("Ready for evaluation on validation set!")

        return model_path
    else:
        logger.error("Training failed!")
        return None


if __name__ == "__main__":
    model_path = main()
    if model_path:
        print(f"\n✅ YOLO-Pose keypoint model trained successfully!")
        print(f"Model path: {model_path}")
    else:
        print("\n❌ Training failed!")