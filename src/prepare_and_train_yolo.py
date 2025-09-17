#!/usr/bin/env python3
"""
Prepare real EndoVis data and train YOLO keypoint model
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import shutil
from ultralytics import YOLO
import yaml
import logging

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndoVisYOLOPreparer:
    """Prepare EndoVis data for YOLO training"""

    def __init__(self, data_root: str = "/Users/scsoc/Desktop/synpase/endovis2025/task_3/data"):
        self.data_root = Path(data_root)
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/models/yolo_keypoints")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Classes from EndoVis
        self.classes = [
            'left_hand_segment',
            'right_hand_segment',
            'scissors',
            'tweezers',
            'needle_holder',
            'needle'
        ]

    def prepare_dataset(self):
        """Prepare YOLO dataset from real EndoVis data"""
        logger.info("Preparing YOLO dataset from EndoVis data...")

        yolo_root = self.output_dir

        # Create directory structure
        for split in ['train', 'val']:
            (yolo_root / split / 'images').mkdir(exist_ok=True, parents=True)
            (yolo_root / split / 'labels').mkdir(exist_ok=True, parents=True)

        # Process training data
        train_frames_dir = self.data_root / "train" / "frames"
        train_mot_dir = self.data_root / "train" / "mot"

        if train_frames_dir.exists():
            self.process_split(train_frames_dir, train_mot_dir, yolo_root / 'train', max_videos=3)

        # Process validation data
        val_frames_dir = self.data_root / "val" / "frames"
        val_mot_dir = self.data_root / "val" / "mot"

        if val_frames_dir.exists():
            self.process_split(val_frames_dir, val_mot_dir, yolo_root / 'val', max_videos=2)

        # Create data.yaml
        self.create_yolo_config(yolo_root)

        return yolo_root

    def process_split(self, frames_dir: Path, mot_dir: Path, output_dir: Path, max_videos: int = None):
        """Process a data split"""
        video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

        if max_videos:
            video_dirs = video_dirs[:max_videos]

        total_images = 0

        for video_dir in video_dirs:
            video_id = video_dir.name
            mot_file = mot_dir / f"{video_id}.txt"

            logger.info(f"Processing video: {video_id}")

            # Parse MOT annotations
            mot_data = {}
            if mot_file.exists():
                with open(mot_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 10:
                            try:
                                frame_id = int(parts[0])
                                track_id = int(parts[1])
                                object_id = int(parts[2])
                                x = float(parts[3])
                                y = float(parts[4])
                                w = float(parts[5])
                                h = float(parts[6])

                                if frame_id not in mot_data:
                                    mot_data[frame_id] = []

                                mot_data[frame_id].append({
                                    'track_id': track_id,
                                    'class_id': object_id % len(self.classes),
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h
                                })
                            except:
                                continue

            # Process frames (limit to reduce training time)
            frame_files = sorted(video_dir.glob("*.png"))[:30]

            for frame_file in frame_files:
                frame_num = int(frame_file.stem.split('_')[-1])

                # Copy image
                img = cv2.imread(str(frame_file))
                if img is None:
                    continue

                h, w = img.shape[:2]

                output_name = f"{video_id}_{frame_file.stem}"
                output_img = output_dir / 'images' / f"{output_name}.jpg"
                cv2.imwrite(str(output_img), img)

                # Create label file
                output_label = output_dir / 'labels' / f"{output_name}.txt"

                with open(output_label, 'w') as f:
                    if frame_num in mot_data:
                        for obj in mot_data[frame_num]:
                            # Convert to YOLO format (normalized)
                            cx = (obj['x'] + obj['w']/2) / w
                            cy = (obj['y'] + obj['h']/2) / h
                            norm_w = obj['w'] / w
                            norm_h = obj['h'] / h

                            # Ensure values are in [0, 1]
                            cx = max(0, min(1, cx))
                            cy = max(0, min(1, cy))
                            norm_w = max(0, min(1, norm_w))
                            norm_h = max(0, min(1, norm_h))

                            # Write YOLO format: class x y w h
                            f.write(f"{obj['class_id']} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

                total_images += 1

        logger.info(f"Processed {total_images} images for {output_dir.name}")

    def create_yolo_config(self, yolo_root: Path):
        """Create YOLO configuration"""
        config = {
            'path': str(yolo_root),
            'train': 'train/images',
            'val': 'val/images',
            'names': self.classes,
            'nc': len(self.classes)
        }

        config_file = yolo_root / 'data.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        logger.info(f"Created config at {config_file}")
        return config_file

class YOLOTrainer:
    """Train YOLO model on prepared data"""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.output_dir = data_path.parent

    def train_detection_model(self, epochs: int = 30):
        """Train YOLO detection model"""
        logger.info("Training YOLO detection model...")

        # Use YOLOv8m for detection
        model = YOLO('yolov8m.pt')

        # Train
        results = model.train(
            data=str(self.data_path / 'data.yaml'),
            epochs=epochs,
            imgsz=640,
            batch=8,
            device='0' if torch.cuda.is_available() else 'cpu',
            project=str(self.output_dir),
            name='yolo_detection',
            exist_ok=True,
            patience=10,
            save=True,
            plots=True,
            optimizer='AdamW',
            lr0=0.001,
            seed=42,
            deterministic=True,
            cache=False  # Don't cache to save memory
        )

        # Save best model
        best_path = self.output_dir / 'yolo_detection' / 'weights' / 'best.pt'
        final_path = self.output_dir / 'yolo_endovis_best.pt'

        if best_path.exists():
            shutil.copy(best_path, final_path)
            logger.info(f"Saved best model to {final_path}")
            return final_path

        return None

    def validate_model(self, model_path: Path):
        """Validate the trained model"""
        logger.info(f"Validating model: {model_path}")

        model = YOLO(str(model_path))

        # Run validation
        metrics = model.val(
            data=str(self.data_path / 'data.yaml'),
            imgsz=640,
            batch=8,
            device='0' if torch.cuda.is_available() else 'cpu',
            plots=True,
            save_json=True
        )

        logger.info(f"Validation Results:")
        logger.info(f"  mAP50: {metrics.box.map50:.3f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.3f}")

        return metrics

def main():
    # Prepare dataset
    preparer = EndoVisYOLOPreparer()
    yolo_root = preparer.prepare_dataset()

    # Check if we have images
    train_images = list((yolo_root / 'train' / 'images').glob('*.jpg'))
    val_images = list((yolo_root / 'val' / 'images').glob('*.jpg'))

    logger.info(f"Found {len(train_images)} training images")
    logger.info(f"Found {len(val_images)} validation images")

    if len(train_images) == 0:
        logger.error("No training images found! Cannot proceed with training.")
        return

    # Train model
    trainer = YOLOTrainer(yolo_root)
    model_path = trainer.train_detection_model(epochs=20)

    # Validate
    if model_path and model_path.exists():
        trainer.validate_model(model_path)
        logger.info(f"\n✅ Training complete! Model saved to: {model_path}")
    else:
        logger.error("Training failed - no model saved")

if __name__ == "__main__":
    main()