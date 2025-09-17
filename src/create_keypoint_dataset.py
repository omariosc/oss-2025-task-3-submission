#!/usr/bin/env python3
"""
Create YOLO-Pose Keypoint Dataset from MOT Annotations
Converts MOT format to YOLO keypoint format for training
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import json
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeypointDatasetCreator:
    """Create YOLO-Pose format dataset from MOT annotations"""

    def __init__(self, data_root="/Users/scsoc/Desktop/synpase/endovis2025/task_3/data"):
        self.data_root = Path(data_root)
        self.output_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/keypoint_dataset")

        # YOLO-Pose requires: class x y w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
        # Where v is visibility (0=not labeled, 1=occluded, 2=visible)

        # We'll use a single class for all objects since we care about keypoints
        self.num_keypoints = 4  # Average 3.8 keypoints per object, use 4

    def parse_mot_file(self, mot_file):
        """Parse MOT file to extract objects and their keypoints"""
        objects_by_frame = defaultdict(list)

        with open(mot_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    frame_num = int(parts[0])
                    obj_id = int(parts[1])
                    track_id = int(parts[2])

                    # Parse keypoints for this object
                    keypoints = []
                    i = 7
                    while i + 2 < len(parts):
                        try:
                            x = float(parts[i])
                            y = float(parts[i+1])
                            v = int(parts[i+2])
                            if v > 0:  # Visible keypoint
                                keypoints.append({'x': x, 'y': y, 'v': 2})  # 2 = visible in YOLO
                            i += 3
                        except:
                            break

                    if keypoints:
                        objects_by_frame[frame_num].append({
                            'obj_id': obj_id,
                            'track_id': track_id,
                            'keypoints': keypoints
                        })

        return objects_by_frame

    def create_yolo_annotation(self, objects, img_width, img_height):
        """Create YOLO-Pose format annotation for a frame"""
        annotations = []

        for obj in objects:
            keypoints = obj['keypoints']

            if len(keypoints) < 2:  # Need at least 2 keypoints
                continue

            # Calculate bounding box from keypoints
            xs = [kp['x'] for kp in keypoints]
            ys = [kp['y'] for kp in keypoints]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            # Add padding to bbox
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img_width, x_max + padding)
            y_max = min(img_height, y_max + padding)

            # Convert to YOLO format (normalized)
            cx = (x_min + x_max) / 2 / img_width
            cy = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height

            # Start annotation with class and bbox
            ann = [0, cx, cy, w, h]  # class 0 (single class for all)

            # Add keypoints (normalized)
            # We need exactly self.num_keypoints keypoints
            for i in range(self.num_keypoints):
                if i < len(keypoints):
                    kp = keypoints[i]
                    kp_x = kp['x'] / img_width
                    kp_y = kp['y'] / img_height
                    kp_v = kp['v']
                else:
                    # Pad with invisible keypoints if needed
                    kp_x = 0
                    kp_y = 0
                    kp_v = 0

                ann.extend([kp_x, kp_y, kp_v])

            annotations.append(ann)

        return annotations

    def create_dataset(self):
        """Create complete YOLO-Pose dataset"""
        logger.info("Creating YOLO-Pose keypoint dataset...")

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        train_images = self.output_root / "images" / "train"
        train_labels = self.output_root / "labels" / "train"
        val_images = self.output_root / "images" / "val"
        val_labels = self.output_root / "labels" / "val"

        train_images.mkdir(parents=True, exist_ok=True)
        train_labels.mkdir(parents=True, exist_ok=True)
        val_images.mkdir(parents=True, exist_ok=True)
        val_labels.mkdir(parents=True, exist_ok=True)

        # Process training data
        train_dir = self.data_root / "train"
        if train_dir.exists():
            logger.info("Processing training data...")
            self.process_split(train_dir, train_images, train_labels, "train")

        # Process validation data (use some for training if needed)
        val_dir = self.data_root / "val"
        if val_dir.exists():
            logger.info("Processing validation data...")
            # Use first video for validation, rest for training
            val_videos = sorted(list((val_dir / "frames").glob("*")))

            for i, video_dir in enumerate(val_videos):
                video_id = video_dir.name
                if i == 0:  # First video for validation
                    self.process_video(video_id, val_dir, val_images, val_labels, "val")
                else:  # Rest for training
                    self.process_video(video_id, val_dir, train_images, train_labels, "train")

        # Create data.yaml configuration
        self.create_config()

        logger.info(f"Dataset created at {self.output_root}")

    def process_split(self, data_dir, images_dir, labels_dir, split_name):
        """Process a data split (train/val)"""
        frames_dir = data_dir / "frames"
        mot_dir = data_dir / "mot"

        if not frames_dir.exists() or not mot_dir.exists():
            logger.warning(f"Missing data for {split_name}")
            return

        # Process each video
        for video_dir in frames_dir.glob("*"):
            video_id = video_dir.name
            self.process_video(video_id, data_dir, images_dir, labels_dir, split_name)

    def process_video(self, video_id, data_dir, images_dir, labels_dir, split_name):
        """Process a single video"""
        frames_dir = data_dir / "frames" / video_id
        mot_file = data_dir / "mot" / f"{video_id}.txt"

        if not mot_file.exists():
            logger.warning(f"No MOT file for {video_id}")
            return

        # Parse MOT annotations
        objects_by_frame = self.parse_mot_file(mot_file)

        # Process each frame
        frame_files = sorted(list(frames_dir.glob("*.png")))
        processed = 0

        for frame_file in frame_files:
            frame_num = int(frame_file.stem.split('_')[-1])

            if frame_num not in objects_by_frame:
                continue

            # Read image to get dimensions
            img = cv2.imread(str(frame_file))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Create YOLO annotation
            objects = objects_by_frame[frame_num]
            annotations = self.create_yolo_annotation(objects, w, h)

            if not annotations:
                continue

            # Copy image
            output_name = f"{video_id}_{frame_file.stem}.jpg"
            output_image = images_dir / output_name
            cv2.imwrite(str(output_image), img)

            # Write label file
            output_label = labels_dir / f"{video_id}_{frame_file.stem}.txt"
            with open(output_label, 'w') as f:
                for ann in annotations:
                    f.write(' '.join(map(str, ann)) + '\n')

            processed += 1

            if processed % 10 == 0:
                logger.info(f"  {split_name}/{video_id}: Processed {processed} frames")

        logger.info(f"  {split_name}/{video_id}: Total {processed} frames with annotations")

    def create_config(self):
        """Create YOLO configuration file"""
        config = {
            'path': str(self.output_root),
            'train': 'images/train',
            'val': 'images/val',

            # Dataset info
            'names': {
                0: 'surgical_tool'  # Single class for all objects
            },
            'nc': 1,  # number of classes

            # Keypoint configuration
            'kpt_shape': [self.num_keypoints, 3],  # 4 keypoints, 3 values each (x, y, v)

            # Skeleton connections (for visualization)
            'skeleton': [
                [0, 1],  # Connect keypoint 0 to 1
                [1, 2],  # Connect keypoint 1 to 2
                [2, 3],  # Connect keypoint 2 to 3
            ]
        }

        yaml_path = self.output_root / "data.yaml"

        # Write as YAML
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created config at {yaml_path}")

    def verify_dataset(self):
        """Verify the created dataset"""
        train_images = len(list((self.output_root / "images" / "train").glob("*.jpg")))
        train_labels = len(list((self.output_root / "labels" / "train").glob("*.txt")))
        val_images = len(list((self.output_root / "images" / "val").glob("*.jpg")))
        val_labels = len(list((self.output_root / "labels" / "val").glob("*.txt")))

        logger.info("\nDataset Statistics:")
        logger.info(f"  Training: {train_images} images, {train_labels} labels")
        logger.info(f"  Validation: {val_images} images, {val_labels} labels")

        # Check a sample annotation
        sample_label = list((self.output_root / "labels" / "train").glob("*.txt"))[0]
        with open(sample_label, 'r') as f:
            lines = f.readlines()
            logger.info(f"\nSample annotation ({sample_label.name}):")
            logger.info(f"  Objects: {len(lines)}")
            if lines:
                parts = lines[0].strip().split()
                logger.info(f"  Values per object: {len(parts)}")
                logger.info(f"  Expected: {5 + self.num_keypoints * 3} (class + bbox + keypoints)")

        return train_images > 0 and train_labels > 0


if __name__ == "__main__":
    creator = KeypointDatasetCreator()
    creator.create_dataset()

    if creator.verify_dataset():
        logger.info("\n✅ Keypoint dataset created successfully!")
    else:
        logger.info("\n❌ Dataset creation failed!")