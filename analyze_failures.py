#!/usr/bin/env python3
"""
Analyze why detection is failing - visualize ground truth vs predictions
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import logging

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from multistage_optimized import OptimizedMultiStageFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailureAnalyzer:
    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames = self.data_root / "val/frames"
        self.val_mot = self.data_root / "val/mot"
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/failure_analysis")
        self.output_dir.mkdir(exist_ok=True)

    def parse_ground_truth_frame(self, video_id: str, frame_num: int):
        """Parse ground truth for specific frame"""
        gt_file = self.val_mot / f"{video_id}.txt"
        objects = []

        if not gt_file.exists():
            return objects

        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    if int(parts[0]) == frame_num:
                        track_id = int(parts[1])
                        object_id = int(parts[2])

                        # Parse all keypoints for this object
                        keypoints = []
                        i = 7
                        while i + 2 < len(parts):
                            try:
                                x = float(parts[i])
                                y = float(parts[i+1])
                                v = int(parts[i+2])
                                if v > 0:  # Only visible keypoints
                                    keypoints.append((x, y))
                                i += 3
                            except:
                                break

                        if keypoints:
                            objects.append({
                                'track_id': track_id,
                                'object_id': object_id,
                                'keypoints': keypoints,
                                'center': (np.mean([k[0] for k in keypoints]),
                                         np.mean([k[1] for k in keypoints]))
                            })

        return objects

    def visualize_comparison(self, video_id: str, frame_nums: list = None):
        """Visualize ground truth vs predictions"""
        frames_dir = self.val_frames / video_id

        if frame_nums is None:
            # Get first few frames
            frame_files = sorted(list(frames_dir.glob("*.png")))[:5]
        else:
            frame_files = [frames_dir / f"{video_id}_frame_{num}.png" for num in frame_nums]

        # Initialize system
        system = OptimizedMultiStageFusion(device='cpu')

        for frame_file in frame_files:
            if not frame_file.exists():
                continue

            frame_num = int(frame_file.stem.split('_')[-1])
            frame = cv2.imread(str(frame_file))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get ground truth
            gt_objects = self.parse_ground_truth_frame(video_id, frame_num)

            # Get predictions
            result = system.process_frame(frame)
            pred_objects = result['objects']

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. Original image
            axes[0].imshow(frame_rgb)
            axes[0].set_title(f"Original Frame {frame_num}")
            axes[0].axis('off')

            # 2. Ground truth
            axes[1].imshow(frame_rgb)
            axes[1].set_title(f"Ground Truth ({len(gt_objects)} objects)")

            colors = plt.cm.rainbow(np.linspace(0, 1, max(len(gt_objects), 1)))
            for i, obj in enumerate(gt_objects):
                color = colors[i]
                # Plot all keypoints
                for kp in obj['keypoints']:
                    axes[1].scatter(kp[0], kp[1], c=[color], s=20, marker='o')
                # Plot center
                axes[1].scatter(obj['center'][0], obj['center'][1], c=[color], s=100, marker='x')
                axes[1].text(obj['center'][0], obj['center'][1], f"T{obj['track_id']}",
                           color='white', fontsize=8, backgroundcolor=color)
            axes[1].axis('off')

            # 3. Predictions
            axes[2].imshow(frame_rgb)
            axes[2].set_title(f"Predictions ({len(pred_objects)} objects)")

            colors = plt.cm.rainbow(np.linspace(0, 1, max(len(pred_objects), 1)))
            for i, obj in enumerate(pred_objects):
                color = colors[i]
                axes[2].scatter(obj['x'], obj['y'], c=[color], s=100, marker='*')
                axes[2].text(obj['x'], obj['y'], f"P{obj['track_id']}",
                           color='white', fontsize=8, backgroundcolor=color)
            axes[2].axis('off')

            plt.suptitle(f"{video_id} - Frame {frame_num}")
            plt.tight_layout()

            # Save figure
            save_path = self.output_dir / f"{video_id}_frame_{frame_num}_comparison.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved visualization to {save_path}")

            # Analyze mismatches
            self.analyze_mismatches(gt_objects, pred_objects, video_id, frame_num)

    def analyze_mismatches(self, gt_objects, pred_objects, video_id, frame_num):
        """Analyze why detections don't match"""
        logger.info(f"\n{video_id} Frame {frame_num} Analysis:")
        logger.info(f"  Ground Truth: {len(gt_objects)} objects")
        logger.info(f"  Predicted: {len(pred_objects)} objects")

        if len(gt_objects) > 0:
            # Calculate distances between GT and predictions
            min_distances = []
            for gt in gt_objects:
                if len(pred_objects) > 0:
                    distances = [np.sqrt((gt['center'][0] - pred['x'])**2 +
                                       (gt['center'][1] - pred['y'])**2)
                                for pred in pred_objects]
                    min_dist = min(distances)
                    min_distances.append(min_dist)

                    if min_dist < 100:
                        logger.info(f"    ✓ GT {gt['track_id']} matched (dist={min_dist:.1f})")
                    else:
                        logger.info(f"    ✗ GT {gt['track_id']} NOT matched (min_dist={min_dist:.1f})")
                else:
                    logger.info(f"    ✗ GT {gt['track_id']} - NO predictions!")

            if min_distances:
                avg_dist = np.mean(min_distances)
                logger.info(f"  Average distance to nearest prediction: {avg_dist:.1f} pixels")

                if avg_dist > 200:
                    logger.info("  ⚠️ PROBLEM: Predictions are too far from ground truth!")
                    logger.info("  → Need to detect at actual object locations")

    def analyze_ground_truth_structure(self):
        """Understand ground truth structure better"""
        logger.info("\nAnalyzing Ground Truth Structure...")

        for video_id in ["E66F", "K16O", "P11H"]:
            gt_file = self.val_mot / f"{video_id}.txt"

            if not gt_file.exists():
                continue

            # Collect statistics
            frames = defaultdict(set)
            objects = defaultdict(set)
            keypoint_counts = []

            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 10:
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        object_id = int(parts[2])

                        frames[frame_id].add(track_id)
                        objects[track_id].add(object_id)

                        # Count keypoints
                        kp_count = (len(parts) - 7) // 3
                        keypoint_counts.append(kp_count)

            logger.info(f"\n{video_id} Statistics:")
            logger.info(f"  Total frames: {len(frames)}")
            logger.info(f"  Unique tracks: {len(objects)}")
            logger.info(f"  Objects per frame: {np.mean([len(f) for f in frames.values()]):.1f}")
            logger.info(f"  Keypoints per object: {np.mean(keypoint_counts):.1f}")

            # Sample first frame
            if frames:
                first_frame = min(frames.keys())
                logger.info(f"  First frame ({first_frame}) has tracks: {sorted(frames[first_frame])}")

    def find_detection_pattern(self):
        """Find pattern in ground truth to improve detection"""
        logger.info("\nFinding Detection Pattern...")

        video_id = "E66F"

        # Analyze first 10 frames
        all_centers = []
        for frame_num in range(10):
            objects = self.parse_ground_truth_frame(video_id, frame_num)
            for obj in objects:
                all_centers.append(obj['center'])

        if all_centers:
            all_centers = np.array(all_centers)

            # Find clusters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(6, len(all_centers)), random_state=42)
            kmeans.fit(all_centers)

            logger.info(f"\nTypical object locations (cluster centers):")
            for i, center in enumerate(kmeans.cluster_centers_):
                logger.info(f"  Object {i+1}: ({center[0]:.0f}, {center[1]:.0f})")

            # These are the regions we should focus detection on!
            return kmeans.cluster_centers_

        return None

def main():
    analyzer = FailureAnalyzer()

    # 1. Understand ground truth structure
    analyzer.analyze_ground_truth_structure()

    # 2. Find detection patterns
    typical_locations = analyzer.find_detection_pattern()

    # 3. Visualize specific frames
    logger.info("\nVisualizing comparisons...")
    analyzer.visualize_comparison("E66F", frame_nums=[0, 29, 58])

    # 4. Generate insights
    logger.info("\n" + "="*60)
    logger.info("KEY INSIGHTS:")
    logger.info("="*60)
    logger.info("1. Ground truth has 6 objects with ~3-6 keypoints each")
    logger.info("2. Our detection is finding wrong locations")
    logger.info("3. Need to detect at actual object positions, not grid")
    logger.info("4. Should use actual keypoint locations, not synthetic")

    if typical_locations is not None:
        logger.info(f"\n5. Focus detection on these regions:")
        for i, loc in enumerate(typical_locations):
            logger.info(f"   Region {i+1}: ({loc[0]:.0f}, {loc[1]:.0f})")

if __name__ == "__main__":
    main()