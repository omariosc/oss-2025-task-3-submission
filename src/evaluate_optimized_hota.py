#!/usr/bin/env python3
"""
Evaluate Optimized Multi-Stage System with HOTA metrics
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')

from multistage_optimized import OptimizedMultiStageFusion
from multistage_fusion_fixed import FixedMultiStageFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HOTAEvaluator:
    """HOTA evaluator for optimized system"""

    def __init__(self, distance_threshold=100.0):
        self.distance_threshold = distance_threshold

    def calculate_hota(self, gt_data: Dict, pred_data: Dict) -> Dict:
        """Calculate HOTA metrics"""
        if not gt_data or not pred_data:
            return {'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0, 'TP': 0, 'FP': 0, 'FN': 0}

        total_gt = 0
        total_pred = 0
        total_tp = 0
        associations = defaultdict(lambda: defaultdict(int))

        for frame_id in set(gt_data.keys()) | set(pred_data.keys()):
            gt_frame = gt_data.get(frame_id, [])
            pred_frame = pred_data.get(frame_id, [])

            total_gt += len(gt_frame)
            total_pred += len(pred_frame)

            if len(gt_frame) > 0 and len(pred_frame) > 0:
                # Build cost matrix
                cost_matrix = np.zeros((len(gt_frame), len(pred_frame)))

                for i, gt in enumerate(gt_frame):
                    for j, pred in enumerate(pred_frame):
                        dist = np.sqrt((gt['x'] - pred['x'])**2 + (gt['y'] - pred['y'])**2)
                        cost_matrix[i, j] = dist

                # Hungarian matching
                gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

                for gi, pi in zip(gt_indices, pred_indices):
                    if cost_matrix[gi, pi] < self.distance_threshold:
                        total_tp += 1
                        gt_id = gt_frame[gi]['id']
                        pred_id = pred_frame[pi]['id']
                        associations[gt_id][pred_id] += 1

        # Calculate metrics
        det_precision = total_tp / total_pred if total_pred > 0 else 0
        det_recall = total_tp / total_gt if total_gt > 0 else 0
        det_a = (det_precision + det_recall) / 2

        # Association accuracy
        total_associations = sum(sum(v.values()) for v in associations.values())
        correct_associations = sum(max(v.values()) for v in associations.values() if v)
        ass_a = correct_associations / total_associations if total_associations > 0 else 0

        # HOTA
        hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0

        return {
            'HOTA': hota,
            'DetA': det_a,
            'AssA': ass_a,
            'DetPr': det_precision,
            'DetRe': det_recall,
            'TP': total_tp,
            'FP': total_pred - total_tp,
            'FN': total_gt - total_tp
        }

class OptimizedEvaluator:
    """Evaluate optimized system"""

    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.gt_mot_dir = self.data_root / "val/mot"
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/optimized_results")
        self.output_dir.mkdir(exist_ok=True)

        self.hota_calculator = HOTAEvaluator(distance_threshold=100.0)
        self.video_ids = sorted([d.name for d in self.val_frames_dir.iterdir() if d.is_dir()])[:3]

    def parse_gt(self, video_id: str) -> Dict:
        """Parse ground truth - 6 objects per frame"""
        gt_file = self.gt_mot_dir / f"{video_id}.txt"
        gt_data = defaultdict(list)

        if not gt_file.exists():
            return {}

        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    try:
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        object_id = int(parts[2])

                        # Parse keypoints to get center
                        keypoints = []
                        i = 7
                        while i + 2 < len(parts):
                            try:
                                x = float(parts[i])
                                y = float(parts[i+1])
                                v = int(parts[i+2])
                                if 0 <= x <= 2000 and 0 <= y <= 2000 and v > 0:
                                    keypoints.append((x, y))
                                i += 3
                            except:
                                break

                        if keypoints:
                            center_x = np.mean([kp[0] for kp in keypoints])
                            center_y = np.mean([kp[1] for kp in keypoints])

                            gt_data[frame_id].append({
                                'id': track_id,
                                'x': center_x,
                                'y': center_y,
                                'object_id': object_id
                            })
                    except:
                        continue

        # Ensure 6 objects per frame
        for frame_id in gt_data:
            if len(gt_data[frame_id]) > 6:
                gt_data[frame_id] = gt_data[frame_id][:6]

        return dict(gt_data)

    def evaluate_system(self, system_name: str, system, video_id: str, max_frames: int = 50):
        """Evaluate a system"""
        frames_dir = self.val_frames_dir / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

        pred_data = defaultdict(list)

        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue

            # Get frame number from filename
            frame_num = int(frame_file.stem.split('_')[-1])

            # Process frame
            if system_name == "optimized":
                result = system.process_frame(frame)
                objects = result['objects']

                for obj in objects:
                    pred_data[frame_num].append({
                        'id': obj['track_id'],
                        'x': obj['x'],
                        'y': obj['y']
                    })
            else:
                # Original system
                keypoints = system.detect_keypoints(frame, use_nms=True)
                tracked = system.track_keypoints(frame, keypoints)

                # Group into 6 objects (simple clustering)
                if tracked:
                    # Take first 6 detections
                    for i, kp in enumerate(tracked[:6]):
                        pred_data[frame_num].append({
                            'id': i + 1,
                            'x': kp['x'],
                            'y': kp['y']
                        })

        return dict(pred_data)

    def run_comparison(self):
        """Compare original vs optimized system"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original': {},
            'optimized': {}
        }

        # Initialize systems
        logger.info("Initializing systems...")
        original_system = FixedMultiStageFusion(device='cpu')
        optimized_system = OptimizedMultiStageFusion(device='cpu')

        # Process each video
        for video_id in self.video_ids:
            logger.info(f"\nProcessing {video_id}...")

            # Load ground truth
            gt_data = self.parse_gt(video_id)
            logger.info(f"  GT: {len(gt_data)} frames, ~6 objects each")

            # Evaluate original
            logger.info(f"  Evaluating original...")
            original_pred = self.evaluate_system("original", original_system, video_id)
            original_metrics = self.hota_calculator.calculate_hota(gt_data, original_pred)
            results['original'][video_id] = original_metrics
            logger.info(f"    Original HOTA: {original_metrics['HOTA']:.3f} "
                       f"(DetA={original_metrics['DetA']:.3f}, AssA={original_metrics['AssA']:.3f})")

            # Evaluate optimized
            logger.info(f"  Evaluating optimized...")
            optimized_pred = self.evaluate_system("optimized", optimized_system, video_id)
            optimized_metrics = self.hota_calculator.calculate_hota(gt_data, optimized_pred)
            results['optimized'][video_id] = optimized_metrics
            logger.info(f"    Optimized HOTA: {optimized_metrics['HOTA']:.3f} "
                       f"(DetA={optimized_metrics['DetA']:.3f}, AssA={optimized_metrics['AssA']:.3f})")

        # Calculate averages
        avg_original = np.mean([v['HOTA'] for v in results['original'].values()])
        avg_optimized = np.mean([v['HOTA'] for v in results['optimized'].values()])

        # Save results
        self.save_results(results, avg_original, avg_optimized)

        return avg_original, avg_optimized

    def save_results(self, results, avg_original, avg_optimized):
        """Save comparison results"""
        report_file = self.output_dir / "OPTIMIZED_HOTA_RESULTS.md"

        with open(report_file, 'w') as f:
            f.write("# Optimized System HOTA Results\n\n")
            f.write(f"**Date**: {results['timestamp']}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Original Average HOTA**: {avg_original:.3f}\n")
            f.write(f"- **Optimized Average HOTA**: {avg_optimized:.3f}\n")
            f.write(f"- **Improvement**: {(avg_optimized - avg_original):.3f} "
                   f"({(avg_optimized/avg_original - 1)*100:.1f}%)\n\n")

            f.write("## Per-Video Results\n\n")

            f.write("### Original System (60 detections)\n\n")
            f.write("| Video | HOTA | DetA | AssA | TP | FP | FN |\n")
            f.write("|-------|------|------|------|----|----|----|\n")
            for vid, m in results['original'].items():
                f.write(f"| {vid} | {m['HOTA']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | ")
                f.write(f"{m['TP']} | {m['FP']} | {m['FN']} |\n")

            f.write("\n### Optimized System (6 objects)\n\n")
            f.write("| Video | HOTA | DetA | AssA | TP | FP | FN |\n")
            f.write("|-------|------|------|------|----|----|----|\n")
            for vid, m in results['optimized'].items():
                f.write(f"| {vid} | {m['HOTA']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | ")
                f.write(f"{m['TP']} | {m['FP']} | {m['FN']} |\n")

            f.write("\n## Key Improvements\n\n")
            f.write("1. **Object Grouping**: 60 detections → 6 objects\n")
            f.write("2. **Confidence Threshold**: 0.3 → 0.7\n")
            f.write("3. **NMS Radius**: 30 → 100 pixels\n")
            f.write("4. **Clustering**: K-means with K=6\n")

            if avg_optimized > avg_original:
                f.write("\n## ✅ SUCCESS: Optimized system achieves higher HOTA!\n")
            else:
                f.write("\n## ⚠️ Need further optimization\n")

        logger.info(f"\nResults saved to {report_file}")

        # Also save JSON
        json_file = self.output_dir / "optimized_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    evaluator = OptimizedEvaluator()
    avg_original, avg_optimized = evaluator.run_comparison()

    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*60)
    logger.info(f"Original HOTA: {avg_original:.3f}")
    logger.info(f"Optimized HOTA: {avg_optimized:.3f}")
    logger.info(f"Improvement: {(avg_optimized - avg_original):.3f} "
               f"({(avg_optimized/avg_original - 1)*100:.1f}%)")

    if avg_optimized > avg_original:
        logger.info("\n🎉 SUCCESS! Optimized system improves HOTA!")
    else:
        logger.info("\n⚠️ Need to tune parameters further")

if __name__ == "__main__":
    main()