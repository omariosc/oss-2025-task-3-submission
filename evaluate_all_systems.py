#!/usr/bin/env python3
"""
Evaluate all tracking systems and compute HOTA metrics on full validation set
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
from scipy.spatial.distance import cdist

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HOTAEvaluator:
    """HOTA evaluator for tracking systems"""

    def __init__(self, distance_threshold=100.0):
        self.distance_threshold = distance_threshold

    def calculate_hota(self, gt_data: Dict, pred_data: Dict) -> Dict:
        """Calculate HOTA metrics"""
        num_frames = max(len(gt_data), len(pred_data))
        if num_frames == 0:
            return {'HOTA': 0, 'DetA': 0, 'AssA': 0}

        total_gt = 0
        total_pred = 0
        total_tp = 0
        associations = defaultdict(lambda: defaultdict(int))

        for frame_id in range(num_frames):
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

class SystemEvaluator:
    """Evaluate different tracking systems"""

    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.gt_mot_dir = self.data_root / "val/mot"
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/final_evaluation")
        self.output_dir.mkdir(exist_ok=True)

        self.hota_calculator = HOTAEvaluator(distance_threshold=100.0)
        self.video_ids = sorted([d.name for d in self.val_frames_dir.iterdir() if d.is_dir()])

        logger.info(f"Found {len(self.video_ids)} validation videos")

    def parse_gt(self, video_id: str) -> Dict:
        """Parse ground truth"""
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
                                'y': center_y
                            })
                    except:
                        continue

        return dict(gt_data)

    def evaluate_yolo(self, video_id: str, max_frames: int = 50):
        """Evaluate YOLO system"""
        try:
            from ultralytics import YOLO

            # Try to load trained model, fallback to pretrained
            model_paths = [
                self.data_root / "yolo11m.pt",
                "/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/models/yolo_trained/yolo_endovis_trained.pt"
            ]

            model = None
            for path in model_paths:
                if path.exists():
                    model = YOLO(str(path))
                    logger.info(f"Loaded YOLO model: {path}")
                    break

            if model is None:
                logger.warning("No YOLO model found, using default")
                model = YOLO('yolov8m.pt')

            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

            pred_data = defaultdict(list)

            for frame_idx, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue

                # YOLO detection with tracking
                results = model.track(frame, persist=True, tracker="botsort.yaml",
                                     conf=0.3, iou=0.5, verbose=False)

                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()

                        if result.boxes.id is not None:
                            track_ids = result.boxes.id.cpu().numpy().astype(int)
                        else:
                            track_ids = list(range(len(boxes)))

                        for box, track_id in zip(boxes, track_ids):
                            pred_data[frame_idx].append({
                                'id': int(track_id),
                                'x': (box[0] + box[2]) / 2,
                                'y': (box[1] + box[3]) / 2
                            })

            return dict(pred_data)

        except Exception as e:
            logger.error(f"YOLO evaluation error: {e}")
            return {}

    def evaluate_multistage(self, video_id: str, max_frames: int = 50):
        """Evaluate Multi-Stage Fusion system"""
        try:
            from multistage_fusion_fixed import FixedMultiStageFusion

            system = FixedMultiStageFusion(device='cpu')

            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

            pred_data = defaultdict(list)

            for frame_idx, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue

                # Detect and track
                keypoints = system.detect_keypoints(frame, use_nms=True)
                tracked = system.track_keypoints(frame, keypoints)

                # Group by class
                class_groups = defaultdict(list)
                for kp in tracked:
                    if 'track_id' in kp:
                        class_id = kp.get('class', 0)
                        class_groups[class_id].append(kp)

                for class_id, kps in class_groups.items():
                    if kps:
                        center_x = np.mean([kp['x'] for kp in kps])
                        center_y = np.mean([kp['y'] for kp in kps])

                        pred_data[frame_idx].append({
                            'id': class_id,
                            'x': center_x,
                            'y': center_y
                        })

            return dict(pred_data)

        except Exception as e:
            logger.error(f"Multi-Stage evaluation error: {e}")
            return {}

    def evaluate_enhanced(self, video_id: str, max_frames: int = 50):
        """Evaluate Enhanced Multi-Stage Fusion with attention"""
        try:
            from enhanced_multistage_fusion import EnhancedMultiStageFusion

            system = EnhancedMultiStageFusion(device='cpu')

            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

            pred_data = defaultdict(list)

            for frame_idx, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue

                # Run enhanced pipeline
                result = system.run_complete_pipeline(frame)

                for det in result['detections']:
                    pred_data[frame_idx].append({
                        'id': det['track_id'],
                        'x': det['x'],
                        'y': det['y']
                    })

            return dict(pred_data)

        except Exception as e:
            logger.error(f"Enhanced evaluation error: {e}")
            return {}

    def run_full_evaluation(self):
        """Run complete evaluation on all systems"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'systems': {
                'yolo': {'videos': {}, 'average': {}},
                'multistage': {'videos': {}, 'average': {}},
                'enhanced': {'videos': {}, 'average': {}}
            }
        }

        # Process each video
        for video_id in self.video_ids:
            logger.info(f"\nProcessing {video_id}...")

            # Load ground truth
            gt_data = self.parse_gt(video_id)
            logger.info(f"  GT frames: {len(gt_data)}")

            # Evaluate YOLO
            logger.info(f"  Evaluating YOLO...")
            yolo_pred = self.evaluate_yolo(video_id)
            yolo_metrics = self.hota_calculator.calculate_hota(gt_data, yolo_pred)
            results['systems']['yolo']['videos'][video_id] = yolo_metrics
            logger.info(f"    YOLO HOTA: {yolo_metrics['HOTA']:.3f}")

            # Evaluate Multi-Stage
            logger.info(f"  Evaluating Multi-Stage...")
            multistage_pred = self.evaluate_multistage(video_id)
            multistage_metrics = self.hota_calculator.calculate_hota(gt_data, multistage_pred)
            results['systems']['multistage']['videos'][video_id] = multistage_metrics
            logger.info(f"    Multi-Stage HOTA: {multistage_metrics['HOTA']:.3f}")

            # Evaluate Enhanced
            logger.info(f"  Evaluating Enhanced...")
            enhanced_pred = self.evaluate_enhanced(video_id)
            enhanced_metrics = self.hota_calculator.calculate_hota(gt_data, enhanced_pred)
            results['systems']['enhanced']['videos'][video_id] = enhanced_metrics
            logger.info(f"    Enhanced HOTA: {enhanced_metrics['HOTA']:.3f}")

        # Calculate averages
        for system in ['yolo', 'multistage', 'enhanced']:
            videos = results['systems'][system]['videos']
            if videos:
                avg_hota = np.mean([v['HOTA'] for v in videos.values()])
                avg_deta = np.mean([v['DetA'] for v in videos.values()])
                avg_assa = np.mean([v['AssA'] for v in videos.values()])

                results['systems'][system]['average'] = {
                    'HOTA': avg_hota,
                    'DetA': avg_deta,
                    'AssA': avg_assa
                }

        # Save results
        self.save_results(results)

        return results

    def save_results(self, results):
        """Save evaluation results"""
        # JSON results
        json_file = self.output_dir / "final_hota_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Markdown report
        md_file = self.output_dir / "FINAL_EVALUATION_REPORT.md"
        with open(md_file, 'w') as f:
            f.write("# Final HOTA Evaluation - All Systems\n\n")
            f.write(f"**Date**: {results['timestamp']}\n\n")

            f.write("## Summary\n\n")
            f.write("| System | Average HOTA | Average DetA | Average AssA |\n")
            f.write("|--------|-------------|--------------|-------------|\n")

            for system in ['yolo', 'multistage', 'enhanced']:
                avg = results['systems'][system]['average']
                f.write(f"| {system.title()} | {avg.get('HOTA', 0):.3f} | ")
                f.write(f"{avg.get('DetA', 0):.3f} | {avg.get('AssA', 0):.3f} |\n")

            # Determine winner
            best_system = max(results['systems'].items(),
                            key=lambda x: x[1]['average'].get('HOTA', 0))

            f.write(f"\n## 🏆 Winner: **{best_system[0].title()}**\n")
            f.write(f"Best HOTA Score: **{best_system[1]['average']['HOTA']:.3f}**\n\n")

            # Per-video results
            f.write("## Detailed Results\n\n")

            for system in ['yolo', 'multistage', 'enhanced']:
                f.write(f"### {system.title()}\n\n")
                f.write("| Video | HOTA | DetA | AssA | TP | FP | FN |\n")
                f.write("|-------|------|------|------|----|----|----|\n")

                for vid, metrics in results['systems'][system]['videos'].items():
                    f.write(f"| {vid} | {metrics['HOTA']:.3f} | {metrics['DetA']:.3f} | ")
                    f.write(f"{metrics['AssA']:.3f} | {metrics.get('TP', 0)} | ")
                    f.write(f"{metrics.get('FP', 0)} | {metrics.get('FN', 0)} |\n")

                f.write("\n")

        logger.info(f"\nResults saved to {self.output_dir}")

def main():
    evaluator = SystemEvaluator()
    results = evaluator.run_full_evaluation()

    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION COMPLETE")
    logger.info("="*60)

    for system in ['yolo', 'multistage', 'enhanced']:
        avg = results['systems'][system]['average']
        logger.info(f"{system.title()}: HOTA={avg.get('HOTA', 0):.3f}")

    # Determine winner
    best_system = max(results['systems'].items(),
                     key=lambda x: x[1]['average'].get('HOTA', 0))
    logger.info(f"\n🏆 Winner: {best_system[0].title()} with HOTA={best_system[1]['average']['HOTA']:.3f}")

if __name__ == "__main__":
    main()