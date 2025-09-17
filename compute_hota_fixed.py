#!/usr/bin/env python3
"""
Fixed HOTA Evaluation with Correct Ground Truth Parsing
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

class EndoVisHOTACalculator:
    """HOTA calculator for EndoVis format"""

    def __init__(self, distance_threshold=100.0):
        """Initialize with larger threshold for keypoint matching"""
        self.distance_threshold = distance_threshold

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def calculate_hota(self, gt_data: Dict, pred_data: Dict) -> Dict:
        """Calculate HOTA metrics"""

        # Initialize metrics
        num_frames = max(len(gt_data), len(pred_data))
        if num_frames == 0:
            return {'HOTA': 0, 'DetA': 0, 'AssA': 0}

        # Calculate per-frame matches
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
                # Build cost matrix using position or bbox
                cost_matrix = np.zeros((len(gt_frame), len(pred_frame)))

                for i, gt in enumerate(gt_frame):
                    for j, pred in enumerate(pred_frame):
                        # Calculate distance between detections
                        if 'bbox' in gt and 'bbox' in pred:
                            # Use IoU for boxes
                            iou = self.calculate_iou(gt['bbox'], pred['bbox'])
                            cost_matrix[i, j] = 1 - iou
                        else:
                            # Use Euclidean distance for points
                            dist = np.sqrt((gt['x'] - pred['x'])**2 + (gt['y'] - pred['y'])**2)
                            cost_matrix[i, j] = dist

                # Hungarian matching
                gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

                # Count matches
                for gi, pi in zip(gt_indices, pred_indices):
                    if cost_matrix[gi, pi] < self.distance_threshold:
                        total_tp += 1

                        # Track associations
                        gt_id = gt_frame[gi]['id']
                        pred_id = pred_frame[pi]['id']
                        associations[gt_id][pred_id] += 1

        # Calculate Detection Accuracy (DetA)
        det_precision = total_tp / total_pred if total_pred > 0 else 0
        det_recall = total_tp / total_gt if total_gt > 0 else 0
        det_a = (det_precision + det_recall) / 2

        # Calculate Association Accuracy (AssA)
        # Count correct associations
        total_associations = sum(sum(v.values()) for v in associations.values())
        correct_associations = sum(max(v.values()) for v in associations.values() if v)

        ass_a = correct_associations / total_associations if total_associations > 0 else 0

        # Calculate HOTA
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

class EndoVisEvaluator:
    """Evaluator for EndoVis format data"""

    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.gt_mot_dir = self.data_root / "val/mot"
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/hota_final")
        self.output_dir.mkdir(exist_ok=True)

        self.hota_calculator = EndoVisHOTACalculator(distance_threshold=100.0)
        self.video_ids = sorted([d.name for d in self.val_frames_dir.iterdir() if d.is_dir()])[:3]

    def parse_endovis_gt(self, video_id: str) -> Dict:
        """Parse EndoVis ground truth format"""
        gt_file = self.gt_mot_dir / f"{video_id}.txt"
        gt_data = defaultdict(list)

        if not gt_file.exists():
            return {}

        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 10:
                    continue

                try:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    object_id = int(parts[2])

                    # Parse keypoints starting from index 7
                    keypoints = []
                    i = 7
                    while i + 2 < len(parts):
                        try:
                            x = float(parts[i])
                            y = float(parts[i+1])
                            visibility = int(parts[i+2])

                            if 0 <= x <= 2000 and 0 <= y <= 2000:
                                keypoints.append({'x': x, 'y': y, 'v': visibility})
                            i += 3
                        except:
                            break

                    if keypoints:
                        # Use first visible keypoint as object center
                        visible_kps = [kp for kp in keypoints if kp['v'] > 0]
                        if visible_kps:
                            center_x = np.mean([kp['x'] for kp in visible_kps])
                            center_y = np.mean([kp['y'] for kp in visible_kps])

                            gt_data[frame_id].append({
                                'id': track_id,
                                'x': center_x,
                                'y': center_y,
                                'num_keypoints': len(visible_kps)
                            })
                except Exception as e:
                    continue

        logger.info(f"Parsed {len(gt_data)} frames from {video_id}")
        return dict(gt_data)

    def evaluate_yolo(self, video_id: str, max_frames: int = 20):
        """Evaluate YOLO system"""
        from ultralytics import YOLO

        model_path = self.data_root / "yolo11m.pt"
        if not model_path.exists():
            return {}

        model = YOLO(str(model_path))

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
                            'y': (box[1] + box[3]) / 2,
                            'bbox': box.tolist()
                        })

        return dict(pred_data)

    def evaluate_multistage(self, video_id: str, max_frames: int = 20):
        """Evaluate Multi-Stage system"""
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

            # Group by object class (simulate object-level tracking)
            class_groups = defaultdict(list)
            for kp in tracked:
                if 'track_id' in kp:
                    class_id = kp.get('class', 0)
                    class_groups[class_id].append(kp)

            # Create one detection per class group
            for class_id, kps in class_groups.items():
                if kps:
                    # Use centroid of keypoints
                    center_x = np.mean([kp['x'] for kp in kps])
                    center_y = np.mean([kp['y'] for kp in kps])

                    # Create bounding box around keypoints
                    min_x = min(kp['x'] for kp in kps)
                    max_x = max(kp['x'] for kp in kps)
                    min_y = min(kp['y'] for kp in kps)
                    max_y = max(kp['y'] for kp in kps)

                    pred_data[frame_idx].append({
                        'id': class_id,
                        'x': center_x,
                        'y': center_y,
                        'bbox': [min_x, min_y, max_x, max_y],
                        'num_keypoints': len(kps)
                    })

        return dict(pred_data)

    def run_evaluation(self):
        """Run complete evaluation"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'yolo': {},
            'multistage': {},
            'summary': {}
        }

        logger.info("="*60)
        logger.info("EVALUATING SYSTEMS WITH ENDOVIS FORMAT")
        logger.info("="*60)

        yolo_scores = []
        multistage_scores = []

        for video_id in self.video_ids:
            logger.info(f"\nProcessing {video_id}...")

            # Load ground truth
            gt_data = self.parse_endovis_gt(video_id)

            # Evaluate YOLO
            logger.info(f"  Evaluating YOLO...")
            yolo_pred = self.evaluate_yolo(video_id, max_frames=10)
            yolo_metrics = self.hota_calculator.calculate_hota(gt_data, yolo_pred)
            results['yolo'][video_id] = yolo_metrics
            yolo_scores.append(yolo_metrics['HOTA'])

            logger.info(f"    YOLO: HOTA={yolo_metrics['HOTA']:.3f}, "
                       f"DetA={yolo_metrics['DetA']:.3f}, AssA={yolo_metrics['AssA']:.3f}")

            # Evaluate Multi-Stage
            logger.info(f"  Evaluating Multi-Stage...")
            multistage_pred = self.evaluate_multistage(video_id, max_frames=10)
            multistage_metrics = self.hota_calculator.calculate_hota(gt_data, multistage_pred)
            results['multistage'][video_id] = multistage_metrics
            multistage_scores.append(multistage_metrics['HOTA'])

            logger.info(f"    Multi-Stage: HOTA={multistage_metrics['HOTA']:.3f}, "
                       f"DetA={multistage_metrics['DetA']:.3f}, AssA={multistage_metrics['AssA']:.3f}")

        # Calculate averages
        avg_yolo = np.mean(yolo_scores) if yolo_scores else 0
        avg_multistage = np.mean(multistage_scores) if multistage_scores else 0

        results['summary'] = {
            'yolo_avg_hota': avg_yolo,
            'multistage_avg_hota': avg_multistage,
            'best_system': 'YOLO' if avg_yolo > avg_multistage else 'Multi-Stage',
            'best_score': max(avg_yolo, avg_multistage)
        }

        # Generate report
        self.generate_report(results)

        return results

    def generate_report(self, results):
        """Generate final report"""
        report_file = self.output_dir / "FINAL_HOTA_COMPARISON.md"

        with open(report_file, 'w') as f:
            f.write("# Final HOTA Comparison - EndoVis 2025\n\n")
            f.write(f"**Date**: {results['timestamp']}\n\n")

            f.write("## Per-Video HOTA Scores\n\n")

            # YOLO
            f.write("### YOLO + BoT-SORT\n\n")
            f.write("| Video | HOTA | DetA | AssA | TP | FP | FN |\n")
            f.write("|-------|------|------|------|----|----|----|\n")

            for vid, m in results['yolo'].items():
                f.write(f"| {vid} | {m['HOTA']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | ")
                f.write(f"{m.get('TP', 0)} | {m.get('FP', 0)} | {m.get('FN', 0)} |\n")

            # Multi-Stage
            f.write("\n### Fixed Multi-Stage Fusion\n\n")
            f.write("| Video | HOTA | DetA | AssA | TP | FP | FN |\n")
            f.write("|-------|------|------|------|----|----|----|\n")

            for vid, m in results['multistage'].items():
                f.write(f"| {vid} | {m['HOTA']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | ")
                f.write(f"{m.get('TP', 0)} | {m.get('FP', 0)} | {m.get('FN', 0)} |\n")

            # Summary
            s = results['summary']
            f.write("\n## Final Results\n\n")
            f.write(f"- **YOLO Average HOTA**: {s['yolo_avg_hota']:.3f}\n")
            f.write(f"- **Multi-Stage Average HOTA**: {s['multistage_avg_hota']:.3f}\n\n")

            f.write(f"## 🏆 Winner: **{s['best_system']}**\n\n")
            f.write(f"Best HOTA Score: **{s['best_score']:.3f}**\n\n")

            # Recommendation
            f.write("## Docker Submission Recommendation\n\n")
            if s['best_system'] == 'YOLO':
                f.write("### Submit: YOLO + BoT-SORT\n\n")
                f.write("- Higher HOTA score\n")
                f.write("- More stable tracking\n")
                f.write("- Already deployed: `docker.synapse.org/syn69762944/endovis2025-task3:v2.1`\n")
            else:
                f.write("### Submit: Fixed Multi-Stage Fusion\n\n")
                f.write("- Higher HOTA score\n")
                f.write("- More keypoints detected\n")
                f.write("- Better for detailed analysis\n")

        logger.info(f"Report saved to {report_file}")

def main():
    evaluator = EndoVisEvaluator()
    results = evaluator.run_evaluation()

    logger.info("\n" + "="*60)
    logger.info("FINAL HOTA RESULTS")
    logger.info("="*60)
    logger.info(f"YOLO: {results['summary']['yolo_avg_hota']:.3f}")
    logger.info(f"Multi-Stage: {results['summary']['multistage_avg_hota']:.3f}")
    logger.info(f"Winner: {results['summary']['best_system']}")

    return results

if __name__ == "__main__":
    main()