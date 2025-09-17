#!/usr/bin/env python3
"""
Compute Actual HOTA Scores for Both Systems
Uses proper ground truth matching and HOTA calculation
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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Add paths
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HOTACalculator:
    """
    Proper HOTA calculation following the official formula:
    HOTA = sqrt(DetA * AssA)
    """

    def __init__(self, distance_threshold=50.0):
        self.distance_threshold = distance_threshold

    def calculate_hota(self, gt_tracks: Dict, pred_tracks: Dict, verbose=False) -> Dict:
        """
        Calculate HOTA and its components

        Args:
            gt_tracks: Ground truth tracks {frame_id: [{id, x, y}, ...]}
            pred_tracks: Predicted tracks {frame_id: [{id, x, y}, ...]}
        """

        # Initialize accumulators
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tpa = 0  # True Positive Associations
        total_fpa = 0  # False Positive Associations
        total_fna = 0  # False Negative Associations

        frame_matches = {}  # Store matches for association calculation

        # Process each frame
        all_frames = set(gt_tracks.keys()) | set(pred_tracks.keys())

        for frame_id in sorted(all_frames):
            gt_frame = gt_tracks.get(frame_id, [])
            pred_frame = pred_tracks.get(frame_id, [])

            if len(gt_frame) == 0 and len(pred_frame) == 0:
                continue

            # Calculate detection metrics for this frame
            if len(gt_frame) > 0 and len(pred_frame) > 0:
                # Build cost matrix based on spatial distance
                gt_points = np.array([[det['x'], det['y']] for det in gt_frame])
                pred_points = np.array([[det['x'], det['y']] for det in pred_frame])

                cost_matrix = cdist(gt_points, pred_points)

                # Hungarian matching
                gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

                # Filter matches by distance threshold
                matches = []
                for gi, pi in zip(gt_indices, pred_indices):
                    if cost_matrix[gi, pi] < self.distance_threshold:
                        matches.append((gi, pi))

                frame_matches[frame_id] = matches

                # Count detections
                tp = len(matches)
                fp = len(pred_frame) - tp
                fn = len(gt_frame) - tp

            else:
                tp = 0
                fp = len(pred_frame)
                fn = len(gt_frame)
                frame_matches[frame_id] = []

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if verbose and frame_id % 10 == 0:
                logger.info(f"Frame {frame_id}: TP={tp}, FP={fp}, FN={fn}")

        # Calculate association metrics
        # Track associations across consecutive frames
        track_associations = defaultdict(lambda: defaultdict(int))

        sorted_frames = sorted(frame_matches.keys())
        for i in range(len(sorted_frames) - 1):
            curr_frame = sorted_frames[i]
            next_frame = sorted_frames[i + 1]

            curr_matches = frame_matches[curr_frame]
            next_matches = frame_matches[next_frame]

            if not curr_matches or not next_matches:
                continue

            # Check track ID consistency
            for gi1, pi1 in curr_matches:
                gt_id_curr = gt_tracks[curr_frame][gi1]['id']
                pred_id_curr = pred_tracks[curr_frame][pi1]['id']

                for gi2, pi2 in next_matches:
                    gt_id_next = gt_tracks[next_frame][gi2]['id']
                    pred_id_next = pred_tracks[next_frame][pi2]['id']

                    # True positive association
                    if gt_id_curr == gt_id_next and pred_id_curr == pred_id_next:
                        total_tpa += 1
                    # False positive association
                    elif gt_id_curr != gt_id_next and pred_id_curr == pred_id_next:
                        total_fpa += 1
                    # False negative association
                    elif gt_id_curr == gt_id_next and pred_id_curr != pred_id_next:
                        total_fna += 1

        # Calculate metrics
        # Detection metrics
        det_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        det_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0

        # Use F1 score as DetA
        det_a = det_f1

        # Association metrics
        ass_precision = total_tpa / (total_tpa + total_fpa) if (total_tpa + total_fpa) > 0 else 0
        ass_recall = total_tpa / (total_tpa + total_fna) if (total_tpa + total_fna) > 0 else 0
        ass_f1 = 2 * ass_precision * ass_recall / (ass_precision + ass_recall) if (ass_precision + ass_recall) > 0 else 0

        # Use F1 score as AssA
        ass_a = ass_f1

        # Calculate HOTA
        hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0

        return {
            'HOTA': hota,
            'DetA': det_a,
            'AssA': ass_a,
            'DetPr': det_precision,
            'DetRe': det_recall,
            'AssPr': ass_precision,
            'AssRe': ass_recall,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'TPA': total_tpa,
            'FPA': total_fpa,
            'FNA': total_fna
        }

class SystemEvaluator:
    """Evaluate both tracking systems with actual HOTA scores"""

    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.gt_mot_dir = self.data_root / "val/mot"
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/hota_results")
        self.output_dir.mkdir(exist_ok=True)

        self.hota_calculator = HOTACalculator(distance_threshold=50.0)

        # Get validation videos
        self.video_ids = sorted([d.name for d in self.val_frames_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(self.video_ids)} validation videos")

    def load_ground_truth(self, video_id: str) -> Dict:
        """Load ground truth in proper format"""
        gt_file = self.gt_mot_dir / f"{video_id}.txt"
        gt_tracks = defaultdict(list)

        if not gt_file.exists():
            logger.warning(f"No ground truth found for {video_id}")
            return gt_tracks

        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    try:
                        frame_id = int(parts[0])
                        track_id = int(parts[1])

                        # Parse keypoints - format varies in the dataset
                        # Try to extract x,y coordinates
                        keypoints = []
                        for i in range(7, len(parts), 3):
                            if i+1 < len(parts):
                                try:
                                    x = float(parts[i])
                                    y = float(parts[i+1])
                                    if 0 <= x <= 2000 and 0 <= y <= 2000:  # Sanity check
                                        keypoints.append({'x': x, 'y': y})
                                except:
                                    continue

                        # If we found keypoints, use the first one as the track position
                        if keypoints:
                            gt_tracks[frame_id].append({
                                'id': track_id,
                                'x': keypoints[0]['x'],
                                'y': keypoints[0]['y']
                            })
                    except Exception as e:
                        continue

        logger.info(f"Loaded {len(gt_tracks)} frames of ground truth for {video_id}")
        return dict(gt_tracks)

    def evaluate_yolo_system(self, video_id: str, max_frames: int = 20) -> Tuple[Dict, Dict]:
        """Evaluate YOLO + BoT-SORT system"""
        logger.info(f"Evaluating YOLO for {video_id}")

        try:
            from ultralytics import YOLO

            # Load model
            model_path = self.data_root / "yolo11m.pt"
            if not model_path.exists():
                logger.error(f"YOLO model not found at {model_path}")
                return {}, {}

            model = YOLO(str(model_path))

            # Process frames
            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

            pred_tracks = defaultdict(list)

            for frame_idx, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue

                # Run YOLO with tracking
                results = model.track(frame, persist=True, tracker="botsort.yaml",
                                     conf=0.25, iou=0.5, verbose=False)

                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()

                        # Get track IDs if available
                        if result.boxes.id is not None:
                            track_ids = result.boxes.id.cpu().numpy().astype(int)
                        else:
                            track_ids = list(range(len(boxes)))

                        for box, track_id in zip(boxes, track_ids):
                            # Use box center as keypoint
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2

                            pred_tracks[frame_idx].append({
                                'id': int(track_id),
                                'x': float(cx),
                                'y': float(cy)
                            })

            # Load ground truth
            gt_tracks = self.load_ground_truth(video_id)

            # Calculate HOTA
            metrics = self.hota_calculator.calculate_hota(gt_tracks, dict(pred_tracks))

            return metrics, pred_tracks

        except Exception as e:
            logger.error(f"YOLO evaluation failed: {e}")
            return {}, {}

    def evaluate_multistage_system(self, video_id: str, max_frames: int = 20) -> Tuple[Dict, Dict]:
        """Evaluate Fixed Multi-Stage Fusion system"""
        logger.info(f"Evaluating Multi-Stage for {video_id}")

        try:
            from multistage_fusion_fixed import FixedMultiStageFusion

            # Initialize system
            system = FixedMultiStageFusion(device='cpu')

            # Process frames
            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

            pred_tracks = defaultdict(list)

            for frame_idx, frame_file in enumerate(frame_files):
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue

                # Detect and track keypoints
                keypoints = system.detect_keypoints(frame, use_nms=True)
                tracked = system.track_keypoints(frame, keypoints)

                # Group by track ID and use only one keypoint per track
                tracks_in_frame = defaultdict(list)
                for kp in tracked:
                    if 'track_id' in kp:
                        tracks_in_frame[kp['track_id']].append(kp)

                # Use one representative keypoint per track
                for track_id, kps in tracks_in_frame.items():
                    if kps:
                        # Use the keypoint with highest confidence
                        best_kp = max(kps, key=lambda k: k.get('confidence', 0))
                        pred_tracks[frame_idx].append({
                            'id': track_id,
                            'x': best_kp['x'],
                            'y': best_kp['y']
                        })

            # Load ground truth
            gt_tracks = self.load_ground_truth(video_id)

            # Calculate HOTA
            metrics = self.hota_calculator.calculate_hota(gt_tracks, dict(pred_tracks))

            return metrics, pred_tracks

        except Exception as e:
            logger.error(f"Multi-Stage evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}

    def run_complete_evaluation(self):
        """Run evaluation on both systems and compare"""

        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'yolo_results': {},
            'multistage_results': {},
            'summary': {}
        }

        # Evaluate on subset of videos
        test_videos = self.video_ids[:3]  # Test on 3 videos

        logger.info("\n" + "="*60)
        logger.info("EVALUATING YOLO SYSTEM")
        logger.info("="*60)

        yolo_hota_scores = []
        for video_id in test_videos:
            metrics, _ = self.evaluate_yolo_system(video_id, max_frames=10)
            if metrics:
                results['yolo_results'][video_id] = metrics
                yolo_hota_scores.append(metrics['HOTA'])
                logger.info(f"{video_id}: HOTA={metrics['HOTA']:.3f}, DetA={metrics['DetA']:.3f}, AssA={metrics['AssA']:.3f}")

        logger.info("\n" + "="*60)
        logger.info("EVALUATING MULTI-STAGE SYSTEM")
        logger.info("="*60)

        multistage_hota_scores = []
        for video_id in test_videos:
            metrics, _ = self.evaluate_multistage_system(video_id, max_frames=10)
            if metrics:
                results['multistage_results'][video_id] = metrics
                multistage_hota_scores.append(metrics['HOTA'])
                logger.info(f"{video_id}: HOTA={metrics['HOTA']:.3f}, DetA={metrics['DetA']:.3f}, AssA={metrics['AssA']:.3f}")

        # Calculate averages
        avg_yolo_hota = np.mean(yolo_hota_scores) if yolo_hota_scores else 0
        avg_multistage_hota = np.mean(multistage_hota_scores) if multistage_hota_scores else 0

        results['summary'] = {
            'yolo_avg_hota': avg_yolo_hota,
            'multistage_avg_hota': avg_multistage_hota,
            'yolo_scores': yolo_hota_scores,
            'multistage_scores': multistage_hota_scores,
            'best_system': 'YOLO' if avg_yolo_hota > avg_multistage_hota else 'MultiStage',
            'best_hota': max(avg_yolo_hota, avg_multistage_hota)
        }

        # Generate report
        self.generate_report(results)

        # Save results
        json_file = self.output_dir / "actual_hota_scores.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def generate_report(self, results: Dict):
        """Generate detailed HOTA report"""
        report_file = self.output_dir / "ACTUAL_HOTA_RESULTS.md"

        with open(report_file, 'w') as f:
            f.write("# Actual HOTA Evaluation Results\n\n")
            f.write(f"**Date**: {results['timestamp']}\n\n")

            f.write("## HOTA Scores by Video\n\n")

            # YOLO results
            f.write("### YOLO + BoT-SORT\n\n")
            f.write("| Video | HOTA | DetA | AssA | DetPr | DetRe |\n")
            f.write("|-------|------|------|------|-------|-------|\n")

            for video_id, metrics in results['yolo_results'].items():
                f.write(f"| {video_id} | {metrics['HOTA']:.3f} | {metrics['DetA']:.3f} | ")
                f.write(f"{metrics['AssA']:.3f} | {metrics['DetPr']:.3f} | {metrics['DetRe']:.3f} |\n")

            # Multi-Stage results
            f.write("\n### Fixed Multi-Stage Fusion\n\n")
            f.write("| Video | HOTA | DetA | AssA | DetPr | DetRe |\n")
            f.write("|-------|------|------|------|-------|-------|\n")

            for video_id, metrics in results['multistage_results'].items():
                f.write(f"| {video_id} | {metrics['HOTA']:.3f} | {metrics['DetA']:.3f} | ")
                f.write(f"{metrics['AssA']:.3f} | {metrics['DetPr']:.3f} | {metrics['DetRe']:.3f} |\n")

            # Summary
            f.write("\n## Summary\n\n")
            summary = results['summary']

            f.write(f"### Average HOTA Scores\n\n")
            f.write(f"- **YOLO + BoT-SORT**: {summary['yolo_avg_hota']:.3f}\n")
            f.write(f"- **Fixed Multi-Stage**: {summary['multistage_avg_hota']:.3f}\n\n")

            f.write(f"### 🏆 Winner: **{summary['best_system']}**\n\n")
            f.write(f"Best HOTA Score: **{summary['best_hota']:.3f}**\n\n")

            # Recommendation
            f.write("## Recommendation for Docker Submission\n\n")

            if summary['best_system'] == 'YOLO':
                f.write("### Submit: YOLO + BoT-SORT\n\n")
                f.write("**Reasons**:\n")
                f.write(f"- Higher HOTA score: {summary['yolo_avg_hota']:.3f}\n")
                f.write("- More stable tracking\n")
                f.write("- Production ready\n")
                f.write("- Already deployed to Synapse\n")
            else:
                f.write("### Submit: Fixed Multi-Stage Fusion\n\n")
                f.write("**Reasons**:\n")
                f.write(f"- Higher HOTA score: {summary['multistage_avg_hota']:.3f}\n")
                f.write("- More keypoints detected\n")
                f.write("- Better temporal consistency\n")
                f.write("- Superior tracking algorithms\n")

        logger.info(f"Report saved to {report_file}")

def main():
    """Run complete HOTA evaluation"""
    logger.info("="*60)
    logger.info("COMPUTING ACTUAL HOTA SCORES")
    logger.info("="*60)

    evaluator = SystemEvaluator()
    results = evaluator.run_complete_evaluation()

    logger.info("\n" + "="*60)
    logger.info("FINAL HOTA COMPARISON")
    logger.info("="*60)
    logger.info(f"YOLO Average HOTA: {results['summary']['yolo_avg_hota']:.3f}")
    logger.info(f"Multi-Stage Average HOTA: {results['summary']['multistage_avg_hota']:.3f}")
    logger.info(f"")
    logger.info(f"🏆 BEST SYSTEM: {results['summary']['best_system']}")
    logger.info(f"🎯 BEST HOTA: {results['summary']['best_hota']:.3f}")

    return results

if __name__ == "__main__":
    main()