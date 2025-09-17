#!/usr/bin/env python3
"""
Full Evaluation of BBox-based Keypoint Extraction
Calculate actual HOTA metrics
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import time

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from bbox_keypoint_extraction import BBoxKeypointExtractor
from multistage_yolo_anchors import YOLOAnchorFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_mot_keypoints(mot_file):
    """Parse individual keypoints from MOT format"""
    keypoints_by_frame = defaultdict(list)

    with open(mot_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame_num = int(parts[0])

                # Parse keypoints starting at index 7
                i = 7
                kp_idx = 0
                while i + 2 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        v = int(parts[i+2])
                        if v > 0:  # Visible keypoint
                            keypoints_by_frame[frame_num].append({
                                'id': kp_idx,
                                'x': x,
                                'y': y
                            })
                            kp_idx += 1
                        i += 3
                    except:
                        break

    return dict(keypoints_by_frame)

def calculate_metrics(gt_keypoints, pred_keypoints, threshold=100):
    """Calculate HOTA metrics"""
    if not gt_keypoints or not pred_keypoints:
        return {
            'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0,
            'Precision': 0.0, 'Recall': 0.0,
            'TP': 0, 'FP': 0, 'FN': 0,
            'avg_distance': float('inf')
        }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_distance = 0
    associations = defaultdict(lambda: defaultdict(int))

    for frame_id in set(gt_keypoints.keys()) | set(pred_keypoints.keys()):
        gt_frame = gt_keypoints.get(frame_id, [])
        pred_frame = pred_keypoints.get(frame_id, [])

        if len(gt_frame) > 0 and len(pred_frame) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(gt_frame), len(pred_frame)))
            for i, gt in enumerate(gt_frame):
                for j, pred in enumerate(pred_frame):
                    dist = np.sqrt((gt['x'] - pred['x'])**2 + (gt['y'] - pred['y'])**2)
                    cost_matrix[i, j] = dist

            # Hungarian matching
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

            matched_gt = set()
            matched_pred = set()

            for gi, pi in zip(gt_indices, pred_indices):
                if cost_matrix[gi, pi] < threshold:
                    total_tp += 1
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    total_distance += cost_matrix[gi, pi]

                    # Track associations
                    gt_id = gt_frame[gi].get('id', gi)
                    pred_id = pred_frame[pi].get('track_id', pi)
                    associations[gt_id][pred_id] += 1

            total_fn += len(gt_frame) - len(matched_gt)
            total_fp += len(pred_frame) - len(matched_pred)
        else:
            total_fn += len(gt_frame)
            total_fp += len(pred_frame)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    det_a = (precision + recall) / 2

    # Association accuracy
    total_associations = sum(sum(v.values()) for v in associations.values())
    correct_associations = sum(max(v.values()) for v in associations.values() if v)
    ass_a = correct_associations / total_associations if total_associations > 0 else 0

    # HOTA
    hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0

    # Average distance
    avg_distance = total_distance / total_tp if total_tp > 0 else float('inf')

    return {
        'HOTA': hota,
        'DetA': det_a,
        'AssA': ass_a,
        'Precision': precision,
        'Recall': recall,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'avg_distance': avg_distance
    }

def evaluate_system(system, system_name, video_ids, val_frames, val_mot, max_frames=50):
    """Evaluate a system"""
    all_results = {}

    for video_id in video_ids:
        logger.info(f"\nEvaluating {system_name} on {video_id}...")

        # Parse ground truth
        gt_file = val_mot / f"{video_id}.txt"
        if not gt_file.exists():
            continue

        gt_keypoints = parse_mot_keypoints(gt_file)

        # Get frame files
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

        # Get actual frame numbers
        frame_numbers = []
        for f in frame_files:
            frame_num = int(f.stem.split('_')[-1])
            frame_numbers.append(frame_num)

        # Process frames
        pred_keypoints = {}
        frames_processed = 0

        for idx, frame_file in enumerate(frame_files):
            frame_num = frame_numbers[idx]

            # Only process if we have GT
            if frame_num not in gt_keypoints:
                continue

            # Read frame
            frame = cv2.imread(str(frame_file))

            # Process
            result = system.process_frame(frame)

            # Store predictions
            pred_keypoints[frame_num] = []
            keypoints = result.get('keypoints', result.get('tracked_keypoints', []))
            for kp in keypoints:
                pred_keypoints[frame_num].append({
                    'track_id': kp.get('track_id', 0),
                    'x': kp.get('x', 0),
                    'y': kp.get('y', 0)
                })

            frames_processed += 1

        # Calculate metrics
        metrics = calculate_metrics(gt_keypoints, pred_keypoints)
        metrics['frames_processed'] = frames_processed
        all_results[video_id] = metrics

        logger.info(f"  Frames: {frames_processed}")
        logger.info(f"  HOTA: {metrics['HOTA']:.4f}")
        logger.info(f"  DetA: {metrics['DetA']:.4f}")
        logger.info(f"  AssA: {metrics['AssA']:.4f}")
        logger.info(f"  Precision: {metrics['Precision']:.4f}")
        logger.info(f"  Recall: {metrics['Recall']:.4f}")

    return all_results

def main():
    """Run full comparison"""

    logger.info("="*80)
    logger.info("FULL EVALUATION: BBOX EXTRACTION vs BASELINE")
    logger.info("="*80)

    # Data paths
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    video_ids = ["E66F", "K16O"]

    # Initialize systems
    logger.info("\nInitializing systems...")
    bbox_system = BBoxKeypointExtractor(device='cpu')
    baseline_system = YOLOAnchorFusion(device='cpu')

    # Evaluate both
    logger.info("\n1. Evaluating BBox Extraction...")
    bbox_results = evaluate_system(
        bbox_system, "BBox Extraction",
        video_ids, val_frames, val_mot, max_frames=50
    )

    logger.info("\n2. Evaluating Baseline (YOLO+Anchors)...")
    baseline_results = evaluate_system(
        baseline_system, "YOLO+Anchors",
        video_ids, val_frames, val_mot, max_frames=50
    )

    # Calculate overall metrics
    logger.info("\n" + "="*80)
    logger.info("OVERALL RESULTS")
    logger.info("="*80)

    for system_name, results in [("BBox Extraction", bbox_results),
                                 ("YOLO+Anchors", baseline_results)]:
        if results:
            avg_hota = np.mean([r['HOTA'] for r in results.values()])
            avg_deta = np.mean([r['DetA'] for r in results.values()])
            avg_assa = np.mean([r['AssA'] for r in results.values()])
            avg_precision = np.mean([r['Precision'] for r in results.values()])
            avg_recall = np.mean([r['Recall'] for r in results.values()])

            logger.info(f"\n{system_name}:")
            logger.info(f"  HOTA: {avg_hota:.4f}")
            logger.info(f"  DetA: {avg_deta:.4f}")
            logger.info(f"  AssA: {avg_assa:.4f}")
            logger.info(f"  Precision: {avg_precision:.4f}")
            logger.info(f"  Recall: {avg_recall:.4f}")

    # Comparison
    if bbox_results and baseline_results:
        bbox_hota = np.mean([r['HOTA'] for r in bbox_results.values()])
        baseline_hota = np.mean([r['HOTA'] for r in baseline_results.values()])

        logger.info("\n" + "="*80)
        logger.info("FINAL COMPARISON")
        logger.info("="*80)
        logger.info(f"Baseline HOTA: {baseline_hota:.4f}")
        logger.info(f"BBox HOTA: {bbox_hota:.4f}")

        if bbox_hota > baseline_hota:
            improvement = (bbox_hota - baseline_hota) / baseline_hota * 100
            logger.info(f"\n🎉 BBox extraction IMPROVED by {improvement:.1f}%!")
        else:
            logger.info(f"\n⚠️  Baseline still better")

    return bbox_results, baseline_results

if __name__ == "__main__":
    bbox, baseline = main()
    print("\n✅ Full evaluation complete!")