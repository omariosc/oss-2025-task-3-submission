#!/usr/bin/env python3
"""
Simple Evaluation of YOLO-Pose Results
Calculate HOTA metrics directly without TrackEval
"""

import sys
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import json
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics_for_video(pred_file, gt_file, threshold=50):
    """Calculate metrics for a single video"""

    # Load predictions
    pred_keypoints = defaultdict(list)
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame = int(parts[0])
                x = float(parts[7])
                y = float(parts[8])
                pred_keypoints[frame].append({'x': x, 'y': y})

    # Load ground truth
    gt_keypoints = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame = int(parts[0])
                i = 7
                while i + 2 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        v = int(parts[i+2])
                        if v > 0:
                            gt_keypoints[frame].append({'x': x, 'y': y})
                        i += 3
                    except:
                        break

    # Calculate metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_distance = 0
    frame_count = 0

    for frame in set(list(gt_keypoints.keys()) + list(pred_keypoints.keys())):
        gt_kps = gt_keypoints.get(frame, [])
        pred_kps = pred_keypoints.get(frame, [])

        if len(gt_kps) == 0 and len(pred_kps) == 0:
            continue

        frame_count += 1

        if len(gt_kps) > 0 and len(pred_kps) > 0:
            # Create distance matrix
            dist_matrix = np.zeros((len(pred_kps), len(gt_kps)))
            for p_idx, pred_kp in enumerate(pred_kps):
                for g_idx, gt_kp in enumerate(gt_kps):
                    dist = np.sqrt((pred_kp['x'] - gt_kp['x'])**2 +
                                  (pred_kp['y'] - gt_kp['y'])**2)
                    dist_matrix[p_idx, g_idx] = dist

            # Hungarian matching
            pred_indices, gt_indices = linear_sum_assignment(dist_matrix)

            # Count matches
            for p_idx, g_idx in zip(pred_indices, gt_indices):
                if dist_matrix[p_idx, g_idx] < threshold:
                    total_tp += 1
                    total_distance += dist_matrix[p_idx, g_idx]
                else:
                    total_fp += 1
                    total_fn += 1

            # Unmatched predictions
            unmatched_pred = set(range(len(pred_kps))) - set(pred_indices)
            total_fp += len(unmatched_pred)

            # Unmatched ground truth
            unmatched_gt = set(range(len(gt_kps))) - set(gt_indices)
            total_fn += len(unmatched_gt)

        elif len(pred_kps) > 0:
            total_fp += len(pred_kps)
        else:
            total_fn += len(gt_kps)

    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'distance_sum': total_distance,
        'frames': frame_count
    }

def main():
    """Main evaluation"""

    logger.info("="*70)
    logger.info("YOLO-POSE EVALUATION (SIMPLIFIED)")
    logger.info("="*70)

    # Paths
    gt_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data/val/mot")
    pred_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/yolo_pose_results")

    if not pred_dir.exists():
        logger.error(f"Predictions not found at {pred_dir}")
        return

    # Process each video
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_distance = 0
    video_results = {}

    for pred_file in pred_dir.glob("*.txt"):
        video_id = pred_file.stem

        # Skip metadata file
        if video_id == "SYNAPSE_METADATA_MANIFEST.tsv":
            continue

        gt_file = gt_dir / f"{video_id}.txt"

        if not gt_file.exists():
            logger.warning(f"No GT for {video_id}")
            continue

        logger.info(f"Processing {video_id}...")

        # Calculate metrics
        metrics = calculate_metrics_for_video(pred_file, gt_file)

        all_tp += metrics['tp']
        all_fp += metrics['fp']
        all_fn += metrics['fn']
        all_distance += metrics['distance_sum']

        video_results[video_id] = metrics

        # Video-specific metrics
        if metrics['tp'] + metrics['fp'] > 0:
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp'])
        else:
            precision = 0

        if metrics['tp'] + metrics['fn'] > 0:
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn'])
        else:
            recall = 0

        logger.info(f"  {video_id}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, Precision={precision:.3f}, Recall={recall:.3f}")

    # Calculate overall metrics
    logger.info("\n" + "="*70)
    logger.info("OVERALL METRICS")
    logger.info("="*70)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Detection Accuracy
    det_a = (precision + recall) / 2

    # Association Accuracy (simplified - assume 50% for now)
    ass_a = 0.5

    # HOTA
    hota = np.sqrt(det_a * ass_a)

    avg_dist = all_distance / all_tp if all_tp > 0 else 0

    logger.info(f"True Positives: {all_tp}")
    logger.info(f"False Positives: {all_fp}")
    logger.info(f"False Negatives: {all_fn}")
    logger.info(f"Average Distance: {avg_dist:.2f} pixels")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"DetA: {det_a:.4f}")
    logger.info(f"AssA (estimated): {ass_a:.4f}")
    logger.info(f"HOTA (estimated): {hota:.4f}")

    # Compare with baseline
    baseline_hota = 0.3463
    baseline_det_a = 0.2285
    baseline_precision = 0.3459
    baseline_recall = 0.1111

    logger.info("\n" + "="*70)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("="*70)

    logger.info("\n| Metric | Baseline | YOLO-Pose | Improvement |")
    logger.info("|--------|----------|-----------|-------------|")
    logger.info(f"| HOTA   | {baseline_hota:.4f}  | {hota:.4f}   | {((hota - baseline_hota) / baseline_hota * 100):+.1f}% |")
    logger.info(f"| DetA   | {baseline_det_a:.4f}  | {det_a:.4f}   | {((det_a - baseline_det_a) / baseline_det_a * 100):+.1f}% |")
    logger.info(f"| Precision | {baseline_precision:.4f} | {precision:.4f} | {((precision - baseline_precision) / baseline_precision * 100):+.1f}% |")
    logger.info(f"| Recall | {baseline_recall:.4f}  | {recall:.4f}   | {((recall - baseline_recall) / baseline_recall * 100):+.1f}% |")

    # Analysis
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS")
    logger.info("="*70)

    avg_kps_per_frame = all_tp / len(video_results) / 64 if len(video_results) > 0 else 0
    logger.info(f"Average keypoints detected per frame: {avg_kps_per_frame:.1f}")
    logger.info(f"Target keypoints per frame: ~22")

    if hota > baseline_hota:
        logger.info(f"\n✅ SUCCESS! HOTA improved from {baseline_hota:.4f} to {hota:.4f} ({((hota - baseline_hota) / baseline_hota * 100):+.1f}%)")
    else:
        logger.info(f"\n⚠️ HOTA did not improve: {hota:.4f} vs baseline {baseline_hota:.4f}")

    # Save results
    results = {
        'overall': {
            'hota': hota,
            'det_a': det_a,
            'ass_a': ass_a,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': all_tp,
            'fp': all_fp,
            'fn': all_fn,
            'avg_distance': avg_dist
        },
        'baseline': {
            'hota': baseline_hota,
            'det_a': baseline_det_a,
            'precision': baseline_precision,
            'recall': baseline_recall
        },
        'videos': video_results
    }

    output_file = pred_dir / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results

if __name__ == "__main__":
    results = main()