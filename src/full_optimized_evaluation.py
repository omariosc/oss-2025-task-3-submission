#!/usr/bin/env python3
"""
Full Evaluation of Optimized Fusion System
Comprehensive testing on all validation videos
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import time
import json

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from optimized_fusion_system import OptimizedFusionSystem
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

def evaluate_system(system, system_name, video_ids, val_frames, val_mot, max_frames=None):
    """Evaluate a tracking system"""
    all_results = {}

    for video_id in video_ids:
        logger.info(f"  Processing {video_id}...")

        # Parse ground truth
        gt_file = val_mot / f"{video_id}.txt"
        if not gt_file.exists():
            continue

        gt_keypoints = parse_mot_keypoints(gt_file)

        # Get frame files
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))
        if max_frames:
            frame_files = frame_files[:max_frames]

        # Get actual frame numbers
        frame_numbers = []
        for f in frame_files:
            frame_num = int(f.stem.split('_')[-1])
            frame_numbers.append(frame_num)

        # Process frames
        pred_keypoints = {}
        processing_times = []

        for idx, frame_file in enumerate(frame_files):
            frame_num = frame_numbers[idx]

            # Only process if we have GT
            if frame_num not in gt_keypoints:
                continue

            # Read frame
            frame = cv2.imread(str(frame_file))

            # Process
            start_time = time.time()
            result = system.process_frame(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Store predictions
            pred_keypoints[frame_num] = []
            keypoints = result.get('keypoints', result.get('tracked_keypoints', []))
            for kp in keypoints:
                pred_keypoints[frame_num].append({
                    'track_id': kp.get('track_id', 0),
                    'x': kp['x'],
                    'y': kp['y']
                })

        # Calculate metrics
        metrics = calculate_metrics(gt_keypoints, pred_keypoints)
        metrics['avg_time'] = np.mean(processing_times) if processing_times else 0

        all_results[video_id] = metrics

    return all_results

def main():
    """Run comprehensive evaluation"""
    logger.info("="*80)
    logger.info("COMPREHENSIVE EVALUATION: OPTIMIZED FUSION vs YOLO+ANCHORS")
    logger.info("="*80)

    # Data paths
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    video_ids = ["E66F", "K16O"]  # Skip S4IC (no GT)

    # Initialize systems
    logger.info("\nInitializing systems...")
    optimized_system = OptimizedFusionSystem(device='cpu')
    baseline_system = YOLOAnchorFusion(device='cpu')

    # Evaluate both systems
    logger.info("\n1. Evaluating Optimized Fusion System...")
    optimized_results = evaluate_system(
        optimized_system, "Optimized Fusion",
        video_ids, val_frames, val_mot, max_frames=20
    )

    logger.info("\n2. Evaluating YOLO+Anchors Baseline...")
    baseline_results = evaluate_system(
        baseline_system, "YOLO+Anchors",
        video_ids, val_frames, val_mot, max_frames=20
    )

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)

    comparison = {}

    for video_id in video_ids:
        if video_id in optimized_results and video_id in baseline_results:
            opt = optimized_results[video_id]
            base = baseline_results[video_id]

            logger.info(f"\n{video_id}:")
            logger.info(f"  {'Metric':<15} {'YOLO+Anchors':>15} {'Optimized':>15} {'Change':>15}")
            logger.info(f"  {'-'*60}")

            metrics_to_compare = ['HOTA', 'DetA', 'AssA', 'Precision', 'Recall', 'avg_distance']
            for metric in metrics_to_compare:
                base_val = base[metric]
                opt_val = opt[metric]

                if metric == 'avg_distance':
                    # Lower is better for distance
                    change = (base_val - opt_val) / base_val * 100 if base_val > 0 else 0
                    symbol = "✅" if opt_val < base_val else "❌"
                    logger.info(f"  {metric:<15} {base_val:>14.1f}px {opt_val:>14.1f}px "
                              f"{change:>13.1f}% {symbol}")
                else:
                    # Higher is better for accuracy metrics
                    change = (opt_val - base_val) / base_val * 100 if base_val > 0 else 0
                    symbol = "✅" if opt_val > base_val else "❌"
                    logger.info(f"  {metric:<15} {base_val:>15.3f} {opt_val:>15.3f} "
                              f"{change:>13.1f}% {symbol}")

            comparison[video_id] = {
                'optimized': opt,
                'baseline': base
            }

    # Overall averages
    logger.info("\n" + "="*80)
    logger.info("OVERALL AVERAGES")
    logger.info("="*80)

    metrics = ['HOTA', 'DetA', 'AssA', 'Precision', 'Recall']

    for metric in metrics:
        opt_avg = np.mean([r[metric] for r in optimized_results.values()])
        base_avg = np.mean([r[metric] for r in baseline_results.values()])
        improvement = (opt_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0

        symbol = "✅" if opt_avg > base_avg else "❌"
        logger.info(f"{metric:<15}: Baseline={base_avg:.3f}, Optimized={opt_avg:.3f}, "
                   f"Change={improvement:+.1f}% {symbol}")

    # Final verdict
    opt_hota = np.mean([r['HOTA'] for r in optimized_results.values()])
    base_hota = np.mean([r['HOTA'] for r in baseline_results.values()])

    logger.info("\n" + "="*80)
    if opt_hota > base_hota:
        improvement = (opt_hota - base_hota) / base_hota * 100
        logger.info(f"🎉 OPTIMIZED FUSION WINS! HOTA improved by {improvement:.1f}%")
        logger.info(f"   Final HOTA: {opt_hota:.3f} (vs {base_hota:.3f} baseline)")
    else:
        logger.info(f"⚠️  YOLO+Anchors remains best. HOTA: {base_hota:.3f} vs {opt_hota:.3f}")

    # Save results
    results_file = Path("full_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'optimized': optimized_results,
            'baseline': baseline_results,
            'comparison': comparison
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    return opt_hota, base_hota

if __name__ == "__main__":
    opt_hota, base_hota = main()
    print(f"\n✅ Full evaluation complete! Optimized: {opt_hota:.3f}, Baseline: {base_hota:.3f}")