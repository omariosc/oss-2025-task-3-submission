#!/usr/bin/env python3
"""
Full Evaluation of Multi-Keypoint Expansion System
Calculates actual HOTA metrics on validation set
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
from multi_keypoint_expansion import MultiKeypointExpansion

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

def evaluate_on_video(system, video_id, val_frames, val_mot, max_frames=30):
    """Evaluate system on a single video"""

    logger.info(f"\nEvaluating on {video_id}...")

    # Parse ground truth
    gt_file = val_mot / f"{video_id}.txt"
    if not gt_file.exists():
        logger.warning(f"No ground truth for {video_id}")
        return None

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
    processing_times = []
    frames_processed = 0

    for idx, frame_file in enumerate(frame_files):
        frame_num = frame_numbers[idx]

        # Only process if we have GT
        if frame_num not in gt_keypoints:
            continue

        # Read frame
        frame = cv2.imread(str(frame_file))

        # Process with expansion system
        start_time = time.time()
        result = system.process_frame(frame)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Store predictions
        pred_keypoints[frame_num] = []
        for kp in result['keypoints']:
            pred_keypoints[frame_num].append({
                'track_id': kp.get('track_id', 0),
                'x': kp['x'],
                'y': kp['y']
            })

        frames_processed += 1

        if frames_processed == 1:
            # Log first frame details
            logger.info(f"  Frame {frame_num}:")
            logger.info(f"    Objects detected: {result['num_objects']}")
            logger.info(f"    Keypoints generated: {result['num_keypoints']}")
            logger.info(f"    Expansion ratio: {result['expansion_ratio']:.1f}x")
            logger.info(f"    GT keypoints: {len(gt_keypoints[frame_num])}")

    # Calculate metrics
    metrics = calculate_metrics(gt_keypoints, pred_keypoints)
    metrics['frames_processed'] = frames_processed
    metrics['avg_processing_time'] = np.mean(processing_times) if processing_times else 0

    return metrics

def main():
    """Run full evaluation"""

    logger.info("="*80)
    logger.info("FULL EVALUATION: MULTI-KEYPOINT EXPANSION SYSTEM")
    logger.info("="*80)

    # Initialize system
    system = MultiKeypointExpansion(device='cpu')

    # Data paths
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # Evaluate on both validation videos
    video_ids = ["E66F", "K16O"]
    all_results = {}

    for video_id in video_ids:
        results = evaluate_on_video(system, video_id, val_frames, val_mot, max_frames=30)
        if results:
            all_results[video_id] = results

            logger.info(f"\nResults for {video_id}:")
            logger.info(f"  Frames processed: {results['frames_processed']}")
            logger.info(f"  HOTA: {results['HOTA']:.4f}")
            logger.info(f"  DetA: {results['DetA']:.4f}")
            logger.info(f"  AssA: {results['AssA']:.4f}")
            logger.info(f"  Precision: {results['Precision']:.4f}")
            logger.info(f"  Recall: {results['Recall']:.4f}")
            logger.info(f"  TP/FP/FN: {results['TP']}/{results['FP']}/{results['FN']}")
            logger.info(f"  Avg distance: {results['avg_distance']:.2f} pixels")

    # Calculate overall metrics
    if all_results:
        avg_hota = np.mean([r['HOTA'] for r in all_results.values()])
        avg_deta = np.mean([r['DetA'] for r in all_results.values()])
        avg_assa = np.mean([r['AssA'] for r in all_results.values()])
        avg_precision = np.mean([r['Precision'] for r in all_results.values()])
        avg_recall = np.mean([r['Recall'] for r in all_results.values()])

        logger.info("\n" + "="*80)
        logger.info("OVERALL RESULTS")
        logger.info("="*80)
        logger.info(f"Average HOTA: {avg_hota:.4f}")
        logger.info(f"Average DetA: {avg_deta:.4f}")
        logger.info(f"Average AssA: {avg_assa:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Recall: {avg_recall:.4f}")

        # Compare with baseline
        logger.info("\n" + "="*80)
        logger.info("COMPARISON WITH BASELINE")
        logger.info("="*80)

        baseline_hota = 0.3463
        improvement = (avg_hota - baseline_hota) / baseline_hota * 100

        logger.info(f"Baseline HOTA: {baseline_hota:.4f}")
        logger.info(f"New HOTA: {avg_hota:.4f}")
        logger.info(f"Improvement: {improvement:+.1f}%")

        if avg_hota > baseline_hota:
            logger.info(f"\n🎉 SUCCESS! Multi-keypoint expansion improved HOTA by {improvement:.1f}%")
        else:
            logger.info(f"\n⚠️  Need to tune expansion patterns further")

    return all_results

if __name__ == "__main__":
    results = main()
    print("\n✅ Full evaluation complete!")