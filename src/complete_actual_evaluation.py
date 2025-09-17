#!/usr/bin/env python3
"""
Complete Actual Evaluation - No Estimates, Only Real Results
Tests all systems on full validation set and reports actual metrics
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
    """Calculate HOTA metrics - actual implementation"""
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

def evaluate_system_fully(system_class, system_name, device='cpu'):
    """Evaluate a system completely on all validation data"""

    logger.info(f"\n{'='*70}")
    logger.info(f"EVALUATING: {system_name}")
    logger.info(f"{'='*70}")

    # Initialize system
    system = system_class(device=device)

    # Data paths
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # Process all validation videos
    video_ids = ["E66F", "K16O"]  # S4IC has no ground truth

    all_results = {}

    for video_id in video_ids:
        logger.info(f"\nProcessing {video_id}...")

        # Parse ground truth
        gt_file = val_mot / f"{video_id}.txt"
        if not gt_file.exists():
            logger.warning(f"No ground truth for {video_id}")
            continue

        gt_keypoints = parse_mot_keypoints(gt_file)

        # Get all frame files
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))

        # Get actual frame numbers
        frame_numbers = []
        for f in frame_files:
            frame_num = int(f.stem.split('_')[-1])
            frame_numbers.append(frame_num)

        # Process ALL frames that have ground truth
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

            # Process with system
            start_time = time.time()
            result = system.process_frame(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Extract keypoints from result
            keypoints = result.get('tracked_keypoints',
                                  result.get('keypoints',
                                           result.get('objects', [])))

            # Store predictions
            pred_keypoints[frame_num] = []
            for kp in keypoints:
                pred_keypoints[frame_num].append({
                    'track_id': kp.get('track_id', kp.get('id', 0)),
                    'x': kp.get('x', 0),
                    'y': kp.get('y', 0)
                })

            frames_processed += 1

            # Log progress every 10 frames
            if frames_processed % 10 == 0:
                logger.info(f"  Processed {frames_processed} frames...")

        # Calculate metrics for this video
        metrics = calculate_metrics(gt_keypoints, pred_keypoints)
        metrics['frames_processed'] = frames_processed
        metrics['avg_processing_time'] = np.mean(processing_times) if processing_times else 0
        metrics['total_gt_keypoints'] = sum(len(kps) for kps in gt_keypoints.values())
        metrics['total_pred_keypoints'] = sum(len(kps) for kps in pred_keypoints.values())

        all_results[video_id] = metrics

        # Log results for this video
        logger.info(f"\nResults for {video_id}:")
        logger.info(f"  Frames processed: {frames_processed}")
        logger.info(f"  HOTA: {metrics['HOTA']:.4f}")
        logger.info(f"  DetA: {metrics['DetA']:.4f}")
        logger.info(f"  AssA: {metrics['AssA']:.4f}")
        logger.info(f"  Precision: {metrics['Precision']:.4f}")
        logger.info(f"  Recall: {metrics['Recall']:.4f}")
        logger.info(f"  TP/FP/FN: {metrics['TP']}/{metrics['FP']}/{metrics['FN']}")
        logger.info(f"  Avg distance: {metrics['avg_distance']:.2f} pixels")
        logger.info(f"  Processing time: {metrics['avg_processing_time']:.3f}s per frame")

    # Calculate overall metrics
    if all_results:
        avg_hota = np.mean([r['HOTA'] for r in all_results.values()])
        avg_deta = np.mean([r['DetA'] for r in all_results.values()])
        avg_assa = np.mean([r['AssA'] for r in all_results.values()])
        avg_precision = np.mean([r['Precision'] for r in all_results.values()])
        avg_recall = np.mean([r['Recall'] for r in all_results.values()])
        avg_distance = np.mean([r['avg_distance'] for r in all_results.values()
                               if r['avg_distance'] != float('inf')])

        logger.info(f"\n{'='*70}")
        logger.info(f"OVERALL RESULTS for {system_name}:")
        logger.info(f"  Average HOTA: {avg_hota:.4f}")
        logger.info(f"  Average DetA: {avg_deta:.4f}")
        logger.info(f"  Average AssA: {avg_assa:.4f}")
        logger.info(f"  Average Precision: {avg_precision:.4f}")
        logger.info(f"  Average Recall: {avg_recall:.4f}")
        logger.info(f"  Average Distance: {avg_distance:.2f} pixels")

        all_results['overall'] = {
            'HOTA': avg_hota,
            'DetA': avg_deta,
            'AssA': avg_assa,
            'Precision': avg_precision,
            'Recall': avg_recall,
            'avg_distance': avg_distance
        }

    return all_results

def main():
    """Run complete evaluation of all systems"""

    logger.info("="*80)
    logger.info("COMPLETE ACTUAL EVALUATION - ALL SYSTEMS")
    logger.info("="*80)

    results = {}

    # 1. Evaluate YOLO + Anchors (current best)
    logger.info("\n1. YOLO + Anchors System")
    from multistage_yolo_anchors import YOLOAnchorFusion
    results['yolo_anchors'] = evaluate_system_fully(
        YOLOAnchorFusion,
        "YOLO + Anchors",
        device='cpu'
    )

    # 2. Evaluate Optimized Fusion
    logger.info("\n2. Optimized Fusion System")
    from optimized_fusion_system import OptimizedFusionSystem
    results['optimized_fusion'] = evaluate_system_fully(
        OptimizedFusionSystem,
        "Optimized Fusion",
        device='cpu'
    )

    # 3. Evaluate Multi-Modal Fusion
    logger.info("\n3. Multi-Modal Fusion System")
    from multimodal_fusion_system import MultiModalFusionSystem
    results['multimodal_fusion'] = evaluate_system_fully(
        MultiModalFusionSystem,
        "Multi-Modal Fusion",
        device='cpu'
    )

    # Final comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON - ACTUAL RESULTS")
    logger.info("="*80)

    logger.info(f"\n{'System':<25} {'HOTA':>10} {'DetA':>10} {'AssA':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-"*75)

    for system_name, system_results in results.items():
        if 'overall' in system_results:
            overall = system_results['overall']
            logger.info(f"{system_name:<25} {overall['HOTA']:>10.4f} {overall['DetA']:>10.4f} "
                       f"{overall['AssA']:>10.4f} {overall['Precision']:>10.4f} {overall['Recall']:>10.4f}")

    # Determine winner
    best_system = max(results.items(),
                     key=lambda x: x[1].get('overall', {}).get('HOTA', 0))
    best_name, best_results = best_system
    best_hota = best_results['overall']['HOTA']

    logger.info("\n" + "="*80)
    logger.info(f"🏆 BEST SYSTEM: {best_name}")
    logger.info(f"   ACTUAL HOTA: {best_hota:.4f}")
    logger.info("="*80)

    # Save all results
    output_file = Path("complete_actual_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nAll results saved to {output_file}")

    return results

if __name__ == "__main__":
    results = main()
    print("\n✅ Complete actual evaluation finished!")