#!/usr/bin/env python3
"""
Evaluate Multi-Modal Fusion System on Validation Set
Combines YOLO+Anchors, Depth, Segmentation, and Dense detection
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
from multimodal_fusion_system import MultiModalFusionSystem

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
            'TP': 0, 'FP': 0, 'FN': 0
        }

    total_tp = 0
    total_fp = 0
    total_fn = 0
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

    return {
        'HOTA': hota,
        'DetA': det_a,
        'AssA': ass_a,
        'Precision': precision,
        'Recall': recall,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn
    }

def evaluate_multimodal_fusion():
    """Evaluate multi-modal fusion system on validation set"""

    logger.info("=" * 80)
    logger.info("MULTI-MODAL FUSION EVALUATION")
    logger.info("=" * 80)

    # Initialize system
    system = MultiModalFusionSystem(device='cpu')

    # Validation data
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # We'll test on all validation videos
    video_ids = ["E66F", "S4IC", "K16O"]

    all_results = {}

    for video_id in video_ids:
        logger.info(f"\nProcessing video: {video_id}")
        logger.info("-" * 40)

        # Parse ground truth
        gt_file = val_mot / f"{video_id}.txt"
        if not gt_file.exists():
            logger.warning(f"No ground truth for {video_id}")
            continue

        gt_keypoints = parse_mot_keypoints(gt_file)

        # Get frame files
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))

        # Get actual frame numbers from filenames
        frame_numbers = []
        for f in frame_files:
            frame_num = int(f.stem.split('_')[-1])
            frame_numbers.append(frame_num)

        # Process frames
        pred_keypoints = {}
        processing_times = []

        for idx, frame_file in enumerate(frame_files[:10]):  # Process first 10 frames
            frame_num = frame_numbers[idx]

            # Only process if we have GT for this frame
            if frame_num not in gt_keypoints:
                continue

            # Read frame
            frame = cv2.imread(str(frame_file))

            # Process with multi-modal fusion
            start_time = time.time()
            result = system.process_frame(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Store predictions
            pred_keypoints[frame_num] = []
            for kp in result['keypoints']:
                pred_keypoints[frame_num].append({
                    'track_id': kp['track_id'],
                    'x': kp['x'],
                    'y': kp['y']
                })

            if idx == 0:
                # Log first frame statistics
                stats = result['statistics']
                logger.info(f"  Frame {frame_num} detection sources:")
                logger.info(f"    YOLO candidates: {stats['yolo']}")
                logger.info(f"    Depth candidates: {stats['depth']}")
                logger.info(f"    Segmentation candidates: {stats['segmentation']}")
                logger.info(f"    Dense candidates: {stats['dense']}")
                logger.info(f"    After fusion: {stats['fused']}")
                logger.info(f"    Final tracked: {stats['tracked']}")

        # Calculate metrics
        metrics = calculate_metrics(gt_keypoints, pred_keypoints, threshold=100)

        # Calculate average detection distance
        total_dist = 0
        total_matches = 0

        for frame_id in set(gt_keypoints.keys()) & set(pred_keypoints.keys()):
            gt_frame = gt_keypoints[frame_id]
            pred_frame = pred_keypoints[frame_id]

            if len(gt_frame) > 0 and len(pred_frame) > 0:
                cost_matrix = np.zeros((len(gt_frame), len(pred_frame)))
                for i, gt in enumerate(gt_frame):
                    for j, pred in enumerate(pred_frame):
                        dist = np.sqrt((gt['x'] - pred['x'])**2 + (gt['y'] - pred['y'])**2)
                        cost_matrix[i, j] = dist

                gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
                for gi, pi in zip(gt_indices, pred_indices):
                    if cost_matrix[gi, pi] < 100:
                        total_dist += cost_matrix[gi, pi]
                        total_matches += 1

        avg_dist = total_dist / total_matches if total_matches > 0 else float('inf')
        avg_time = np.mean(processing_times)

        # Store results
        all_results[video_id] = {
            'metrics': metrics,
            'avg_distance': avg_dist,
            'avg_time': avg_time,
            'num_gt_keypoints': sum(len(kps) for kps in gt_keypoints.values()),
            'num_pred_keypoints': sum(len(kps) for kps in pred_keypoints.values())
        }

        # Log results
        logger.info(f"\n  Results for {video_id}:")
        logger.info(f"    HOTA: {metrics['HOTA']:.3f}")
        logger.info(f"    DetA: {metrics['DetA']:.3f}")
        logger.info(f"    AssA: {metrics['AssA']:.3f}")
        logger.info(f"    Precision: {metrics['Precision']:.3f}")
        logger.info(f"    Recall: {metrics['Recall']:.3f}")
        logger.info(f"    TP/FP/FN: {metrics['TP']}/{metrics['FP']}/{metrics['FN']}")
        logger.info(f"    Avg Distance: {avg_dist:.1f} pixels")
        logger.info(f"    Processing Time: {avg_time:.3f}s per frame")

    # Calculate overall metrics
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 80)

    avg_hota = np.mean([r['metrics']['HOTA'] for r in all_results.values()])
    avg_deta = np.mean([r['metrics']['DetA'] for r in all_results.values()])
    avg_assa = np.mean([r['metrics']['AssA'] for r in all_results.values()])
    avg_precision = np.mean([r['metrics']['Precision'] for r in all_results.values()])
    avg_recall = np.mean([r['metrics']['Recall'] for r in all_results.values()])

    logger.info(f"Average HOTA: {avg_hota:.3f}")
    logger.info(f"Average DetA: {avg_deta:.3f}")
    logger.info(f"Average AssA: {avg_assa:.3f}")
    logger.info(f"Average Precision: {avg_precision:.3f}")
    logger.info(f"Average Recall: {avg_recall:.3f}")

    # Compare with previous best
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON WITH PREVIOUS METHODS")
    logger.info("=" * 80)

    previous_results = {
        'Original Baseline': 0.127,
        'K-means Clustering': 0.187,
        'Dense Keypoints': 0.308,
        'YOLO + Anchors (Best)': 0.437
    }

    for method, hota in previous_results.items():
        improvement = ((avg_hota - hota) / hota * 100) if hota > 0 else 0
        symbol = "✅" if improvement > 0 else "❌"
        logger.info(f"{method:25s}: HOTA={hota:.3f} -> {avg_hota:.3f} ({improvement:+.1f}%) {symbol}")

    logger.info("\nMulti-Modal Fusion: HOTA={:.3f}".format(avg_hota))

    if avg_hota > 0.437:
        improvement = (avg_hota - 0.437) / 0.437 * 100
        logger.info(f"\n🎉 NEW BEST RESULT! Improved by {improvement:.1f}% over YOLO+Anchors")
    else:
        logger.info(f"\n⚠️  Multi-modal fusion HOTA ({avg_hota:.3f}) did not exceed YOLO+Anchors (0.437)")
        logger.info("   This may be due to:")
        logger.info("   - Missing segmentation masks for validation")
        logger.info("   - Need for parameter tuning")
        logger.info("   - Computational overhead of fusion")

    return all_results

if __name__ == "__main__":
    results = evaluate_multimodal_fusion()
    print("\n✅ Multi-modal fusion evaluation complete!")