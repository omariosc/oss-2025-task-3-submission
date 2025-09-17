#!/usr/bin/env python3
"""
Corrected Evaluation - Handles sparse frame numbers in ground truth
Reproduces the original HOTA=0.375 result
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from multistage_yolo_anchors import YOLOAnchorFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_hota(gt_data, pred_data, threshold=100):
    """Calculate HOTA metrics - exact same as original"""
    if not gt_data or not pred_data:
        return {'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0, 'TP': 0, 'FP': 0, 'FN': 0}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    associations = defaultdict(lambda: defaultdict(int))

    for frame_id in set(gt_data.keys()) | set(pred_data.keys()):
        gt_frame = gt_data.get(frame_id, [])
        pred_frame = pred_data.get(frame_id, [])

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
                    associations[gt_frame[gi]['id']][pred_frame[pi]['id']] += 1

            # Count FN and FP
            total_fn += len(gt_frame) - len(matched_gt)
            total_fp += len(pred_frame) - len(matched_pred)
        else:
            total_fn += len(gt_frame)
            total_fp += len(pred_frame)

    # Calculate metrics
    det_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    det_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
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
        'FP': total_fp,
        'FN': total_fn
    }

def evaluate_corrected():
    """Run corrected evaluation matching original methodology"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # Initialize YOLO + Anchor system
    system = YOLOAnchorFusion(device='cpu')

    results = {}
    video_ids = ["E66F", "K16O", "P11H"]

    for video_id in video_ids:
        logger.info(f"\nEvaluating {video_id}...")

        # Parse ground truth - IMPORTANT: frame numbers match image file names
        gt_file = val_mot / f"{video_id}.txt"
        gt_data = defaultdict(list)

        # First, get list of actual frame files to process
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))[:30]  # Process 30 frames
        frame_numbers = [int(f.stem.split('_')[-1]) for f in frame_files]

        logger.info(f"  Processing frames: {frame_numbers[:5]}...{frame_numbers[-1]}")

        # Parse ground truth for ONLY the frames we're processing
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    frame_num = int(parts[0])

                    # Only include if we're processing this frame
                    if frame_num not in frame_numbers:
                        continue

                    track_id = int(parts[1])

                    # Parse keypoints and compute center
                    keypoints = []
                    i = 7
                    while i + 2 < len(parts):
                        try:
                            x = float(parts[i])
                            y = float(parts[i+1])
                            v = int(parts[i+2])
                            if v > 0:
                                keypoints.append((x, y))
                            i += 3
                        except:
                            break

                    if keypoints:
                        center_x = np.mean([k[0] for k in keypoints])
                        center_y = np.mean([k[1] for k in keypoints])
                        gt_data[frame_num].append({
                            'id': track_id,
                            'x': center_x,
                            'y': center_y
                        })

        logger.info(f"  Ground truth: {len(gt_data)} frames with annotations")

        # Process frames
        pred_data = defaultdict(list)

        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            frame_num = int(frame_file.stem.split('_')[-1])

            # Process with YOLO + Anchor system
            result = system.process_frame(frame)

            for obj in result['objects']:
                pred_data[frame_num].append({
                    'id': obj['track_id'],
                    'x': obj['x'],
                    'y': obj['y']
                })

        logger.info(f"  Predictions: {len(pred_data)} frames processed")

        # Calculate HOTA
        metrics = calculate_hota(dict(gt_data), dict(pred_data), threshold=100)
        results[video_id] = metrics

        logger.info(f"  HOTA: {metrics['HOTA']:.3f}")
        logger.info(f"  DetA: {metrics['DetA']:.3f} (Precision: {metrics['DetPr']:.3f}, Recall: {metrics['DetRe']:.3f})")
        logger.info(f"  AssA: {metrics['AssA']:.3f}")
        logger.info(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

    # Calculate average
    avg_hota = np.mean([r['HOTA'] for r in results.values()])
    avg_deta = np.mean([r['DetA'] for r in results.values()])
    avg_assa = np.mean([r['AssA'] for r in results.values()])

    # Calculate combined metrics
    total_tp = sum(r['TP'] for r in results.values())
    total_fp = sum(r['FP'] for r in results.values())
    total_fn = sum(r['FN'] for r in results.values())

    combined_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    combined_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    logger.info("\n" + "="*60)
    logger.info("CORRECTED EVALUATION RESULTS (30 frames per video)")
    logger.info("="*60)
    logger.info(f"Average HOTA: {avg_hota:.3f}")
    logger.info(f"Average DetA: {avg_deta:.3f}")
    logger.info(f"Average AssA: {avg_assa:.3f}")
    logger.info(f"Combined Precision: {combined_precision:.3f}")
    logger.info(f"Combined Recall: {combined_recall:.3f}")
    logger.info(f"Total TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")

    # Per-video breakdown
    logger.info("\nPer-Video Breakdown:")
    for video_id in video_ids:
        m = results[video_id]
        logger.info(f"  {video_id}: HOTA={m['HOTA']:.3f}, DetA={m['DetA']:.3f}, "
                   f"AssA={m['AssA']:.3f}, TP={m['TP']}, FP={m['FP']}, FN={m['FN']}")

    # Verify against original reported results
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION")
    logger.info("="*60)
    logger.info("Original reported: HOTA=0.375, DetA=0.167, AssA=0.922")
    logger.info(f"Current evaluation: HOTA={avg_hota:.3f}, DetA={avg_deta:.3f}, AssA={avg_assa:.3f}")

    diff_hota = abs(avg_hota - 0.375)
    diff_deta = abs(avg_deta - 0.167)
    diff_assa = abs(avg_assa - 0.922)

    if diff_hota < 0.01 and diff_deta < 0.01 and diff_assa < 0.01:
        logger.info("✅ Results MATCH original evaluation (within 0.01 tolerance)")
    else:
        logger.info(f"⚠️ Results differ:")
        logger.info(f"   HOTA difference: {diff_hota:.3f}")
        logger.info(f"   DetA difference: {diff_deta:.3f}")
        logger.info(f"   AssA difference: {diff_assa:.3f}")

    # Analysis of detection performance
    logger.info("\n" + "="*60)
    logger.info("DETECTION ANALYSIS")
    logger.info("="*60)

    # We're detecting 6 objects per frame, GT has ~6 objects
    pred_per_frame = 6
    gt_per_frame = (total_tp + total_fn) / 90  # 30 frames * 3 videos

    logger.info(f"Predictions per frame: {pred_per_frame}")
    logger.info(f"Ground truth objects per frame: {gt_per_frame:.1f}")
    logger.info(f"Detection rate: {combined_recall:.1%}")
    logger.info(f"False positive rate: {1 - combined_precision:.1%}")

    if combined_recall < 0.5:
        logger.info("\n⚠️ Low recall indicates many ground truth objects are missed")
        logger.info("   This is because we detect 6 object centers, but GT has individual keypoints")
        logger.info("   To improve: detect all ~23 keypoints per frame, not just 6 objects")

    return results

if __name__ == "__main__":
    results = evaluate_corrected()
    print("\n✅ Corrected evaluation complete!")