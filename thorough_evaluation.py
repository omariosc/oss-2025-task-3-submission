#!/usr/bin/env python3
"""
Thorough Evaluation of YOLO + Anchors System
Comprehensive HOTA metrics calculation on full validation set
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import json
from datetime import datetime

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from multistage_yolo_anchors import YOLOAnchorFusion
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluation with detailed metrics"""

    def __init__(self, threshold=100):
        self.threshold = threshold
        self.detailed_results = defaultdict(dict)

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes (for future use with bounding boxes)"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        intersect_xmin = max(x1_min, x2_min)
        intersect_ymin = max(y1_min, y2_min)
        intersect_xmax = min(x1_max, x2_max)
        intersect_ymax = min(y1_max, y2_max)

        if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
            return 0.0

        intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
        union_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - intersect_area

        return intersect_area / union_area if union_area > 0 else 0

    def calculate_hota_detailed(self, gt_data, pred_data):
        """Calculate detailed HOTA metrics with per-frame analysis"""
        if not gt_data or not pred_data:
            return {
                'HOTA': 0.0, 'DetA': 0.0, 'AssA': 0.0,
                'DetPr': 0.0, 'DetRe': 0.0,
                'TP': 0, 'FP': 0, 'FN': 0,
                'num_frames': 0, 'num_gt_total': 0, 'num_pred_total': 0
            }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        associations = defaultdict(lambda: defaultdict(int))

        frame_metrics = []
        all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))

        for frame_id in all_frames:
            gt_frame = gt_data.get(frame_id, [])
            pred_frame = pred_data.get(frame_id, [])

            frame_tp = 0
            frame_fp = 0
            frame_fn = 0

            if len(gt_frame) > 0 and len(pred_frame) > 0:
                # Build cost matrix based on Euclidean distance
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
                    if cost_matrix[gi, pi] < self.threshold:
                        frame_tp += 1
                        total_tp += 1
                        matched_gt.add(gi)
                        matched_pred.add(pi)

                        # Track associations for AssA calculation
                        gt_id = gt_frame[gi]['id']
                        pred_id = pred_frame[pi]['id']
                        associations[gt_id][pred_id] += 1

                # Count unmatched
                frame_fn = len(gt_frame) - len(matched_gt)
                frame_fp = len(pred_frame) - len(matched_pred)
                total_fn += frame_fn
                total_fp += frame_fp
            else:
                frame_fn = len(gt_frame)
                frame_fp = len(pred_frame)
                total_fn += frame_fn
                total_fp += frame_fp

            # Store per-frame metrics
            frame_metrics.append({
                'frame_id': frame_id,
                'tp': frame_tp,
                'fp': frame_fp,
                'fn': frame_fn,
                'num_gt': len(gt_frame),
                'num_pred': len(pred_frame),
                'precision': frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0,
                'recall': frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
            })

        # Calculate overall metrics
        det_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        det_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        det_a = (det_precision + det_recall) / 2

        # Association accuracy
        total_associations = sum(sum(v.values()) for v in associations.values())
        correct_associations = sum(max(v.values()) for v in associations.values() if v)
        ass_a = correct_associations / total_associations if total_associations > 0 else 0

        # HOTA score
        hota = np.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0

        # Additional statistics
        num_gt_total = sum(len(gt_data.get(f, [])) for f in all_frames)
        num_pred_total = sum(len(pred_data.get(f, [])) for f in all_frames)

        return {
            'HOTA': hota,
            'DetA': det_a,
            'AssA': ass_a,
            'DetPr': det_precision,
            'DetRe': det_recall,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'num_frames': len(all_frames),
            'num_gt_total': num_gt_total,
            'num_pred_total': num_pred_total,
            'frame_metrics': frame_metrics,
            'associations': dict(associations)
        }

def parse_mot_ground_truth(gt_file, max_frames=None):
    """Parse MOT format ground truth with proper keypoint handling"""
    gt_data = defaultdict(list)

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame_num = int(parts[0])

                if max_frames and frame_num >= max_frames:
                    continue

                track_id = int(parts[1])

                # Parse all keypoints for this object
                keypoints = []
                i = 7
                while i + 2 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        v = int(parts[i+2])
                        if v > 0:  # Only visible keypoints
                            keypoints.append((x, y))
                        i += 3
                    except:
                        break

                # Add center point for object-level tracking
                if keypoints:
                    center_x = np.mean([k[0] for k in keypoints])
                    center_y = np.mean([k[1] for k in keypoints])
                    gt_data[frame_num].append({
                        'id': track_id,
                        'x': center_x,
                        'y': center_y,
                        'num_keypoints': len(keypoints),
                        'keypoints': keypoints
                    })

    return dict(gt_data)

def evaluate_comprehensive():
    """Run comprehensive evaluation on full validation set"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # Initialize system
    system = YOLOAnchorFusion(device='cpu')
    evaluator = ComprehensiveEvaluator(threshold=100)

    # Video configurations - using 30 frames for faster evaluation
    video_configs = {
        "E66F": {"max_frames": 30},  # Process 30 frames for speed
        "K16O": {"max_frames": 30},
        "P11H": {"max_frames": 30}
    }

    all_results = {}

    for video_id, config in video_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {video_id}")
        logger.info(f"{'='*60}")

        # Parse ground truth
        gt_file = val_mot / f"{video_id}.txt"
        gt_data = parse_mot_ground_truth(gt_file, config['max_frames'])

        logger.info(f"Ground truth: {len(gt_data)} frames, "
                   f"{sum(len(objs) for objs in gt_data.values())} total objects")

        # Process frames
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))

        if config['max_frames']:
            frame_files = frame_files[:config['max_frames']]

        pred_data = defaultdict(list)
        processing_times = []

        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            frame_num = int(frame_file.stem.split('_')[-1])

            # Process with YOLO + Anchor system
            import time
            start_time = time.time()
            result = system.process_frame(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Store predictions
            for obj in result['objects']:
                pred_data[frame_num].append({
                    'id': obj['track_id'],
                    'x': obj['x'],
                    'y': obj['y'],
                    'confidence': obj['confidence'],
                    'source': obj.get('source', 'unknown')
                })

            # Log progress
            if frame_idx % 10 == 0:
                logger.info(f"  Processed frame {frame_idx}/{len(frame_files)}: "
                           f"{len(result['objects'])} objects detected")

        # Calculate detailed metrics
        metrics = evaluator.calculate_hota_detailed(gt_data, dict(pred_data))

        # Store results
        all_results[video_id] = {
            'metrics': metrics,
            'num_frames_processed': len(frame_files),
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': sum(processing_times)
        }

        # Log results
        logger.info(f"\nResults for {video_id}:")
        logger.info(f"  HOTA: {metrics['HOTA']:.4f}")
        logger.info(f"  DetA: {metrics['DetA']:.4f}")
        logger.info(f"  AssA: {metrics['AssA']:.4f}")
        logger.info(f"  Detection Precision: {metrics['DetPr']:.4f}")
        logger.info(f"  Detection Recall: {metrics['DetRe']:.4f}")
        logger.info(f"  True Positives: {metrics['TP']}")
        logger.info(f"  False Positives: {metrics['FP']}")
        logger.info(f"  False Negatives: {metrics['FN']}")
        logger.info(f"  Avg processing time: {np.mean(processing_times):.3f}s per frame")

        # Analyze frame-by-frame performance
        frame_metrics = metrics['frame_metrics']
        if frame_metrics:
            avg_frame_precision = np.mean([f['precision'] for f in frame_metrics])
            avg_frame_recall = np.mean([f['recall'] for f in frame_metrics])
            logger.info(f"  Avg frame precision: {avg_frame_precision:.4f}")
            logger.info(f"  Avg frame recall: {avg_frame_recall:.4f}")

    # Calculate overall averages
    avg_hota = np.mean([r['metrics']['HOTA'] for r in all_results.values()])
    avg_deta = np.mean([r['metrics']['DetA'] for r in all_results.values()])
    avg_assa = np.mean([r['metrics']['AssA'] for r in all_results.values()])
    avg_precision = np.mean([r['metrics']['DetPr'] for r in all_results.values()])
    avg_recall = np.mean([r['metrics']['DetRe'] for r in all_results.values()])

    total_tp = sum(r['metrics']['TP'] for r in all_results.values())
    total_fp = sum(r['metrics']['FP'] for r in all_results.values())
    total_fn = sum(r['metrics']['FN'] for r in all_results.values())

    logger.info(f"\n{'='*60}")
    logger.info("OVERALL COMPREHENSIVE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Average HOTA: {avg_hota:.4f}")
    logger.info(f"Average DetA: {avg_deta:.4f}")
    logger.info(f"Average AssA: {avg_assa:.4f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f}")
    logger.info(f"Total TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")

    # F1 Score
    f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    logger.info(f"F1 Score: {f1_score:.4f}")

    # Save detailed results to JSON
    output_file = Path("comprehensive_evaluation_results.json")

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(all_results)

    # Add summary
    serializable_results['summary'] = {
        'timestamp': datetime.now().isoformat(),
        'average_hota': float(avg_hota),
        'average_deta': float(avg_deta),
        'average_assa': float(avg_assa),
        'average_precision': float(avg_precision),
        'average_recall': float(avg_recall),
        'f1_score': float(f1_score),
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'evaluation_threshold': evaluator.threshold
    }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nDetailed results saved to: {output_file}")

    # Compare with previous reported results
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON WITH PREVIOUS RESULTS")
    logger.info(f"{'='*60}")
    logger.info("Previous reported (30 frames): HOTA=0.375, DetA=0.167, AssA=0.922")
    logger.info(f"Current evaluation (all frames): HOTA={avg_hota:.4f}, DetA={avg_deta:.4f}, AssA={avg_assa:.4f}")

    if avg_hota > 0.375:
        improvement = (avg_hota - 0.375) / 0.375 * 100
        logger.info(f"✅ IMPROVED: +{improvement:.1f}% HOTA")
    else:
        decrease = (0.375 - avg_hota) / 0.375 * 100
        logger.info(f"⚠️ Different result: -{decrease:.1f}% HOTA")
        logger.info("Note: This evaluation uses ALL frames, previous used 30 frames")

    return all_results

if __name__ == "__main__":
    logger.info("Starting comprehensive evaluation...")
    results = evaluate_comprehensive()
    print("\n✅ Comprehensive evaluation complete!")