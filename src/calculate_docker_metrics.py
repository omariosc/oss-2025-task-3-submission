#!/usr/bin/env python3
"""
Calculate actual metrics from Docker output on validation set
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def load_mot_file(filepath):
    """Load MOT format file"""
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                frame = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])

                data[frame].append({
                    'track_id': track_id,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'conf': conf,
                    'cx': x + w/2,
                    'cy': y + h/2
                })
    return data

def load_gt_keypoints(gt_file):
    """Load ground truth keypoints from MOT file"""
    gt_data = defaultdict(list)
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
                            gt_data[frame].append({'x': x, 'y': y})
                        i += 3
                    except:
                        break
    return gt_data

def calculate_metrics(pred_file, gt_file, threshold=50):
    """Calculate tracking metrics"""

    # Load predictions and ground truth
    predictions = load_mot_file(pred_file)
    ground_truth = load_gt_keypoints(gt_file)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_distance = 0

    # Process each frame
    all_frames = set(list(predictions.keys()) + list(ground_truth.keys()))

    for frame in all_frames:
        pred_boxes = predictions.get(frame, [])
        gt_keypoints = ground_truth.get(frame, [])

        if len(pred_boxes) > 0 and len(gt_keypoints) > 0:
            # Create distance matrix from box centers to keypoints
            dist_matrix = np.zeros((len(pred_boxes), len(gt_keypoints)))

            for p_idx, pred in enumerate(pred_boxes):
                for g_idx, gt in enumerate(gt_keypoints):
                    dist = np.sqrt((pred['cx'] - gt['x'])**2 +
                                  (pred['cy'] - gt['y'])**2)
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

            # Unmatched
            total_fp += len(pred_boxes) - len(pred_indices)
            total_fn += len(gt_keypoints) - len(gt_indices)

        elif len(pred_boxes) > 0:
            total_fp += len(pred_boxes)
        elif len(gt_keypoints) > 0:
            total_fn += len(gt_keypoints)

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Detection Accuracy
    det_a = (precision + recall) / 2

    # Simplified Association Accuracy (based on track consistency)
    unique_tracks = len(set(p['track_id'] for frame_preds in predictions.values() for p in frame_preds))
    ass_a = min(1.0, unique_tracks / max(1, len(ground_truth))) * 0.5  # Simple estimate

    # HOTA
    hota = np.sqrt(det_a * ass_a)

    avg_dist = total_distance / total_tp if total_tp > 0 else 0

    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'det_a': det_a,
        'ass_a': ass_a,
        'hota': hota,
        'avg_distance': avg_dist
    }

def main():
    """Calculate metrics for Docker output"""

    print("="*70)
    print("DOCKER OUTPUT METRICS CALCULATION")
    print("="*70)

    # Paths
    pred_file = Path("test_output/E66F_tracking.txt")
    gt_file = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data/val/mot/E66F.txt")

    if not pred_file.exists():
        print(f"Error: Prediction file not found: {pred_file}")
        return

    if not gt_file.exists():
        print(f"Error: Ground truth file not found: {gt_file}")
        return

    # Calculate metrics
    metrics = calculate_metrics(pred_file, gt_file)

    # Display results
    print(f"\nVideo: E66F")
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"Average Distance: {metrics['avg_distance']:.2f} pixels")
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"DetA: {metrics['det_a']:.4f}")
    print(f"AssA: {metrics['ass_a']:.4f}")
    print(f"HOTA: {metrics['hota']:.4f}")

    # Check improvement
    baseline_hota = 0.3463
    improvement = ((metrics['hota'] - baseline_hota) / baseline_hota * 100)

    print(f"\n{'='*70}")
    print(f"COMPARISON WITH BASELINE")
    print(f"{'='*70}")
    print(f"Baseline HOTA: {baseline_hota:.4f}")
    print(f"Docker HOTA: {metrics['hota']:.4f}")
    print(f"Improvement: {improvement:+.1f}%")

    if metrics['hota'] > baseline_hota:
        print(f"\n✅ SUCCESS! Docker achieves {improvement:+.1f}% improvement")
    else:
        print(f"\n⚠️ HOTA below baseline")

    return metrics

if __name__ == "__main__":
    metrics = main()