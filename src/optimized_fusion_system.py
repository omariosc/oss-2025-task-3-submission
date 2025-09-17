#!/usr/bin/env python3
"""
Optimized Fusion System
Uses YOLO+Anchors as primary with selective depth enhancement
"""

import sys
import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import logging
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from multistage_yolo_anchors import YOLOAnchorFusion
from depth_guided_detection import DepthGuidedKeypointDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFusionSystem:
    """Optimized fusion using YOLO+Anchors with selective depth enhancement"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Optimized Fusion System on {device}")

        # Primary detection (proven best)
        self.yolo_anchor = YOLOAnchorFusion(device=device)

        # Secondary enhancement
        self.depth_detector = DepthGuidedKeypointDetector(device=device)

        # Fusion parameters
        self.depth_confidence_threshold = 0.8  # Only use high-confidence depth
        self.fusion_distance_threshold = 50    # Max distance for fusion
        self.depth_weight = 0.2                # Weight for depth contribution

        # Target keypoints
        self.target_keypoints = 23

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with optimized fusion"""

        # 1. Get primary detections from YOLO+Anchors (best method)
        yolo_result = self.yolo_anchor.process_frame(frame)
        primary_keypoints = yolo_result.get('tracked_keypoints', yolo_result.get('keypoints', []))

        # 2. If we have enough keypoints, just return them
        if len(primary_keypoints) >= self.target_keypoints:
            return {
                'keypoints': primary_keypoints[:self.target_keypoints],
                'num_keypoints': min(len(primary_keypoints), self.target_keypoints),
                'source': 'yolo_only'
            }

        # 3. Otherwise, selectively enhance with depth
        depth_map = self.depth_detector.depth_estimator.estimate_depth(frame)
        depth_keypoints = self.depth_detector.detect_keypoints_with_depth(frame, depth_map)

        # 4. Filter depth keypoints to only high-confidence ones
        high_conf_depth = [kp for kp in depth_keypoints
                          if kp.get('confidence', 0) > self.depth_confidence_threshold]

        # 5. Smart fusion - add depth keypoints that don't overlap with YOLO
        enhanced_keypoints = list(primary_keypoints)  # Start with YOLO

        for depth_kp in high_conf_depth:
            # Check if depth keypoint is far enough from existing ones
            too_close = False
            for existing_kp in enhanced_keypoints:
                dist = np.sqrt(
                    (depth_kp['x'] - existing_kp['x'])**2 +
                    (depth_kp['y'] - existing_kp['y'])**2
                )
                if dist < self.fusion_distance_threshold:
                    too_close = True
                    break

            if not too_close:
                # Add depth keypoint with reduced confidence
                depth_kp['confidence'] *= self.depth_weight
                depth_kp['source'] = 'depth_enhanced'
                enhanced_keypoints.append(depth_kp)

                if len(enhanced_keypoints) >= self.target_keypoints:
                    break

        # 6. If still not enough, use anchor fallback
        if len(enhanced_keypoints) < self.target_keypoints:
            # Add more anchors
            for anchor in self.yolo_anchor.anchor_locations[:self.target_keypoints]:
                if len(enhanced_keypoints) >= self.target_keypoints:
                    break

                ax, ay = anchor
                too_close = False
                for kp in enhanced_keypoints:
                    dist = np.sqrt((ax - kp['x'])**2 + (ay - kp['y'])**2)
                    if dist < 50:
                        too_close = True
                        break

                if not too_close:
                    enhanced_keypoints.append({
                        'track_id': len(enhanced_keypoints) + 1,
                        'x': ax,
                        'y': ay,
                        'confidence': 0.3,
                        'source': 'anchor_fallback'
                    })

        # 7. Ensure proper track IDs
        final_keypoints = enhanced_keypoints[:self.target_keypoints]
        for i, kp in enumerate(final_keypoints):
            if 'track_id' not in kp:
                kp['track_id'] = i + 1

        return {
            'keypoints': final_keypoints,
            'num_keypoints': len(final_keypoints),
            'source': 'optimized_fusion',
            'yolo_count': len(primary_keypoints),
            'depth_added': len([k for k in final_keypoints if k.get('source') == 'depth_enhanced'])
        }

def evaluate_optimized():
    """Quick evaluation of optimized fusion"""
    from pathlib import Path

    logger.info("\n" + "="*60)
    logger.info("OPTIMIZED FUSION EVALUATION")
    logger.info("="*60)

    system = OptimizedFusionSystem(device='cpu')

    # Test on validation frames
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    video_id = "E66F"
    frames_dir = val_frames / video_id
    gt_file = val_mot / f"{video_id}.txt"

    # Parse ground truth
    gt_keypoints = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame_num = int(parts[0])
                i = 7
                while i + 2 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        v = int(parts[i+2])
                        if v > 0:
                            gt_keypoints[frame_num].append({'x': x, 'y': y})
                        i += 3
                    except:
                        break

    # Process first 5 frames
    frame_files = sorted(list(frames_dir.glob("*.png")))[:5]
    pred_keypoints = defaultdict(list)

    for frame_file in frame_files:
        frame_num = int(frame_file.stem.split('_')[-1])
        if frame_num not in gt_keypoints:
            continue

        frame = cv2.imread(str(frame_file))
        result = system.process_frame(frame)

        for kp in result['keypoints']:
            pred_keypoints[frame_num].append({
                'x': kp['x'],
                'y': kp['y']
            })

        logger.info(f"Frame {frame_num}: {result['num_keypoints']} keypoints "
                   f"(YOLO: {result['yolo_count']}, Depth added: {result['depth_added']})")

    # Quick metric calculation
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for frame_num in gt_keypoints:
        if frame_num in pred_keypoints:
            gt = gt_keypoints[frame_num]
            pred = pred_keypoints[frame_num]

            if len(gt) > 0 and len(pred) > 0:
                cost_matrix = np.zeros((len(gt), len(pred)))
                for i, g in enumerate(gt):
                    for j, p in enumerate(pred):
                        dist = np.sqrt((g['x'] - p['x'])**2 + (g['y'] - p['y'])**2)
                        cost_matrix[i, j] = dist

                gt_idx, pred_idx = linear_sum_assignment(cost_matrix)
                for gi, pi in zip(gt_idx, pred_idx):
                    if cost_matrix[gi, pi] < 100:
                        total_tp += 1

                total_fn += len(gt) - len([i for i, j in zip(gt_idx, pred_idx)
                                          if cost_matrix[i, j] < 100])
                total_fp += len(pred) - len([j for i, j in zip(gt_idx, pred_idx)
                                            if cost_matrix[i, j] < 100])
            else:
                total_fn += len(gt)
                total_fp += len(pred)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    det_a = (precision + recall) / 2

    logger.info(f"\nQuick Results:")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall: {recall:.3f}")
    logger.info(f"  DetA: {det_a:.3f}")
    logger.info(f"  TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")

    # Estimate HOTA (assuming good association from YOLO tracking)
    estimated_hota = np.sqrt(det_a * 0.922)  # Using AssA from YOLO+Anchors
    logger.info(f"  Estimated HOTA: {estimated_hota:.3f}")

    if estimated_hota > 0.437:
        logger.info(f"  ✅ Potential improvement over YOLO+Anchors!")
    else:
        logger.info(f"  ⚠️  Similar to YOLO+Anchors baseline")

    return estimated_hota

if __name__ == "__main__":
    hota = evaluate_optimized()
    print(f"\n✅ Optimized fusion evaluation complete! Estimated HOTA: {hota:.3f}")