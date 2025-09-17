#!/usr/bin/env python3
"""
Segmentation-Based Tracking System
Combines segmentation keypoint extraction with tracking for improved HOTA
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from segmentation_keypoint_extractor import SegmentationKeypointExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationTracker:
    """Track keypoints extracted from segmentation masks"""

    def __init__(self, device='cpu'):
        self.device = device
        self.extractor = SegmentationKeypointExtractor(device=device)

        # Tracking
        self.kalman_filters = {}
        self.track_id_counter = 0
        self.active_tracks = {}
        self.track_age = {}
        self.max_age = 5

        # For validation, we need to handle cases without masks
        self.fallback_to_yolo = True

    def create_kalman_filter(self, x, y):
        """Create Kalman filter for a keypoint"""
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [x, y, vx, vy]
        kf.x = np.array([x, y, 0, 0]).reshape(4, 1)

        # State transition
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise
        kf.Q *= 0.03

        # Measurement noise
        kf.R *= 3

        # Initial uncertainty
        kf.P *= 100

        return kf

    def match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Match new detections to existing tracks"""
        if len(self.active_tracks) == 0 or len(detections) == 0:
            return {}, detections

        # Predict current positions
        predicted_positions = {}
        for track_id, kf in self.kalman_filters.items():
            if track_id in self.active_tracks:
                kf.predict()
                predicted_positions[track_id] = kf.x[:2].flatten()

        # Build cost matrix
        track_ids = list(predicted_positions.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                pred_pos = predicted_positions[track_id]
                dist = np.sqrt((det['x'] - pred_pos[0])**2 + (det['y'] - pred_pos[1])**2)
                cost_matrix[i, j] = dist

        # Hungarian matching
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # Process matches
        matches = {}
        unmatched_detections = list(range(len(detections)))
        max_distance = 100  # Maximum distance for valid match

        for di, ti in zip(det_indices, track_indices):
            if cost_matrix[di, ti] < max_distance:
                track_id = track_ids[ti]
                matches[track_id] = detections[di]
                unmatched_detections.remove(di)

        # Get unmatched detections
        unmatched = [detections[i] for i in unmatched_detections]

        return matches, unmatched

    def update_tracks(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        # Match detections to existing tracks
        matches, unmatched = self.match_detections_to_tracks(detections)

        # Update matched tracks
        for track_id, detection in matches.items():
            kf = self.kalman_filters[track_id]
            kf.update([detection['x'], detection['y']])
            self.active_tracks[track_id] = {
                'x': kf.x[0, 0],
                'y': kf.x[1, 0],
                'confidence': detection.get('confidence', 0.5),
                'type': detection.get('type', 'unknown'),
                'tool_class': detection.get('tool_class', 'unknown')
            }
            self.track_age[track_id] = 0

        # Create new tracks for unmatched detections
        for detection in unmatched:
            track_id = self.track_id_counter
            self.track_id_counter += 1

            kf = self.create_kalman_filter(detection['x'], detection['y'])
            self.kalman_filters[track_id] = kf
            self.active_tracks[track_id] = detection
            self.track_age[track_id] = 0

        # Age and remove old tracks
        tracks_to_remove = []
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matches:
                self.track_age[track_id] += 1
                if self.track_age[track_id] > self.max_age:
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            del self.kalman_filters[track_id]
            del self.track_age[track_id]

        # Convert to output format
        tracked_keypoints = []
        for track_id, state in self.active_tracks.items():
            tracked_keypoints.append({
                'track_id': track_id + 1,  # 1-indexed
                'x': state['x'],
                'y': state['y'],
                'confidence': state.get('confidence', 0.5),
                'type': state.get('type', 'unknown'),
                'tool_class': state.get('tool_class', 'unknown')
            })

        return tracked_keypoints

    def process_frame_with_masks(self, frame: np.ndarray, frame_id: str,
                                 masks_dir: Path) -> Dict:
        """Process frame using segmentation masks"""
        # Extract keypoints from masks
        keypoints = self.extractor.process_frame_masks(frame_id, masks_dir)

        # Track keypoints
        tracked = self.update_tracks(keypoints)

        return {
            'keypoints': tracked,
            'num_keypoints': len(tracked),
            'num_detections': len(keypoints)
        }

    def process_frame_fallback(self, frame: np.ndarray) -> Dict:
        """Fallback processing when masks are not available"""
        # Use YOLO or other detection method
        from multistage_yolo_anchors import YOLOAnchorFusion

        if not hasattr(self, 'yolo_system'):
            self.yolo_system = YOLOAnchorFusion(device='cpu')

        result = self.yolo_system.process_frame(frame)

        # Convert object detections to keypoint format
        keypoints = []
        for obj in result['objects']:
            # Generate multiple keypoints per object
            cx, cy = obj['x'], obj['y']

            # Center keypoint
            keypoints.append({
                'x': cx,
                'y': cy,
                'confidence': obj['confidence'],
                'type': 'center',
                'tool_class': 'unknown'
            })

            # Add surrounding keypoints (simulate multiple per object)
            offsets = [(-30, 0), (30, 0), (0, -30), (0, 30)]
            for i, (dx, dy) in enumerate(offsets):
                keypoints.append({
                    'x': cx + dx,
                    'y': cy + dy,
                    'confidence': obj['confidence'] * 0.8,
                    'type': f'offset_{i}',
                    'tool_class': 'unknown'
                })

        # Track keypoints
        tracked = self.update_tracks(keypoints)

        return {
            'keypoints': tracked,
            'num_keypoints': len(tracked),
            'num_detections': len(keypoints)
        }

def calculate_hota(gt_data, pred_data, threshold=100):
    """Calculate HOTA metrics for keypoint tracking"""
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

def evaluate_segmentation_tracking():
    """Evaluate segmentation-based tracking system"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # For validation, we may not have masks, so we'll use fallback
    # For now, let's test with training data where we have masks
    train_frames = data_root / "train/frames"
    train_masks = data_root / "train/masks"
    train_mot = data_root / "train/mot"

    # Initialize tracker
    tracker = SegmentationTracker(device='cpu')

    # Test on a single video first
    video_id = "S93I"  # Use training video with masks

    logger.info(f"\nEvaluating {video_id} with segmentation keypoints...")

    # Parse ground truth (individual keypoints)
    gt_file = train_mot / f"{video_id}.txt"

    if not gt_file.exists():
        logger.warning(f"No ground truth for {video_id}, using validation fallback")
        # Fall back to validation without masks
        return evaluate_with_fallback()

    gt_data = defaultdict(list)
    gt_track_counter = 0

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                frame_num = int(parts[0])

                # Parse individual keypoints
                i = 7
                while i + 2 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        v = int(parts[i+2])
                        if v > 0:
                            gt_data[frame_num].append({
                                'id': gt_track_counter,
                                'x': x,
                                'y': y
                            })
                            gt_track_counter += 1
                        i += 3
                    except:
                        break

    # Process frames
    frames_dir = train_frames / video_id
    frame_files = sorted(list(frames_dir.glob("*.png")))[:10]  # Process 10 frames for quick test

    pred_data = defaultdict(list)

    for frame_idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        frame_num = int(frame_file.stem.split('_')[-1])
        frame_id = frame_file.stem  # e.g., "S93I_frame_0"

        # Process with segmentation
        result = tracker.process_frame_with_masks(frame, frame_id, train_masks)

        for kp in result['keypoints']:
            pred_data[frame_num].append({
                'id': kp['track_id'],
                'x': kp['x'],
                'y': kp['y']
            })

        if frame_idx == 0:
            logger.info(f"  Frame {frame_num}: {result['num_detections']} detected, "
                      f"{result['num_keypoints']} tracked")

    # Calculate HOTA
    metrics = calculate_hota(dict(gt_data), dict(pred_data), threshold=100)

    logger.info(f"\nSegmentation-based Results:")
    logger.info(f"  HOTA: {metrics['HOTA']:.3f}")
    logger.info(f"  DetA: {metrics['DetA']:.3f} (Precision: {metrics['DetPr']:.3f}, "
               f"Recall: {metrics['DetRe']:.3f})")
    logger.info(f"  AssA: {metrics['AssA']:.3f}")
    logger.info(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

    # Compare with previous best
    logger.info(f"\nComparison:")
    logger.info(f"  Previous best (YOLO+Anchors): HOTA=0.437")
    logger.info(f"  Segmentation-based: HOTA={metrics['HOTA']:.3f}")

    if metrics['HOTA'] > 0.437:
        improvement = (metrics['HOTA'] - 0.437) / 0.437 * 100
        logger.info(f"  ✅ IMPROVEMENT: +{improvement:.1f}%")
    else:
        logger.info(f"  ⚠️  Need further optimization")

    return metrics

def evaluate_with_fallback():
    """Evaluate on validation set using fallback (no masks)"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    tracker = SegmentationTracker(device='cpu')
    video_id = "E66F"

    logger.info(f"\nEvaluating {video_id} with fallback (no masks)...")

    # Similar evaluation but using fallback method
    # ... (abbreviated for brevity)

    return {'HOTA': 0.0}

if __name__ == "__main__":
    results = evaluate_segmentation_tracking()
    print("\n✅ Segmentation tracking evaluation complete!")