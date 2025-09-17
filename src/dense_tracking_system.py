#!/usr/bin/env python3
"""
Dense Keypoint Tracking System
Integrates dense detection with tracking for improved HOTA
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
from dense_keypoint_detector import DenseKeypointDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DenseKeypointTracker:
    """Track individual keypoints across frames"""

    def __init__(self, device='cpu'):
        self.device = device
        self.detector = DenseKeypointDetector(device=device)

        # Tracking
        self.kalman_filters = {}
        self.track_id_counter = 0
        self.active_tracks = {}
        self.track_age = {}
        self.max_age = 5

        # Expected keypoints per frame
        self.expected_keypoints = 23

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
        kf.R *= 5

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
        max_distance = 50  # Maximum distance for valid match

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
                'confidence': detection['confidence']
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
                'confidence': state.get('confidence', 0.5)
            })

        return tracked_keypoints

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame"""
        # Detect keypoints
        detections = self.detector.detect_keypoints(frame)

        # Track keypoints
        tracked = self.update_tracks(detections)

        return {
            'keypoints': tracked,
            'num_keypoints': len(tracked),
            'num_detections': len(detections)
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


def evaluate_dense_tracking():
    """Evaluate dense keypoint tracking system"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames = data_root / "val/frames"
    val_mot = data_root / "val/mot"

    # Initialize tracker
    tracker = DenseKeypointTracker(device='cpu')

    results = {}
    video_ids = ["E66F", "K16O", "P11H"]

    for video_id in video_ids:
        logger.info(f"\nEvaluating {video_id}...")

        # Parse ground truth (individual keypoints)
        gt_file = val_mot / f"{video_id}.txt"
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

        # Reset tracker for each video
        tracker.active_tracks = {}
        tracker.kalman_filters = {}
        tracker.track_id_counter = 0

        # Process frames
        frames_dir = val_frames / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))[:5]  # Process only 5 frames for quick test

        pred_data = defaultdict(list)

        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            frame_num = int(frame_file.stem.split('_')[-1])

            # Process with dense tracker
            result = tracker.process_frame(frame)

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
        results[video_id] = metrics

        logger.info(f"  HOTA: {metrics['HOTA']:.3f}")
        logger.info(f"  DetA: {metrics['DetA']:.3f} (Precision: {metrics['DetPr']:.3f}, "
                   f"Recall: {metrics['DetRe']:.3f})")
        logger.info(f"  AssA: {metrics['AssA']:.3f}")
        logger.info(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}")

    # Calculate average
    avg_hota = np.mean([r['HOTA'] for r in results.values()])
    avg_deta = np.mean([r['DetA'] for r in results.values()])
    avg_assa = np.mean([r['AssA'] for r in results.values()])

    logger.info("\n" + "="*60)
    logger.info("DENSE KEYPOINT TRACKING RESULTS")
    logger.info("="*60)
    logger.info(f"Average HOTA: {avg_hota:.3f}")
    logger.info(f"Average DetA: {avg_deta:.3f}")
    logger.info(f"Average AssA: {avg_assa:.3f}")

    # Compare with previous results
    logger.info("\nComparison with previous systems:")
    logger.info(f"  YOLO + Anchors (6 objects):  HOTA=0.375, DetA=0.167")
    logger.info(f"  Dense Keypoints (23 points): HOTA={avg_hota:.3f}, DetA={avg_deta:.3f}")

    improvement = (avg_hota - 0.375) / 0.375 * 100
    if improvement > 0:
        logger.info(f"\n✅ NEW BEST! Improvement: +{improvement:.1f}% over YOLO+Anchors")
    else:
        logger.info(f"\n❌ No improvement: {improvement:.1f}% from YOLO+Anchors")

    return results


if __name__ == "__main__":
    results = evaluate_dense_tracking()
    print("\n✅ Dense tracking evaluation complete!")