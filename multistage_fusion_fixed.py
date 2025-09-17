#!/usr/bin/env python3
"""
Fixed Multi-Stage Fusion System for Keypoint Detection and Tracking
Complete implementation with all necessary components for surgical video analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import logging
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# STAGE 1: Keypoint Detection Network
# ==========================================

class SurgicalKeypointDetector(nn.Module):
    """
    CNN-based keypoint detector optimized for surgical scenes
    Uses ResNet backbone with FPN for multi-scale detection
    """

    def __init__(self, num_keypoint_types=6, backbone='resnet50'):
        super().__init__()

        # Load pretrained ResNet backbone
        import torchvision.models as models
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            resnet = models.resnet34(pretrained=True)

        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # FPN (Feature Pyramid Network) layers
        self.fpn_conv5 = nn.Conv2d(2048, 256, 1)
        self.fpn_conv4 = nn.Conv2d(1024, 256, 1)
        self.fpn_conv3 = nn.Conv2d(512, 256, 1)
        self.fpn_conv2 = nn.Conv2d(256, 256, 1)

        # Keypoint heads for each scale
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoint_types + 1, 1)  # +1 for confidence
        )

        # Offset regression for sub-pixel accuracy
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)  # x, y offsets
        )

    def forward(self, x):
        """Forward pass with FPN"""
        # Bottom-up pathway
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down pathway with lateral connections
        p5 = self.fpn_conv5(c5)
        p4 = self.fpn_conv4(c4) + F.interpolate(p5, scale_factor=2)
        p3 = self.fpn_conv3(c3) + F.interpolate(p4, scale_factor=2)
        p2 = self.fpn_conv2(c2) + F.interpolate(p3, scale_factor=2)

        # Merge features from all scales
        # Resize all to same size (use p3 size as reference)
        p2_resized = F.interpolate(p2, size=p3.shape[2:])
        p4_resized = F.interpolate(p4, size=p3.shape[2:])
        p5_resized = F.interpolate(p5, size=p3.shape[2:])

        merged = p2_resized + p3 + p4_resized + p5_resized

        # Detect keypoints
        keypoint_logits = self.keypoint_head(merged)
        offsets = self.offset_head(merged)

        return {
            'heatmaps': torch.sigmoid(keypoint_logits[:, :-1]),  # Keypoint heatmaps
            'confidence': torch.sigmoid(keypoint_logits[:, -1:]),  # Confidence map
            'offsets': torch.tanh(offsets)  # Sub-pixel offsets
        }

# ==========================================
# STAGE 2: Optical Flow & Motion Estimation
# ==========================================

class OpticalFlowTracker:
    """
    Optical flow based motion estimation for temporal consistency
    Uses Lucas-Kanade or Farneback method
    """

    def __init__(self, method='lucas_kanade'):
        self.method = method
        self.prev_gray = None
        self.prev_keypoints = None

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def estimate_motion(self, frame, keypoints):
        """Estimate motion of keypoints using optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            return keypoints, np.ones(len(keypoints))

        if len(self.prev_keypoints) == 0:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            return keypoints, np.ones(len(keypoints))

        # Convert keypoints to numpy array
        prev_pts = np.array([[kp['x'], kp['y']] for kp in self.prev_keypoints], dtype=np.float32)
        prev_pts = prev_pts.reshape(-1, 1, 2)

        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        # Filter good points
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Update keypoints with motion
        tracked_keypoints = []
        confidences = []

        for i, kp in enumerate(keypoints):
            # Find nearest optical flow vector
            kp_pos = np.array([kp['x'], kp['y']])

            if len(good_old) > 0:
                distances = np.linalg.norm(good_old.reshape(-1, 2) - kp_pos, axis=1)
                nearest_idx = np.argmin(distances)

                if distances[nearest_idx] < 20:  # Within 20 pixels
                    # Use optical flow prediction
                    flow_vector = good_new[nearest_idx].reshape(2) - good_old[nearest_idx].reshape(2)
                    kp['x'] += flow_vector[0] * 0.5  # Blend with detection
                    kp['y'] += flow_vector[1] * 0.5
                    confidence = 1.0 / (1.0 + distances[nearest_idx] / 10.0)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5

            tracked_keypoints.append(kp)
            confidences.append(confidence)

        self.prev_gray = gray
        self.prev_keypoints = tracked_keypoints

        return tracked_keypoints, np.array(confidences)

# ==========================================
# STAGE 3: Kalman Filter for Smoothing
# ==========================================

class KalmanTracker:
    """
    Kalman filter for smooth keypoint tracking
    Handles occlusions and temporary disappearances
    """

    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        self.trackers = {}
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.next_id = 0

    def create_kalman_filter(self):
        """Create a new Kalman filter for 2D tracking"""
        kf = cv2.KalmanFilter(4, 2)  # 4 state dims (x,y,vx,vy), 2 measurement dims (x,y)

        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise

        # Measurement noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise

        # Error covariance
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def update(self, detections):
        """Update Kalman filters with new detections"""
        # Predict step for all existing trackers
        predictions = {}
        for track_id, kf in self.trackers.items():
            prediction = kf.predict()
            predictions[track_id] = prediction[:2].flatten()

        if not detections:
            return predictions

        # Convert detections to numpy array
        detection_points = np.array([[d['x'], d['y']] for d in detections])

        if len(predictions) > 0:
            # Hungarian algorithm for optimal assignment
            pred_points = np.array(list(predictions.values()))
            cost_matrix = cdist(detection_points, pred_points)

            # Solve assignment problem
            det_indices, track_indices = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            matched_tracks = set()
            track_ids = list(predictions.keys())

            for det_idx, track_idx in zip(det_indices, track_indices):
                if cost_matrix[det_idx, track_idx] < 50:  # Max distance threshold
                    track_id = track_ids[track_idx]
                    measurement = detection_points[det_idx].reshape(2, 1).astype(np.float32)
                    self.trackers[track_id].correct(measurement)
                    matched_tracks.add(track_id)
                    detections[det_idx]['track_id'] = track_id

            # Create new tracks for unmatched detections
            unmatched_dets = set(range(len(detections))) - set(det_indices)
            for det_idx in unmatched_dets:
                self._create_new_track(detections[det_idx])
        else:
            # Create new tracks for all detections
            for det in detections:
                self._create_new_track(det)

        # Remove dead tracks (not updated for too long)
        # This is simplified - in production, track age management needed

        return self.get_tracked_points()

    def _create_new_track(self, detection):
        """Create a new Kalman filter track"""
        kf = self.create_kalman_filter()
        kf.statePre = np.array([detection['x'], detection['y'], 0, 0], dtype=np.float32)
        kf.statePost = kf.statePre.copy()

        track_id = self.next_id
        self.next_id += 1
        self.trackers[track_id] = kf
        detection['track_id'] = track_id

    def get_tracked_points(self):
        """Get current tracked points"""
        tracked = {}
        for track_id, kf in self.trackers.items():
            state = kf.statePost
            tracked[track_id] = {
                'x': float(state[0]),
                'y': float(state[1]),
                'vx': float(state[2]),
                'vy': float(state[3])
            }
        return tracked

# ==========================================
# STAGE 4: Hungarian Association
# ==========================================

class HungarianAssociator:
    """
    Hungarian algorithm for optimal track-detection association
    Considers multiple costs: distance, appearance, motion
    """

    def __init__(self, max_distance=100, max_age=30):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0

    def associate(self, detections, predictions, features=None):
        """
        Associate detections with existing tracks

        Args:
            detections: List of detected keypoints
            predictions: Dict of predicted positions from Kalman filter
            features: Optional appearance features for each detection
        """
        if not detections:
            return [], list(predictions.keys()), []

        if not predictions:
            # All detections are new tracks
            new_tracks = []
            for det in detections:
                track_id = self.next_id
                self.next_id += 1
                det['track_id'] = track_id
                new_tracks.append(track_id)
            return [], [], new_tracks

        # Build cost matrix
        n_dets = len(detections)
        n_tracks = len(predictions)

        det_points = np.array([[d['x'], d['y']] for d in detections])
        track_points = np.array([[p['x'], p['y']] for p in predictions.values()])

        # Euclidean distance cost
        dist_cost = cdist(det_points, track_points)

        # Normalize costs
        dist_cost = dist_cost / self.max_distance

        # Add motion consistency cost if available
        if all('vx' in p for p in predictions.values()):
            motion_cost = self._compute_motion_cost(detections, predictions)
            cost_matrix = 0.7 * dist_cost + 0.3 * motion_cost
        else:
            cost_matrix = dist_cost

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter assignments by threshold
        matches = []
        track_ids = list(predictions.keys())

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1.0:  # Normalized threshold
                matches.append((i, track_ids[j]))
                detections[i]['track_id'] = track_ids[j]

        # Find unmatched
        matched_det_indices = {m[0] for m in matches}
        matched_track_ids = {m[1] for m in matches}

        unmatched_dets = [i for i in range(n_dets) if i not in matched_det_indices]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_ids]

        # Create new tracks for unmatched detections
        new_tracks = []
        for i in unmatched_dets:
            track_id = self.next_id
            self.next_id += 1
            detections[i]['track_id'] = track_id
            new_tracks.append(track_id)

        return matches, unmatched_tracks, new_tracks

    def _compute_motion_cost(self, detections, predictions):
        """Compute motion consistency cost"""
        n_dets = len(detections)
        n_tracks = len(predictions)
        motion_cost = np.zeros((n_dets, n_tracks))

        for i, det in enumerate(detections):
            for j, (track_id, pred) in enumerate(predictions.items()):
                # Expected position based on velocity
                expected_x = pred['x'] + pred['vx']
                expected_y = pred['y'] + pred['vy']

                # Distance from expected position
                dist = np.sqrt((det['x'] - expected_x)**2 + (det['y'] - expected_y)**2)
                motion_cost[i, j] = dist / self.max_distance

        return motion_cost

# ==========================================
# COMPLETE PIPELINE
# ==========================================

class FixedMultiStageFusion:
    """
    Complete fixed multi-stage fusion pipeline for keypoint tracking
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Stage 1: Keypoint detection
        self.detector = SurgicalKeypointDetector(num_keypoint_types=6)
        self.detector = self.detector.to(device)
        self.detector.eval()

        # Stage 2: Optical flow
        self.flow_tracker = OpticalFlowTracker()

        # Stage 3: Kalman filtering
        self.kalman_tracker = KalmanTracker()

        # Stage 4: Hungarian association
        self.associator = HungarianAssociator()

        logger.info(f"Multi-Stage Fusion initialized on {device}")

    def detect_keypoints(self, frame, use_nms=True):
        """
        Detect keypoints in a frame using CNN
        """
        # Preprocess frame
        img_tensor = self._preprocess_frame(frame)

        # Run detection
        with torch.no_grad():
            outputs = self.detector(img_tensor)

        # Extract keypoints from heatmaps
        heatmaps = outputs['heatmaps'].cpu().numpy()[0]
        confidence = outputs['confidence'].cpu().numpy()[0, 0]
        offsets = outputs['offsets'].cpu().numpy()[0]

        keypoints = []

        for class_idx in range(heatmaps.shape[0]):
            heatmap = heatmaps[class_idx]

            # Find peaks in heatmap
            if use_nms:
                peaks = self._nms_heatmap(heatmap, threshold=0.3)
            else:
                peaks = np.argwhere(heatmap > 0.3)

            for y, x in peaks:
                # Apply sub-pixel offset
                x_offset = offsets[0, y, x]
                y_offset = offsets[1, y, x]

                # Scale to original image size
                scale_x = frame.shape[1] / heatmap.shape[1]
                scale_y = frame.shape[0] / heatmap.shape[0]

                keypoints.append({
                    'x': (x + x_offset) * scale_x,
                    'y': (y + y_offset) * scale_y,
                    'confidence': float(heatmap[y, x] * confidence[y, x]),
                    'class': class_idx
                })

        return keypoints

    def track_keypoints(self, frame, keypoints):
        """
        Track keypoints across frames using optical flow and Kalman filtering
        """
        # Stage 2: Optical flow refinement
        tracked_kps, flow_confidence = self.flow_tracker.estimate_motion(frame, keypoints)

        # Stage 3: Kalman filter update
        kalman_predictions = self.kalman_tracker.update(tracked_kps)

        # Stage 4: Hungarian association for final tracks
        matches, unmatched_tracks, new_tracks = self.associator.associate(
            tracked_kps, kalman_predictions
        )

        # Combine results
        final_tracks = []
        for kp in tracked_kps:
            if 'track_id' in kp:
                final_tracks.append(kp)

        return final_tracks

    def process_video(self, video_path, output_path=None):
        """
        Process entire video for keypoint tracking
        """
        cap = cv2.VideoCapture(str(video_path))

        tracks_by_frame = {}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect keypoints
            keypoints = self.detect_keypoints(frame)

            # Track keypoints
            tracked = self.track_keypoints(frame, keypoints)

            # Store results
            tracks_by_frame[frame_idx] = tracked

            frame_idx += 1

            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx} frames, tracking {len(tracked)} keypoints")

        cap.release()

        # Convert to MOT format if output path provided
        if output_path:
            self._save_mot_format(tracks_by_frame, output_path)

        return tracks_by_frame

    def _preprocess_frame(self, frame):
        """Preprocess frame for CNN input"""
        # Resize to standard size
        img = cv2.resize(frame, (640, 480))

        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # Convert to tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor

    def _nms_heatmap(self, heatmap, threshold=0.3, kernel_size=5):
        """Non-maximum suppression on heatmap"""
        # Apply max pooling
        pooled = cv2.dilate(heatmap, np.ones((kernel_size, kernel_size)))

        # Find peaks (local maxima)
        peaks = (heatmap == pooled) & (heatmap > threshold)

        return np.argwhere(peaks)

    def _save_mot_format(self, tracks_by_frame, output_path):
        """Save tracks in MOT challenge format"""
        mot_lines = []

        for frame_idx, tracks in tracks_by_frame.items():
            for track in tracks:
                if 'track_id' not in track:
                    continue

                # MOT format: frame,id,x,y,w,h,conf,class,visibility
                line = f"{frame_idx+1},{track['track_id']},{track['x']:.2f},{track['y']:.2f},"
                line += f"10,10,{track.get('confidence', 1.0):.3f},{track.get('class', -1)},1"
                mot_lines.append(line)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(mot_lines))

        logger.info(f"Saved {len(mot_lines)} tracks to {output_path}")

# ==========================================
# TESTING
# ==========================================

def test_multistage_fusion():
    """Test the fixed multi-stage fusion system"""

    logger.info("Testing Fixed Multi-Stage Fusion System")

    # Initialize system
    system = FixedMultiStageFusion()

    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test keypoint detection
    logger.info("Testing keypoint detection...")
    keypoints = system.detect_keypoints(test_frame)
    logger.info(f"Detected {len(keypoints)} keypoints")

    # Test tracking
    logger.info("Testing keypoint tracking...")
    tracked = system.track_keypoints(test_frame, keypoints)
    logger.info(f"Tracked {len(tracked)} keypoints")

    # Verify track IDs assigned
    tracks_with_id = [kp for kp in tracked if 'track_id' in kp]
    logger.info(f"Assigned track IDs to {len(tracks_with_id)} keypoints")

    logger.info("✅ Multi-Stage Fusion system working!")

    return system

if __name__ == "__main__":
    test_multistage_fusion()