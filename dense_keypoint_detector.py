#!/usr/bin/env python3
"""
Dense Keypoint Detection with Learned Filtering
Detects thousands of candidate keypoints and filters to find true surgical keypoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeypointFilterNet(nn.Module):
    """Neural network to classify candidate keypoints as true/false"""

    def __init__(self, input_channels=3, patch_size=32):
        super().__init__()
        self.patch_size = patch_size

        # CNN for patch feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

        # Calculate feature size after convolutions
        feat_size = (patch_size // 8) * (patch_size // 8) * 128

        # Classifier
        self.fc1 = nn.Linear(feat_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return torch.sigmoid(x)


class DenseKeypointDetector:
    """Detect dense grid of keypoints and filter to find true surgical keypoints"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        logger.info(f"Dense Keypoint Detector on {device}")

        # Grid parameters
        self.grid_spacing = 30  # Pixels between grid points (reduced for speed)
        self.multi_scale = [1.0]  # Single scale for speed

        # Initialize filter network
        self.filter_net = KeypointFilterNet().to(self.device)
        self.filter_net.eval()

        # Traditional feature extractor for initial filtering
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=5000)

        # Random forest for fast initial filtering
        self.rf_classifier = None

        # Expected number of keypoints
        self.expected_keypoints = 23

    def generate_dense_candidates(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Generate dense grid of candidate keypoints"""
        h, w = frame.shape[:2]
        candidates = []

        # 1. Uniform grid sampling
        for scale in self.multi_scale:
            scaled_spacing = int(self.grid_spacing * scale)
            for y in range(scaled_spacing, h - scaled_spacing, scaled_spacing):
                for x in range(scaled_spacing, w - scaled_spacing, scaled_spacing):
                    candidates.append((x, y))

        # 2. Interest point detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # SIFT keypoints
        sift_kps = self.sift.detect(gray, None)
        for kp in sift_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            candidates.append((x, y))

        # ORB keypoints
        orb_kps = self.orb.detect(gray, None)
        for kp in orb_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            candidates.append((x, y))

        # 3. Edge-based sampling
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))

        # Sample from edge points
        if len(edge_points) > 0:
            n_samples = min(200, len(edge_points))  # Reduced for speed
            indices = np.random.choice(len(edge_points), n_samples, replace=False)
            for idx in indices:
                y, x = edge_points[idx]
                candidates.append((x, y))

        # 4. Color-based sampling (surgical tools)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue gloves
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Metallic tools
        gray_scaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, metallic_mask = cv2.threshold(gray_scaled, 180, 255, cv2.THRESH_BINARY)

        # Sample from colored regions
        for mask in [blue_mask, metallic_mask]:
            mask_points = np.column_stack(np.where(mask > 0))
            if len(mask_points) > 0:
                n_samples = min(100, len(mask_points))  # Reduced for speed
                indices = np.random.choice(len(mask_points), n_samples, replace=False)
                for idx in indices:
                    y, x = mask_points[idx]
                    candidates.append((x, y))

        # Remove duplicates
        candidates = list(set(candidates))

        logger.info(f"Generated {len(candidates)} candidate keypoints")
        return candidates

    def extract_patch_features(self, frame: np.ndarray, x: int, y: int,
                              patch_size: int = 32) -> np.ndarray:
        """Extract features from a patch around a candidate point"""
        h, w = frame.shape[:2]
        half_patch = patch_size // 2

        # Get patch boundaries
        x1 = max(0, x - half_patch)
        x2 = min(w, x + half_patch)
        y1 = max(0, y - half_patch)
        y2 = min(h, y + half_patch)

        # Extract patch
        patch = frame[y1:y2, x1:x2]

        # Pad if necessary
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        # Compute features
        features = []

        # Color features
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        features.extend(hsv_patch.mean(axis=(0, 1)))  # Mean HSV
        features.extend(hsv_patch.std(axis=(0, 1)))   # Std HSV

        # Texture features (using gray)
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Gradients
        dx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
        features.append(np.abs(dx).mean())
        features.append(np.abs(dy).mean())

        # Laplacian (for sharpness)
        laplacian = cv2.Laplacian(gray_patch, cv2.CV_64F)
        features.append(np.abs(laplacian).mean())

        # Corner response
        corners = cv2.cornerHarris(gray_patch, 2, 3, 0.04)
        features.append(corners.max())

        # Local contrast
        features.append(gray_patch.std())

        return np.array(features)

    def filter_candidates_neural(self, frame: np.ndarray,
                                 candidates: List[Tuple[int, int]]) -> List[Dict]:
        """Filter candidates using neural network"""
        if len(candidates) == 0:
            return []

        # Prepare patches for batch processing
        patches = []
        patch_size = 32

        for x, y in candidates:
            patch = self.extract_patch_for_nn(frame, x, y, patch_size)
            patches.append(patch)

        # Batch process
        batch_size = 256
        filtered = []

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_tensor = torch.stack(batch).to(self.device)

                # Get predictions
                probs = self.filter_net(batch_tensor).squeeze()

                # Add high-confidence candidates
                for j, prob in enumerate(probs):
                    if prob > 0.3:  # Lower threshold to get more candidates
                        idx = i + j
                        x, y = candidates[idx]
                        filtered.append({
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(prob),
                            'source': 'neural_filter'
                        })

        return filtered

    def extract_patch_for_nn(self, frame: np.ndarray, x: int, y: int,
                             patch_size: int) -> torch.Tensor:
        """Extract and preprocess patch for neural network"""
        h, w = frame.shape[:2]
        half_patch = patch_size // 2

        # Get patch
        x1 = max(0, x - half_patch)
        x2 = min(w, x + half_patch)
        y1 = max(0, y - half_patch)
        y2 = min(h, y + half_patch)

        patch = frame[y1:y2, x1:x2]

        # Resize if needed
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        # Convert to tensor and normalize
        patch_tensor = torch.from_numpy(patch).float() / 255.0
        patch_tensor = patch_tensor.permute(2, 0, 1)  # HWC to CHW

        return patch_tensor

    def filter_candidates_traditional(self, frame: np.ndarray,
                                     candidates: List[Tuple[int, int]]) -> List[Dict]:
        """Filter candidates using traditional CV methods"""
        filtered = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for x, y in candidates:
            score = 0.0

            # Check if on an edge
            edges = cv2.Canny(gray, 50, 150)
            if edges[y, x] > 0:
                score += 0.3

            # Check color (blue for gloves, metallic for tools)
            h, s, v = hsv[y, x]
            if 100 <= h <= 130 and s > 50:  # Blue
                score += 0.4
            if v > 180 and s < 50:  # Metallic
                score += 0.4

            # Check local gradient strength
            patch_size = 5
            y1 = max(0, y - patch_size)
            y2 = min(gray.shape[0], y + patch_size)
            x1 = max(0, x - patch_size)
            x2 = min(gray.shape[1], x + patch_size)

            local_patch = gray[y1:y2, x1:x2]
            if local_patch.size > 0:
                gradient = np.std(local_patch)
                if gradient > 20:
                    score += 0.3

            if score > 0.3:
                filtered.append({
                    'x': float(x),
                    'y': float(y),
                    'confidence': score,
                    'source': 'traditional_filter'
                })

        return filtered

    def apply_spatial_constraints(self, keypoints: List[Dict]) -> List[Dict]:
        """Apply spatial constraints to reduce false positives"""
        if len(keypoints) <= self.expected_keypoints:
            return keypoints

        # Sort by confidence
        keypoints.sort(key=lambda x: x['confidence'], reverse=True)

        # Non-maximum suppression
        filtered = []
        min_distance = 30  # Minimum distance between keypoints

        for kp in keypoints:
            too_close = False
            for fkp in filtered:
                dist = np.sqrt((kp['x'] - fkp['x'])**2 + (kp['y'] - fkp['y'])**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                filtered.append(kp)

            if len(filtered) >= self.expected_keypoints * 2:
                break

        return filtered

    def hungarian_matching(self, predictions: List[Dict],
                          ground_truth: List[Dict]) -> Tuple[List[Tuple], float]:
        """Hungarian algorithm for optimal matching"""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return [], 0.0

        # Build cost matrix
        cost_matrix = np.zeros((len(predictions), len(ground_truth)))

        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                dist = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
                cost_matrix[i, j] = dist

        # Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        # Get matches within threshold
        matches = []
        threshold = 100
        total_cost = 0

        for pi, gi in zip(pred_indices, gt_indices):
            if cost_matrix[pi, gi] < threshold:
                matches.append((pi, gi))
                total_cost += cost_matrix[pi, gi]

        avg_cost = total_cost / len(matches) if matches else float('inf')

        return matches, avg_cost

    def detect_keypoints(self, frame: np.ndarray) -> List[Dict]:
        """Main detection pipeline"""
        # 1. Generate dense candidates
        candidates = self.generate_dense_candidates(frame)

        # 2. Filter candidates (use traditional for speed in this demo)
        filtered = self.filter_candidates_traditional(frame, candidates)

        # Skip neural filtering for speed (commented out for now)
        # if len(filtered) > 100 and hasattr(self, 'filter_net'):
        #     # Take top candidates for neural filtering
        #     top_candidates = [(int(kp['x']), int(kp['y'])) for kp in filtered[:500]]
        #     neural_filtered = self.filter_candidates_neural(frame, top_candidates)
        #     if neural_filtered:
        #         filtered = neural_filtered

        # 3. Apply spatial constraints
        filtered = self.apply_spatial_constraints(filtered)

        # 4. Ensure we have reasonable number of keypoints
        if len(filtered) < self.expected_keypoints:
            # Add some high-gradient points
            logger.warning(f"Only {len(filtered)} keypoints found, adding more")
            filtered = self.add_fallback_keypoints(frame, filtered)

        # Limit to reasonable number
        filtered = filtered[:self.expected_keypoints * 2]

        return filtered

    def add_fallback_keypoints(self, frame: np.ndarray,
                               existing: List[Dict]) -> List[Dict]:
        """Add fallback keypoints if not enough detected"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use corner detection
        corners = cv2.goodFeaturesToTrack(gray,
                                         maxCorners=self.expected_keypoints * 2,
                                         qualityLevel=0.01,
                                         minDistance=30)

        if corners is not None:
            for corner in corners:
                x, y = corner[0]
                existing.append({
                    'x': float(x),
                    'y': float(y),
                    'confidence': 0.2,
                    'source': 'fallback_corner'
                })

        return existing

def test_dense_detector():
    """Test the dense keypoint detector"""
    from pathlib import Path

    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    detector = DenseKeypointDetector(device='cpu')

    # Load test frame
    frame_file = data_root / "val/frames/E66F/E66F_frame_0.png"
    frame = cv2.imread(str(frame_file))

    # Detect keypoints
    keypoints = detector.detect_keypoints(frame)

    logger.info(f"\nDense Detection Results:")
    logger.info(f"  Keypoints detected: {len(keypoints)}")

    # Group by source
    by_source = defaultdict(list)
    for kp in keypoints:
        by_source[kp['source']].append(kp)

    for source, kps in by_source.items():
        logger.info(f"  {source}: {len(kps)} keypoints")

    # Load ground truth for comparison
    gt_file = data_root / "val/mot/E66F.txt"
    gt_keypoints = []

    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == '0':  # Frame 0
                # Extract keypoints
                i = 7
                while i + 2 < len(parts):
                    try:
                        x = float(parts[i])
                        y = float(parts[i+1])
                        v = int(parts[i+2])
                        if v > 0:
                            gt_keypoints.append({'x': x, 'y': y})
                        i += 3
                    except:
                        break

    logger.info(f"\nGround Truth: {len(gt_keypoints)} keypoints")

    # Match predictions to ground truth
    matches, avg_dist = detector.hungarian_matching(keypoints, gt_keypoints)

    logger.info(f"\nMatching Results:")
    logger.info(f"  Matched: {len(matches)}/{len(gt_keypoints)}")
    logger.info(f"  Average distance: {avg_dist:.1f} pixels")
    logger.info(f"  Recall: {len(matches)/len(gt_keypoints):.3f}")
    logger.info(f"  Precision: {len(matches)/len(keypoints):.3f}")

    return keypoints

if __name__ == "__main__":
    keypoints = test_dense_detector()
    print(f"\n✅ Dense detector ready! Detected {len(keypoints)} keypoints")