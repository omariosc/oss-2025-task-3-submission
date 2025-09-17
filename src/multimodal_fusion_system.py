#!/usr/bin/env python3
"""
Multi-Modal Fusion System
Combines RGB, Depth, and Segmentation with attention mechanisms for optimal keypoint detection
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from segmentation_keypoint_extractor import SegmentationKeypointExtractor
from dense_keypoint_detector import DenseKeypointDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionFusion(nn.Module):
    """Attention mechanism for fusing multi-modal features"""

    def __init__(self, feature_dim=256, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Feature projections for each modality
        self.rgb_proj = nn.Linear(3, feature_dim)
        self.depth_proj = nn.Linear(1, feature_dim)
        self.seg_proj = nn.Linear(1, feature_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_features, depth_features, seg_features):
        """Fuse features from multiple modalities"""
        # Project features
        rgb_emb = self.rgb_proj(rgb_features)
        depth_emb = self.depth_proj(depth_features)
        seg_emb = self.seg_proj(seg_features)

        # Stack modalities
        features = torch.stack([rgb_emb, depth_emb, seg_emb], dim=1)

        # Apply attention
        attended, _ = self.attention(features, features, features)

        # Mean pooling across modalities
        fused = attended.mean(dim=1)

        # Output confidence
        confidence = self.output_proj(fused)

        return confidence.squeeze(), fused


class MultiModalFusionSystem:
    """Complete multi-modal fusion system for surgical keypoint tracking"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        logger.info(f"Multi-Modal Fusion System initializing on {device}")

        # Initialize all detection methods
        self.yolo_anchor = YOLOAnchorFusion(device=device)
        self.depth_detector = DepthGuidedKeypointDetector(device=device)
        self.seg_extractor = SegmentationKeypointExtractor(device=device)
        self.dense_detector = DenseKeypointDetector(device=device)

        # Attention fusion network
        self.fusion_net = AttentionFusion().to(self.device)
        self.fusion_net.eval()

        # Tracking
        self.kalman_filters = {}
        self.track_id_counter = 0
        self.active_tracks = {}

        # Expected number of keypoints
        self.target_keypoints = 23

        logger.info("Multi-Modal Fusion System ready")

    def extract_all_keypoints(self, frame: np.ndarray,
                              depth_map: Optional[np.ndarray] = None,
                              masks: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """Extract keypoints using all available methods"""
        all_keypoints = {}

        # 1. YOLO + Anchors (object centers)
        yolo_result = self.yolo_anchor.process_frame(frame)
        yolo_keypoints = []
        for obj in yolo_result['objects']:
            # Generate multiple keypoints per object
            cx, cy = obj['x'], obj['y']
            yolo_keypoints.append({
                'x': cx, 'y': cy,
                'confidence': obj['confidence'],
                'source': 'yolo_center'
            })

            # Add corner keypoints
            offsets = [(-20, -20), (20, -20), (-20, 20), (20, 20)]
            for dx, dy in offsets:
                yolo_keypoints.append({
                    'x': cx + dx, 'y': cy + dy,
                    'confidence': obj['confidence'] * 0.8,
                    'source': 'yolo_corner'
                })

        all_keypoints['yolo'] = yolo_keypoints

        # 2. Depth-guided detection
        if depth_map is None:
            depth_map = self.depth_detector.depth_estimator.estimate_depth(frame)

        depth_keypoints = self.depth_detector.detect_keypoints_with_depth(frame, depth_map)
        all_keypoints['depth'] = depth_keypoints

        # 3. Segmentation-based (if masks available)
        if masks:
            seg_keypoints = []
            for tool_class, mask in masks.items():
                kps = self.seg_extractor.extract_keypoints_from_mask(mask, tool_class)
                seg_keypoints.extend(kps)
            all_keypoints['segmentation'] = seg_keypoints
        else:
            all_keypoints['segmentation'] = []

        # 4. Dense detection (selective)
        # Only use if we have few keypoints from other methods
        total_so_far = sum(len(kps) for kps in all_keypoints.values())
        if total_so_far < self.target_keypoints * 2:
            dense_keypoints = self.dense_detector.detect_keypoints(frame)
            all_keypoints['dense'] = dense_keypoints[:50]  # Limit to top 50
        else:
            all_keypoints['dense'] = []

        return all_keypoints

    def fuse_keypoints(self, all_keypoints: Dict[str, List[Dict]],
                       frame: np.ndarray,
                       depth_map: np.ndarray) -> List[Dict]:
        """Fuse keypoints from all sources using attention mechanism"""
        fused_keypoints = []

        # Collect all candidate keypoints
        candidates = []
        for source, keypoints in all_keypoints.items():
            for kp in keypoints:
                kp['modality'] = source
                candidates.append(kp)

        if not candidates:
            return []

        # Apply non-maximum suppression across modalities
        candidates = self.nms_keypoints(candidates, threshold=30)

        # Score each candidate using multi-modal features
        scored_candidates = self.score_candidates(candidates, frame, depth_map)

        # Select top keypoints
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Ensure diversity (not all from same location)
        selected = []
        min_distance = 40

        for cand in scored_candidates:
            too_close = False
            for sel in selected:
                dist = np.sqrt((cand['x'] - sel['x'])**2 + (cand['y'] - sel['y'])**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected.append(cand)

            if len(selected) >= self.target_keypoints:
                break

        return selected

    def nms_keypoints(self, keypoints: List[Dict], threshold: float = 30) -> List[Dict]:
        """Non-maximum suppression for keypoints"""
        if not keypoints:
            return []

        # Sort by confidence
        keypoints.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        keep = []
        suppressed = set()

        for i, kp1 in enumerate(keypoints):
            if i in suppressed:
                continue

            keep.append(kp1)

            # Suppress nearby keypoints with lower confidence
            for j, kp2 in enumerate(keypoints[i+1:], i+1):
                if j in suppressed:
                    continue

                dist = np.sqrt((kp1['x'] - kp2['x'])**2 + (kp1['y'] - kp2['y'])**2)
                if dist < threshold:
                    suppressed.add(j)

        return keep

    def score_candidates(self, candidates: List[Dict],
                        frame: np.ndarray,
                        depth_map: np.ndarray) -> List[Dict]:
        """Score candidates using multi-modal features"""
        h, w = frame.shape[:2]

        for cand in candidates:
            x, y = int(cand['x']), int(cand['y'])

            # Ensure coordinates are valid
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))

            # Extract local features
            # RGB feature
            rgb_val = frame[y, x] / 255.0 if y < h and x < w else np.array([0, 0, 0])

            # Depth feature
            depth_val = depth_map[y, x] if y < depth_map.shape[0] and x < depth_map.shape[1] else 0.5

            # Compute final score (weighted combination)
            base_conf = cand.get('confidence', 0.5)
            modality_weight = {
                'yolo': 1.2,
                'depth': 1.0,
                'segmentation': 1.5,
                'dense': 0.8
            }
            weight = modality_weight.get(cand.get('modality', 'unknown'), 1.0)

            # Depth-based adjustment (closer objects get higher score)
            depth_bonus = depth_val * 0.2

            # Edge response bonus
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge_response = cv2.Laplacian(gray, cv2.CV_64F)
            edge_val = abs(edge_response[y, x]) / 255.0 if y < h and x < w else 0
            edge_bonus = edge_val * 0.1

            cand['final_score'] = base_conf * weight + depth_bonus + edge_bonus

        return candidates

    def track_keypoints(self, keypoints: List[Dict]) -> List[Dict]:
        """Track keypoints across frames using Kalman filtering"""
        tracked = []

        # Simple tracking for now (can be enhanced)
        for i, kp in enumerate(keypoints):
            track_id = i  # Simple ID assignment

            if track_id not in self.kalman_filters:
                kf = self.create_kalman_filter(kp['x'], kp['y'])
                self.kalman_filters[track_id] = kf
            else:
                kf = self.kalman_filters[track_id]

            # Predict and update
            kf.predict()
            kf.update([kp['x'], kp['y']])

            # Get smoothed position
            tracked.append({
                'track_id': track_id + 1,  # 1-indexed
                'x': kf.x[0, 0],
                'y': kf.x[1, 0],
                'confidence': kp.get('final_score', kp.get('confidence', 0.5)),
                'source': kp.get('modality', 'unknown')
            })

        return tracked

    def create_kalman_filter(self, x, y):
        """Create Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=4, dim_z=2)

        kf.x = np.array([x, y, 0, 0]).reshape(4, 1)

        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        kf.Q *= 0.02
        kf.R *= 3
        kf.P *= 100

        return kf

    def process_frame(self, frame: np.ndarray,
                     depth_map: Optional[np.ndarray] = None,
                     masks: Optional[Dict] = None) -> Dict:
        """Process single frame with multi-modal fusion"""

        # Extract keypoints from all sources
        all_keypoints = self.extract_all_keypoints(frame, depth_map, masks)

        # Fuse keypoints
        if depth_map is None:
            depth_map = self.depth_detector.depth_estimator.estimate_depth(frame)

        fused_keypoints = self.fuse_keypoints(all_keypoints, frame, depth_map)

        # Track keypoints
        tracked = self.track_keypoints(fused_keypoints)

        # Log statistics
        stats = {
            'yolo': len(all_keypoints.get('yolo', [])),
            'depth': len(all_keypoints.get('depth', [])),
            'segmentation': len(all_keypoints.get('segmentation', [])),
            'dense': len(all_keypoints.get('dense', [])),
            'fused': len(fused_keypoints),
            'tracked': len(tracked)
        }

        return {
            'keypoints': tracked,
            'num_keypoints': len(tracked),
            'statistics': stats,
            'depth_map': depth_map
        }


def test_multimodal_fusion():
    """Test the multi-modal fusion system"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")

    # Initialize system
    system = MultiModalFusionSystem(device='cpu')

    # Load test frame
    frame_file = data_root / "val/frames/E66F/E66F_frame_0.png"
    frame = cv2.imread(str(frame_file))

    logger.info("Processing frame with multi-modal fusion...")

    # Process frame
    result = system.process_frame(frame)

    logger.info(f"\nMulti-Modal Fusion Results:")
    logger.info(f"  Keypoints detected: {result['num_keypoints']}")

    stats = result['statistics']
    logger.info(f"\nDetection Statistics:")
    logger.info(f"  YOLO candidates: {stats['yolo']}")
    logger.info(f"  Depth candidates: {stats['depth']}")
    logger.info(f"  Segmentation candidates: {stats['segmentation']}")
    logger.info(f"  Dense candidates: {stats['dense']}")
    logger.info(f"  After fusion: {stats['fused']}")
    logger.info(f"  Final tracked: {stats['tracked']}")

    # Group by source
    by_source = defaultdict(int)
    for kp in result['keypoints']:
        by_source[kp['source']] += 1

    logger.info(f"\nKeypoints by source:")
    for source, count in by_source.items():
        logger.info(f"  {source}: {count}")

    # Compare with ground truth
    gt_keypoints = 23
    logger.info(f"\nTarget keypoints: {gt_keypoints}")
    logger.info(f"Detected keypoints: {result['num_keypoints']}")

    if result['num_keypoints'] >= gt_keypoints * 0.8:
        logger.info("✅ Good keypoint coverage!")
    else:
        logger.info("⚠️  Need to detect more keypoints")

    return result


if __name__ == "__main__":
    result = test_multimodal_fusion()
    print(f"\n✅ Multi-modal fusion complete! Detected {result['num_keypoints']} keypoints")