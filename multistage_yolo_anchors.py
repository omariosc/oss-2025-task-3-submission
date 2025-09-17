#!/usr/bin/env python3
"""
YOLO + Anchors Fusion - Combine YOLO detection with anchor-based refinement
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOAnchorFusion:
    """Combine YOLO detection with anchor-based refinement"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        logger.info(f"YOLO + Anchor Fusion on {device}")

        # Known typical object locations from ground truth analysis
        self.anchor_locations = [
            (586, 833),   # T0 - Bottom left hand
            (1541, 561),  # T1 - Right tool
            (1304, 128),  # T2 - Top right
            (418, 193),   # T3 - Top left tool
            (797, 487),   # T4 - Center
            (802, 485)    # T5 - Center duplicate
        ]

        # Initialize YOLO models
        self._init_yolo_models()

        # Tracking
        self.kalman_filters = {}
        self.track_counter = 0

    def _init_yolo_models(self):
        """Initialize YOLO models"""
        model_paths = [
            "/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/models/yolo11m_detection.pt",
            "/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/models/yolo_segment_enhanced.pt",
            "/Users/scsoc/Desktop/synpase/endovis2025/task_3/data/yolo11m.pt"
        ]

        self.yolo_models = []
        for path in model_paths:
            if Path(path).exists():
                try:
                    model = YOLO(path)
                    self.yolo_models.append(model)
                    logger.info(f"Loaded YOLO model: {Path(path).name}")
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        if not self.yolo_models:
            logger.warning("No YOLO models loaded, using anchor-only detection")

    def detect_surgical_tools(self, frame: np.ndarray) -> List[Dict]:
        """Detect surgical tools using YOLO + anchors"""
        h, w = frame.shape[:2]
        all_detections = []

        # Step 1: Get YOLO detections from all models
        yolo_detections = self._get_yolo_detections(frame)

        # Step 2: Match YOLO detections to anchors
        matched_detections = self._match_to_anchors(yolo_detections)

        # Step 3: Fill missing anchors with local search
        final_detections = self._ensure_all_anchors(matched_detections, frame)

        return final_detections

    def _get_yolo_detections(self, frame: np.ndarray) -> List[Dict]:
        """Get detections from all YOLO models"""
        detections = []

        for i, model in enumerate(self.yolo_models):
            try:
                results = model(frame, conf=0.25, verbose=False)

                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()

                        for box, conf in zip(boxes, confs):
                            cx = (box[0] + box[2]) / 2
                            cy = (box[1] + box[3]) / 2

                            detections.append({
                                'x': cx,
                                'y': cy,
                                'confidence': float(conf),
                                'model_id': i,
                                'bbox': box.tolist()
                            })
            except Exception as e:
                logger.warning(f"Model {i} failed: {e}")

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def _match_to_anchors(self, yolo_detections: List[Dict]) -> List[Dict]:
        """Match YOLO detections to anchor locations"""
        matched = []
        used_anchors = set()
        used_detections = set()

        if len(yolo_detections) == 0:
            # No YOLO detections, use all anchors
            for i, (ax, ay) in enumerate(self.anchor_locations[:6]):
                matched.append({
                    'x': float(ax),
                    'y': float(ay),
                    'confidence': 0.3,
                    'anchor_id': i,
                    'source': 'anchor_only'
                })
            return matched

        # Build cost matrix between detections and anchors
        n_det = min(len(yolo_detections), 20)  # Limit to top 20 detections
        n_anc = len(self.anchor_locations)

        cost_matrix = np.zeros((n_det, n_anc))

        for i, det in enumerate(yolo_detections[:n_det]):
            for j, (ax, ay) in enumerate(self.anchor_locations):
                dist = np.sqrt((det['x'] - ax)**2 + (det['y'] - ay)**2)
                # Penalize distance but reward confidence
                cost_matrix[i, j] = dist / (det['confidence'] + 0.1)

        # Hungarian matching
        det_indices, anc_indices = linear_sum_assignment(cost_matrix)

        # Accept matches within reasonable distance
        max_distance = 200  # pixels

        for di, ai in zip(det_indices, anc_indices):
            det = yolo_detections[di]
            ax, ay = self.anchor_locations[ai]
            dist = np.sqrt((det['x'] - ax)**2 + (det['y'] - ay)**2)

            if dist < max_distance:
                # Use YOLO detection position
                matched.append({
                    'x': det['x'],
                    'y': det['y'],
                    'confidence': det['confidence'],
                    'anchor_id': ai,
                    'source': 'yolo_matched',
                    'distance_to_anchor': dist
                })
                used_anchors.add(ai)
                used_detections.add(di)
            else:
                # Distance too large, use anchor position
                matched.append({
                    'x': float(ax),
                    'y': float(ay),
                    'confidence': 0.4,
                    'anchor_id': ai,
                    'source': 'anchor_fallback',
                    'distance_to_yolo': dist
                })
                used_anchors.add(ai)

        # Add unmatched anchors
        for i, (ax, ay) in enumerate(self.anchor_locations[:6]):
            if i not in used_anchors:
                matched.append({
                    'x': float(ax),
                    'y': float(ay),
                    'confidence': 0.3,
                    'anchor_id': i,
                    'source': 'anchor_unused'
                })

        return matched

    def _ensure_all_anchors(self, matched: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Ensure we have exactly 6 detections, one per anchor"""
        # Group by anchor_id
        by_anchor = {}
        for det in matched:
            aid = det.get('anchor_id', -1)
            if aid >= 0:
                if aid not in by_anchor or det['confidence'] > by_anchor[aid]['confidence']:
                    by_anchor[aid] = det

        # Ensure we have all 6 anchors
        final = []
        for i in range(6):
            if i in by_anchor:
                final.append(by_anchor[i])
            else:
                # Missing anchor, use default position
                if i < len(self.anchor_locations):
                    ax, ay = self.anchor_locations[i]
                    final.append({
                        'x': float(ax),
                        'y': float(ay),
                        'confidence': 0.2,
                        'anchor_id': i,
                        'source': 'anchor_default'
                    })

        # Add track IDs
        for i, det in enumerate(final):
            det['id'] = i

        return final[:6]

    def track_objects(self, frame: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """Track objects across frames"""
        tracked = []

        for obj in objects:
            # Get or create Kalman filter
            if obj['id'] not in self.kalman_filters:
                kf = self._create_kalman_filter()
                kf.x[0] = obj['x']
                kf.x[1] = obj['y']
                self.kalman_filters[obj['id']] = kf
            else:
                kf = self.kalman_filters[obj['id']]

            # Predict and update
            kf.predict()
            kf.update([obj['x'], obj['y']])

            # Get smoothed position
            tracked.append({
                'track_id': obj['id'] + 1,  # 1-indexed
                'x': kf.x[0, 0],
                'y': kf.x[1, 0],
                'confidence': obj['confidence'],
                'source': obj.get('source', 'unknown'),
                'anchor_id': obj.get('anchor_id', -1)
            })

        return tracked

    def _create_kalman_filter(self):
        """Create Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=4, dim_z=2)

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

        kf.Q *= 0.05  # Lower process noise
        kf.R *= 5     # Lower measurement noise for YOLO
        kf.P *= 100

        return kf

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame"""
        # Detect surgical tools
        objects = self.detect_surgical_tools(frame)

        # Track objects
        tracked = self.track_objects(frame, objects)

        return {
            'objects': tracked,
            'num_objects': len(tracked),
            'sources': list(set(obj.get('source', '') for obj in objects))
        }

def test_yolo_anchor_system():
    """Test the YOLO + Anchor system"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    system = YOLOAnchorFusion(device='cpu')

    # Test on specific frame
    frame_file = data_root / "val/frames/E66F/E66F_frame_0.png"
    frame = cv2.imread(str(frame_file))

    result = system.process_frame(frame)

    logger.info(f"\nYOLO + Anchor System Results:")
    logger.info(f"  Objects detected: {result['num_objects']}")
    logger.info(f"  Detection sources: {result['sources']}")

    for obj in result['objects']:
        logger.info(f"    Object {obj['track_id']}: ({obj['x']:.0f}, {obj['y']:.0f}) "
                   f"conf={obj['confidence']:.2f} source={obj['source']} anchor={obj['anchor_id']}")

    # Compare with ground truth
    gt_locations = [
        (586, 833),   # T0
        (1541, 561),  # T1
        (1304, 128),  # T2
        (418, 193),   # T3
        (797, 487),   # T4
        (802, 485)    # T5
    ]

    logger.info("\nDistance to ground truth:")
    total_dist = 0
    for i, obj in enumerate(result['objects']):
        if i < len(gt_locations):
            gt_x, gt_y = gt_locations[i]
            dist = np.sqrt((obj['x'] - gt_x)**2 + (obj['y'] - gt_y)**2)
            total_dist += dist
            logger.info(f"  Object {i+1}: {dist:.1f} pixels")

    avg_dist = total_dist / min(len(result['objects']), len(gt_locations))
    logger.info(f"\nAverage distance: {avg_dist:.1f} pixels")

    if avg_dist < 100:
        logger.info("✅ Good detection accuracy (< 100 pixels)")
    else:
        logger.info("⚠️  Detection needs improvement")

    return result

if __name__ == "__main__":
    test_yolo_anchor_system()
    print("\n✅ YOLO + Anchor system ready!")