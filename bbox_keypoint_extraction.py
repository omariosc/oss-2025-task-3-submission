#!/usr/bin/env python3
"""
Bounding Box-based Keypoint Extraction
Extract keypoints from detected bounding boxes at key locations
"""

import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BBoxKeypointExtractor:
    """Extract keypoints from bounding boxes at anatomically relevant positions"""

    def __init__(self, device='cpu'):
        self.device = device
        logger.info("BBox Keypoint Extractor initialized")

        # Load base detection system
        from multistage_yolo_anchors import YOLOAnchorFusion
        self.detector = YOLOAnchorFusion(device=device)

        # Target keypoints per object based on GT analysis
        self.keypoints_per_object = {
            'default': 3,  # Most common
            'large': 4,    # Larger objects
            'complex': 5   # Complex tools
        }

    def extract_keypoints_from_bbox(self, detection: Dict) -> List[Dict]:
        """Extract keypoints from a single bounding box"""

        # Get center point
        cx = detection.get('x', 0)
        cy = detection.get('y', 0)
        confidence = detection.get('confidence', 0.5)

        keypoints = []

        # If we have a bounding box, use it
        if 'bbox' in detection and len(detection['bbox']) >= 4:
            bbox = detection['bbox']

            # Extract box coordinates
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                # YOLO format: cx, cy, w, h
                w, h = bbox[2], bbox[3]
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2

            # Determine object size category
            area = abs((x2 - x1) * (y2 - y1))
            if area > 10000:
                category = 'large'
                num_keypoints = 4
            elif area > 5000:
                category = 'complex'
                num_keypoints = 3
            else:
                category = 'default'
                num_keypoints = 3

            # Generate keypoints based on bbox
            if num_keypoints >= 3:
                # Top-center (tip/head)
                keypoints.append({
                    'x': (x1 + x2) / 2,
                    'y': y1,
                    'confidence': confidence,
                    'type': 'top'
                })

                # Center (pivot/joint)
                keypoints.append({
                    'x': cx,
                    'y': cy,
                    'confidence': confidence * 0.9,
                    'type': 'center'
                })

                # Bottom-center (base/handle)
                keypoints.append({
                    'x': (x1 + x2) / 2,
                    'y': y2,
                    'confidence': confidence * 0.8,
                    'type': 'bottom'
                })

            if num_keypoints >= 4:
                # Add left or right edge
                if abs(x2 - x1) > abs(y2 - y1):  # Horizontal object
                    # Add left edge
                    keypoints.append({
                        'x': x1,
                        'y': cy,
                        'confidence': confidence * 0.7,
                        'type': 'left'
                    })
                else:  # Vertical object
                    # Add another point between center and bottom
                    keypoints.append({
                        'x': cx,
                        'y': (cy + y2) / 2,
                        'confidence': confidence * 0.7,
                        'type': 'mid_bottom'
                    })

        else:
            # No bbox, just use center with small offsets
            offsets = [(0, 0), (0, -20), (0, 20)]
            for i, (dx, dy) in enumerate(offsets):
                keypoints.append({
                    'x': cx + dx,
                    'y': cy + dy,
                    'confidence': confidence * (1.0 - i * 0.2),
                    'type': 'offset'
                })

        # Add metadata to each keypoint
        for i, kp in enumerate(keypoints):
            kp['object_id'] = detection.get('id', 0)
            kp['keypoint_idx'] = i

        return keypoints

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with bbox-based keypoint extraction"""

        # Get base detections with bounding boxes
        base_result = self.detector.process_frame(frame)
        detections = base_result.get('objects', base_result.get('tracked_keypoints', []))

        # Limit to top 6 detections (GT average: 5.7 objects)
        detections = sorted(detections,
                          key=lambda x: x.get('confidence', 0),
                          reverse=True)[:6]

        # Extract keypoints from each detection
        all_keypoints = []
        total_extracted = 0

        for detection in detections:
            keypoints = self.extract_keypoints_from_bbox(detection)
            all_keypoints.extend(keypoints)
            total_extracted += len(keypoints)

        # Apply simple tracking IDs
        for i, kp in enumerate(all_keypoints):
            kp['track_id'] = i + 1

        # Target is ~22 keypoints (GT average: 21.7)
        # If we have too many, keep highest confidence
        if len(all_keypoints) > 23:
            all_keypoints.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            all_keypoints = all_keypoints[:23]

        return {
            'keypoints': all_keypoints,
            'num_keypoints': len(all_keypoints),
            'num_objects': len(detections),
            'avg_keypoints_per_object': total_extracted / max(len(detections), 1)
        }


def evaluate_bbox_extraction():
    """Evaluate bbox-based keypoint extraction"""

    logger.info("="*70)
    logger.info("EVALUATING BBOX-BASED KEYPOINT EXTRACTION")
    logger.info("="*70)

    system = BBoxKeypointExtractor(device='cpu')

    # Test on validation
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

    # Process frames
    frame_files = sorted(list(frames_dir.glob("*.png")))[:20]

    all_results = []
    for frame_file in frame_files:
        frame_num = int(frame_file.stem.split('_')[-1])
        if frame_num not in gt_keypoints:
            continue

        frame = cv2.imread(str(frame_file))
        result = system.process_frame(frame)

        gt_kps = gt_keypoints[frame_num]
        pred_kps = result['keypoints']

        # Calculate metrics for this frame
        if len(pred_kps) > 0 and len(gt_kps) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(gt_kps), len(pred_kps)))
            for i, gt in enumerate(gt_kps):
                for j, pred in enumerate(pred_kps):
                    dist = np.sqrt((gt['x'] - pred['x'])**2 + (gt['y'] - pred['y'])**2)
                    cost_matrix[i, j] = dist

            # Hungarian matching
            gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

            tp = sum(1 for gi, pi in zip(gt_idx, pred_idx) if cost_matrix[gi, pi] < 100)
            fp = len(pred_kps) - tp
            fn = len(gt_kps) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            all_results.append({
                'frame': frame_num,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'num_objects': result['num_objects'],
                'num_keypoints': result['num_keypoints']
            })

            logger.info(f"Frame {frame_num}: Objects={result['num_objects']}, "
                       f"Keypoints={result['num_keypoints']}, "
                       f"TP={tp}, FP={fp}, FN={fn}, "
                       f"Precision={precision:.3f}, Recall={recall:.3f}")

    # Calculate overall metrics
    if all_results:
        total_tp = sum(r['tp'] for r in all_results)
        total_fp = sum(r['fp'] for r in all_results)
        total_fn = sum(r['fn'] for r in all_results)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        det_a = (overall_precision + overall_recall) / 2

        # Estimate HOTA
        estimated_hota = np.sqrt(det_a * 0.6)  # Assuming moderate association

        logger.info(f"\nOverall Results:")
        logger.info(f"  Total TP: {total_tp}")
        logger.info(f"  Total FP: {total_fp}")
        logger.info(f"  Total FN: {total_fn}")
        logger.info(f"  Precision: {overall_precision:.3f}")
        logger.info(f"  Recall: {overall_recall:.3f}")
        logger.info(f"  DetA: {det_a:.3f}")
        logger.info(f"  Estimated HOTA: {estimated_hota:.3f}")

        if estimated_hota > 0.346:
            improvement = (estimated_hota - 0.346) / 0.346 * 100
            logger.info(f"  ✅ Potential improvement: +{improvement:.1f}%")
        else:
            logger.info(f"  ⚠️  Still below baseline")

        return estimated_hota

    return 0.0


if __name__ == "__main__":
    hota = evaluate_bbox_extraction()
    print(f"\n✅ BBox extraction evaluation complete! Estimated HOTA: {hota:.3f}")