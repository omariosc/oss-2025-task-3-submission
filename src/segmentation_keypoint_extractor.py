#!/usr/bin/env python3
"""
Segmentation-Based Keypoint Extraction
Extracts anatomical keypoints from segmentation masks for surgical tools
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import ndimage
from skimage import morphology, measure
from skimage.morphology import skeletonize
import torch
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationKeypointExtractor:
    """Extract anatomical keypoints from segmentation masks"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Segmentation Keypoint Extractor initialized on {device}")

        # Tool classes and expected keypoints
        self.tool_classes = [
            'left_hand_segment',
            'right_hand_segment',
            'scissors',
            'tweezers',
            'needle_holder',
            'needle'
        ]

        # Expected keypoints per tool type
        self.keypoint_configs = {
            'left_hand_segment': ['palm_center', 'thumb_tip', 'index_tip', 'wrist'],
            'right_hand_segment': ['palm_center', 'thumb_tip', 'index_tip', 'wrist'],
            'scissors': ['blade_tip1', 'blade_tip2', 'pivot', 'handle1', 'handle2'],
            'tweezers': ['tip1', 'tip2', 'grasp_point', 'base'],
            'needle_holder': ['jaw_tip1', 'jaw_tip2', 'lock', 'handle'],
            'needle': ['tip', 'eye', 'center']
        }

    def extract_keypoints_from_mask(self, mask: np.ndarray, tool_class: str) -> List[Dict]:
        """Extract keypoints from a single segmentation mask"""
        keypoints = []

        if mask.sum() == 0:  # Empty mask
            return keypoints

        # Clean the mask
        mask_clean = self.clean_mask(mask)

        # Get contours
        contours, _ = cv2.findContours(mask_clean.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return keypoints

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Extract keypoints based on tool type
        if 'hand' in tool_class:
            keypoints = self.extract_hand_keypoints(mask_clean, contour)
        elif tool_class == 'scissors':
            keypoints = self.extract_scissors_keypoints(mask_clean, contour)
        elif tool_class == 'tweezers':
            keypoints = self.extract_tweezers_keypoints(mask_clean, contour)
        elif tool_class == 'needle_holder':
            keypoints = self.extract_needle_holder_keypoints(mask_clean, contour)
        elif tool_class == 'needle':
            keypoints = self.extract_needle_keypoints(mask_clean, contour)
        else:
            # Generic keypoint extraction
            keypoints = self.extract_generic_keypoints(mask_clean, contour)

        return keypoints

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean segmentation mask using morphological operations"""
        # Convert to binary
        mask_binary = (mask > 0).astype(np.uint8) * 255

        # Remove small components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        # Fill holes
        mask_filled = ndimage.binary_fill_holes(mask_clean > 0)

        return mask_filled.astype(np.uint8) * 255

    def extract_hand_keypoints(self, mask: np.ndarray, contour: np.ndarray) -> List[Dict]:
        """Extract keypoints from hand segmentation"""
        keypoints = []

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Palm center (centroid of mask)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.append({
                'x': cx, 'y': cy,
                'confidence': 0.9,
                'type': 'palm_center'
            })

        # Wrist (bottom of bounding box)
        wrist_x = x + w // 2
        wrist_y = y + h
        keypoints.append({
            'x': wrist_x, 'y': wrist_y,
            'confidence': 0.8,
            'type': 'wrist'
        })

        # Find fingertips using distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # Skeleton for finding extremities
        skeleton = skeletonize(mask > 0)

        # Find endpoints of skeleton (likely fingertips)
        endpoints = self.find_skeleton_endpoints(skeleton)

        # Add top fingertips
        if endpoints:
            # Sort by y-coordinate (top points)
            endpoints_sorted = sorted(endpoints, key=lambda p: p[1])[:2]
            for i, (ex, ey) in enumerate(endpoints_sorted):
                keypoints.append({
                    'x': ex, 'y': ey,
                    'confidence': 0.7,
                    'type': f'finger_tip_{i}'
                })

        return keypoints

    def extract_scissors_keypoints(self, mask: np.ndarray, contour: np.ndarray) -> List[Dict]:
        """Extract keypoints from scissors segmentation"""
        keypoints = []

        # Get convex hull for overall shape
        hull = cv2.convexHull(contour)

        # Find the two blade tips (furthest points)
        distances = []
        for i in range(len(hull)):
            for j in range(i+1, len(hull)):
                p1 = hull[i][0]
                p2 = hull[j][0]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                distances.append((dist, p1, p2))

        if distances:
            distances.sort(key=lambda x: x[0], reverse=True)
            # Two furthest points are likely blade tips
            _, tip1, tip2 = distances[0]
            keypoints.append({
                'x': int(tip1[0]), 'y': int(tip1[1]),
                'confidence': 0.85,
                'type': 'blade_tip1'
            })
            keypoints.append({
                'x': int(tip2[0]), 'y': int(tip2[1]),
                'confidence': 0.85,
                'type': 'blade_tip2'
            })

            # Pivot point (center of mass)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                keypoints.append({
                    'x': cx, 'y': cy,
                    'confidence': 0.9,
                    'type': 'pivot'
                })

        return keypoints

    def extract_tweezers_keypoints(self, mask: np.ndarray, contour: np.ndarray) -> List[Dict]:
        """Extract keypoints from tweezers segmentation"""
        keypoints = []

        # Fit a line to find principal axis
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Find extremities along the principal axis
        points = contour.reshape(-1, 2)
        projections = []

        for point in points:
            # Project point onto line
            t = ((point[0] - x) * vx + (point[1] - y) * vy) / (vx**2 + vy**2)
            proj_x = x + t * vx
            proj_y = y + t * vy
            projections.append((t[0], point))

        # Sort by projection value
        projections.sort(key=lambda x: x[0])

        # Tips are at the extremes
        if len(projections) >= 2:
            tip1 = projections[0][1]
            tip2 = projections[-1][1]

            keypoints.append({
                'x': int(tip1[0]), 'y': int(tip1[1]),
                'confidence': 0.9,
                'type': 'tip1'
            })
            keypoints.append({
                'x': int(tip2[0]), 'y': int(tip2[1]),
                'confidence': 0.9,
                'type': 'tip2'
            })

            # Grasp point (center)
            grasp_x = (tip1[0] + tip2[0]) // 2
            grasp_y = (tip1[1] + tip2[1]) // 2
            keypoints.append({
                'x': grasp_x, 'y': grasp_y,
                'confidence': 0.85,
                'type': 'grasp_point'
            })

        return keypoints

    def extract_needle_holder_keypoints(self, mask: np.ndarray, contour: np.ndarray) -> List[Dict]:
        """Extract keypoints from needle holder segmentation"""
        keypoints = []

        # Similar to tweezers but with lock mechanism
        # Get bounding box
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Find jaw tips (narrow end)
        center = rect[0]
        width, height = rect[1]
        angle = rect[2]

        # Determine orientation
        if width < height:
            # Vertical orientation
            tip_offset = height / 2
        else:
            # Horizontal orientation
            tip_offset = width / 2

        # Calculate jaw tips based on orientation
        tip1_x = center[0] + tip_offset * np.cos(np.radians(angle))
        tip1_y = center[1] + tip_offset * np.sin(np.radians(angle))
        tip2_x = center[0] - tip_offset * np.cos(np.radians(angle))
        tip2_y = center[1] - tip_offset * np.sin(np.radians(angle))

        keypoints.append({
            'x': int(tip1_x), 'y': int(tip1_y),
            'confidence': 0.8,
            'type': 'jaw_tip1'
        })
        keypoints.append({
            'x': int(tip2_x), 'y': int(tip2_y),
            'confidence': 0.8,
            'type': 'jaw_tip2'
        })

        # Lock mechanism (center of mass)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.append({
                'x': cx, 'y': cy,
                'confidence': 0.85,
                'type': 'lock'
            })

        return keypoints

    def extract_needle_keypoints(self, mask: np.ndarray, contour: np.ndarray) -> List[Dict]:
        """Extract keypoints from needle segmentation"""
        keypoints = []

        # Needles are small and curved
        # Find the endpoints
        perimeter = cv2.arcLength(contour, False)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, False)

        if len(approx) >= 2:
            # Tip and eye are the endpoints
            tip = approx[0][0]
            eye = approx[-1][0]

            keypoints.append({
                'x': int(tip[0]), 'y': int(tip[1]),
                'confidence': 0.9,
                'type': 'tip'
            })
            keypoints.append({
                'x': int(eye[0]), 'y': int(eye[1]),
                'confidence': 0.9,
                'type': 'eye'
            })

            # Center
            center_x = (tip[0] + eye[0]) // 2
            center_y = (tip[1] + eye[1]) // 2
            keypoints.append({
                'x': center_x, 'y': center_y,
                'confidence': 0.85,
                'type': 'center'
            })

        return keypoints

    def extract_generic_keypoints(self, mask: np.ndarray, contour: np.ndarray) -> List[Dict]:
        """Generic keypoint extraction for unknown objects"""
        keypoints = []

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            keypoints.append({
                'x': cx, 'y': cy,
                'confidence': 0.7,
                'type': 'center'
            })

        # Convex hull points
        hull = cv2.convexHull(contour)

        # Sample 4 points from hull
        if len(hull) >= 4:
            step = len(hull) // 4
            for i in range(0, len(hull), step)[:4]:
                point = hull[i][0]
                keypoints.append({
                    'x': int(point[0]), 'y': int(point[1]),
                    'confidence': 0.6,
                    'type': f'hull_point_{i//step}'
                })

        return keypoints

    def find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints of a skeleton"""
        endpoints = []

        # Convolve with kernel to find endpoints
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])

        convolved = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')

        # Endpoints have value 11 (1 neighbor + 10 for center)
        endpoint_coords = np.where((convolved == 11) & skeleton)

        for y, x in zip(endpoint_coords[0], endpoint_coords[1]):
            endpoints.append((x, y))

        return endpoints

    def process_frame_masks(self, frame_id: str, masks_dir: Path) -> List[Dict]:
        """Process all masks for a single frame"""
        all_keypoints = []

        for tool_class in self.tool_classes:
            mask_file = masks_dir / f"{frame_id}_{tool_class}_mask.png"

            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                keypoints = self.extract_keypoints_from_mask(mask, tool_class)

                # Add tool class to each keypoint
                for kp in keypoints:
                    kp['tool_class'] = tool_class
                    all_keypoints.append(kp)

        return all_keypoints

def test_segmentation_extraction():
    """Test segmentation keypoint extraction"""
    extractor = SegmentationKeypointExtractor(device='cpu')

    # Test with training masks
    masks_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data/train/masks")

    # Get a sample frame
    mask_files = list(masks_dir.glob("*_frame_0_*_mask.png"))

    if mask_files:
        # Group by frame
        frames = defaultdict(list)
        for mask_file in mask_files:
            parts = mask_file.stem.split('_')
            frame_id = f"{parts[0]}_frame_{parts[2]}"
            frames[frame_id].append(mask_file)

        # Process first frame
        first_frame = list(frames.keys())[0]
        logger.info(f"Processing frame: {first_frame}")

        keypoints = extractor.process_frame_masks(first_frame, masks_dir)

        logger.info(f"Extracted {len(keypoints)} keypoints")

        # Group by tool class
        by_class = defaultdict(list)
        for kp in keypoints:
            by_class[kp['tool_class']].append(kp)

        for tool_class, kps in by_class.items():
            logger.info(f"  {tool_class}: {len(kps)} keypoints")
            for kp in kps:
                logger.info(f"    - {kp['type']}: ({kp['x']}, {kp['y']}) conf={kp['confidence']:.2f}")

    return keypoints

if __name__ == "__main__":
    keypoints = test_segmentation_extraction()
    print(f"\n✅ Segmentation keypoint extraction ready!")