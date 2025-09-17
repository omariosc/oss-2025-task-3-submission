#!/usr/bin/env python3
"""
Depth-Guided Detection System
Integrates MiDaS depth estimation for improved keypoint detection and occlusion handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthEstimator:
    """MiDaS-based depth estimation for surgical scenes"""

    def __init__(self, model_type='DPT_Large', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model_type = model_type
        logger.info(f"Initializing MiDaS depth estimator ({model_type}) on {device}")

        try:
            # Try to import MiDaS
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas.to(self.device)
            self.midas.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            logger.info("MiDaS loaded successfully")
            self.available = True

        except Exception as e:
            logger.warning(f"Could not load MiDaS: {e}")
            logger.info("Installing fallback depth estimation")
            self.available = False
            self.midas = None

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from RGB frame"""
        if not self.available:
            return self.estimate_depth_fallback(frame)

        # Prepare image
        input_batch = self.transform(frame).to(self.device)

        # Generate depth map
        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-1 range
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_map

    def estimate_depth_fallback(self, frame: np.ndarray) -> np.ndarray:
        """Simple depth estimation using image cues when MiDaS is not available"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use multiple cues for depth
        # 1. Blur as depth cue (sharper = closer)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.abs(laplacian)

        # 2. Brightness as depth cue (brighter = closer in surgical scenes)
        brightness = gray.astype(np.float32) / 255.0

        # 3. Color saturation (more saturated = closer)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:,:,1].astype(np.float32) / 255.0

        # Combine cues
        depth_map = 0.4 * sharpness + 0.3 * brightness + 0.3 * saturation

        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)

        # Normalize
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        return depth_map


class DepthGuidedKeypointDetector:
    """Detect keypoints using depth information for better accuracy"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.depth_estimator = DepthEstimator(device=device)

        # Depth thresholds for object segmentation
        self.depth_threshold_near = 0.7  # Objects closer than this
        self.depth_threshold_far = 0.3   # Background further than this

        logger.info("Depth-guided keypoint detector initialized")

    def extract_depth_features(self, depth_map: np.ndarray) -> Dict:
        """Extract useful features from depth map"""
        features = {}

        # 1. Depth discontinuities (object boundaries)
        depth_grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        depth_grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        depth_edges = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
        features['depth_edges'] = depth_edges

        # 2. Depth layers (foreground, midground, background)
        features['foreground_mask'] = (depth_map > self.depth_threshold_near).astype(np.uint8) * 255
        features['midground_mask'] = ((depth_map > self.depth_threshold_far) &
                                      (depth_map <= self.depth_threshold_near)).astype(np.uint8) * 255
        features['background_mask'] = (depth_map <= self.depth_threshold_far).astype(np.uint8) * 255

        # 3. Surface normals from depth
        normals = self.compute_surface_normals(depth_map)
        features['normals'] = normals

        # 4. Occlusion boundaries
        occlusion_mask = self.detect_occlusions(depth_map, depth_edges)
        features['occlusion_mask'] = occlusion_mask

        # 5. Depth statistics per region
        features['depth_mean'] = np.mean(depth_map)
        features['depth_std'] = np.std(depth_map)

        return features

    def compute_surface_normals(self, depth_map: np.ndarray) -> np.ndarray:
        """Compute surface normals from depth map"""
        # Gradients
        dz_dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

        # Normal vector components
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(depth_map)

        # Normalize
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        normals = np.stack([nx/norm, ny/norm, nz/norm], axis=-1)

        return normals

    def detect_occlusions(self, depth_map: np.ndarray, depth_edges: np.ndarray) -> np.ndarray:
        """Detect occlusion boundaries from depth discontinuities"""
        # Strong depth edges indicate occlusions
        edge_threshold = np.percentile(depth_edges, 90)
        occlusion_candidates = depth_edges > edge_threshold

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_mask = cv2.morphologyEx(occlusion_candidates.astype(np.uint8) * 255,
                                          cv2.MORPH_CLOSE, kernel)

        return occlusion_mask

    def detect_keypoints_with_depth(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> List[Dict]:
        """Detect keypoints using both RGB and depth information"""

        # Estimate depth if not provided
        if depth_map is None:
            depth_map = self.depth_estimator.estimate_depth(frame)

        # Extract depth features
        depth_features = self.extract_depth_features(depth_map)

        keypoints = []

        # 1. Detect keypoints on foreground objects (surgical tools/hands)
        foreground_mask = depth_features['foreground_mask']

        # Find contours in foreground
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Skip small regions
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Extract keypoints for this foreground object
            object_keypoints = self.extract_object_keypoints(frame[y:y+h, x:x+w],
                                                            depth_map[y:y+h, x:x+w],
                                                            offset=(x, y))
            keypoints.extend(object_keypoints)

        # 2. Use depth edges for precise tool boundaries
        depth_edges = depth_features['depth_edges']
        edge_keypoints = self.extract_edge_keypoints(depth_edges, depth_map)
        keypoints.extend(edge_keypoints)

        # 3. Handle occlusions
        occlusion_mask = depth_features['occlusion_mask']
        keypoints = self.refine_keypoints_with_occlusion(keypoints, occlusion_mask, depth_map)

        return keypoints

    def extract_object_keypoints(self, object_region: np.ndarray,
                                 depth_region: np.ndarray,
                                 offset: Tuple[int, int]) -> List[Dict]:
        """Extract keypoints from a single object region"""
        keypoints = []
        ox, oy = offset

        if object_region.size == 0:
            return keypoints

        # Find depth peaks (closest points, likely tool tips)
        depth_peaks = self.find_depth_peaks(depth_region)

        for px, py, confidence in depth_peaks:
            keypoints.append({
                'x': px + ox,
                'y': py + oy,
                'confidence': confidence,
                'type': 'depth_peak',
                'depth': depth_region[py, px] if py < depth_region.shape[0] and px < depth_region.shape[1] else 0
            })

        # Find depth valleys (furthest points, likely handles)
        depth_valleys = self.find_depth_valleys(depth_region)

        for vx, vy, confidence in depth_valleys:
            keypoints.append({
                'x': vx + ox,
                'y': vy + oy,
                'confidence': confidence * 0.8,  # Lower confidence for valleys
                'type': 'depth_valley',
                'depth': depth_region[vy, vx] if vy < depth_region.shape[0] and vx < depth_region.shape[1] else 0
            })

        return keypoints

    def find_depth_peaks(self, depth_region: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find local maxima in depth (closest points)"""
        peaks = []

        if depth_region.size == 0:
            return peaks

        # Apply Gaussian blur to smooth
        depth_smooth = cv2.GaussianBlur(depth_region, (5, 5), 0)

        # Find local maxima
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        local_max = cv2.dilate(depth_smooth, kernel)
        peak_mask = (depth_smooth == local_max) & (depth_smooth > np.percentile(depth_smooth, 75))

        # Get peak coordinates
        peak_coords = np.column_stack(np.where(peak_mask))

        for py, px in peak_coords[:5]:  # Limit to top 5 peaks
            confidence = depth_smooth[py, px]
            peaks.append((px, py, float(confidence)))

        return peaks

    def find_depth_valleys(self, depth_region: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find local minima in depth (furthest points)"""
        valleys = []

        if depth_region.size == 0:
            return valleys

        # Invert depth for valley detection
        depth_inv = 1.0 - depth_region

        # Apply same peak detection on inverted depth
        return self.find_depth_peaks(depth_inv)

    def extract_edge_keypoints(self, depth_edges: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        """Extract keypoints from depth discontinuities"""
        keypoints = []

        # Threshold edges
        edge_threshold = np.percentile(depth_edges, 85)
        strong_edges = depth_edges > edge_threshold

        # Find corners in edge map
        corners = cv2.goodFeaturesToTrack(strong_edges.astype(np.uint8) * 255,
                                         maxCorners=20,
                                         qualityLevel=0.1,
                                         minDistance=30)

        if corners is not None:
            for corner in corners:
                x, y = corner[0]
                keypoints.append({
                    'x': float(x),
                    'y': float(y),
                    'confidence': 0.7,
                    'type': 'depth_edge',
                    'depth': depth_map[int(y), int(x)] if int(y) < depth_map.shape[0] and int(x) < depth_map.shape[1] else 0
                })

        return keypoints

    def refine_keypoints_with_occlusion(self, keypoints: List[Dict],
                                       occlusion_mask: np.ndarray,
                                       depth_map: np.ndarray) -> List[Dict]:
        """Refine keypoints based on occlusion information"""
        refined = []

        for kp in keypoints:
            x, y = int(kp['x']), int(kp['y'])

            # Check if keypoint is at occlusion boundary
            if 0 <= y < occlusion_mask.shape[0] and 0 <= x < occlusion_mask.shape[1]:
                if occlusion_mask[y, x] > 0:
                    # Keypoint is at occlusion, might be partially visible
                    kp['confidence'] *= 0.7
                    kp['occluded'] = True
                else:
                    kp['occluded'] = False

                # Add depth information
                kp['depth'] = depth_map[y, x]

            refined.append(kp)

        # Sort by depth (closer objects first)
        refined.sort(key=lambda x: x.get('depth', 0), reverse=True)

        return refined


def test_depth_guided_detection():
    """Test depth-guided keypoint detection"""
    from pathlib import Path

    detector = DepthGuidedKeypointDetector(device='cpu')

    # Test on validation frame
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    frame_file = data_root / "val/frames/E66F/E66F_frame_0.png"

    if frame_file.exists():
        frame = cv2.imread(str(frame_file))

        logger.info("Estimating depth map...")
        depth_map = detector.depth_estimator.estimate_depth(frame)

        logger.info("Detecting keypoints with depth guidance...")
        keypoints = detector.detect_keypoints_with_depth(frame, depth_map)

        logger.info(f"\nDetected {len(keypoints)} keypoints with depth")

        # Group by type
        by_type = defaultdict(list)
        for kp in keypoints:
            by_type[kp['type']].append(kp)

        for kp_type, kps in by_type.items():
            logger.info(f"  {kp_type}: {len(kps)} keypoints")

            # Show top 3 by confidence
            kps_sorted = sorted(kps, key=lambda x: x['confidence'], reverse=True)[:3]
            for kp in kps_sorted:
                logger.info(f"    ({kp['x']:.0f}, {kp['y']:.0f}) conf={kp['confidence']:.2f} depth={kp.get('depth', 0):.2f}")

        # Visualize depth map (optional)
        save_viz = True
        if save_viz:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Frame")
            axes[0].axis('off')

            # Depth map
            axes[1].imshow(depth_map, cmap='plasma')
            axes[1].set_title("Estimated Depth")
            axes[1].axis('off')

            # Keypoints on depth
            axes[2].imshow(depth_map, cmap='plasma')
            for kp in keypoints[:30]:  # Show top 30
                color = 'red' if kp['type'] == 'depth_peak' else 'blue'
                axes[2].scatter(kp['x'], kp['y'], c=color, s=20, alpha=0.7)
            axes[2].set_title("Depth-Guided Keypoints")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/depth_guided_detection.png')
            plt.close()
            logger.info("  Saved visualization to depth_guided_detection.png")

    return keypoints


if __name__ == "__main__":
    keypoints = test_depth_guided_detection()
    print(f"\n✅ Depth-guided detection ready! Detected {len(keypoints) if keypoints else 0} keypoints")