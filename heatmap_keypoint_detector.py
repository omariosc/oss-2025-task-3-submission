#!/usr/bin/env python3
"""
Heatmap-based Keypoint Detection
Directly regresses 23 heatmaps for expected keypoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
from pathlib import Path
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeatmapNet(nn.Module):
    """Network to predict keypoint heatmaps"""

    def __init__(self, num_keypoints=23):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Encoder (downsampling path)
        self.enc1 = self._conv_block(3, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Decoder (upsampling path)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 256 from up3 + 256 from enc3

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)  # 128 from up2 + 128 from enc2

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)   # 64 from up1 + 64 from enc1

        # Final heatmap prediction
        self.heatmap = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Decoder with skip connections
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Generate heatmaps
        heatmaps = self.heatmap(d1)
        heatmaps = torch.sigmoid(heatmaps)  # Ensure [0, 1] range

        return heatmaps


class HeatmapKeypointDetector:
    """Detect keypoints using heatmap regression"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 num_keypoints=23):
        self.device = torch.device(device)
        self.num_keypoints = num_keypoints
        logger.info(f"Heatmap Keypoint Detector on {device}")

        # Initialize network
        self.model = HeatmapNet(num_keypoints=num_keypoints).to(self.device)
        self.model.eval()

        # Image preprocessing parameters
        self.input_size = (640, 480)  # Standard size for processing
        self.sigma = 6  # Gaussian sigma for heatmap generation

        # Load pretrained weights if available
        self.load_pretrained()

    def load_pretrained(self):
        """Load pretrained weights if available"""
        model_path = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/models/heatmap_model.pth")
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Loaded pretrained heatmap model")
        else:
            logger.info("No pretrained model found, using random initialization")

    def create_ground_truth_heatmap(self, keypoints: List[Tuple[float, float]],
                                   size: Tuple[int, int]) -> np.ndarray:
        """Create ground truth heatmap for training"""
        h, w = size
        heatmaps = np.zeros((self.num_keypoints, h, w), dtype=np.float32)

        for i, (x, y) in enumerate(keypoints[:self.num_keypoints]):
            if 0 <= x < w and 0 <= y < h:
                # Create Gaussian peak at keypoint location
                y_int, x_int = int(y), int(x)

                # Create small Gaussian kernel
                kernel_size = int(6 * self.sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                # Generate 2D Gaussian
                ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))

                # Place kernel in heatmap
                y1 = max(0, y_int - kernel_size // 2)
                y2 = min(h, y_int + kernel_size // 2 + 1)
                x1 = max(0, x_int - kernel_size // 2)
                x2 = min(w, x_int + kernel_size // 2 + 1)

                # Adjust kernel bounds
                ky1 = max(0, kernel_size // 2 - y_int)
                ky2 = kernel_size - max(0, (y_int + kernel_size // 2 + 1) - h)
                kx1 = max(0, kernel_size // 2 - x_int)
                kx2 = kernel_size - max(0, (x_int + kernel_size // 2 + 1) - w)

                if y2 > y1 and x2 > x1:
                    heatmaps[i, y1:y2, x1:x2] = kernel[ky1:ky2, kx1:kx2]

        return heatmaps

    def extract_keypoints_from_heatmap(self, heatmaps: torch.Tensor,
                                      threshold: float = 0.3) -> List[Dict]:
        """Extract keypoint locations from heatmaps"""
        keypoints = []
        heatmaps_np = heatmaps.cpu().numpy()

        for i in range(heatmaps_np.shape[0]):
            hmap = heatmaps_np[i]

            # Apply Gaussian filter to smooth
            hmap = gaussian_filter(hmap, sigma=2)

            # Find peak
            max_val = hmap.max()

            if max_val > threshold:
                # Get peak location
                y, x = np.unravel_index(hmap.argmax(), hmap.shape)

                # Sub-pixel refinement using weighted average around peak
                y1 = max(0, y - 2)
                y2 = min(hmap.shape[0], y + 3)
                x1 = max(0, x - 2)
                x2 = min(hmap.shape[1], x + 3)

                local_patch = hmap[y1:y2, x1:x2]
                if local_patch.sum() > 0:
                    yy, xx = np.mgrid[y1:y2, x1:x2]
                    y_refined = (yy * local_patch).sum() / local_patch.sum()
                    x_refined = (xx * local_patch).sum() / local_patch.sum()
                else:
                    y_refined, x_refined = float(y), float(x)

                keypoints.append({
                    'x': x_refined,
                    'y': y_refined,
                    'confidence': float(max_val),
                    'keypoint_id': i
                })

        return keypoints

    def detect_keypoints(self, frame: np.ndarray) -> List[Dict]:
        """Detect keypoints in frame using heatmap regression"""
        h_orig, w_orig = frame.shape[:2]

        # Resize to standard size
        frame_resized = cv2.resize(frame, self.input_size)

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Generate heatmaps
        with torch.no_grad():
            heatmaps = self.model(frame_tensor)

        # Extract keypoints
        keypoints = self.extract_keypoints_from_heatmap(heatmaps[0])

        # Scale back to original size
        scale_x = w_orig / self.input_size[0]
        scale_y = h_orig / self.input_size[1]

        for kp in keypoints:
            kp['x'] *= scale_x
            kp['y'] *= scale_y

        return keypoints

    def train_step(self, frames: List[np.ndarray],
                  keypoints_list: List[List[Tuple[float, float]]]) -> float:
        """Single training step (if we had training data)"""
        self.model.train()
        batch_size = len(frames)

        # Prepare batch
        batch_frames = []
        batch_heatmaps = []

        for frame, keypoints in zip(frames, keypoints_list):
            # Resize frame
            frame_resized = cv2.resize(frame, self.input_size)
            frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)
            batch_frames.append(frame_tensor)

            # Create ground truth heatmap
            # Scale keypoints to resized frame
            h_orig, w_orig = frame.shape[:2]
            scale_x = self.input_size[0] / w_orig
            scale_y = self.input_size[1] / h_orig

            scaled_keypoints = [(x * scale_x, y * scale_y) for x, y in keypoints]
            gt_heatmap = self.create_ground_truth_heatmap(scaled_keypoints, self.input_size[::-1])
            batch_heatmaps.append(torch.from_numpy(gt_heatmap))

        # Stack batch
        batch_frames = torch.stack(batch_frames).to(self.device)
        batch_heatmaps = torch.stack(batch_heatmaps).to(self.device)

        # Forward pass
        pred_heatmaps = self.model(batch_frames)

        # MSE loss
        loss = F.mse_loss(pred_heatmaps, batch_heatmaps)

        self.model.eval()
        return loss.item()


def test_heatmap_detector():
    """Test the heatmap keypoint detector"""
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    detector = HeatmapKeypointDetector(device='cpu')

    # Load test frame
    frame_file = data_root / "val/frames/E66F/E66F_frame_0.png"
    frame = cv2.imread(str(frame_file))

    # Detect keypoints
    keypoints = detector.detect_keypoints(frame)

    logger.info(f"\nHeatmap Detection Results:")
    logger.info(f"  Keypoints detected: {len(keypoints)}")

    # Load ground truth
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
                            gt_keypoints.append((x, y))
                        i += 3
                    except:
                        break

    logger.info(f"  Ground Truth: {len(gt_keypoints)} keypoints")

    # Compare with ground truth
    if len(keypoints) > 0 and len(gt_keypoints) > 0:
        # Simple nearest neighbor matching
        total_dist = 0
        matches = 0

        for kp in keypoints:
            min_dist = float('inf')
            for gt_x, gt_y in gt_keypoints:
                dist = np.sqrt((kp['x'] - gt_x)**2 + (kp['y'] - gt_y)**2)
                min_dist = min(min_dist, dist)

            if min_dist < 100:
                matches += 1
                total_dist += min_dist

        recall = matches / len(gt_keypoints)
        precision = matches / len(keypoints) if len(keypoints) > 0 else 0
        avg_dist = total_dist / matches if matches > 0 else float('inf')

        logger.info(f"\nMatching Results:")
        logger.info(f"  Matched: {matches}/{len(gt_keypoints)}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Average distance: {avg_dist:.1f} pixels")

    # Visualize heatmaps (optional)
    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        # Generate heatmaps for visualization
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, detector.input_size)
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(detector.device)

        with torch.no_grad():
            heatmaps = detector.model(frame_tensor)

        # Show first few heatmaps
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(min(8, heatmaps.shape[1])):
            hmap = heatmaps[0, i].cpu().numpy()
            axes[i].imshow(hmap, cmap='hot')
            axes[i].set_title(f'Keypoint {i}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/heatmaps_viz.png')
        plt.close()
        logger.info("  Saved heatmap visualization")

    return keypoints


if __name__ == "__main__":
    keypoints = test_heatmap_detector()
    print(f"\n✅ Heatmap detector ready! Detected {len(keypoints)} keypoints")