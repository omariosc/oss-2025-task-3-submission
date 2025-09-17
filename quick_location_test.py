#!/usr/bin/env python3
"""
Quick test to verify detection locations match ground truth
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Compare detection locations
def test_locations():
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")

    # Load one frame
    frame_file = data_root / "val/frames/E66F/E66F_frame_0.png"
    frame = cv2.imread(str(frame_file))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ground truth locations for frame 0 (from our analysis)
    gt_locations = [
        (586, 833),   # T0 - Bottom left hand
        (1541, 561),  # T1 - Right tool
        (1304, 128),  # T2 - Top right
        (418, 193),   # T3 - Top left tool
        (797, 487),   # T4 - Center
        (802, 485)    # T5 - Center duplicate
    ]

    # Our current detections (grid-based - WRONG)
    grid_detections = [
        (240, 240),
        (720, 240),
        (1200, 240),
        (240, 720),
        (720, 720),
        (1200, 720)
    ]

    # Use YOLO to get better detections
    from ultralytics import YOLO
    yolo_path = "/Users/scsoc/Desktop/synpase/endovis2025/task_3/data/yolo11m.pt"
    model = YOLO(yolo_path)

    results = model(frame, conf=0.25, verbose=False)
    yolo_detections = []

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                yolo_detections.append((cx, cy))

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Ground Truth
    axes[0].imshow(frame_rgb)
    axes[0].set_title("Ground Truth Locations")
    for i, (x, y) in enumerate(gt_locations):
        axes[0].scatter(x, y, c='red', s=200, marker='o')
        axes[0].text(x, y, f'GT{i}', color='white', fontsize=12, backgroundcolor='red')

    # 2. Grid Detection (Wrong)
    axes[1].imshow(frame_rgb)
    axes[1].set_title("Grid Detection (WRONG)")
    for i, (x, y) in enumerate(grid_detections):
        axes[1].scatter(x, y, c='blue', s=200, marker='x')
        axes[1].text(x, y, f'G{i}', color='white', fontsize=12, backgroundcolor='blue')

    # 3. YOLO Detection
    axes[2].imshow(frame_rgb)
    axes[2].set_title(f"YOLO Detection ({len(yolo_detections)} objects)")
    for i, (x, y) in enumerate(yolo_detections[:6]):
        axes[2].scatter(x, y, c='green', s=200, marker='*')
        axes[2].text(x, y, f'Y{i}', color='white', fontsize=12, backgroundcolor='green')

    plt.tight_layout()
    plt.savefig("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/detection_comparison.png", dpi=150)
    plt.show()

    # Calculate distances
    print("\nDistance Analysis:")
    print("="*50)

    # Grid to GT distances
    grid_distances = []
    for gx, gy in grid_detections[:len(gt_locations)]:
        min_dist = min(np.sqrt((gx - gtx)**2 + (gy - gty)**2) for gtx, gty in gt_locations)
        grid_distances.append(min_dist)

    print(f"Grid Detection Average Distance to GT: {np.mean(grid_distances):.1f} pixels")

    # YOLO to GT distances
    if yolo_detections:
        yolo_distances = []
        for yx, yy in yolo_detections[:len(gt_locations)]:
            min_dist = min(np.sqrt((yx - gtx)**2 + (yy - gty)**2) for gtx, gty in gt_locations)
            yolo_distances.append(min_dist)

        print(f"YOLO Detection Average Distance to GT: {np.mean(yolo_distances):.1f} pixels")

        # Calculate potential HOTA improvement
        grid_matches = sum(1 for d in grid_distances if d < 100)
        yolo_matches = sum(1 for d in yolo_distances if d < 100)

        print(f"\nPotential matches (< 100 pixels):")
        print(f"  Grid: {grid_matches}/6 = {grid_matches/6:.1%}")
        print(f"  YOLO: {yolo_matches}/6 = {yolo_matches/6:.1%}")

        print(f"\nExpected DetA improvement:")
        print(f"  Grid: {grid_matches/6 * 0.5:.3f}")
        print(f"  YOLO: {yolo_matches/6 * 0.5:.3f}")

        print(f"\nExpected HOTA (with perfect AssA):")
        print(f"  Grid: {np.sqrt(grid_matches/6 * 0.5):.3f}")
        print(f"  YOLO: {np.sqrt(yolo_matches/6 * 0.5):.3f}")

if __name__ == "__main__":
    test_locations()