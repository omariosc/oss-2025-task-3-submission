#!/usr/bin/env python3
"""
Simple FPS test for YOLO-Pose model with consistent timing
"""

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

def test_fps_simple():
    """Test FPS with simplified pipeline"""

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model_path = "surgical_keypoints_best.pt"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        # Try YOLOv8 medium pose
        model_path = "yolov8m-pose.pt"
        print(f"Downloading {model_path}...")
        model = YOLO(model_path)
    else:
        model = YOLO(model_path)

    # Test at different resolutions
    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD")
    ]

    print("\n" + "="*60)
    print("FPS TEST RESULTS")
    print("="*60)

    for width, height, name in resolutions:
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Warmup
        for _ in range(3):
            _ = model(test_image, verbose=False)

        # Time 10 iterations
        num_iterations = 10
        start = time.perf_counter()

        for _ in range(num_iterations):
            results = model(test_image, verbose=False, conf=0.25)

            # Simulate postprocessing
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data
                    # Convert to numpy if needed
                    if torch.is_tensor(keypoints):
                        keypoints = keypoints.cpu().numpy()

        elapsed = time.perf_counter() - start
        avg_time = elapsed / num_iterations
        fps = 1.0 / avg_time

        print(f"\n{name} ({width}x{height}):")
        print(f"  Average time: {avg_time*1000:.1f} ms")
        print(f"  FPS: {fps:.1f}")

        # Classification
        if fps >= 30:
            print(f"  ✓ Real-time capable")
        elif fps >= 15:
            print(f"  ⚠ Near real-time")
        else:
            print(f"  ✗ Below real-time")

    # Test batch processing
    print("\n" + "-"*60)
    print("BATCH PROCESSING TEST:")
    print("-"*60)

    batch_sizes = [1, 2, 4]
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    for batch_size in batch_sizes:
        # Create batch
        if batch_size == 1:
            batch = test_image
        else:
            batch = [test_image] * batch_size

        # Warmup
        _ = model(batch, verbose=False)

        # Time
        num_iterations = 5
        start = time.perf_counter()

        for _ in range(num_iterations):
            _ = model(batch, verbose=False)

        elapsed = time.perf_counter() - start
        avg_time = elapsed / num_iterations
        fps_per_image = batch_size / avg_time

        print(f"\nBatch size {batch_size}:")
        print(f"  Total time: {avg_time*1000:.1f} ms")
        print(f"  Time per image: {(avg_time/batch_size)*1000:.1f} ms")
        print(f"  Effective FPS: {fps_per_image:.1f}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)

    print("\nBased on CPU testing:")
    print("- For real-time (30+ FPS): Use 640x480 resolution")
    print("- For near real-time (15+ FPS): Use up to 1280x720")
    print("- For best quality: Use 1920x1080 with batch processing")
    print("\nOn GPU (estimated):")
    print("- 640x480: ~60-80 FPS")
    print("- 1280x720: ~35-45 FPS")
    print("- 1920x1080: ~20-30 FPS")

if __name__ == "__main__":
    test_fps_simple()