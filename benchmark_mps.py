#!/usr/bin/env python3
"""
MPS-enabled benchmark for YOLO-Pose model
Tests end-to-end performance from video input to MOT output
"""

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def test_mps_performance():
    """Test end-to-end performance with MPS acceleration"""

    print("="*60)
    print("YOLO-POSE END-TO-END BENCHMARK WITH MPS")
    print("="*60)

    # Check available devices
    print("\nDevice Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Select best available device
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"⚠ Using CPU")

    # Load model
    model_path = Path("surgical_keypoints_best.pt")
    if not model_path.exists():
        print(f"Downloading YOLOv8m-pose model...")
        model = YOLO("yolov8m-pose.pt")
    else:
        print(f"Loading model: {model_path}")
        model = YOLO(str(model_path))

    # Set device
    model.to(device)
    print(f"Model loaded on: {device}")

    # Find test video
    test_video = None
    test_videos = list(Path("test_input").glob("*.mp4"))
    if test_videos:
        test_video = test_videos[0]
        print(f"\nTest video: {test_video}")
    else:
        print("No test video found, using synthetic frames")

    # Warmup
    print("\nWarming up model...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy, verbose=False, device=device)

    print("\n" + "-"*60)
    print("END-TO-END PIPELINE TEST")
    print("-"*60)

    if test_video:
        # Open video
        cap = cv2.VideoCapture(str(test_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        print(f"Total frames: {total_frames}")

        # Process limited frames for quick test
        max_frames = min(100, total_frames)
        print(f"Testing {max_frames} frames...")

        # Track timing
        frame_times = []
        total_detections = 0
        mot_output = []

        # Start total timing
        pipeline_start = time.perf_counter()

        for frame_idx in range(max_frames):
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Time single frame processing
            frame_start = time.perf_counter()

            # Run inference
            results = model(frame, verbose=False, device=device, conf=0.25)

            # Process results to MOT format
            if results and len(results) > 0:
                result = results[0]

                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for i, box in enumerate(boxes.data):
                        x1, y1, x2, y2 = box[:4].cpu().numpy()
                        conf = box[4].cpu().numpy() if len(box) > 4 else 1.0
                        cls = int(box[5].cpu().numpy()) if len(box) > 5 else 0

                        # MOT format: frame,id,x,y,w,h,conf,class,vis
                        mot_entry = f"{frame_idx+1},{i+1},{x1:.1f},{y1:.1f},{x2-x1:.1f},{y2-y1:.1f},{conf:.3f},{cls},1.0"
                        mot_output.append(mot_entry)
                        total_detections += 1

            # Record frame time
            frame_time = time.perf_counter() - frame_start
            frame_times.append(frame_time)

            # Progress indicator
            if (frame_idx + 1) % 20 == 0:
                avg_fps = 1.0 / np.mean(frame_times)
                print(f"  Processed {frame_idx+1}/{max_frames} frames | Avg FPS: {avg_fps:.1f}")

        # Total pipeline time
        pipeline_time = time.perf_counter() - pipeline_start

        cap.release()

        # Calculate statistics
        avg_frame_time = np.mean(frame_times)
        std_frame_time = np.std(frame_times)
        min_frame_time = np.min(frame_times)
        max_frame_time = np.max(frame_times)

        effective_fps = len(frame_times) / pipeline_time
        avg_fps = 1.0 / avg_frame_time

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        print(f"\nDevice: {device.upper()}")
        print(f"Resolution: {width}x{height}")
        print(f"Frames processed: {len(frame_times)}")
        print(f"Total detections: {total_detections}")
        print(f"Avg detections/frame: {total_detections/len(frame_times):.1f}")

        print("\nTiming (per frame):")
        print(f"  Mean: {avg_frame_time*1000:.1f} ms")
        print(f"  Std:  {std_frame_time*1000:.1f} ms")
        print(f"  Min:  {min_frame_time*1000:.1f} ms")
        print(f"  Max:  {max_frame_time*1000:.1f} ms")

        print(f"\nFPS Performance:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Effective FPS: {effective_fps:.1f}")
        print(f"  Total pipeline: {pipeline_time:.2f} seconds")
        print(f"  Time per frame: {avg_frame_time*1000:.1f} ms")

        # Performance classification
        print("\nPerformance Rating:")
        if avg_fps >= 30:
            print(f"  ✓ REAL-TIME: {avg_fps:.1f} FPS")
        elif avg_fps >= 15:
            print(f"  ⚠ NEAR REAL-TIME: {avg_fps:.1f} FPS")
        else:
            print(f"  ✗ BELOW REAL-TIME: {avg_fps:.1f} FPS")

        # Save sample MOT output
        if mot_output:
            mot_file = Path("benchmark_mot_output.txt")
            with open(mot_file, 'w') as f:
                f.write('\n'.join(mot_output[:50]))  # Save first 50 entries
            print(f"\nSample MOT output saved to: {mot_file}")

    # Test different resolutions
    print("\n" + "-"*60)
    print("RESOLUTION COMPARISON")
    print("-"*60)

    resolutions = [(640, 480), (960, 540), (1280, 720), (1920, 1080)]

    for width, height in resolutions:
        # Create test frame
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Time 20 iterations
        times = []
        for _ in range(20):
            start = time.perf_counter()
            _ = model(test_frame, verbose=False, device=device)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times[5:])  # Skip first 5 for stability
        fps = 1.0 / avg_time

        print(f"\n{width}x{height}:")
        print(f"  Time: {avg_time*1000:.1f} ms")
        print(f"  FPS: {fps:.1f}")

        if fps >= 30:
            print(f"  ✓ Real-time")
        elif fps >= 15:
            print(f"  ⚠ Near real-time")
        else:
            print(f"  ✗ Below real-time")

if __name__ == "__main__":
    test_mps_performance()