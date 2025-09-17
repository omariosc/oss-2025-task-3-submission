#!/usr/bin/env python3
"""
FPS Benchmark Script for YOLO-Pose Model
Tests end-to-end inference speed from input to output
"""

import cv2
import torch
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import json

def benchmark_yolo_pose(video_path=None, model_path=None, num_frames=100):
    """
    Benchmark YOLO-Pose model FPS with complete pipeline
    Includes: preprocessing, inference, postprocessing
    """

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}")

    # Load model
    if model_path is None:
        model_path = "surgical_keypoints_best.pt"
        if not Path(model_path).exists():
            model_path = "yolov8m-pose.pt"

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Warmup GPU
    if device == 'cuda':
        print("Warming up GPU...")
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _ = model(dummy_img, verbose=False)
        torch.cuda.synchronize()

    # Prepare video or test frames
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_frames)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"Testing {total_frames} frames")
    else:
        # Generate test frames
        print("Using synthetic test frames")
        total_frames = num_frames
        frame_width, frame_height = 1280, 720
        cap = None

    # Benchmark metrics
    frame_times = []
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    total_keypoints = 0
    total_detections = 0

    print("\nStarting benchmark...")
    pbar = tqdm(total=total_frames, desc="Processing frames")

    for frame_idx in range(total_frames):
        # Get frame
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate synthetic frame
            frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)

        # Start total timing
        start_total = time.perf_counter()

        # Preprocessing timing
        start_prep = time.perf_counter()
        # YOLO handles preprocessing internally, but we measure the overhead
        prep_frame = frame.copy()  # Simulate any preprocessing
        preprocessing_time = time.perf_counter() - start_prep

        # Inference timing (includes YOLO's internal preprocessing)
        start_inf = time.perf_counter()
        if device == 'cuda':
            torch.cuda.synchronize()

        # Run inference
        results = model(prep_frame, verbose=False, conf=0.25, iou=0.45)

        if device == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_inf

        # Postprocessing timing
        start_post = time.perf_counter()

        # Extract keypoints and convert to tracking format
        if results and len(results) > 0:
            result = results[0]

            # Process keypoints
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data
                num_detections = len(keypoints_data)
                num_keypoints = keypoints_data.shape[1] if len(keypoints_data) > 0 else 0

                total_detections += num_detections
                total_keypoints += num_detections * num_keypoints

                # Simulate MOT format conversion
                for det_idx in range(num_detections):
                    kps = keypoints_data[det_idx]
                    visible_kps = kps[kps[:, 2] > 0.3]

                    if len(visible_kps) > 0:
                        # Calculate bounding box from keypoints
                        x_coords = visible_kps[:, 0].cpu().numpy()
                        y_coords = visible_kps[:, 1].cpu().numpy()
                        x1, x2 = np.min(x_coords), np.max(x_coords)
                        y1, y2 = np.min(y_coords), np.max(y_coords)

                        # MOT format: frame,id,x,y,w,h,conf,class,vis
                        mot_entry = f"{frame_idx},{det_idx},{x1},{y1},{x2-x1},{y2-y1},0.8,1,1.0"

        postprocessing_time = time.perf_counter() - start_post

        # Total frame time
        total_time = time.perf_counter() - start_total

        # Store timings
        frame_times.append(total_time)
        preprocessing_times.append(preprocessing_time)
        inference_times.append(inference_time)
        postprocessing_times.append(postprocessing_time)

        pbar.update(1)

    pbar.close()

    if cap:
        cap.release()

    # Calculate statistics
    results = {
        "device": device,
        "model": str(model_path),
        "frame_resolution": f"{frame_width}x{frame_height}",
        "total_frames": len(frame_times),
        "total_detections": total_detections,
        "total_keypoints": total_keypoints,
        "timings": {
            "preprocessing_ms": {
                "mean": np.mean(preprocessing_times) * 1000,
                "std": np.std(preprocessing_times) * 1000,
                "min": np.min(preprocessing_times) * 1000,
                "max": np.max(preprocessing_times) * 1000
            },
            "inference_ms": {
                "mean": np.mean(inference_times) * 1000,
                "std": np.std(inference_times) * 1000,
                "min": np.min(inference_times) * 1000,
                "max": np.max(inference_times) * 1000
            },
            "postprocessing_ms": {
                "mean": np.mean(postprocessing_times) * 1000,
                "std": np.std(postprocessing_times) * 1000,
                "min": np.min(postprocessing_times) * 1000,
                "max": np.max(postprocessing_times) * 1000
            },
            "total_ms": {
                "mean": np.mean(frame_times) * 1000,
                "std": np.std(frame_times) * 1000,
                "min": np.min(frame_times) * 1000,
                "max": np.max(frame_times) * 1000
            }
        },
        "fps": {
            "mean": 1.0 / np.mean(frame_times),
            "std": 1.0 / np.mean(frame_times) - 1.0 / (np.mean(frame_times) + np.std(frame_times)),
            "min": 1.0 / np.max(frame_times),
            "max": 1.0 / np.min(frame_times)
        }
    }

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Device: {results['device']}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: {results['model']}")
    print(f"Resolution: {results['frame_resolution']}")
    print(f"Frames processed: {results['total_frames']}")
    print(f"Total detections: {results['total_detections']}")
    print(f"Total keypoints: {results['total_keypoints']}")
    print(f"Avg detections/frame: {results['total_detections']/results['total_frames']:.1f}")

    print("\n" + "-"*60)
    print("TIMING BREAKDOWN (ms per frame):")
    print("-"*60)
    print(f"Preprocessing:   {results['timings']['preprocessing_ms']['mean']:.2f} ± {results['timings']['preprocessing_ms']['std']:.2f}")
    print(f"Inference:       {results['timings']['inference_ms']['mean']:.2f} ± {results['timings']['inference_ms']['std']:.2f}")
    print(f"Postprocessing:  {results['timings']['postprocessing_ms']['mean']:.2f} ± {results['timings']['postprocessing_ms']['std']:.2f}")
    print(f"TOTAL:          {results['timings']['total_ms']['mean']:.2f} ± {results['timings']['total_ms']['std']:.2f}")

    print("\n" + "-"*60)
    print("FPS (Frames Per Second):")
    print("-"*60)
    print(f"Mean FPS:    {results['fps']['mean']:.1f}")
    print(f"Min FPS:     {results['fps']['min']:.1f}")
    print(f"Max FPS:     {results['fps']['max']:.1f}")

    print("\n" + "-"*60)
    print("PERFORMANCE SUMMARY:")
    print("-"*60)

    # Determine actual performance tier
    mean_fps = results['fps']['mean']
    if mean_fps >= 60:
        print(f"✓ REAL-TIME+ Performance: {mean_fps:.1f} FPS")
    elif mean_fps >= 30:
        print(f"✓ REAL-TIME Performance: {mean_fps:.1f} FPS")
    elif mean_fps >= 15:
        print(f"⚠ NEAR REAL-TIME Performance: {mean_fps:.1f} FPS")
    else:
        print(f"✗ BELOW REAL-TIME Performance: {mean_fps:.1f} FPS")

    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO-Pose FPS')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, help='Path to YOLO model')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to test')

    args = parser.parse_args()

    # Find a test video if not specified
    if not args.video:
        test_videos = list(Path("test_input").glob("*.mp4"))
        if test_videos:
            args.video = str(test_videos[0])
            print(f"Using test video: {args.video}")

    # Run benchmark
    benchmark_yolo_pose(
        video_path=args.video,
        model_path=args.model,
        num_frames=args.frames
    )

if __name__ == "__main__":
    main()