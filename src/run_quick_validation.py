#!/usr/bin/env python3
"""
Quick Validation Evaluation - Processes limited frames for rapid results
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
import json
from datetime import datetime

# Add paths
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3/candidate_submission/src')

def evaluate_all_videos():
    """Run quick evaluation on all validation videos"""

    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    val_frames_dir = data_root / "val/frames"
    masks_dir = data_root / "class_masks"
    output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/quick_validation_results")
    output_dir.mkdir(exist_ok=True)

    # Get all videos
    video_ids = sorted([d.name for d in val_frames_dir.iterdir() if d.is_dir()])
    print(f"Found {len(video_ids)} videos: {video_ids}")

    # Results storage
    results = {
        'multistage': {},
        'yolo': {},
        'timestamp': datetime.now().isoformat()
    }

    # Initialize markdown output
    md_file = output_dir / "quick_results.md"
    with open(md_file, 'w') as f:
        f.write("# Quick Validation Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Videos**: {len(video_ids)}\n")
        f.write(f"**Frames per video**: 3\n\n")

    # 1. EVALUATE MULTI-STAGE FUSION
    print("\n" + "="*60)
    print("EVALUATING MULTI-STAGE FUSION")
    print("="*60)

    try:
        from candidate_submission.src.keypoint_detector import UltraDenseKeypointDetector

        # Very fast config
        config = {
            'grid_sizes': [(64, 48)],  # Single small grid
            'segmentation_weight': 2.0,
            'nms_radius': 2,
            'confidence_threshold': 0.3
        }
        detector = UltraDenseKeypointDetector(config)

        with open(md_file, 'a') as f:
            f.write("## Multi-Stage Fusion Results\n\n")
            f.write("| Video | Frames | Keypoints | KP/Frame | Time(s) | FPS |\n")
            f.write("|-------|--------|-----------|----------|---------|-----|\n")

        multistage_total_kp = 0
        multistage_total_frames = 0
        multistage_total_time = 0

        for video_id in video_ids:
            print(f"\nProcessing {video_id}...")
            frames_dir = val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:3]  # Only 3 frames

            start_time = time.time()
            video_keypoints = 0
            frames_processed = 0

            for frame_file in frame_files:
                try:
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue

                    # Load masks
                    masks = {}
                    for class_name in ['left_hand_segment', 'right_hand_segment']:
                        mask_path = masks_dir / class_name / frame_file.name
                        if mask_path.exists():
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            masks[class_name] = mask if mask is not None else np.zeros((480, 640), dtype=np.uint8)
                        else:
                            masks[class_name] = np.zeros((480, 640), dtype=np.uint8)

                    keypoints = detector.detect(frame, masks)
                    video_keypoints += len(keypoints)
                    frames_processed += 1
                    print(f"  Frame {frames_processed}: {len(keypoints)} keypoints")

                except Exception as e:
                    print(f"  Frame error: {e}")
                    continue

            processing_time = time.time() - start_time
            fps = frames_processed / processing_time if processing_time > 0 else 0
            kp_per_frame = video_keypoints / frames_processed if frames_processed > 0 else 0

            results['multistage'][video_id] = {
                'frames': frames_processed,
                'keypoints': video_keypoints,
                'kp_per_frame': kp_per_frame,
                'time': processing_time,
                'fps': fps
            }

            multistage_total_kp += video_keypoints
            multistage_total_frames += frames_processed
            multistage_total_time += processing_time

            with open(md_file, 'a') as f:
                f.write(f"| {video_id} | {frames_processed} | {video_keypoints} | "
                       f"{kp_per_frame:.0f} | {processing_time:.2f} | {fps:.2f} |\n")

            print(f"  Result: {frames_processed} frames, {video_keypoints} keypoints, {fps:.2f} FPS")

    except Exception as e:
        print(f"Multi-Stage error: {e}")
        with open(md_file, 'a') as f:
            f.write(f"\n**Error**: {e}\n")

    # Write Multi-Stage summary
    with open(md_file, 'a') as f:
        f.write(f"\n### Multi-Stage Summary\n")
        f.write(f"- **Total Frames**: {multistage_total_frames}\n")
        f.write(f"- **Total Keypoints**: {multistage_total_kp}\n")
        f.write(f"- **Avg KP/Frame**: {multistage_total_kp/multistage_total_frames if multistage_total_frames > 0 else 0:.0f}\n")
        f.write(f"- **Total Time**: {multistage_total_time:.2f}s\n")
        f.write(f"- **Avg FPS**: {multistage_total_frames/multistage_total_time if multistage_total_time > 0 else 0:.2f}\n\n")

    # 2. EVALUATE YOLO
    print("\n" + "="*60)
    print("EVALUATING YOLO")
    print("="*60)

    try:
        from ultralytics import YOLO

        model_path = data_root / "yolo11m.pt"
        model = YOLO(str(model_path))

        with open(md_file, 'a') as f:
            f.write("## YOLO Results\n\n")
            f.write("| Video | Frames | Objects | Obj/Frame | Time(s) | FPS |\n")
            f.write("|-------|--------|---------|-----------|---------|-----|\n")

        yolo_total_obj = 0
        yolo_total_frames = 0
        yolo_total_time = 0

        for video_id in video_ids:
            print(f"\nProcessing {video_id}...")
            frames_dir = val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:3]  # Only 3 frames

            start_time = time.time()
            video_objects = 0
            frames_processed = 0

            for frame_file in frame_files:
                try:
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue

                    results_yolo = model(frame, verbose=False)

                    frame_objects = 0
                    for result in results_yolo:
                        if result.boxes is not None:
                            frame_objects += len(result.boxes)

                    video_objects += frame_objects
                    frames_processed += 1
                    print(f"  Frame {frames_processed}: {frame_objects} objects")

                except Exception as e:
                    print(f"  Frame error: {e}")
                    continue

            processing_time = time.time() - start_time
            fps = frames_processed / processing_time if processing_time > 0 else 0
            obj_per_frame = video_objects / frames_processed if frames_processed > 0 else 0

            results['yolo'][video_id] = {
                'frames': frames_processed,
                'objects': video_objects,
                'obj_per_frame': obj_per_frame,
                'time': processing_time,
                'fps': fps
            }

            yolo_total_obj += video_objects
            yolo_total_frames += frames_processed
            yolo_total_time += processing_time

            with open(md_file, 'a') as f:
                f.write(f"| {video_id} | {frames_processed} | {video_objects} | "
                       f"{obj_per_frame:.1f} | {processing_time:.2f} | {fps:.2f} |\n")

            print(f"  Result: {frames_processed} frames, {video_objects} objects, {fps:.2f} FPS")

    except Exception as e:
        print(f"YOLO error: {e}")
        with open(md_file, 'a') as f:
            f.write(f"\n**Error**: {e}\n")

    # Write YOLO summary
    with open(md_file, 'a') as f:
        f.write(f"\n### YOLO Summary\n")
        f.write(f"- **Total Frames**: {yolo_total_frames}\n")
        f.write(f"- **Total Objects**: {yolo_total_obj}\n")
        f.write(f"- **Avg Obj/Frame**: {yolo_total_obj/yolo_total_frames if yolo_total_frames > 0 else 0:.1f}\n")
        f.write(f"- **Total Time**: {yolo_total_time:.2f}s\n")
        f.write(f"- **Avg FPS**: {yolo_total_frames/yolo_total_time if yolo_total_time > 0 else 0:.2f}\n\n")

    # 3. COMPARISON
    with open(md_file, 'a') as f:
        f.write("## Comparison\n\n")

        if multistage_total_frames > 0 and yolo_total_frames > 0:
            ms_kp_per_frame = multistage_total_kp / multistage_total_frames
            yolo_obj_per_frame = yolo_total_obj / yolo_total_frames
            ms_fps = multistage_total_frames / multistage_total_time if multistage_total_time > 0 else 0
            yolo_fps = yolo_total_frames / yolo_total_time if yolo_total_time > 0 else 0

            f.write(f"### Key Metrics\n\n")
            f.write(f"| Metric | Multi-Stage | YOLO | Ratio |\n")
            f.write(f"|--------|-------------|------|-------|\n")
            f.write(f"| Features/Frame | {ms_kp_per_frame:.0f} | {yolo_obj_per_frame:.1f} | "
                   f"{ms_kp_per_frame/yolo_obj_per_frame if yolo_obj_per_frame > 0 else 0:.0f}x |\n")
            f.write(f"| Processing Speed (FPS) | {ms_fps:.2f} | {yolo_fps:.2f} | "
                   f"{yolo_fps/ms_fps if ms_fps > 0 else 0:.1f}x faster |\n")
            f.write(f"| Total Features | {multistage_total_kp} | {yolo_total_obj} | "
                   f"{multistage_total_kp/yolo_total_obj if yolo_total_obj > 0 else 0:.0f}x |\n")

            f.write(f"\n### Conclusions\n\n")
            f.write(f"- **Multi-Stage Fusion** detects **{ms_kp_per_frame/yolo_obj_per_frame if yolo_obj_per_frame > 0 else 0:.0f}x more features** than YOLO\n")
            f.write(f"- **YOLO** is **{yolo_fps/ms_fps if ms_fps > 0 else 0:.1f}x faster** than Multi-Stage Fusion\n")
            f.write(f"- **Multi-Stage** is better for detailed analysis (research)\n")
            f.write(f"- **YOLO** is better for real-time applications (production)\n")

    # Save JSON results
    json_file = output_dir / "quick_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to:")
    print(f"  - {md_file}")
    print(f"  - {json_file}")

    return results

if __name__ == "__main__":
    evaluate_all_videos()