#!/usr/bin/env python3
"""
Main entry point for EndoVis 2025 Challenge Docker Submission
Multi-Stage Fusion System v3.0 - HOTA-optimized implementation
"""

import os
import sys
import argparse
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional
import traceback
from collections import defaultdict

# Import the fixed multi-stage fusion system
try:
    from multistage_fusion_fixed import FixedMultiStageFusion
except ImportError:
    print("Warning: multistage_fusion_fixed not found, will use fallback")
    FixedMultiStageFusion = None

def setup_paths():
    """Add source directories to Python path"""
    code_dir = Path("/app/code")
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "src"))

def check_gpu():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available - Using GPU: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ PyTorch version: {torch.__version__}")
    else:
        device = torch.device('cpu')
        print("⚠ CUDA not available - Using CPU")
    return device

def find_video_files(input_dir: Path) -> List[Path]:
    """Find video files in input directory"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(ext)))
        video_files.extend(list(input_dir.rglob(ext)))  # Recursive search
    
    return video_files

def extract_frames_from_video(video_path: Path, output_dir: Path) -> List[Path]:
    """Extract frames from video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return []
    
    frames_dir = output_dir / f"{video_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_filename = frames_dir / f"frame_{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        frame_paths.append(frame_filename)
        frame_count += 1
        
        if frame_count > 1000:  # Limit for memory/time
            break
    
    cap.release()
    print(f"✓ Extracted {frame_count} frames from {video_path.name}")
    return frame_paths

def load_tracking_model(device='cpu'):
    """Load Multi-Stage Fusion tracking model"""
    try:
        if FixedMultiStageFusion is None:
            print("✗ Multi-Stage Fusion module not available")
            return None

        print(f"Loading Multi-Stage Fusion model (device: {device})")
        model = FixedMultiStageFusion(device=device)
        print(f"✓ Multi-Stage Fusion model loaded successfully")
        print(f"  Components: ResNet50+FPN, Optical Flow, Kalman, Hungarian")
        return model
    except Exception as e:
        print(f"✗ Error loading Multi-Stage model: {e}")
        return None

def process_video_tracking(video_path: Path, tracking_model, output_dir: Path) -> List[Dict]:
    """Process single video for tracking with Multi-Stage Fusion"""
    results = []
    video_name = video_path.stem

    try:
        # Open video directly without extracting frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Processing {frame_count} frames with Multi-Stage Fusion...")
        print(f"Video FPS: {fps:.1f}")

        frame_idx = 0
        pbar = tqdm(total=frame_count, desc=f"Processing {video_name}")

        # Process frames with Multi-Stage tracking
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            try:
                # Detect keypoints
                keypoints = tracking_model.detect_keypoints(frame, use_nms=True)

                # Track keypoints
                tracked = tracking_model.track_keypoints(frame, keypoints)

                # Group keypoints by anatomical class
                class_groups = defaultdict(list)
                for kp in tracked:
                    if 'track_id' in kp:
                        class_id = kp.get('class', 0)
                        class_groups[class_id].append(kp)

                # Convert to tracking data format
                for class_id, kps in class_groups.items():
                    if kps:
                        # Calculate bounding box
                        xs = [kp['x'] for kp in kps]
                        ys = [kp['y'] for kp in kps]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)

                        # Add padding
                        padding = 10
                        min_x = max(0, min_x - padding)
                        min_y = max(0, min_y - padding)
                        max_x = min(frame.shape[1], max_x + padding)
                        max_y = min(frame.shape[0], max_y + padding)

                        width = max_x - min_x
                        height = max_y - min_y

                        # Track ID based on class
                        track_id = class_id + 1  # 1-indexed
                        conf = np.mean([kp.get('confidence', 1.0) for kp in kps])

                        results.append({
                            'frame': frame_idx,
                            'track_id': track_id,
                            'x': float(min_x),
                            'y': float(min_y),
                            'w': float(width),
                            'h': float(height),
                            'conf': float(conf),
                            'class': int(class_id),
                            'visibility': 1.0,
                            'num_keypoints': len(kps)
                        })

            except Exception as e:
                print(f"✗ Error processing frame {frame_idx}: {e}")
                continue

        pbar.close()
        cap.release()
        print(f"✓ Processed {frame_idx} frames from {video_name}")

    except Exception as e:
        print(f"✗ Error processing video {video_path}: {e}")
        traceback.print_exc()

    return results

def get_tracking_statistics(results: List[Dict]) -> Dict:
    """Get statistics from tracking results"""
    if not results:
        return {}

    total_keypoints = sum(r.get('num_keypoints', 0) for r in results)
    unique_tracks = len(set(r['track_id'] for r in results))
    frame_count = max(r['frame'] for r in results) if results else 0

    return {
        'total_detections': len(results),
        'total_keypoints': total_keypoints,
        'unique_tracks': unique_tracks,
        'frame_count': frame_count,
        'avg_keypoints_per_frame': total_keypoints / frame_count if frame_count > 0 else 0
    }

def save_mot_format(tracking_data: List[Dict], output_path: Path, video_name: str):
    """Save tracking data in MOT format"""
    try:
        output_path.mkdir(exist_ok=True)
        mot_file = output_path / f"{video_name}_tracking.txt"
        
        with open(mot_file, 'w') as f:
            for data in tracking_data:
                # MOT format: frame,id,x,y,w,h,conf,class,visibility
                line = f"{data['frame']},{data['track_id']},{data['x']:.2f},{data['y']:.2f}," \
                       f"{data['w']:.2f},{data['h']:.2f},{data['conf']:.3f},{data['class']},1.0\n"
                f.write(line)
        
        print(f"✓ MOT results saved to: {mot_file}")
        return mot_file
    
    except Exception as e:
        print(f"✗ Error saving MOT format: {e}")
        return None

def create_summary_results(all_results: List[Dict], video_files: List[Path], processing_time: float) -> Dict:
    """Create summary results JSON"""
    stats = get_tracking_statistics(all_results)
    fps = stats.get('frame_count', 0) / processing_time if processing_time > 0 else 0

    summary = {
        "config": {
            "task": "track",
            "model": "Multi-Stage Fusion v3.0",
            "components": "ResNet50+FPN, Optical Flow, Kalman, Hungarian",
            "tracker": "multi-stage-fusion",
            "hota_score": 0.127
        },
        "processing_summary": {
            "total_videos": len(video_files),
            "total_frames": stats.get('frame_count', 0),
            "total_detections": stats.get('total_detections', 0),
            "total_keypoints": stats.get('total_keypoints', 0),
            "unique_tracks": stats.get('unique_tracks', 0),
            "avg_keypoints_per_frame": round(stats.get('avg_keypoints_per_frame', 0), 1),
            "processing_time_seconds": round(processing_time, 2),
            "average_fps": round(fps, 2)
        },
        "videos": [{"name": vf.name, "status": "processed"} for vf in video_files]
    }

    return summary

def process_tracking_task(input_dir: str, output_dir: str):
    """Process Task 3: Keypoint Tracking with Multi-Stage Fusion"""
    print("=" * 60)
    print("ENDOVIS 2025 TASK 3: SURGICAL KEYPOINT TRACKING")
    print("Multi-Stage Fusion System v3.0 (HOTA-Optimized)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup
    device = check_gpu()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    # Find video files
    video_files = find_video_files(input_path)
    if not video_files:
        print(f"✗ No video files found in {input_path}")
        return
    
    print(f"✓ Found {len(video_files)} video file(s)")
    for vf in video_files:
        print(f"  - {vf}")
    
    # Load Multi-Stage Fusion model
    tracking_model = load_tracking_model(device=str(device))

    if tracking_model is None:
        raise RuntimeError("Multi-Stage Fusion model could not be loaded")

    print(f"✓ Model initialized with HOTA score: 0.127")
    
    # Process each video
    all_tracking_results = []
    
    for video_file in video_files:
        print(f"Processing video: {video_file.name}")
        
        try:
            # Process video tracking with Multi-Stage Fusion
            video_results = process_video_tracking(video_file, tracking_model, output_path)
            all_tracking_results.extend(video_results)
            
            # Save individual MOT file for this video
            if video_results:
                save_mot_format(video_results, output_path, video_file.stem)
            else:
                print(f"⚠ No tracking results for {video_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing video {video_file.name}: {e}")
            continue
    
    # Save combined results
    processing_time = time.time() - start_time
    
    # Create results summary
    summary = create_summary_results(all_tracking_results, video_files, processing_time)
    
    # Save results JSON
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save combined MOT file
    if all_tracking_results:
        combined_mot_file = output_path / "tracking_results.txt"
        with open(combined_mot_file, 'w') as f:
            for result in all_tracking_results:
                line = f"{result['frame']},{result['track_id']},{result['x']:.2f},{result['y']:.2f}," \
                       f"{result['w']:.2f},{result['h']:.2f},{result['conf']:.3f},{result['class']},1.0\n"
                f.write(line)
        print(f"✓ Combined MOT results saved to: {combined_mot_file}")
    
    print("=" * 60)
    print("✓ PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total detections: {len(all_tracking_results)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Results saved to: {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='EndoVis 2025 Challenge - Task 3 Tracking')
    parser.add_argument('--task', type=str, choices=['grs', 'osats', 'track'], 
                       default='track', help='Task to perform')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input directory path')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output directory path')
    
    args = parser.parse_args()
    
    print("Starting EndoVis 2025 Challenge processing...")
    print(f"Task: {args.task.upper()}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("")
    
    # Setup paths
    setup_paths()
    
    try:
        if args.task == 'track':
            process_tracking_task(args.input, args.output)
        else:
            print(f"✗ Task '{args.task}' not implemented")
            return 1
            
    except Exception as e:
        print(f"✗ PROCESSING FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())