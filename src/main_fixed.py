#!/usr/bin/env python3
"""
Main entry point for EndoVis 2025 Challenge Docker Submission
Fixed version with working YOLO + BoT-SORT tracking pipeline
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
from ultralytics import YOLO
import traceback

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

def load_yolo_model(model_path: str):
    """Load YOLO model"""
    try:
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            return None
            
        print(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        print(f"✓ YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading YOLO model: {e}")
        return None

def process_video_tracking(video_path: Path, yolo_model, output_dir: Path) -> List[Dict]:
    """Process single video for tracking"""
    results = []
    video_name = video_path.stem
    
    try:
        # Extract frames
        temp_frames_dir = output_dir / "temp_frames"
        temp_frames_dir.mkdir(exist_ok=True)
        
        frame_paths = extract_frames_from_video(video_path, temp_frames_dir)
        if not frame_paths:
            print(f"✗ No frames extracted from {video_path}")
            return []
        
        print(f"Processing {len(frame_paths)} frames with YOLO tracking...")
        
        # Process frames with YOLO tracking
        for i, frame_path in enumerate(tqdm(frame_paths, desc=f"Processing {video_name}")):
            try:
                # Run YOLO tracking on frame
                track_results = yolo_model.track(
                    source=str(frame_path),
                    persist=True,
                    tracker="botsort.yaml",
                    conf=0.3,
                    iou=0.5,
                    verbose=False
                )
                
                # Extract tracking information
                frame_data = extract_tracking_data(track_results, i + 1)
                if frame_data:
                    results.extend(frame_data)
                    
            except Exception as e:
                print(f"✗ Error processing frame {i}: {e}")
                continue
        
        # Cleanup temp frames
        import shutil
        if temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)
            
    except Exception as e:
        print(f"✗ Error processing video {video_path}: {e}")
        traceback.print_exc()
    
    return results

def extract_tracking_data(track_results, frame_num: int) -> List[Dict]:
    """Extract tracking data from YOLO results"""
    tracking_data = []
    
    try:
        if track_results and len(track_results) > 0:
            result = track_results[0]  # First result
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # Handle track IDs
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy()
                else:
                    track_ids = np.arange(len(boxes)) + 1
                
                # Handle confidences
                if result.boxes.conf is not None:
                    confidences = result.boxes.conf.cpu().numpy()
                else:
                    confidences = np.ones(len(boxes))
                
                # Handle classes
                if result.boxes.cls is not None:
                    classes = result.boxes.cls.cpu().numpy()
                else:
                    classes = np.zeros(len(boxes))
                
                for i, (box, track_id, conf, cls) in enumerate(zip(boxes, track_ids, confidences, classes)):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    tracking_data.append({
                        'frame': frame_num,
                        'track_id': int(track_id),
                        'x': float(x1),
                        'y': float(y1),
                        'w': float(w),
                        'h': float(h),
                        'conf': float(conf),
                        'class': int(cls),
                        'visibility': 1.0
                    })
    
    except Exception as e:
        print(f"✗ Error extracting tracking data from frame {frame_num}: {e}")
    
    return tracking_data

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
    total_detections = len(all_results)
    unique_track_ids = set()
    frame_count = 0
    
    for result in all_results:
        unique_track_ids.add(result['track_id'])
        frame_count = max(frame_count, result['frame'])
    
    fps = frame_count / processing_time if processing_time > 0 else 0
    
    summary = {
        "config": {
            "task": "track",
            "model": "YOLO + BoT-SORT",
            "model_path": "/app/code/models/yolo_segment_enhanced.pt",
            "tracker": "botsort"
        },
        "processing_summary": {
            "total_videos": len(video_files),
            "total_frames": frame_count,
            "total_detections": total_detections,
            "unique_tracks": len(unique_track_ids),
            "processing_time_seconds": round(processing_time, 2),
            "average_fps": round(fps, 2)
        },
        "videos": [{"name": vf.name, "status": "processed"} for vf in video_files]
    }
    
    return summary

def process_tracking_task(input_dir: str, output_dir: str):
    """Process Task 3: Keypoint Tracking"""
    print("=" * 60)
    print("ENDOVIS 2025 TASK 3: SURGICAL KEYPOINT TRACKING")
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
    
    # Load YOLO model - try multiple model options
    model_paths = [
        "/app/code/models/yolo_segment_enhanced.pt",
        "/app/code/models/yolo11m_detection.pt",
        "/app/code/models/yolo_keypoints.pt"
    ]
    
    yolo_model = None
    for model_path in model_paths:
        yolo_model = load_yolo_model(model_path)
        if yolo_model is not None:
            print(f"✓ Using model: {model_path}")
            break
    
    if yolo_model is None:
        raise RuntimeError("No YOLO model could be loaded")
    
    # Process each video
    all_tracking_results = []
    
    for video_file in video_files:
        print(f"Processing video: {video_file.name}")
        
        try:
            # Process video tracking
            video_results = process_video_tracking(video_file, yolo_model, output_path)
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