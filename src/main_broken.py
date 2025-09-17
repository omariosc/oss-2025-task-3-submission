#!/usr/bin/env python3
"""
Main entry point for EndoVis 2025 Challenge Docker Submission
Supports Task 3 (Tracking) with ultra-dense keypoint detection and HOTA-optimized tracking
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
import yaml
import time
from typing import Dict, List, Tuple, Optional

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

def validate_directories(input_dir: str, output_dir: str) -> Tuple[Path, Path]:
    """Validate input and output directories"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = list(input_path.glob("*.mp4"))
    if not video_files:
        print(f"⚠ No MP4 files found in {input_path}")
        # Also check for image sequences
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        if image_files:
            print(f"✓ Found {len(image_files)} image files")
        else:
            raise FileNotFoundError("No video or image files found in input directory")
    else:
        print(f"✓ Found {len(video_files)} MP4 files")
    
    return input_path, output_path

def load_tracking_pipeline():
    """Load the surgical keypoint tracking pipeline"""
    try:
        from pipeline import SurgicalKeypointPipeline, PipelineConfig
        print("✓ Successfully imported tracking pipeline")
        return SurgicalKeypointPipeline, PipelineConfig
    except ImportError as e:
        print(f"✗ Failed to import tracking pipeline: {e}")
        # Fallback to TrackFormer approach
        try:
            from trackformer_tracker import TrackFormerTracker
            print("✓ Using TrackFormer fallback")
            return TrackFormerTracker, None
        except ImportError:
            raise ImportError("Neither candidate nor TrackFormer pipeline available")

def create_pipeline_config(input_dir: str, output_dir: str) -> dict:
    """Create pipeline configuration"""
    config = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        
        # Detection settings - optimized for Docker environment
        'detection_method': 'segmentation_guided',  # More efficient than ultra_dense
        'segmentation_model': '/app/code/models/yolo11m_segmentation.pt',
        'yolo_model': '/app/code/models/yolo11m_detection.pt',
        
        # Grid sizes - reduced for Docker efficiency
        'grid_sizes': [(128, 96), (64, 48)],  # Smaller grids for speed
        'segmentation_weight': 15.0,
        
        # Tracking settings
        'tracker_type': 'hota_optimized',
        'track_thresh': 0.3,
        'match_thresh': 0.8,
        'track_buffer': 30,
        
        # Preprocessing
        'enable_preprocessing': True,
        'intensity_normalization': True,
        'clahe_enabled': True,
        'denoise_enabled': False,  # Disable for speed
        
        # Output settings
        'save_visualizations': False,  # Disable for Docker submission
        'save_videos': False,
        'evaluate': False,
    }
    return config

def process_tracking_task(input_dir: str, output_dir: str):
    """Process Task 3: Keypoint Tracking"""
    print("=" * 60)
    print("ENDOVIS 2025 TASK 3: SURGICAL KEYPOINT TRACKING")
    print("=" * 60)
    
    # Setup
    device = check_gpu()
    input_path, output_path = validate_directories(input_dir, output_dir)
    
    # Load pipeline
    PipelineClass, ConfigClass = load_tracking_pipeline()
    
    # Create configuration
    config_dict = create_pipeline_config(input_dir, output_dir)
    
    if ConfigClass:
        # Use candidate submission pipeline
        print("Using ultra-dense keypoint tracking pipeline...")
        config = ConfigClass(**config_dict)
        pipeline = PipelineClass(config)
        
        # Process dataset
        start_time = time.time()
        results = pipeline.process_dataset()
        processing_time = time.time() - start_time
        
        print(f"✓ Processing completed in {processing_time:.2f} seconds")
        print(f"✓ Results saved to: {output_path}")
        
        # Ensure MOT format output
        mot_output = output_path / "tracking_results.txt"
        if not mot_output.exists():
            # Convert results to MOT format if needed
            print("Converting results to MOT format...")
            pipeline.save_mot_format(results, str(mot_output))
            
    else:
        # Use TrackFormer fallback
        print("Using TrackFormer tracking...")
        model_path = "/app/code/models/trackformer_model.pth"
        tracker = PipelineClass(model_path)
        
        # Process all videos
        start_time = time.time()
        results = tracker.process_videos(input_path, output_path)
        processing_time = time.time() - start_time
        
        print(f"✓ Processing completed in {processing_time:.2f} seconds")
    
    # Validate output format
    validate_mot_output(output_path)

def validate_mot_output(output_dir: Path):
    """Validate that MOT format output exists and is correctly formatted"""
    expected_files = ["tracking_results.txt"]
    
    for filename in expected_files:
        output_file = output_dir / filename
        if output_file.exists():
            print(f"✓ Output file created: {filename}")
            
            # Check format
            with open(output_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    first_line = lines[0].strip()
                    parts = first_line.split(',')
                    if len(parts) >= 7:  # Frame,TrackID,ClassID,Bbox(4),KP(variable)
                        print(f"✓ MOT format validation passed ({len(lines)} detections)")
                    else:
                        print(f"⚠ MOT format may be incorrect (parts: {len(parts)})")
                else:
                    print("⚠ Output file is empty")
        else:
            print(f"✗ Missing output file: {filename}")

def process_grs_task(input_dir: str, output_dir: str):
    """Process Task 1: Global Rating Score (GRS) - Placeholder"""
    print("Task 1 (GRS) not implemented in this submission")
    # Create empty CSV for compatibility
    output_file = Path(output_dir) / "grs_results.csv"
    with open(output_file, 'w') as f:
        f.write("VIDEO,GRS\n")
    print(f"Created placeholder file: {output_file}")

def process_osats_task(input_dir: str, output_dir: str):
    """Process Task 2: OSATS Component Scores - Placeholder"""
    print("Task 2 (OSATS) not implemented in this submission")
    # Create empty CSV for compatibility
    output_file = Path(output_dir) / "osats_results.csv"
    with open(output_file, 'w') as f:
        f.write("VIDEO,OSATS_RESPECT,OSATS_MOTION,OSATS_INSTRUMENT,OSATS_SUTURE,OSATS_FLOW,OSATS_KNOWLEDGE,OSATS_PERFORMANCE,OSATSFINALQUALITY\n")
    print(f"Created placeholder file: {output_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="EndoVis 2025 Challenge Docker Submission")
    parser.add_argument("--task", choices=["grs", "osats", "track"], default="track",
                        help="Task type to execute")
    parser.add_argument("--input", required=True, help="Input directory path")
    parser.add_argument("--output", required=True, help="Output directory path")
    
    args = parser.parse_args()
    
    print(f"Starting EndoVis 2025 Challenge processing...")
    print(f"Task: {args.task.upper()}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Setup paths
    setup_paths()
    
    try:
        if args.task == "track":
            process_tracking_task(args.input, args.output)
        elif args.task == "grs":
            process_grs_task(args.input, args.output)
        elif args.task == "osats":
            process_osats_task(args.input, args.output)
        
        print()
        print("=" * 60)
        print("✓ PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print("✗ PROCESSING FAILED")
        print(f"Error: {e}")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()