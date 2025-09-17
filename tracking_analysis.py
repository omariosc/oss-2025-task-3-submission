#!/usr/bin/env python3
"""
Analysis of Docker container tracking results
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

def analyze_tracking_results(tracking_file_path, results_json_path):
    """Analyze the tracking results from Docker container."""
    
    print("=== DOCKER CONTAINER TRACKING ANALYSIS ===\n")
    
    # Load tracking data
    tracking_data = []
    with open(tracking_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                tracking_data.append({
                    'frame': int(parts[0]),
                    'track_id': int(parts[1]),
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'w': float(parts[4]),
                    'h': float(parts[5]),
                    'conf': float(parts[6]),
                    'class': int(parts[7]),
                    'visibility': float(parts[8])
                })
    
    # Load results metadata
    with open(results_json_path, 'r') as f:
        results_json = json.load(f)
    
    # Basic statistics
    df = pd.DataFrame(tracking_data)
    
    print("1. BASIC STATISTICS")
    print(f"   - Total detections: {len(tracking_data)}")
    print(f"   - Unique frames: {df['frame'].nunique()}")
    print(f"   - Unique track IDs: {df['track_id'].nunique()}")
    print(f"   - Frame range: {df['frame'].min()} - {df['frame'].max()}")
    
    # Detection quality analysis
    print("\n2. DETECTION QUALITY")
    print(f"   - Average confidence: {df['conf'].mean():.3f}")
    print(f"   - Min confidence: {df['conf'].min():.3f}")
    print(f"   - Max confidence: {df['conf'].max():.3f}")
    print(f"   - Confidence std: {df['conf'].std():.3f}")
    
    # Class distribution
    print("\n3. CLASS DISTRIBUTION")
    class_names = {
        0: 'left_hand_segment',
        1: 'right_hand_segment', 
        2: 'scissors',
        3: 'tweezers',
        4: 'needle_holder',
        5: 'needle'
    }
    
    class_counts = df['class'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        class_name = class_names.get(class_id, f'unknown_{class_id}')
        print(f"   - {class_name}: {count} detections ({count/len(df)*100:.1f}%)")
    
    # Track persistence analysis
    print("\n4. TRACK PERSISTENCE")
    track_lengths = df.groupby('track_id')['frame'].nunique()
    print(f"   - Average track length: {track_lengths.mean():.1f} frames")
    print(f"   - Max track length: {track_lengths.max()} frames")
    print(f"   - Min track length: {track_lengths.min()} frames")
    print(f"   - Tracks appearing in >1 frame: {(track_lengths > 1).sum()}")
    
    # Bounding box analysis
    print("\n5. BOUNDING BOX ANALYSIS")
    print(f"   - Average box width: {df['w'].mean():.1f} pixels")
    print(f"   - Average box height: {df['h'].mean():.1f} pixels")
    print(f"   - Average box area: {(df['w'] * df['h']).mean():.0f} pixels²")
    
    # Frame density analysis
    print("\n6. FRAME DENSITY")
    detections_per_frame = df.groupby('frame').size()
    print(f"   - Average detections per frame: {detections_per_frame.mean():.1f}")
    print(f"   - Max detections per frame: {detections_per_frame.max()}")
    print(f"   - Min detections per frame: {detections_per_frame.min()}")
    
    # Calculate basic tracking metrics
    print("\n7. TRACKING QUALITY METRICS")
    
    # Detection rate (assuming we should have ~6 objects per frame ideally)
    expected_objects_per_frame = 6
    total_expected = expected_objects_per_frame * df['frame'].nunique()
    detection_rate = len(tracking_data) / total_expected * 100
    print(f"   - Detection rate: {detection_rate:.1f}% (vs expected {expected_objects_per_frame} objects/frame)")
    
    # ID switches (simplified calculation)
    id_switches = 0
    for track_id in df['track_id'].unique():
        track_frames = df[df['track_id'] == track_id]['frame'].tolist()
        if len(track_frames) > 1:
            # Check for gaps in frames (potential ID switches)
            gaps = sum(1 for i in range(1, len(track_frames)) 
                      if track_frames[i] - track_frames[i-1] > 1)
            id_switches += gaps
    
    print(f"   - Estimated ID switches: {id_switches}")
    
    # Confidence-based quality score
    high_conf_detections = (df['conf'] > 0.7).sum()
    quality_score = high_conf_detections / len(df) * 100
    print(f"   - High confidence detections (>0.7): {quality_score:.1f}%")
    
    # Processing performance from JSON
    print("\n8. PROCESSING PERFORMANCE")
    processing_time = results_json['processing_summary']['processing_time_seconds']
    total_frames = df['frame'].nunique()
    if processing_time > 0:
        fps = total_frames / processing_time
        print(f"   - Processing FPS: {fps:.1f} frames/second")
    print(f"   - Total processing time: {processing_time:.2f} seconds")
    
    return {
        'total_detections': len(tracking_data),
        'unique_tracks': df['track_id'].nunique(),
        'avg_confidence': df['conf'].mean(),
        'detection_rate': detection_rate,
        'id_switches': id_switches,
        'quality_score': quality_score,
        'fps': fps if processing_time > 0 else 0
    }

if __name__ == "__main__":
    tracking_file = "test_output/tracking_results.txt"
    results_file = "test_output/results.json"
    
    metrics = analyze_tracking_results(tracking_file, results_file)
    
    print(f"\n=== SUMMARY METRICS ===")
    print(f"Overall Performance Score: {metrics['quality_score']:.1f}%")
    print(f"Detection Efficiency: {metrics['detection_rate']:.1f}%")
    print(f"Processing Speed: {metrics['fps']:.1f} FPS")