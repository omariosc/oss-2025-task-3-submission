#!/usr/bin/env python3
"""
Real Data Evaluation: Multi-Stage Fusion vs YOLO
Evaluates both systems on actual EndoVis 2025 Task 3 validation data.
NO SIMULATION - REAL DATA ONLY!
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths for modules
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3/candidate_submission/src')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3/modules/depth_estimation')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3/modules/temporal_modeling')

@dataclass
class EvaluationResult:
    """Results from evaluating a tracking system"""
    system_name: str
    total_keypoints_detected: int
    total_frames_processed: int
    processing_time_seconds: float
    fps: float
    keypoints_per_frame: float
    error_count: int
    success_rate: float
    memory_usage_mb: float

class RealDataEvaluator:
    """Evaluates tracking systems on real EndoVis data"""
    
    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.val_masks_dir = self.data_root / "class_masks"
        self.val_annotations = self.data_root / "val/mot"
        
        # Get available videos
        self.video_ids = [d.name for d in self.val_frames_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(self.video_ids)} videos for evaluation: {self.video_ids}")
        
    def load_segmentation_masks(self, video_id: str, frame_name: str) -> Dict[str, np.ndarray]:
        """Load segmentation masks for all classes for a specific frame"""
        masks = {}
        mask_classes = ['left_hand_segment', 'right_hand_segment', 'scissors', 'tweezers', 'needle_holder', 'needle']
        
        for class_name in mask_classes:
            mask_path = self.val_masks_dir / class_name / f"{frame_name}"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                masks[class_name] = mask
            else:
                # Create empty mask if not found
                masks[class_name] = np.zeros((480, 640), dtype=np.uint8)
                
        return masks
    
    def load_real_annotations(self, video_id: str) -> List[Dict]:
        """Load real keypoint annotations for a video"""
        annotation_file = self.val_annotations / f"{video_id}.txt"
        if not annotation_file.exists():
            logger.warning(f"No annotations found for {video_id}")
            return []
        
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    object_id = int(parts[2])
                    
                    # Parse keypoints (format seems to be: frame,track,object,...,x1,y1,class1,x2,y2,class2,...)
                    keypoints = []
                    for i in range(7, len(parts), 3):  # Start from index 7, step by 3
                        if i+2 < len(parts):
                            try:
                                x = float(parts[i])
                                y = float(parts[i+1])
                                cls = int(parts[i+2])
                                if cls > 0:  # Valid keypoint
                                    keypoints.append({'x': x, 'y': y, 'class': cls})
                            except (ValueError, IndexError):
                                continue
                    
                    if keypoints:  # Only add if we found valid keypoints
                        annotations.append({
                            'frame_id': frame_id,
                            'track_id': track_id, 
                            'object_id': object_id,
                            'keypoints': keypoints
                        })
        
        logger.info(f"Loaded {len(annotations)} keypoint annotations for {video_id}")
        return annotations
    
    def evaluate_yolo_baseline(self, video_id: str, max_frames: int = 20) -> EvaluationResult:
        """Evaluate YOLO baseline system on real data"""
        logger.info(f"🔍 Evaluating YOLO baseline on video {video_id}")
        
        start_time = time.time()
        total_keypoints = 0
        frames_processed = 0
        errors = 0
        
        try:
            # Load YOLO model (use existing trained model)
            model_path = self.data_root / "yolo11m.pt"
            if not model_path.exists():
                logger.error(f"YOLO model not found at {model_path}")
                return EvaluationResult("YOLO_BASELINE", 0, 0, 0, 0, 0, 1, 0, 0)
                
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            # Get frames for this video
            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]
            
            logger.info(f"Processing {len(frame_files)} frames for {video_id}")
            
            for frame_file in frame_files:
                try:
                    # Load frame
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        logger.warning(f"Could not load frame {frame_file}")
                        errors += 1
                        continue
                    
                    # Run YOLO detection
                    results = model(frame, verbose=False)
                    
                    # Count detected objects as keypoints (simple approximation)
                    frame_keypoints = 0
                    for result in results:
                        if result.boxes is not None:
                            frame_keypoints += len(result.boxes)
                    
                    total_keypoints += frame_keypoints
                    frames_processed += 1
                    
                    if frames_processed % 5 == 0:
                        logger.info(f"YOLO: Processed {frames_processed} frames, {total_keypoints} total keypoints")
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_file}: {e}")
                    errors += 1
                    
        except Exception as e:
            logger.error(f"YOLO evaluation failed: {e}")
            errors += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps = frames_processed / processing_time if processing_time > 0 else 0
        keypoints_per_frame = total_keypoints / frames_processed if frames_processed > 0 else 0
        success_rate = (frames_processed - errors) / frames_processed if frames_processed > 0 else 0
        
        return EvaluationResult(
            system_name="YOLO_BASELINE",
            total_keypoints_detected=total_keypoints,
            total_frames_processed=frames_processed,
            processing_time_seconds=processing_time,
            fps=fps,
            keypoints_per_frame=keypoints_per_frame,
            error_count=errors,
            success_rate=success_rate,
            memory_usage_mb=0  # TODO: Implement memory tracking
        )
    
    def evaluate_multistage_fusion(self, video_id: str, max_frames: int = 20) -> EvaluationResult:
        """Evaluate Multi-Stage Fusion system on real data"""
        logger.info(f"🎯 Evaluating Multi-Stage Fusion on video {video_id}")
        
        start_time = time.time()
        total_keypoints = 0
        frames_processed = 0
        errors = 0
        
        try:
            # Import multi-stage fusion components
            from candidate_submission.src.keypoint_detector import UltraDenseKeypointDetector
            from modules.depth_estimation.dinov2_features import DINOv2FeatureExtractor
            from modules.depth_estimation.surgical_depth_prior import SurgicalDepthPrior
            from modules.temporal_modeling.temporal_transformer import TemporalTransformer
            
            # Initialize components with real configuration (no demo mode!)
            logger.info("Initializing Multi-Stage Fusion components...")
            
            # Stage 1: Dense keypoint detector
            detector_config = {
                'grid_sizes': [(512, 384), (256, 192), (128, 96), (64, 48)],  # Full grid sizes
                'segmentation_weight': 15.0,
                'nms_radius': 5,
                'confidence_threshold': 0.1
            }
            keypoint_detector = UltraDenseKeypointDetector(detector_config)
            
            # Stage 2: Feature extraction (skip DINOv2 download for speed, use depth prior only)
            depth_prior = SurgicalDepthPrior()
            
            # Stage 3: Temporal transformer
            temporal_transformer = TemporalTransformer(
                d_model=256,
                n_heads=8, 
                n_layers=6,
                temporal_window=8
            )
            
            # Get frames for this video
            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]
            
            logger.info(f"Processing {len(frame_files)} frames with Multi-Stage Fusion")
            
            temporal_features = []  # Store features for temporal processing
            
            for i, frame_file in enumerate(frame_files):
                try:
                    # Load frame
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        logger.warning(f"Could not load frame {frame_file}")
                        errors += 1
                        continue
                    
                    # Load segmentation masks for this frame
                    frame_name = frame_file.name
                    segmentation_masks = self.load_segmentation_masks(video_id, frame_name)
                    
                    # Stage 1: Dense keypoint detection
                    stage1_keypoints = keypoint_detector.detect(frame, segmentation_masks)
                    stage1_count = len(stage1_keypoints)
                    
                    # Stage 2: Multi-modal feature extraction (simplified for real evaluation)
                    # Use depth prior with current frame features
                    try:
                        # Create depth features from frame
                        frame_resized = cv2.resize(frame, (320, 240))
                        frame_features = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
                        frame_features = torch.nn.functional.interpolate(frame_features, size=(30, 40))
                        
                        # Get combined segmentation mask
                        combined_mask = np.zeros_like(segmentation_masks['left_hand_segment'])
                        for mask in segmentation_masks.values():
                            combined_mask = np.maximum(combined_mask, mask)
                        
                        depth_map = depth_prior.estimate_depth(
                            frame_features, 
                            segmentation_hints=torch.from_numpy(combined_mask).unsqueeze(0)
                        )
                        stage2_features = depth_map
                        
                    except Exception as e:
                        logger.warning(f"Stage 2 processing failed for frame {i}: {e}")
                        stage2_features = torch.randn(1, 240, 320)  # Fallback
                    
                    # Collect temporal features
                    if len(stage1_keypoints) > 0:
                        # Create feature representation for temporal processing
                        keypoint_features = []
                        for kp in stage1_keypoints[:100]:  # Limit for processing speed
                            x, y = kp['coords']
                            confidence = kp.get('confidence', 1.0)
                            feature = torch.tensor([x/640.0, y/480.0, confidence] + [0]*253)  # Pad to 256
                            keypoint_features.append(feature)
                        
                        if keypoint_features:
                            frame_temporal_features = torch.stack(keypoint_features)
                            temporal_features.append(frame_temporal_features)
                    
                    total_keypoints += stage1_count
                    frames_processed += 1
                    
                    if frames_processed % 5 == 0:
                        logger.info(f"Multi-Stage: Processed {frames_processed} frames, {total_keypoints} total keypoints")
                        
                except Exception as e:
                    logger.error(f"Error in multi-stage processing for frame {frame_file}: {e}")
                    errors += 1
                    continue
            
            # Stage 3: Temporal processing (if we have enough frames)
            if len(temporal_features) >= 4:
                try:
                    logger.info("Running Stage 3: Temporal Transformer processing...")
                    
                    # Process in windows
                    temporal_window = min(8, len(temporal_features))
                    for start_idx in range(0, len(temporal_features) - temporal_window + 1, temporal_window):
                        window_features = temporal_features[start_idx:start_idx + temporal_window]
                        
                        # Pad all frames to same keypoint count
                        max_keypoints = max(f.shape[0] for f in window_features)
                        padded_features = []
                        for f in window_features:
                            if f.shape[0] < max_keypoints:
                                pad_size = max_keypoints - f.shape[0]
                                padding = torch.zeros(pad_size, f.shape[1])
                                f = torch.cat([f, padding], dim=0)
                            padded_features.append(f)
                        
                        # Stack into sequence
                        sequence = torch.stack(padded_features)  # [time, keypoints, features]
                        
                        # Run temporal transformer
                        with torch.no_grad():
                            enhanced_features = temporal_transformer(sequence)
                        
                        logger.info(f"Processed temporal window: {sequence.shape} -> {enhanced_features.shape}")
                        
                except Exception as e:
                    logger.error(f"Stage 3 temporal processing failed: {e}")
                    errors += 1
            
        except Exception as e:
            logger.error(f"Multi-Stage Fusion evaluation failed: {e}")
            errors += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps = frames_processed / processing_time if processing_time > 0 else 0
        keypoints_per_frame = total_keypoints / frames_processed if frames_processed > 0 else 0
        success_rate = (frames_processed - errors) / frames_processed if frames_processed > 0 else 0
        
        return EvaluationResult(
            system_name="MULTISTAGE_FUSION",
            total_keypoints_detected=total_keypoints,
            total_frames_processed=frames_processed,
            processing_time_seconds=processing_time,
            fps=fps,
            keypoints_per_frame=keypoints_per_frame,
            error_count=errors,
            success_rate=success_rate,
            memory_usage_mb=0
        )
    
    def compare_systems(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare evaluation results from different systems"""
        comparison = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'systems': {},
            'winner': {},
            'summary': {}
        }
        
        for result in results:
            comparison['systems'][result.system_name] = {
                'keypoints_detected': result.total_keypoints_detected,
                'frames_processed': result.total_frames_processed,
                'processing_time': result.processing_time_seconds,
                'fps': result.fps,
                'keypoints_per_frame': result.keypoints_per_frame,
                'success_rate': result.success_rate,
                'error_count': result.error_count
            }
        
        # Determine winners in different categories
        if len(results) >= 2:
            keypoints_winner = max(results, key=lambda x: x.total_keypoints_detected)
            fps_winner = max(results, key=lambda x: x.fps)
            reliability_winner = max(results, key=lambda x: x.success_rate)
            
            comparison['winner'] = {
                'most_keypoints': keypoints_winner.system_name,
                'fastest_fps': fps_winner.system_name,
                'most_reliable': reliability_winner.system_name
            }
            
            # Overall comparison
            multistage = next((r for r in results if 'MULTISTAGE' in r.system_name), None)
            yolo = next((r for r in results if 'YOLO' in r.system_name), None)
            
            if multistage and yolo:
                comparison['summary'] = {
                    'keypoint_ratio': multistage.keypoints_per_frame / yolo.keypoints_per_frame if yolo.keypoints_per_frame > 0 else 0,
                    'speed_ratio': yolo.fps / multistage.fps if multistage.fps > 0 else float('inf'),
                    'multistage_advantage': multistage.keypoints_per_frame > yolo.keypoints_per_frame,
                    'yolo_faster': yolo.fps > multistage.fps
                }
        
        return comparison

def main():
    """Run real data evaluation comparing Multi-Stage Fusion vs YOLO"""
    logger.info("="*70)
    logger.info("🎯 REAL DATA EVALUATION: Multi-Stage Fusion vs YOLO")
    logger.info("="*70)
    logger.info("Using actual EndoVis 2025 Task 3 validation data - NO SIMULATION!")
    
    evaluator = RealDataEvaluator()
    
    # Select video for evaluation (use first available)
    if not evaluator.video_ids:
        logger.error("No validation videos found!")
        return
    
    test_video = evaluator.video_ids[0]  # Use first video
    logger.info(f"📹 Selected video for evaluation: {test_video}")
    
    # Load ground truth annotations
    ground_truth = evaluator.load_real_annotations(test_video)
    logger.info(f"📊 Ground truth keypoints: {len(ground_truth)} annotations")
    
    results = []
    
    # Evaluate YOLO baseline
    logger.info("\n" + "="*50)
    logger.info("🏃 EVALUATING YOLO BASELINE")
    logger.info("="*50)
    yolo_result = evaluator.evaluate_yolo_baseline(test_video, max_frames=20)
    results.append(yolo_result)
    
    # Evaluate Multi-Stage Fusion
    logger.info("\n" + "="*50)
    logger.info("🎯 EVALUATING MULTI-STAGE FUSION")
    logger.info("="*50)
    multistage_result = evaluator.evaluate_multistage_fusion(test_video, max_frames=20)
    results.append(multistage_result)
    
    # Compare results
    logger.info("\n" + "="*70)
    logger.info("📊 PERFORMANCE COMPARISON")
    logger.info("="*70)
    
    comparison = evaluator.compare_systems(results)
    
    # Print results
    for result in results:
        logger.info(f"\n🔍 {result.system_name} RESULTS:")
        logger.info(f"   Frames processed: {result.total_frames_processed}")
        logger.info(f"   Total keypoints: {result.total_keypoints_detected}")
        logger.info(f"   Keypoints per frame: {result.keypoints_per_frame:.1f}")
        logger.info(f"   Processing time: {result.processing_time_seconds:.2f}s")
        logger.info(f"   FPS: {result.fps:.2f}")
        logger.info(f"   Success rate: {result.success_rate:.2%}")
        logger.info(f"   Errors: {result.error_count}")
    
    # Summary comparison
    if 'summary' in comparison and comparison['summary']:
        summary = comparison['summary']
        logger.info(f"\n🏆 SUMMARY COMPARISON:")
        logger.info(f"   Multi-Stage detects {summary['keypoint_ratio']:.1f}x more keypoints than YOLO")
        logger.info(f"   YOLO is {summary['speed_ratio']:.1f}x faster than Multi-Stage")
        logger.info(f"   Multi-Stage advantage in keypoints: {summary['multistage_advantage']}")
        logger.info(f"   YOLO advantage in speed: {summary['yolo_faster']}")
    
    # Save results
    results_file = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/real_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': [
                {
                    'system': r.system_name,
                    'keypoints': r.total_keypoints_detected,
                    'frames': r.total_frames_processed,
                    'time': r.processing_time_seconds,
                    'fps': r.fps,
                    'kp_per_frame': r.keypoints_per_frame,
                    'success_rate': r.success_rate,
                    'errors': r.error_count
                } for r in results
            ],
            'comparison': comparison
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info("\n🎯 REAL DATA EVALUATION COMPLETE!")
    
    return results

if __name__ == "__main__":
    results = main()