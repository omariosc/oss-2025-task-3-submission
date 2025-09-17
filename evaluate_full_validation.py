#!/usr/bin/env python3
"""
Full Validation Set Evaluation with Continuous Output
Evaluates both YOLO and Multi-Stage Fusion on all validation videos
Writes results continuously to avoid data loss
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025')
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/task_3/candidate_submission/src')

class ContinuousEvaluator:
    """Evaluator that continuously writes results to avoid data loss"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data paths
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.masks_dir = self.data_root / "class_masks"

        # Get all validation videos first
        self.video_ids = sorted([d.name for d in self.val_frames_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(self.video_ids)} validation videos: {self.video_ids}")

        # Output files
        self.multistage_output = self.output_dir / "multistage_results.json"
        self.yolo_output = self.output_dir / "yolo_results.json"
        self.results_md = self.output_dir / "validation_results.md"

        # Initialize output files
        self._init_output_files()

    def _init_output_files(self):
        """Initialize output files with headers"""
        # Initialize JSON files
        for json_file in [self.multistage_output, self.yolo_output]:
            with open(json_file, 'w') as f:
                json.dump({
                    'start_time': datetime.now().isoformat(),
                    'videos': {},
                    'summary': {}
                }, f, indent=2)

        # Initialize markdown file
        with open(self.results_md, 'w') as f:
            f.write("# Full Validation Set Results\n\n")
            f.write(f"**Evaluation Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Dataset Information\n\n")
            f.write(f"- **Data Path**: {self.val_frames_dir}\n")
            f.write(f"- **Videos**: {len(self.video_ids)}\n")
            f.write(f"- **Video IDs**: {', '.join(self.video_ids)}\n\n")

    def _write_continuous_result(self, system: str, video_id: str, result: Dict):
        """Write result immediately to prevent data loss"""
        output_file = self.multistage_output if system == 'multistage' else self.yolo_output

        # Read current data
        with open(output_file, 'r') as f:
            data = json.load(f)

        # Update with new result
        data['videos'][video_id] = result
        data['last_updated'] = datetime.now().isoformat()

        # Write back immediately
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Also append to markdown
        self._append_to_markdown(system, video_id, result)

    def _append_to_markdown(self, system: str, video_id: str, result: Dict):
        """Append result to markdown file"""
        with open(self.results_md, 'a') as f:
            if result.get('success', False):
                f.write(f"\n### {system.upper()} - {video_id}\n")
                f.write(f"- Frames: {result.get('frames_processed', 0)}\n")
                f.write(f"- Keypoints: {result.get('total_keypoints', 0)}\n")
                f.write(f"- KP/Frame: {result.get('keypoints_per_frame', 0):.1f}\n")
                f.write(f"- Time: {result.get('processing_time', 0):.2f}s\n")
                f.write(f"- FPS: {result.get('fps', 0):.2f}\n")
            else:
                f.write(f"\n### {system.upper()} - {video_id} [FAILED]\n")
                f.write(f"- Error: {result.get('error', 'Unknown')}\n")

    def evaluate_multistage_video(self, video_id: str, max_frames: int = None) -> Dict:
        """Evaluate Multi-Stage Fusion on a single video"""
        logger.info(f"🎯 Evaluating Multi-Stage on {video_id}")

        try:
            from candidate_submission.src.keypoint_detector import UltraDenseKeypointDetector

            # Optimized configuration for full validation
            config = {
                'grid_sizes': [(128, 96)],  # Single grid for speed
                'segmentation_weight': 5.0,
                'nms_radius': 3,
                'confidence_threshold': 0.2
            }

            detector = UltraDenseKeypointDetector(config)

            # Get frames for this video
            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))
            if max_frames:
                frame_files = frame_files[:max_frames]

            start_time = time.time()
            total_keypoints = 0
            frames_processed = 0

            # Process frames
            for i, frame_file in enumerate(frame_files):
                try:
                    # Load frame
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue

                    # Load masks
                    frame_name = frame_file.name
                    masks = {}
                    mask_classes = ['left_hand_segment', 'right_hand_segment', 'scissors',
                                  'tweezers', 'needle_holder', 'needle']

                    for class_name in mask_classes:
                        mask_path = self.masks_dir / class_name / frame_name
                        if mask_path.exists():
                            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                            masks[class_name] = mask if mask is not None else np.zeros((480, 640), dtype=np.uint8)
                        else:
                            masks[class_name] = np.zeros((480, 640), dtype=np.uint8)

                    # Detect keypoints
                    keypoints = detector.detect(frame, masks)
                    frame_keypoints = len(keypoints)

                    total_keypoints += frame_keypoints
                    frames_processed += 1

                    # Log progress every 10 frames
                    if (i + 1) % 10 == 0:
                        logger.info(f"  {video_id}: Processed {i+1}/{len(frame_files)} frames")

                        # Write intermediate result
                        intermediate_result = {
                            'frames_processed': frames_processed,
                            'total_keypoints': total_keypoints,
                            'status': 'processing'
                        }
                        self._write_continuous_result('multistage', video_id, intermediate_result)

                except Exception as e:
                    logger.warning(f"Frame {i} failed: {e}")
                    continue

            # Calculate final metrics
            end_time = time.time()
            processing_time = end_time - start_time

            result = {
                'success': True,
                'frames_processed': frames_processed,
                'total_frames': len(frame_files),
                'total_keypoints': total_keypoints,
                'keypoints_per_frame': total_keypoints / frames_processed if frames_processed > 0 else 0,
                'processing_time': processing_time,
                'fps': frames_processed / processing_time if processing_time > 0 else 0,
                'status': 'completed'
            }

            logger.info(f"✅ {video_id}: {frames_processed} frames, {total_keypoints} keypoints, {result['fps']:.2f} FPS")

        except Exception as e:
            logger.error(f"❌ Multi-Stage failed for {video_id}: {e}")
            result = {
                'success': False,
                'error': str(e),
                'status': 'failed'
            }

        # Write final result
        self._write_continuous_result('multistage', video_id, result)
        return result

    def evaluate_yolo_video(self, video_id: str, max_frames: int = None) -> Dict:
        """Evaluate YOLO on a single video"""
        logger.info(f"🔍 Evaluating YOLO on {video_id}")

        try:
            from ultralytics import YOLO

            # Load YOLO model
            model_path = self.data_root / "yolo11m.pt"
            model = YOLO(str(model_path))

            # Get frames
            frames_dir = self.val_frames_dir / video_id
            frame_files = sorted(list(frames_dir.glob("*.png")))
            if max_frames:
                frame_files = frame_files[:max_frames]

            start_time = time.time()
            total_objects = 0
            frames_processed = 0

            # Process frames
            for i, frame_file in enumerate(frame_files):
                try:
                    frame = cv2.imread(str(frame_file))
                    if frame is None:
                        continue

                    # Run YOLO detection
                    results = model(frame, verbose=False)

                    # Count detections
                    frame_objects = 0
                    for result in results:
                        if result.boxes is not None:
                            frame_objects += len(result.boxes)

                    total_objects += frame_objects
                    frames_processed += 1

                    # Log progress every 10 frames
                    if (i + 1) % 10 == 0:
                        logger.info(f"  {video_id}: Processed {i+1}/{len(frame_files)} frames")

                        # Write intermediate result
                        intermediate_result = {
                            'frames_processed': frames_processed,
                            'total_objects': total_objects,
                            'status': 'processing'
                        }
                        self._write_continuous_result('yolo', video_id, intermediate_result)

                except Exception as e:
                    logger.warning(f"Frame {i} failed: {e}")
                    continue

            # Calculate final metrics
            end_time = time.time()
            processing_time = end_time - start_time

            result = {
                'success': True,
                'frames_processed': frames_processed,
                'total_frames': len(frame_files),
                'total_objects': total_objects,
                'total_keypoints': total_objects,  # For consistency
                'keypoints_per_frame': total_objects / frames_processed if frames_processed > 0 else 0,
                'processing_time': processing_time,
                'fps': frames_processed / processing_time if processing_time > 0 else 0,
                'status': 'completed'
            }

            logger.info(f"✅ {video_id}: {frames_processed} frames, {total_objects} objects, {result['fps']:.2f} FPS")

        except Exception as e:
            logger.error(f"❌ YOLO failed for {video_id}: {e}")
            result = {
                'success': False,
                'error': str(e),
                'status': 'failed'
            }

        # Write final result
        self._write_continuous_result('yolo', video_id, result)
        return result

    def evaluate_full_validation(self, system: str, max_frames_per_video: int = None):
        """Evaluate full validation set for a system"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 FULL VALIDATION SET EVALUATION: {system.upper()}")
        logger.info(f"{'='*80}")

        all_results = {}

        for video_id in self.video_ids:
            logger.info(f"\nProcessing video {video_id}...")

            if system == 'multistage':
                result = self.evaluate_multistage_video(video_id, max_frames_per_video)
            else:
                result = self.evaluate_yolo_video(video_id, max_frames_per_video)

            all_results[video_id] = result

        # Calculate summary statistics
        summary = self._calculate_summary(all_results)

        # Write final summary
        output_file = self.multistage_output if system == 'multistage' else self.yolo_output
        with open(output_file, 'r') as f:
            data = json.load(f)

        data['summary'] = summary
        data['end_time'] = datetime.now().isoformat()

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Write summary to markdown
        self._write_summary_to_markdown(system, summary)

        return all_results, summary

    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        successful = [r for r in results.values() if r.get('success', False)]

        if not successful:
            return {'error': 'No successful evaluations'}

        total_frames = sum(r['frames_processed'] for r in successful)
        total_keypoints = sum(r['total_keypoints'] for r in successful)
        total_time = sum(r['processing_time'] for r in successful)

        return {
            'videos_processed': len(successful),
            'videos_failed': len(results) - len(successful),
            'total_frames': total_frames,
            'total_keypoints': total_keypoints,
            'avg_keypoints_per_frame': total_keypoints / total_frames if total_frames > 0 else 0,
            'total_processing_time': total_time,
            'avg_fps': total_frames / total_time if total_time > 0 else 0,
            'per_video_stats': {
                vid: {
                    'frames': r['frames_processed'],
                    'keypoints': r['total_keypoints'],
                    'kp_per_frame': r['keypoints_per_frame'],
                    'fps': r['fps']
                }
                for vid, r in results.items() if r.get('success', False)
            }
        }

    def _write_summary_to_markdown(self, system: str, summary: Dict):
        """Write summary to markdown file"""
        with open(self.results_md, 'a') as f:
            f.write(f"\n## {system.upper()} - Summary\n\n")

            if 'error' in summary:
                f.write(f"**Error**: {summary['error']}\n")
            else:
                f.write(f"- **Videos Processed**: {summary['videos_processed']}\n")
                f.write(f"- **Videos Failed**: {summary['videos_failed']}\n")
                f.write(f"- **Total Frames**: {summary['total_frames']}\n")
                f.write(f"- **Total Keypoints**: {summary['total_keypoints']}\n")
                f.write(f"- **Avg Keypoints/Frame**: {summary['avg_keypoints_per_frame']:.1f}\n")
                f.write(f"- **Total Time**: {summary['total_processing_time']:.2f}s\n")
                f.write(f"- **Avg FPS**: {summary['avg_fps']:.2f}\n")

                # Per-video breakdown table
                f.write(f"\n### Per-Video Breakdown\n\n")
                f.write("| Video | Frames | Keypoints | KP/Frame | FPS |\n")
                f.write("|-------|--------|-----------|----------|-----|\n")

                for vid, stats in summary.get('per_video_stats', {}).items():
                    f.write(f"| {vid} | {stats['frames']} | {stats['keypoints']} | "
                           f"{stats['kp_per_frame']:.1f} | {stats['fps']:.2f} |\n")

def main():
    """Run full validation set evaluation"""
    # Create evaluator
    output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/full_validation_results")
    evaluator = ContinuousEvaluator(output_dir)

    # Configure evaluation
    max_frames_per_video = 10  # Limit frames for faster evaluation (set to None for all frames)

    logger.info("Starting full validation set evaluation...")
    logger.info(f"Max frames per video: {max_frames_per_video if max_frames_per_video else 'All'}")

    # Evaluate Multi-Stage Fusion
    logger.info("\n" + "="*80)
    logger.info("STARTING MULTI-STAGE FUSION EVALUATION")
    logger.info("="*80)
    multistage_results, multistage_summary = evaluator.evaluate_full_validation('multistage', max_frames_per_video)

    # Evaluate YOLO
    logger.info("\n" + "="*80)
    logger.info("STARTING YOLO EVALUATION")
    logger.info("="*80)
    yolo_results, yolo_summary = evaluator.evaluate_full_validation('yolo', max_frames_per_video)

    # Final comparison
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"- Multi-Stage results: {evaluator.multistage_output}")
    logger.info(f"- YOLO results: {evaluator.yolo_output}")
    logger.info(f"- Summary markdown: {evaluator.results_md}")

    return multistage_results, yolo_results

if __name__ == "__main__":
    main()