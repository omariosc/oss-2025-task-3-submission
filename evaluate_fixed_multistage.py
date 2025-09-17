#!/usr/bin/env python3
"""
Evaluate the Fixed Multi-Stage Fusion System on Real EndoVis Data
"""

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import time
import logging
from typing import Dict, List

# Add path for the fixed system
sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')
from multistage_fusion_fixed import FixedMultiStageFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiStageEvaluator:
    """Evaluate fixed multi-stage fusion on real data"""

    def __init__(self):
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.val_frames_dir = self.data_root / "val/frames"
        self.output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/multistage_evaluation")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize the fixed system
        logger.info("Initializing Fixed Multi-Stage Fusion System...")
        self.system = FixedMultiStageFusion(device='cpu')  # Use CPU for stability

        # Get validation videos
        self.video_ids = sorted([d.name for d in self.val_frames_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(self.video_ids)} validation videos")

    def evaluate_video(self, video_id: str, max_frames: int = 10):
        """Evaluate on a single video"""
        logger.info(f"\nEvaluating video: {video_id}")

        frames_dir = self.val_frames_dir / video_id
        frame_files = sorted(list(frames_dir.glob("*.png")))[:max_frames]

        if not frame_files:
            logger.warning(f"No frames found for {video_id}")
            return None

        results = {
            'video_id': video_id,
            'frames_processed': 0,
            'total_keypoints_detected': 0,
            'total_tracks': 0,
            'avg_keypoints_per_frame': 0,
            'processing_time': 0,
            'fps': 0,
            'tracks_by_frame': {}
        }

        start_time = time.time()
        all_track_ids = set()

        for frame_idx, frame_file in enumerate(frame_files):
            try:
                # Load frame
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue

                # Detect keypoints
                keypoints = self.system.detect_keypoints(frame, use_nms=True)

                # Track keypoints
                tracked = self.system.track_keypoints(frame, keypoints)

                # Store results
                results['tracks_by_frame'][frame_idx] = {
                    'frame_file': frame_file.name,
                    'num_keypoints': len(keypoints),
                    'num_tracked': len(tracked),
                    'tracks': [
                        {
                            'track_id': kp.get('track_id', -1),
                            'x': kp['x'],
                            'y': kp['y'],
                            'confidence': kp.get('confidence', 0.5),
                            'class': kp.get('class', -1)
                        }
                        for kp in tracked
                    ]
                }

                # Update statistics
                results['frames_processed'] += 1
                results['total_keypoints_detected'] += len(keypoints)

                # Collect unique track IDs
                for kp in tracked:
                    if 'track_id' in kp:
                        all_track_ids.add(kp['track_id'])

                logger.info(f"  Frame {frame_idx+1}/{len(frame_files)}: "
                           f"{len(keypoints)} keypoints, {len(tracked)} tracked")

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
                continue

        # Calculate final metrics
        end_time = time.time()
        results['processing_time'] = end_time - start_time
        results['total_tracks'] = len(all_track_ids)

        if results['frames_processed'] > 0:
            results['avg_keypoints_per_frame'] = results['total_keypoints_detected'] / results['frames_processed']
            results['fps'] = results['frames_processed'] / results['processing_time']

        logger.info(f"Completed {video_id}:")
        logger.info(f"  - Frames: {results['frames_processed']}")
        logger.info(f"  - Total keypoints: {results['total_keypoints_detected']}")
        logger.info(f"  - Unique tracks: {results['total_tracks']}")
        logger.info(f"  - Avg keypoints/frame: {results['avg_keypoints_per_frame']:.1f}")
        logger.info(f"  - FPS: {results['fps']:.2f}")

        return results

    def save_mot_format(self, results: Dict, output_file: Path):
        """Save tracking results in MOT format"""
        mot_lines = []

        for frame_idx, frame_data in results['tracks_by_frame'].items():
            for track in frame_data['tracks']:
                # MOT format: frame,id,x,y,w,h,conf,class,visibility
                line = f"{frame_idx+1},{track['track_id']},{track['x']:.2f},{track['y']:.2f},"
                line += f"10,10,{track['confidence']:.3f},{track['class']},1"
                mot_lines.append(line)

        with open(output_file, 'w') as f:
            f.write('\n'.join(mot_lines))

        logger.info(f"Saved {len(mot_lines)} MOT entries to {output_file}")

    def evaluate_all(self, max_videos: int = 3, max_frames_per_video: int = 10):
        """Evaluate on all validation videos"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATING FIXED MULTI-STAGE FUSION")
        logger.info("="*60)

        all_results = {}
        summary = {
            'total_videos': 0,
            'total_frames': 0,
            'total_keypoints': 0,
            'total_tracks': 0,
            'avg_keypoints_per_frame': 0,
            'avg_fps': 0,
            'total_time': 0
        }

        # Process videos
        for video_id in self.video_ids[:max_videos]:
            result = self.evaluate_video(video_id, max_frames_per_video)

            if result:
                all_results[video_id] = result

                # Update summary
                summary['total_videos'] += 1
                summary['total_frames'] += result['frames_processed']
                summary['total_keypoints'] += result['total_keypoints_detected']
                summary['total_tracks'] += result['total_tracks']
                summary['total_time'] += result['processing_time']

                # Save MOT format for this video
                mot_file = self.output_dir / f"{video_id}_tracking.txt"
                self.save_mot_format(result, mot_file)

        # Calculate averages
        if summary['total_frames'] > 0:
            summary['avg_keypoints_per_frame'] = summary['total_keypoints'] / summary['total_frames']

        if summary['total_time'] > 0:
            summary['avg_fps'] = summary['total_frames'] / summary['total_time']

        # Save results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'per_video_results': {
                    vid: {
                        'frames': r['frames_processed'],
                        'keypoints': r['total_keypoints_detected'],
                        'tracks': r['total_tracks'],
                        'avg_kp_per_frame': r['avg_keypoints_per_frame'],
                        'fps': r['fps']
                    }
                    for vid, r in all_results.items()
                }
            }, f, indent=2)

        # Generate markdown report
        self.generate_report(summary, all_results)

        return summary, all_results

    def generate_report(self, summary: Dict, all_results: Dict):
        """Generate evaluation report"""
        report_file = self.output_dir / "evaluation_report.md"

        with open(report_file, 'w') as f:
            f.write("# Fixed Multi-Stage Fusion Evaluation Report\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Videos Evaluated**: {summary['total_videos']}\n")
            f.write(f"- **Total Frames**: {summary['total_frames']}\n")
            f.write(f"- **Total Keypoints**: {summary['total_keypoints']:,}\n")
            f.write(f"- **Total Tracks**: {summary['total_tracks']}\n")
            f.write(f"- **Avg Keypoints/Frame**: {summary['avg_keypoints_per_frame']:.1f}\n")
            f.write(f"- **Avg FPS**: {summary['avg_fps']:.2f}\n")
            f.write(f"- **Total Time**: {summary['total_time']:.2f}s\n\n")

            f.write("## Per-Video Results\n\n")
            f.write("| Video | Frames | Keypoints | Tracks | KP/Frame | FPS |\n")
            f.write("|-------|--------|-----------|--------|----------|-----|\n")

            for video_id, result in all_results.items():
                f.write(f"| {video_id} | {result['frames_processed']} | "
                       f"{result['total_keypoints_detected']} | {result['total_tracks']} | "
                       f"{result['avg_keypoints_per_frame']:.1f} | {result['fps']:.2f} |\n")

            f.write("\n## System Components\n\n")
            f.write("### Stage 1: CNN Keypoint Detection\n")
            f.write("- **Backbone**: ResNet50 with FPN\n")
            f.write("- **Multi-scale detection**: Yes\n")
            f.write("- **Sub-pixel refinement**: Yes\n")
            f.write("- **NMS applied**: Yes\n\n")

            f.write("### Stage 2: Optical Flow Tracking\n")
            f.write("- **Method**: Lucas-Kanade\n")
            f.write("- **Temporal consistency**: Enhanced\n")
            f.write("- **Motion prediction**: Yes\n\n")

            f.write("### Stage 3: Kalman Filtering\n")
            f.write("- **State model**: Constant velocity\n")
            f.write("- **Smoothing**: Applied\n")
            f.write("- **Occlusion handling**: Yes\n\n")

            f.write("### Stage 4: Hungarian Association\n")
            f.write("- **Cost matrix**: Distance + Motion\n")
            f.write("- **Optimal assignment**: Yes\n")
            f.write("- **Track management**: Automatic\n\n")

            f.write("## Comparison with YOLO\n\n")
            f.write("| Metric | Fixed Multi-Stage | YOLO Baseline |\n")
            f.write("|--------|------------------|---------------|\n")
            f.write(f"| Keypoints/Frame | {summary['avg_keypoints_per_frame']:.1f} | ~4 |\n")
            f.write(f"| FPS | {summary['avg_fps']:.2f} | 1.5 |\n")
            f.write(f"| Tracking Method | Kalman+Hungarian | BoT-SORT |\n")
            f.write(f"| Temporal Consistency | Optical Flow | Appearance |\n\n")

            ratio = summary['avg_keypoints_per_frame'] / 4.0 if summary['avg_keypoints_per_frame'] > 0 else 0
            f.write(f"**Keypoint Detection Improvement**: {ratio:.1f}x more keypoints than YOLO\n")

        logger.info(f"Report saved to {report_file}")

def main():
    """Run evaluation"""
    evaluator = MultiStageEvaluator()

    # Evaluate on subset of videos
    summary, results = evaluator.evaluate_all(
        max_videos=3,
        max_frames_per_video=5  # Limited for faster testing
    )

    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Average Keypoints/Frame: {summary['avg_keypoints_per_frame']:.1f}")
    logger.info(f"Average FPS: {summary['avg_fps']:.2f}")
    logger.info(f"Total Unique Tracks: {summary['total_tracks']}")
    logger.info(f"Results saved to: {evaluator.output_dir}")

    return summary

if __name__ == "__main__":
    main()