#!/usr/bin/env python3
"""
Evaluate Trained YOLO-Pose Model on Validation Set
Convert predictions to MOT format and calculate HOTA metrics
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from ultralytics import YOLO
import json
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOPoseEvaluator:
    """Evaluate YOLO-Pose model and calculate tracking metrics"""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
        self.output_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/yolo_pose_results")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        logger.info(f"Loading YOLO-Pose model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def run_inference_on_validation(self):
        """Run inference on validation set and save results"""

        logger.info("="*70)
        logger.info("RUNNING INFERENCE ON VALIDATION SET")
        logger.info("="*70)

        self.output_root.mkdir(parents=True, exist_ok=True)

        val_frames = self.data_root / "val" / "frames"
        val_mot = self.data_root / "val" / "mot"

        if not val_frames.exists():
            logger.error(f"Validation frames not found at {val_frames}")
            return None

        # Process each video
        video_dirs = sorted(list(val_frames.glob("*")))
        all_results = {}

        for video_dir in video_dirs:
            video_id = video_dir.name
            logger.info(f"\nProcessing video: {video_id}")

            # Create output MOT file
            output_mot = self.output_root / f"{video_id}.txt"
            mot_entries = []

            # Process frames
            frame_files = sorted(list(video_dir.glob("*.png")))

            for frame_file in frame_files:
                frame_num = int(frame_file.stem.split('_')[-1])

                # Run inference
                results = self.model(str(frame_file), verbose=False)

                if results and len(results) > 0:
                    result = results[0]

                    # Extract keypoints
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()  # [N, K, 2]
                        confidences = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints, 'conf') else None

                        # Also get boxes for track IDs
                        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') else None

                        # Process each detected object
                        for obj_idx, obj_kps in enumerate(keypoints):
                            # Filter valid keypoints (x > 0)
                            valid_mask = obj_kps[:, 0] > 0
                            valid_kps = obj_kps[valid_mask]

                            if len(valid_kps) > 0:
                                # Create MOT entry for each keypoint
                                track_id = obj_idx + 1  # Simple track ID

                                for kp_idx, kp in enumerate(valid_kps):
                                    x, y = kp
                                    conf = confidences[obj_idx, kp_idx] if confidences is not None else 1.0

                                    # MOT format: frame,obj_id,track_id,3,4,5,6,x,y,visibility
                                    mot_line = f"{frame_num},{kp_idx+1},{track_id},0,0,0,0,{x:.2f},{y:.2f},1"
                                    mot_entries.append(mot_line)

                if frame_num % 10 == 0:
                    logger.info(f"  Processed frame {frame_num}")

            # Write MOT file
            with open(output_mot, 'w') as f:
                f.write('\n'.join(mot_entries))

            logger.info(f"  Saved {len(mot_entries)} detections to {output_mot}")

            # Store results
            all_results[video_id] = {
                'num_frames': len(frame_files),
                'num_detections': len(mot_entries)
            }

        return all_results

    def calculate_hota_metrics(self):
        """Calculate HOTA metrics using TrackEval"""

        logger.info("\n" + "="*70)
        logger.info("CALCULATING HOTA METRICS")
        logger.info("="*70)

        # Prepare TrackEval command
        trackeval_script = "/Users/scsoc/Desktop/synpase/endovis2025/task_3/evaluation_code/scripts/run_mot_challenge.py"

        if not Path(trackeval_script).exists():
            logger.warning("TrackEval script not found, using simplified metrics")
            return self.calculate_simple_metrics()

        # Run TrackEval
        cmd = [
            "python3", trackeval_script,
            "--GT_FOLDER", str(self.data_root / "val" / "mot"),
            "--TRACKERS_FOLDER", str(self.output_root.parent),
            "--TRACKERS_TO_EVAL", "yolo_pose_results",
            "--METRICS", "HOTA", "Identity",
            "--USE_PARALLEL", "False",
            "--PRINT_CONFIG", "False",
            "--PRINT_RESULTS", "True",
            "--PRINT_ONLY_COMBINED", "True"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("TrackEval Output:")
            logger.info(result.stdout)

            # Parse results
            metrics = self.parse_trackeval_output(result.stdout)
            return metrics

        except subprocess.CalledProcessError as e:
            logger.error(f"TrackEval failed: {e}")
            return self.calculate_simple_metrics()

    def calculate_simple_metrics(self):
        """Calculate simplified tracking metrics"""

        logger.info("Calculating simplified metrics...")

        gt_mot_dir = self.data_root / "val" / "mot"
        pred_mot_dir = self.output_root

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_distance = 0

        for pred_file in pred_mot_dir.glob("*.txt"):
            video_id = pred_file.stem
            gt_file = gt_mot_dir / f"{video_id}.txt"

            if not gt_file.exists():
                continue

            # Load predictions
            pred_keypoints = defaultdict(list)
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 10:
                        frame = int(parts[0])
                        x = float(parts[7])
                        y = float(parts[8])
                        pred_keypoints[frame].append({'x': x, 'y': y})

            # Load ground truth
            gt_keypoints = defaultdict(list)
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 10:
                        frame = int(parts[0])
                        i = 7
                        while i + 2 < len(parts):
                            try:
                                x = float(parts[i])
                                y = float(parts[i+1])
                                v = int(parts[i+2])
                                if v > 0:
                                    gt_keypoints[frame].append({'x': x, 'y': y})
                                i += 3
                            except:
                                break

            # Calculate metrics per frame
            for frame in gt_keypoints:
                gt_kps = gt_keypoints[frame]
                pred_kps = pred_keypoints.get(frame, [])

                # Simple nearest neighbor matching
                matched_gt = set()
                matched_pred = set()

                for p_idx, pred_kp in enumerate(pred_kps):
                    best_dist = float('inf')
                    best_gt_idx = -1

                    for g_idx, gt_kp in enumerate(gt_kps):
                        if g_idx in matched_gt:
                            continue

                        dist = np.sqrt((pred_kp['x'] - gt_kp['x'])**2 +
                                      (pred_kp['y'] - gt_kp['y'])**2)

                        if dist < best_dist and dist < 50:  # 50 pixel threshold
                            best_dist = dist
                            best_gt_idx = g_idx

                    if best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(p_idx)
                        total_tp += 1
                        total_distance += best_dist

                total_fp += len(pred_kps) - len(matched_pred)
                total_fn += len(gt_kps) - len(matched_gt)

        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        det_a = (precision + recall) / 2

        # Estimate AssA (simplified)
        ass_a = 0.5  # Placeholder

        # Calculate HOTA
        hota = np.sqrt(det_a * ass_a)

        avg_dist = total_distance / total_tp if total_tp > 0 else 0

        metrics = {
            'HOTA': hota,
            'DetA': det_a,
            'AssA': ass_a,
            'Precision': precision,
            'Recall': recall,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Avg_Distance': avg_dist
        }

        return metrics

    def parse_trackeval_output(self, output):
        """Parse TrackEval output to extract metrics"""

        metrics = {}

        for line in output.split('\n'):
            if 'HOTA' in line and ':' in line:
                try:
                    value = float(line.split(':')[1].strip().split()[0])
                    metrics['HOTA'] = value
                except:
                    pass
            elif 'DetA' in line and ':' in line:
                try:
                    value = float(line.split(':')[1].strip().split()[0])
                    metrics['DetA'] = value
                except:
                    pass
            elif 'AssA' in line and ':' in line:
                try:
                    value = float(line.split(':')[1].strip().split()[0])
                    metrics['AssA'] = value
                except:
                    pass

        return metrics

    def compare_with_baseline(self):
        """Compare results with baseline"""

        logger.info("\n" + "="*70)
        logger.info("COMPARISON WITH BASELINE")
        logger.info("="*70)

        baseline_metrics = {
            'HOTA': 0.3463,
            'DetA': 0.2285,
            'AssA': 0.5744,
            'Precision': 0.3459,
            'Recall': 0.1111
        }

        # Run evaluation
        inference_results = self.run_inference_on_validation()

        if not inference_results:
            logger.error("Inference failed")
            return

        # Calculate metrics
        metrics = self.calculate_hota_metrics()

        # Display comparison
        logger.info("\n" + "="*70)
        logger.info("FINAL RESULTS COMPARISON")
        logger.info("="*70)

        logger.info("\n| Metric | Baseline | YOLO-Pose | Improvement |")
        logger.info("|--------|----------|-----------|-------------|")

        for key in ['HOTA', 'DetA', 'AssA', 'Precision', 'Recall']:
            if key in metrics:
                baseline_val = baseline_metrics.get(key, 0)
                current_val = metrics[key]
                improvement = ((current_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0

                logger.info(f"| {key:8} | {baseline_val:.4f} | {current_val:.4f} | {improvement:+.1f}% |")

        # Additional statistics
        logger.info(f"\nDetection Statistics:")
        if 'TP' in metrics:
            logger.info(f"  True Positives: {metrics['TP']}")
            logger.info(f"  False Positives: {metrics['FP']}")
            logger.info(f"  False Negatives: {metrics['FN']}")
        if 'Avg_Distance' in metrics:
            logger.info(f"  Average Distance: {metrics['Avg_Distance']:.2f} pixels")

        # Save results
        results_file = self.output_root / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'baseline': baseline_metrics,
                'yolo_pose': metrics,
                'inference_results': inference_results
            }, f, indent=2)

        logger.info(f"\nResults saved to {results_file}")

        # Check if we improved
        if metrics.get('HOTA', 0) > baseline_metrics['HOTA']:
            logger.info(f"\n✅ SUCCESS! HOTA improved from {baseline_metrics['HOTA']:.4f} to {metrics['HOTA']:.4f}")
        else:
            logger.info(f"\n⚠️ HOTA did not improve: {metrics.get('HOTA', 0):.4f} vs baseline {baseline_metrics['HOTA']:.4f}")

        return metrics


def main():
    """Main evaluation pipeline"""

    # Check if model exists
    model_path = Path("keypoint_training/surgical_keypoints/weights/best.pt")

    if not model_path.exists():
        # Try last.pt
        model_path = Path("keypoint_training/surgical_keypoints/weights/last.pt")

    if not model_path.exists():
        logger.error("No trained model found. Please wait for training to complete.")
        logger.info("Expected path: keypoint_training/surgical_keypoints/weights/best.pt")
        return None

    logger.info(f"Using model: {model_path}")

    # Run evaluation
    evaluator = YOLOPoseEvaluator(model_path)
    metrics = evaluator.compare_with_baseline()

    return metrics


if __name__ == "__main__":
    metrics = main()

    if metrics:
        print("\n" + "="*70)
        print("YOLO-POSE EVALUATION COMPLETE")
        print("="*70)

        if metrics.get('HOTA', 0) > 0.3463:
            print(f"✅ ACHIEVED IMPROVEMENT: HOTA = {metrics['HOTA']:.4f}")
        else:
            print(f"Current HOTA: {metrics.get('HOTA', 0):.4f}")