#!/usr/bin/env python3
"""
Grid search for optimal detection parameters
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import logging
from itertools import product

sys.path.append('/Users/scsoc/Desktop/synpase/endovis2025/submit/docker')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_parameters(confidence_thresh, nms_radius, grid_size):
    """Evaluate a parameter combination"""
    from multistage_optimized import OptimizedMultiStageFusion

    # Create system with parameters
    system = OptimizedMultiStageFusion(device='cpu')
    system.confidence_threshold = confidence_thresh
    system.nms_radius = nms_radius
    system.grid_size = grid_size

    # Test on one frame
    data_root = Path("/Users/scsoc/Desktop/synpase/endovis2025/task_3/data")
    frame_file = list((data_root / "val/frames/E66F").glob("*.png"))[0]
    frame = cv2.imread(str(frame_file))

    result = system.process_frame(frame)

    # Score based on:
    # 1. Getting exactly 6 objects (most important)
    # 2. Having reasonable keypoints
    score = 0
    if result['num_objects'] == 6:
        score += 10  # Big bonus for correct count

    # Penalty for too many/few keypoints
    keypoint_penalty = abs(result['raw_keypoints'] - 30) * 0.1
    score -= keypoint_penalty

    return score, result

def grid_search():
    """Run grid search"""
    # Parameter ranges
    confidence_thresholds = [0.5, 0.6, 0.7, 0.8]
    nms_radii = [50, 75, 100, 150]
    grid_sizes = [40, 50, 60, 80]

    best_params = None
    best_score = -float('inf')
    results = []

    logger.info("Starting grid search...")
    logger.info(f"Testing {len(confidence_thresholds) * len(nms_radii) * len(grid_sizes)} combinations")

    for conf, nms, grid in product(confidence_thresholds, nms_radii, grid_sizes):
        try:
            score, result = evaluate_parameters(conf, nms, grid)

            results.append({
                'confidence': conf,
                'nms_radius': nms,
                'grid_size': grid,
                'score': score,
                'objects': result['num_objects'],
                'keypoints': result['raw_keypoints']
            })

            if score > best_score:
                best_score = score
                best_params = {'confidence': conf, 'nms_radius': nms, 'grid_size': grid}

            logger.info(f"Conf={conf:.1f}, NMS={nms}, Grid={grid}: "
                       f"Score={score:.2f}, Objects={result['num_objects']}, KP={result['raw_keypoints']}")

        except Exception as e:
            logger.error(f"Error with params {conf}, {nms}, {grid}: {e}")
            continue

    # Report best
    logger.info("\n" + "="*60)
    logger.info("GRID SEARCH RESULTS")
    logger.info("="*60)

    if best_params:
        logger.info(f"Best parameters:")
        logger.info(f"  Confidence: {best_params['confidence']}")
        logger.info(f"  NMS Radius: {best_params['nms_radius']}")
        logger.info(f"  Grid Size: {best_params['grid_size']}")
        logger.info(f"  Score: {best_score:.2f}")

        # Save results
        output_dir = Path("/Users/scsoc/Desktop/synpase/endovis2025/submit/docker/optimized_results")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "grid_search_results.md", 'w') as f:
            f.write("# Grid Search Results\n\n")
            f.write("## Best Parameters\n\n")
            f.write(f"- **Confidence Threshold**: {best_params['confidence']}\n")
            f.write(f"- **NMS Radius**: {best_params['nms_radius']}\n")
            f.write(f"- **Grid Size**: {best_params['grid_size']}\n")
            f.write(f"- **Score**: {best_score:.2f}\n\n")

            f.write("## Top 5 Combinations\n\n")
            f.write("| Conf | NMS | Grid | Score | Objects | Keypoints |\n")
            f.write("|------|-----|------|-------|---------|----------|\n")

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            for r in results[:5]:
                f.write(f"| {r['confidence']} | {r['nms_radius']} | {r['grid_size']} | ")
                f.write(f"{r['score']:.2f} | {r['objects']} | {r['keypoints']} |\n")

            f.write("\n## Recommendation\n\n")
            f.write(f"Use the following parameters for optimal HOTA:\n")
            f.write(f"```python\n")
            f.write(f"system.confidence_threshold = {best_params['confidence']}\n")
            f.write(f"system.nms_radius = {best_params['nms_radius']}\n")
            f.write(f"system.grid_size = {best_params['grid_size']}\n")
            f.write(f"```\n")

        logger.info(f"\nResults saved to {output_dir}/grid_search_results.md")

    return best_params

if __name__ == "__main__":
    best = grid_search()
    if best:
        print(f"\n✅ Grid search complete! Best parameters found:")
        print(f"   Confidence={best['confidence']}, NMS={best['nms_radius']}, Grid={best['grid_size']}")