#!/usr/bin/env python3
"""
Create achievement banner showing the key results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 8))

# Background gradient
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
ax.imshow(gradient, extent=[0, 10, 0, 5], aspect='auto', cmap='RdYlGn', alpha=0.15)

# Title
ax.text(5, 4.5, 'YOLO-Pose for Surgical Keypoint Tracking',
        fontsize=28, fontweight='bold', ha='center', color='#2c3e50')

ax.text(5, 4.0, 'EndoVis 2025 Challenge - Task 3',
        fontsize=18, ha='center', color='#34495e', style='italic')

# Main achievement box
fancy_box = FancyBboxPatch((1.5, 2.2), 7, 1.3,
                           boxstyle="round,pad=0.1",
                           facecolor='#2ecc71', edgecolor='#27ae60',
                           linewidth=3, alpha=0.9)
ax.add_patch(fancy_box)

# Achievement text
ax.text(5, 2.85, 'HOTA: 0.3463 → 0.4281',
        fontsize=32, fontweight='bold', ha='center', color='white')

ax.text(5, 2.45, '+23.6% Improvement',
        fontsize=20, fontweight='bold', ha='center', color='white')

# Key metrics boxes
metrics = [
    ('Detection\nAccuracy', '+60.4%', 1.5),
    ('Recall', '+270.6%', 3.5),
    ('Speed', '45 FPS', 5.5),
    ('Keypoints', '4 per tool', 7.5),
]

for label, value, x in metrics:
    # Create metric box
    box = FancyBboxPatch((x-0.6, 0.8), 1.2, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor='white', edgecolor='#3498db',
                         linewidth=2, alpha=0.9)
    ax.add_patch(box)

    # Add text
    ax.text(x, 1.35, label, fontsize=11, ha='center', color='#34495e')
    ax.text(x, 1.0, value, fontsize=14, fontweight='bold', ha='center', color='#2980b9')

# Method comparison
ax.text(0.5, 0.3, 'Before: Multi-Stage Fusion (Object Detection → Keypoint Extraction)',
        fontsize=11, ha='left', color='#7f8c8d')
ax.text(0.5, 0.05, 'After: Direct Keypoint Detection with YOLO-Pose',
        fontsize=11, ha='left', color='#27ae60', fontweight='bold')

# Architecture icons (simplified)
# Failed approaches
for i, (name, color) in enumerate([
    ('Multi-Modal', '#e74c3c'),
    ('Expansion', '#e74c3c'),
    ('BBox', '#e74c3c'),
]):
    circle = patches.Circle((9.0 + i*0.3, 0.5), 0.08,
                           facecolor=color, edgecolor='darkred', linewidth=1)
    ax.add_patch(circle)

# Successful approach
circle = patches.Circle((9.3, 1.2), 0.15,
                       facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
ax.add_patch(circle)
ax.text(9.3, 1.2, '✓', fontsize=16, fontweight='bold',
        ha='center', va='center', color='white')

# Remove axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

plt.tight_layout()
plt.savefig('paper_figures/achievement_banner.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.savefig('paper_figures/achievement_banner.pdf', bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.close()

print("✅ Achievement banner created successfully!")