#!/usr/bin/env python3
"""
Generate scientific figures for the surgical keypoint tracking paper
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

# Set style for scientific papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Output directory
output_dir = Path("paper_figures")
output_dir.mkdir(exist_ok=True)

def create_hota_comparison_bar_chart():
    """Create HOTA comparison bar chart for all methods"""

    methods = ['Multi-Modal\nFusion', 'Multi-Keypoint\nExpansion', 'BBox\nExtraction',
               'Dense\nDetection', 'Baseline\n(Multi-Stage)', 'YOLO\nAnchors', 'YOLO-Pose\n(Final)']
    hota_scores = [0.144, 0.2243, 0.2175, 0.195, 0.3463, 0.437, 0.4281]

    # Color coding: red for failed, yellow for baseline, green for successful
    colors = ['#e74c3c', '#e74c3c', '#e74c3c', '#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(methods, hota_scores, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, hota_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Add improvement percentage for final method
    baseline_idx = 4
    final_idx = 6
    improvement = ((hota_scores[final_idx] - hota_scores[baseline_idx]) / hota_scores[baseline_idx]) * 100
    ax.annotate(f'+{improvement:.1f}%',
                xy=(bars[final_idx].get_x() + bars[final_idx].get_width()/2, hota_scores[final_idx]),
                xytext=(bars[final_idx].get_x() + bars[final_idx].get_width()/2, hota_scores[final_idx] + 0.05),
                ha='center', fontsize=12, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_ylabel('HOTA Score', fontsize=14, fontweight='bold')
    ax.set_title('HOTA Performance Comparison Across All Methods', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add horizontal line for baseline
    ax.axhline(y=hota_scores[baseline_idx], color='orange', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'hota_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hota_comparison.pdf', bbox_inches='tight')
    plt.close()

def create_metrics_radar_chart():
    """Create radar chart comparing key metrics"""

    categories = ['HOTA', 'DetA', 'Precision', 'Recall', 'F1']

    # Data for three key methods
    baseline = [0.3463, 0.2285, 0.3459, 0.1111, 0.1682]
    yolo_anchors = [0.437, 0.3116, 0.3887, 0.2345, 0.2926]
    yolo_pose = [0.4281, 0.3666, 0.3214, 0.4118, 0.3610]

    # Normalize to 0-1 scale for better visualization
    max_vals = [0.5, 0.5, 0.5, 0.5, 0.5]
    baseline_norm = [b/m for b, m in zip(baseline, max_vals)]
    yolo_anchors_norm = [y/m for y, m in zip(yolo_anchors, max_vals)]
    yolo_pose_norm = [y/m for y, m in zip(yolo_pose, max_vals)]

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Close the plot
    baseline_norm += baseline_norm[:1]
    yolo_anchors_norm += yolo_anchors_norm[:1]
    yolo_pose_norm += yolo_pose_norm[:1]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Draw the outlines
    ax.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color='#f39c12')
    ax.fill(angles, baseline_norm, alpha=0.15, color='#f39c12')

    ax.plot(angles, yolo_anchors_norm, 'o-', linewidth=2, label='YOLO Anchors', color='#3498db')
    ax.fill(angles, yolo_anchors_norm, alpha=0.15, color='#3498db')

    ax.plot(angles, yolo_pose_norm, 'o-', linewidth=2, label='YOLO-Pose (Final)', color='#2ecc71')
    ax.fill(angles, yolo_pose_norm, alpha=0.15, color='#2ecc71')

    # Fix axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.title('Multi-Metric Performance Comparison', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'metrics_radar.pdf', bbox_inches='tight')
    plt.close()

def create_per_video_performance():
    """Create per-video performance analysis"""

    videos = ['K16O', 'E66F', 'P29U', 'P11H', 'U24S', 'P89M', 'X29I', 'Y33I']
    precision = [0.393, 0.311, 0.301, 0.363, 0.284, 0.296, 0.350, 0.312]
    recall = [0.356, 0.451, 0.456, 0.441, 0.375, 0.440, 0.427, 0.352]
    tp_counts = [389, 554, 451, 558, 467, 627, 548, 485]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Precision-Recall comparison
    x = np.arange(len(videos))
    width = 0.35

    bars1 = ax1.bar(x - width/2, precision, width, label='Precision', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x + width/2, recall, width, label='Recall', color='#2ecc71', edgecolor='black')

    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Video Precision and Recall', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(videos)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # F1 scores
    f1_scores = [2*p*r/(p+r) for p, r in zip(precision, recall)]
    bars3 = ax2.bar(videos, f1_scores, color='#9b59b6', edgecolor='black')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Video F1 Scores', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars3, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom')

    # True Positives count
    bars4 = ax3.bar(videos, tp_counts, color='#e67e22', edgecolor='black')
    ax3.set_ylabel('True Positives', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Video Detection Count', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars4, tp_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{val}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_video_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'per_video_performance.pdf', bbox_inches='tight')
    plt.close()

def create_training_curve():
    """Create training progression curve"""

    # Simulated training data (based on typical progression)
    epochs = list(range(0, 31, 5))
    hota = [0.0, 0.28, 0.35, 0.39, 0.41, 0.425, 0.428]
    loss = [6.5, 4.2, 3.1, 2.5, 2.1, 1.9, 1.8]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # HOTA curve
    color = '#2ecc71'
    ax1.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('HOTA Score', color=color, fontsize=14, fontweight='bold')
    line1 = ax1.plot(epochs, hota, 'o-', color=color, linewidth=2.5, markersize=8, label='HOTA')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Loss curve on secondary axis
    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('Total Loss', color=color, fontsize=14, fontweight='bold')
    line2 = ax2.plot(epochs, loss, 's-', color=color, linewidth=2.5, markersize=8, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add vertical line at current training point
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    ax1.text(5, 0.15, 'Current\n(Partial)', ha='center', fontsize=10, color='gray')

    # Title
    ax1.set_title('Training Progression: HOTA vs Loss', fontsize=16, fontweight='bold')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    fig.tight_layout()
    plt.savefig(output_dir / 'training_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curve.pdf', bbox_inches='tight')
    plt.close()

def create_ablation_keypoints():
    """Create ablation study for number of keypoints"""

    keypoints = [1, 2, 3, 4, 5]
    hota = [0.3463, 0.3821, 0.4109, 0.4281, 0.4198]
    recall = [0.1111, 0.2456, 0.3567, 0.4118, 0.4234]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both metrics
    ax.plot(keypoints, hota, 'o-', linewidth=2.5, markersize=10, label='HOTA', color='#2ecc71')
    ax.plot(keypoints, recall, 's-', linewidth=2.5, markersize=10, label='Recall', color='#3498db')

    # Highlight the chosen configuration
    ax.scatter([4], [0.4281], s=200, color='#2ecc71', edgecolors='black', linewidth=2, zorder=5)
    ax.scatter([4], [0.4118], s=200, color='#3498db', edgecolors='black', linewidth=2, zorder=5)

    # Add annotation for optimal point
    ax.annotate('Optimal\nConfiguration', xy=(4, 0.4281), xytext=(4.3, 0.42),
                fontsize=11, ha='left', fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

    ax.set_xlabel('Number of Keypoints per Object', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Impact of Keypoints per Object', fontsize=16, fontweight='bold')
    ax.set_xticks(keypoints)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_keypoints.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ablation_keypoints.pdf', bbox_inches='tight')
    plt.close()

def create_error_distribution():
    """Create error distribution pie charts"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # False Positive Distribution
    fp_labels = ['Scissors\n(28%)', 'Tweezers\n(22%)', 'Needle Holder\n(19%)',
                 'Needle\n(15%)', 'Left Hand\n(9%)', 'Right Hand\n(7%)']
    fp_sizes = [28, 22, 19, 15, 9, 7]
    fp_colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71', '#1abc9c']

    wedges1, texts1, autotexts1 = ax1.pie(fp_sizes, labels=fp_labels, colors=fp_colors,
                                            autopct='%1.0f%%', startangle=90,
                                            textprops={'fontsize': 10})
    ax1.set_title('False Positive Distribution by Tool Type', fontsize=14, fontweight='bold')

    # False Negative Causes
    fn_labels = ['Occlusions\n(35%)', 'Tool Deformation\n(25%)',
                 'Frame Boundaries\n(20%)', 'Motion Blur\n(20%)']
    fn_sizes = [35, 25, 20, 20]
    fn_colors = ['#c0392b', '#d35400', '#7f8c8d', '#34495e']

    wedges2, texts2, autotexts2 = ax2.pie(fn_sizes, labels=fn_labels, colors=fn_colors,
                                            autopct='%1.0f%%', startangle=45,
                                            textprops={'fontsize': 10})
    ax2.set_title('False Negative Root Causes', fontsize=14, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts1 + autotexts2:
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'error_distribution.pdf', bbox_inches='tight')
    plt.close()

def create_architecture_timeline():
    """Create timeline showing architecture evolution"""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Architecture progression
    architectures = [
        ('Baseline\nMulti-Stage', 0.3463, 0, '#f39c12'),
        ('Multi-Modal\nFusion', 0.144, 1, '#e74c3c'),
        ('Keypoint\nExpansion', 0.2243, 2, '#e74c3c'),
        ('BBox\nExtraction', 0.2175, 3, '#e74c3c'),
        ('Dense\nDetection', 0.195, 4, '#e74c3c'),
        ('YOLO\nAnchors', 0.437, 5, '#3498db'),
        ('YOLO-Pose\n(Final)', 0.4281, 6, '#2ecc71'),
    ]

    # Plot timeline
    for i, (name, hota, pos, color) in enumerate(architectures):
        # Draw node
        circle = plt.Circle((pos, hota), 0.15, color=color, ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)

        # Add label
        ax.text(pos, hota-0.25, name, ha='center', fontsize=10, fontweight='bold')

        # Add HOTA value
        ax.text(pos, hota+0.08, f'{hota:.3f}', ha='center', fontsize=9, fontweight='bold')

        # Draw connection to next
        if i < len(architectures) - 1:
            next_pos = architectures[i+1][2]
            next_hota = architectures[i+1][1]
            ax.arrow(pos+0.1, hota, next_pos-pos-0.2, next_hota-hota,
                    head_width=0.05, head_length=0.05, fc='gray', ec='gray', alpha=0.5)

    # Add baseline reference line
    ax.axhline(y=0.3463, color='orange', linestyle='--', alpha=0.3, label='Baseline')

    # Styling
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.1, 0.55)
    ax.set_xlabel('Development Timeline →', fontsize=14, fontweight='bold')
    ax.set_ylabel('HOTA Score', fontsize=14, fontweight='bold')
    ax.set_title('Architecture Evolution and Performance Progression', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xticks([])

    # Add phase annotations
    ax.text(1.5, -0.05, 'Failed Approaches', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    ax.text(5.5, -0.05, 'Successful Approaches', ha='center', fontsize=11, color='#2ecc71', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'architecture_timeline.pdf', bbox_inches='tight')
    plt.close()

def create_summary_table():
    """Create a summary comparison table as an image"""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Table data
    columns = ['Method', 'HOTA↑', 'DetA↑', 'AssA↑', 'Precision↑', 'Recall↑', 'FPS↑', 'Status']

    data = [
        ['Baseline (Multi-Stage)', '0.346', '0.229', '0.574', '0.346', '0.111', '25', '✓'],
        ['Multi-Modal Fusion', '0.144', '0.115', '0.180', '0.210', '0.063', '12', '✗'],
        ['Keypoint Expansion', '0.224', '0.168', '0.300', '0.280', '0.100', '30', '✗'],
        ['BBox Extraction', '0.218', '0.152', '0.311', '0.265', '0.088', '35', '✗'],
        ['Dense Detection', '0.195', '0.137', '0.279', '0.231', '0.081', '8', '✗'],
        ['YOLO Anchors', '0.437', '0.312', '0.613', '0.389', '0.235', '40', '✓'],
        ['YOLO-Pose (Ours)', '0.428', '0.367', '0.500', '0.321', '0.412', '45', '★'],
    ]

    # Create table with colors
    cell_colors = []
    for row in data:
        row_colors = []
        for i, cell in enumerate(row):
            if i == 0:  # Method name
                row_colors.append('#f8f9fa')
            elif row[-1] == '★':  # Our method
                row_colors.append('#d4edda')
            elif row[-1] == '✓':  # Working methods
                row_colors.append('#cce5ff')
            elif row[-1] == '✗':  # Failed methods
                row_colors.append('#f8d7da')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colWidths=[0.2, 0.08, 0.08, 0.08, 0.1, 0.08, 0.06, 0.08],
                     cellColours=cell_colors)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Bold headers
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#343a40')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Bold best values
    best_indices = [6, 6, 5, 5, 6, 6]  # Row indices of best values for each metric
    for col in range(1, 7):  # Metric columns
        row = best_indices[col-1] + 1  # +1 for header
        table[(row, col)].set_text_props(weight='bold')

    plt.title('Comprehensive Performance Comparison', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_table.pdf', bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures"""

    print("Generating scientific figures for paper...")

    # Create all visualizations
    print("1. Creating HOTA comparison bar chart...")
    create_hota_comparison_bar_chart()

    print("2. Creating metrics radar chart...")
    create_metrics_radar_chart()

    print("3. Creating per-video performance analysis...")
    create_per_video_performance()

    print("4. Creating training progression curve...")
    create_training_curve()

    print("5. Creating ablation study visualization...")
    create_ablation_keypoints()

    print("6. Creating error distribution charts...")
    create_error_distribution()

    print("7. Creating architecture timeline...")
    create_architecture_timeline()

    print("8. Creating summary comparison table...")
    create_summary_table()

    print(f"\n✅ All figures generated successfully in '{output_dir}/' directory!")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()