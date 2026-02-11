#!/usr/bin/env python3
"""
Label Fusion Pipeline
Combines weak labels from SPINEPS with human-refined labels from med students
Prioritizes human labels, falls back to weak labels
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_yolo_labels(label_path):
    """Load YOLO format labels from file"""
    labels = []
    
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                labels.append({
                    'class': class_id,
                    'x': x_center,
                    'y': y_center,
                    'w': width,
                    'h': height,
                })
    
    return labels

def save_yolo_labels(labels, label_path):
    """Save YOLO format labels to file"""
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(f"{label['class']} {label['x']:.6f} {label['y']:.6f} {label['w']:.6f} {label['h']:.6f}\n")

def iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_min = box1['x'] - box1['w'] / 2
    y1_min = box1['y'] - box1['h'] / 2
    x1_max = box1['x'] + box1['w'] / 2
    y1_max = box1['y'] + box1['h'] / 2
    
    x2_min = box2['x'] - box2['w'] / 2
    y2_min = box2['y'] - box2['h'] / 2
    x2_max = box2['x'] + box2['w'] / 2
    y2_max = box2['y'] + box2['h'] / 2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def fuse_labels(weak_labels, human_labels, iou_threshold=0.3):
    """
    Fuse weak and human labels
    Priority: Human labels > Weak labels
    
    Strategy:
    1. Keep all human labels
    2. For weak labels:
       - If overlaps with human label (IoU > threshold): discard (human is better)
       - If no overlap: keep (human missed it)
    """
    
    fused = list(human_labels)  # Start with all human labels
    
    for weak_box in weak_labels:
        # Check if this weak box overlaps with any human box of same class
        overlaps = False
        
        for human_box in human_labels:
            if weak_box['class'] == human_box['class']:
                if iou(weak_box, human_box) > iou_threshold:
                    overlaps = True
                    break
        
        if not overlaps:
            # Weak label doesn't overlap with human - keep it
            fused.append(weak_box)
    
    return fused

def calculate_fusion_metrics(weak_dir, human_dir, output_dir):
    """Calculate metrics about the fusion process"""
    
    metrics = {
        'total_images': 0,
        'images_with_human_labels': 0,
        'images_with_weak_only': 0,
        'total_weak_boxes': 0,
        'total_human_boxes': 0,
        'total_fused_boxes': 0,
        'weak_boxes_retained': 0,
        'weak_boxes_replaced': 0,
        'human_additions': 0,
        'per_class_metrics': defaultdict(lambda: {
            'weak': 0,
            'human': 0,
            'fused': 0,
        }),
    }
    
    weak_files = list(weak_dir.glob("*.txt"))
    human_files = list(human_dir.glob("*.txt"))
    fused_files = list(output_dir.glob("*.txt"))
    
    metrics['total_images'] = len(fused_files)
    metrics['images_with_human_labels'] = len(human_files)
    metrics['images_with_weak_only'] = len(weak_files) - len(human_files)
    
    # Count boxes
    for weak_file in weak_files:
        weak_labels = load_yolo_labels(weak_file)
        metrics['total_weak_boxes'] += len(weak_labels)
        for label in weak_labels:
            metrics['per_class_metrics'][label['class']]['weak'] += 1
    
    for human_file in human_files:
        human_labels = load_yolo_labels(human_file)
        metrics['total_human_boxes'] += len(human_labels)
        for label in human_labels:
            metrics['per_class_metrics'][label['class']]['human'] += 1
    
    for fused_file in fused_files:
        fused_labels = load_yolo_labels(fused_file)
        metrics['total_fused_boxes'] += len(fused_labels)
        for label in fused_labels:
            metrics['per_class_metrics'][label['class']]['fused'] += 1
    
    metrics['weak_boxes_retained'] = metrics['total_fused_boxes'] - metrics['total_human_boxes']
    metrics['weak_boxes_replaced'] = metrics['total_weak_boxes'] - metrics['weak_boxes_retained']
    metrics['human_additions'] = max(0, metrics['total_human_boxes'] - metrics['total_weak_boxes'])
    
    # Calculate improvement rates
    metrics['human_correction_rate'] = (
        metrics['weak_boxes_replaced'] / metrics['total_weak_boxes'] 
        if metrics['total_weak_boxes'] > 0 else 0
    )
    
    metrics['human_addition_rate'] = (
        metrics['human_additions'] / metrics['total_human_boxes']
        if metrics['total_human_boxes'] > 0 else 0
    )
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Fuse weak and human labels')
    parser.add_argument('--weak_labels_dir', type=str, required=True,
                       help='Directory with weak labels (from SPINEPS)')
    parser.add_argument('--human_labels_dir', type=str, required=True,
                       help='Directory with human-refined labels')
    parser.add_argument('--weak_images_dir', type=str, required=True,
                       help='Directory with images for weak labels')
    parser.add_argument('--human_images_dir', type=str, required=True,
                       help='Directory with images for human labels')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for fused dataset')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                       help='IoU threshold for considering boxes as overlapping')
    
    args = parser.parse_args()
    
    weak_labels_dir = Path(args.weak_labels_dir)
    human_labels_dir = Path(args.human_labels_dir)
    weak_images_dir = Path(args.weak_images_dir)
    human_images_dir = Path(args.human_images_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    output_labels_dir = output_dir / 'labels' / 'train'
    output_images_dir = output_dir / 'images' / 'train'
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LABEL FUSION PIPELINE")
    print("="*60)
    print(f"Weak labels:  {weak_labels_dir}")
    print(f"Human labels: {human_labels_dir}")
    print(f"Output:       {output_dir}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("="*60)
    
    # Get all image files
    weak_images = list(weak_images_dir.glob("*.jpg"))
    human_images = list(human_images_dir.glob("*.jpg"))
    
    print(f"\nWeak images: {len(weak_images)}")
    print(f"Human-labeled images: {len(human_images)}")
    
    # Process images
    fusion_count = 0
    weak_only_count = 0
    
    all_images = set([f.name for f in weak_images] + [f.name for f in human_images])
    
    for image_name in all_images:
        weak_label_file = weak_labels_dir / f"{Path(image_name).stem}.txt"
        human_label_file = human_labels_dir / f"{Path(image_name).stem}.txt"
        
        weak_image_file = weak_images_dir / image_name
        human_image_file = human_images_dir / image_name
        
        # Load labels
        weak_labels = load_yolo_labels(weak_label_file)
        human_labels = load_yolo_labels(human_label_file)
        
        # Determine output
        if len(human_labels) > 0:
            # Fuse labels
            fused_labels = fuse_labels(weak_labels, human_labels, args.iou_threshold)
            fusion_count += 1
        else:
            # No human labels, use weak only
            fused_labels = weak_labels
            weak_only_count += 1
        
        # Save fused labels
        output_label_file = output_labels_dir / f"{Path(image_name).stem}.txt"
        save_yolo_labels(fused_labels, output_label_file)
        
        # Copy image (prefer human-labeled version if exists)
        if human_image_file.exists():
            shutil.copy(human_image_file, output_images_dir / image_name)
        elif weak_image_file.exists():
            shutil.copy(weak_image_file, output_images_dir / image_name)
    
    print(f"\n✓ Processed {len(all_images)} images")
    print(f"  Fused (weak + human): {fusion_count}")
    print(f"  Weak only: {weak_only_count}")
    
    # Calculate metrics
    print("\nCalculating fusion metrics...")
    metrics = calculate_fusion_metrics(weak_labels_dir, human_labels_dir, output_labels_dir)
    
    print("\n" + "="*60)
    print("FUSION METRICS")
    print("="*60)
    print(f"Total images: {metrics['total_images']}")
    print(f"Human-labeled images: {metrics['images_with_human_labels']}")
    print(f"Weak-only images: {metrics['images_with_weak_only']}")
    print(f"\nBox counts:")
    print(f"  Weak boxes (total):      {metrics['total_weak_boxes']}")
    print(f"  Human boxes (total):     {metrics['total_human_boxes']}")
    print(f"  Fused boxes (total):     {metrics['total_fused_boxes']}")
    print(f"\nFusion details:")
    print(f"  Weak boxes retained:     {metrics['weak_boxes_retained']}")
    print(f"  Weak boxes replaced:     {metrics['weak_boxes_replaced']}")
    print(f"  Human additions (new):   {metrics['human_additions']}")
    print(f"\nImprovement rates:")
    print(f"  Human correction rate:   {metrics['human_correction_rate']*100:.1f}%")
    print(f"  Human addition rate:     {metrics['human_addition_rate']*100:.1f}%")
    print("="*60)
    
    # Per-class breakdown
    print("\nPer-class breakdown:")
    class_names = {
        0: 't12_vertebra',
        1: 't12_rib',
        2: 'l5_vertebra',
        3: 'l5_transverse_process',
        4: 'sacrum',
        5: 'l4_vertebra',
        6: 'l5_s1_disc',
    }
    
    for class_id in sorted(metrics['per_class_metrics'].keys()):
        class_name = class_names.get(class_id, f'class_{class_id}')
        weak_count = metrics['per_class_metrics'][class_id]['weak']
        human_count = metrics['per_class_metrics'][class_id]['human']
        fused_count = metrics['per_class_metrics'][class_id]['fused']
        
        improvement = fused_count - weak_count
        improvement_pct = (improvement / weak_count * 100) if weak_count > 0 else 0
        
        print(f"  {class_name:25s}: Weak={weak_count:4d}, Human={human_count:4d}, "
              f"Fused={fused_count:4d} ({improvement:+4d}, {improvement_pct:+5.1f}%)")
    
    # Save metrics
    metrics_path = output_dir / 'fusion_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=int)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Create dataset YAML
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names,
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Dataset config: {yaml_path}")
    
    print("\n" + "="*60)
    print("FUSION COMPLETE!")
    print("="*60)
    print(f"Fused dataset ready at: {output_dir}")
    print("\nNext steps:")
    print("  1. Review fusion metrics")
    print("  2. Run train/val split on fused dataset")
    print("  3. Train YOLOv11 on fused labels")
    print("  4. Compare with weak-only baseline")
    print("="*60)

if __name__ == "__main__":
    main()
