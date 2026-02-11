#!/usr/bin/env python3
"""
Evaluate YOLOv11 LSTV detection model
"""

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import torch
from tqdm import tqdm
import cv2

def load_model(weights_path):
    """Load trained YOLO model"""
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    return model

def evaluate_on_dataset(model, data_yaml, conf_threshold=0.25):
    """Run validation and collect detailed metrics"""
    print("\nRunning validation...")
    
    metrics = model.val(data=data_yaml, conf=conf_threshold)
    
    results = {
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'per_class_ap': {},
    }
    
    class_names = ['t12_rib', 'l5_vertebra', 'l5_transverse_process', 'sacrum', 'l4_vertebra']
    
    for i, name in enumerate(class_names):
        if i < len(metrics.box.ap50):
            results['per_class_ap'][name] = {
                'ap50': float(metrics.box.ap50[i]),
                'ap50_95': float(metrics.box.ap[i]),
            }
    
    return results, metrics

def create_visualizations(results, output_dir):
    """Create evaluation visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-class AP50 bar chart
    class_names = list(results['per_class_ap'].keys())
    ap50_values = [results['per_class_ap'][name]['ap50'] for name in class_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, ap50_values)
    plt.xlabel('Class')
    plt.ylabel('AP@50')
    plt.title('Average Precision by Class')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    
    for bar, ap in zip(bars, ap50_values):
        if ap > 0.7:
            bar.set_color('green')
        elif ap > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_ap50.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Visualizations saved to {output_dir}")

def test_on_lstv_cases(model, test_images_dir, output_dir, conf_threshold=0.25):
    """Test model on specific LSTV candidate images"""
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(test_images_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"\nTesting on {len(image_files)} images...")
    
    detections_summary = []
    
    for img_file in tqdm(image_files[:20], desc="Testing"):
        results = model(str(img_file), conf=conf_threshold)[0]
        boxes = results.boxes
        
        detection_info = {
            'image': img_file.name,
            'detections': [],
        }
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                detection_info['detections'].append({
                    'class_id': cls_id,
                    'class_name': model.names[cls_id],
                    'confidence': conf,
                })
        
        detections_summary.append(detection_info)
        
        annotated = results.plot()
        cv2.imwrite(str(output_dir / img_file.name), annotated)
    
    with open(output_dir / 'detections_summary.json', 'w') as f:
        json.dump(detections_summary, f, indent=2)
    
    class_counts = {}
    for det in detections_summary:
        for d in det['detections']:
            class_name = d['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nDetection Statistics:")
    print("--------------------")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name:25s}: {count:4d} detections")
    
    print(f"\n✓ Annotated images saved to {output_dir}")

def generate_report(results, output_path):
    """Generate detailed evaluation report"""
    
    report = f"""# LSTV Detection Model - Evaluation Report

## Overall Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | {results['map50']:.4f} |
| mAP@50-95 | {results['map50_95']:.4f} |
| Precision | {results['precision']:.4f} |
| Recall | {results['recall']:.4f} |

## Per-Class Performance

"""
    
    for class_name, metrics in results['per_class_ap'].items():
        report += f"\n### {class_name}\n\n"
        report += f"- AP@50: {metrics['ap50']:.4f}\n"
        report += f"- AP@50-95: {metrics['ap50_95']:.4f}\n"
    
    report += f"""

## Interpretation

### Overall Performance
"""
    
    if results['map50'] > 0.7:
        report += "- ✅ Excellent detection performance (mAP@50 > 0.7)\n"
    elif results['map50'] > 0.5:
        report += "- ⚠️  Good detection performance (mAP@50 > 0.5)\n"
    else:
        report += "- ❌ Poor detection performance (mAP@50 < 0.5) - needs improvement\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Evaluation report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTV detection model')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/evaluation')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--test-images', type=str, default=None)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LSTV DETECTION MODEL EVALUATION")
    print("="*60)
    print(f"Weights: {args.weights}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    
    model = load_model(args.weights)
    results, metrics = evaluate_on_dataset(model, args.data, args.conf)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP@50:    {results['map50']:.4f}")
    print(f"mAP@50-95: {results['map50_95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print("\nPer-class AP@50:")
    for class_name, class_metrics in results['per_class_ap'].items():
        print(f"  {class_name:25s}: {class_metrics['ap50']:.4f}")
    print("="*60)
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")
    
    create_visualizations(results, output_dir / 'plots')
    generate_report(results, output_dir / 'EVALUATION_REPORT.md')
    
    if args.test_images:
        test_on_lstv_cases(model, args.test_images, output_dir / 'test_predictions', args.conf)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
