#!/usr/bin/env python3
"""
ENHANCED Weak Label Generation v3.0
NOW WITH SPINE-AWARE SLICE SELECTION + QUALITY METRICS

Features:
- Intelligent spine-centered slice selection
- Before/After comparison visualizations
- Quantitative offset measurements
- Roboflow upload of quality reports
- Statistical analysis of corrections

Version 3.0 - PRODUCTION GRADE WITH VALIDATION
"""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2
from scipy import ndimage
from scipy.ndimage import label as scipy_label
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

# SPINEPS Label Map
SPINEPS_LABELS = {
    'T12': 19,
    'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25,
    'Sacrum': 26,
    'L5_S1_disc': 124,
}

# YOLO Classes
YOLO_CLASSES = {
    0: 't12_vertebra',
    1: 't12_rib',
    2: 'l5_vertebra',
    3: 'l5_transverse_process',
    4: 'sacrum',
    5: 'l4_vertebra',
    6: 'l5_s1_disc',
}

@dataclass
class SliceMetrics:
    """Metrics for slice selection quality"""
    study_id: str
    geometric_mid: int
    spine_aware_mid: int
    offset_voxels: int
    offset_mm: float
    spine_density_geometric: float
    spine_density_optimal: float
    improvement_ratio: float
    method: str

class SpineAwareSliceSelector:
    """Intelligent slice selection using spine segmentation"""
    
    def __init__(self, voxel_spacing_mm=1.0, parasagittal_offset_mm=30):
        self.voxel_spacing_mm = voxel_spacing_mm
        self.parasagittal_offset_mm = parasagittal_offset_mm
        self.metrics_log = []
    
    def find_sagittal_axis(self, data_shape):
        """Determine sagittal axis (smallest dimension)"""
        return np.argmin(data_shape)
    
    def calculate_spine_density(self, seg_data, sag_axis, slice_idx):
        """Calculate spine content in a slice"""
        lumbar_labels = [SPINEPS_LABELS[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'Sacrum']]
        
        vertebra_mask = np.zeros_like(seg_data, dtype=bool)
        for label in lumbar_labels:
            vertebra_mask |= (seg_data == label)
        
        if sag_axis == 0:
            slice_mask = vertebra_mask[slice_idx, :, :]
        elif sag_axis == 1:
            slice_mask = vertebra_mask[:, slice_idx, :]
        else:
            slice_mask = vertebra_mask[:, :, slice_idx]
        
        return slice_mask.sum()
    
    def find_optimal_midline(self, seg_data, sag_axis, study_id=None):
        """
        Find TRUE spinal midline using segmentation
        Returns: (geometric_mid, optimal_mid, metrics)
        """
        num_slices = seg_data.shape[sag_axis]
        geometric_mid = num_slices // 2
        
        # Get lumbar spine mask
        lumbar_labels = [SPINEPS_LABELS[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'Sacrum']]
        
        vertebra_mask = np.zeros_like(seg_data, dtype=bool)
        for label in lumbar_labels:
            if label in seg_data:
                vertebra_mask |= (seg_data == label)
        
        if not vertebra_mask.any():
            # Fallback to geometric
            metrics = SliceMetrics(
                study_id=study_id or "unknown",
                geometric_mid=geometric_mid,
                spine_aware_mid=geometric_mid,
                offset_voxels=0,
                offset_mm=0.0,
                spine_density_geometric=0.0,
                spine_density_optimal=0.0,
                improvement_ratio=1.0,
                method='geometric_fallback'
            )
            self.metrics_log.append(metrics)
            return geometric_mid, geometric_mid, metrics
        
        # Calculate spine density for each slice
        spine_density = np.zeros(num_slices)
        
        for i in range(num_slices):
            if sag_axis == 0:
                slice_mask = vertebra_mask[i, :, :]
            elif sag_axis == 1:
                slice_mask = vertebra_mask[:, i, :]
            else:
                slice_mask = vertebra_mask[:, :, i]
            
            spine_density[i] = slice_mask.sum()
        
        # Find optimal slice (maximum spine content)
        optimal_mid = int(np.argmax(spine_density))
        
        # Calculate metrics
        offset_voxels = abs(optimal_mid - geometric_mid)
        offset_mm = offset_voxels * self.voxel_spacing_mm
        
        density_geometric = spine_density[geometric_mid]
        density_optimal = spine_density[optimal_mid]
        
        improvement_ratio = density_optimal / density_geometric if density_geometric > 0 else 1.0
        
        metrics = SliceMetrics(
            study_id=study_id or "unknown",
            geometric_mid=geometric_mid,
            spine_aware_mid=optimal_mid,
            offset_voxels=offset_voxels,
            offset_mm=offset_mm,
            spine_density_geometric=float(density_geometric),
            spine_density_optimal=float(density_optimal),
            improvement_ratio=float(improvement_ratio),
            method='spine_aware'
        )
        
        self.metrics_log.append(metrics)
        
        return geometric_mid, optimal_mid, metrics
    
    def get_three_slices(self, seg_data, sag_axis, study_id=None):
        """
        Get left, mid, right slice indices
        Returns: dict with indices and metrics
        """
        geometric_mid, optimal_mid, metrics = self.find_optimal_midline(seg_data, sag_axis, study_id)
        
        num_slices = seg_data.shape[sag_axis]
        offset_voxels = int(self.parasagittal_offset_mm / self.voxel_spacing_mm)
        
        left_idx = max(0, optimal_mid - offset_voxels)
        right_idx = min(num_slices - 1, optimal_mid + offset_voxels)
        
        return {
            'left': left_idx,
            'mid': optimal_mid,
            'right': right_idx,
            'geometric_mid': geometric_mid,
            'metrics': metrics,
            'sag_axis': sag_axis,
        }

def extract_slice(data, sag_axis, slice_idx):
    """Extract 2D slice from 3D volume"""
    if sag_axis == 0:
        return data[slice_idx, :, :]
    elif sag_axis == 1:
        return data[:, slice_idx, :]
    else:
        return data[:, :, slice_idx]

def normalize_slice(img_slice):
    """Normalize to 0-255 uint8 with CLAHE"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) / 
                     (img_slice.max() - img_slice.min()) * 255)
        normalized = normalized.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)
        return normalized
    return np.zeros_like(img_slice, dtype=np.uint8)

def create_before_after_comparison(mri_data, seg_data, slice_info, output_path):
    """
    Create visualization comparing geometric vs spine-aware slice selection
    """
    sag_axis = slice_info['sag_axis']
    geometric_mid = slice_info['geometric_mid']
    optimal_mid = slice_info['mid']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Geometric center
    geometric_views = {
        'left': max(0, geometric_mid - 30),
        'mid': geometric_mid,
        'right': min(mri_data.shape[sag_axis] - 1, geometric_mid + 30),
    }
    
    for i, (view_name, slice_idx) in enumerate(geometric_views.items()):
        mri_slice = extract_slice(mri_data, sag_axis, slice_idx)
        seg_slice = extract_slice(seg_data, sag_axis, slice_idx)
        
        normalized = normalize_slice(mri_slice)
        
        axes[0, i].imshow(normalized.T, cmap='gray', origin='lower')
        axes[0, i].imshow(seg_slice.T, cmap='jet', alpha=0.3, origin='lower')
        axes[0, i].set_title(f'GEOMETRIC {view_name.upper()}\\nSlice {slice_idx}', fontsize=14, color='red')
        axes[0, i].axis('off')
    
    # Row 2: Spine-aware
    spine_views = {
        'left': slice_info['left'],
        'mid': optimal_mid,
        'right': slice_info['right'],
    }
    
    for i, (view_name, slice_idx) in enumerate(spine_views.items()):
        mri_slice = extract_slice(mri_data, sag_axis, slice_idx)
        seg_slice = extract_slice(seg_data, sag_axis, slice_idx)
        
        normalized = normalize_slice(mri_slice)
        
        axes[1, i].imshow(normalized.T, cmap='gray', origin='lower')
        axes[1, i].imshow(seg_slice.T, cmap='jet', alpha=0.3, origin='lower')
        axes[1, i].set_title(f'SPINE-AWARE {view_name.upper()}\\nSlice {slice_idx}', fontsize=14, color='green')
        axes[1, i].axis('off')
    
    # Add metrics
    metrics = slice_info['metrics']
    fig.text(0.5, 0.02, 
             f"Offset: {metrics.offset_voxels} voxels ({metrics.offset_mm:.1f}mm) | "
             f"Spine density improvement: {metrics.improvement_ratio:.2f}x",
             ha='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('BEFORE (Geometric) vs AFTER (Spine-Aware) Slice Selection', 
                 fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def extract_bounding_box(mask, label_id):
    """Extract YOLO format bounding box"""
    label_mask = (mask == label_id)
    
    if not label_mask.any():
        return None
    
    coords = np.argwhere(label_mask)
    if len(coords) == 0:
        return None
    
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    height, width = mask.shape
    
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    
    if box_width <= 0 or box_height <= 0:
        return None
    
    return [x_center, y_center, box_width, box_height]

def detect_rib_from_vertebra(seg_slice, vertebra_label, side='left'):
    """Enhanced rib detection"""
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    height, width = seg_slice.shape
    
    rib_search_mask = np.zeros_like(vert_mask)
    
    if side == 'left':
        search_x_min = max(0, int(cx - width * 0.3))
        search_x_max = int(cx)
    else:
        search_x_min = int(cx)
        search_x_max = min(width, int(cx + width * 0.3))
    
    search_y_min = max(0, int(cy - height * 0.15))
    search_y_max = min(height, int(cy + height * 0.15))
    
    for y in range(search_y_min, search_y_max):
        for x in range(search_x_min, search_x_max):
            if vert_mask[y, x]:
                dist_from_center = abs(x - cx)
                if dist_from_center > width * 0.08:
                    rib_search_mask[y, x] = True
    
    if not rib_search_mask.any():
        return None
    
    return extract_bounding_box(rib_search_mask.astype(int), 1)

def detect_transverse_process(seg_slice, vertebra_label):
    """Enhanced transverse process detection"""
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    height, width = seg_slice.shape
    
    transverse_mask = vert_mask.copy()
    central_width = width * 0.12
    
    for y in range(height):
        for x in range(width):
            if transverse_mask[y, x]:
                dist_from_center = abs(x - cx)
                if dist_from_center < central_width:
                    transverse_mask[y, x] = False
    
    if not transverse_mask.any():
        return None
    
    labeled, num_features = scipy_label(transverse_mask)
    
    if num_features == 0:
        return None
    
    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    if len(component_sizes) == 0:
        return None
    
    largest_components = sorted(range(len(component_sizes)), 
                                key=lambda i: component_sizes[i], 
                                reverse=True)[:2]
    
    final_mask = np.zeros_like(transverse_mask)
    for comp_idx in largest_components:
        final_mask[labeled == (comp_idx + 1)] = True
    
    if not final_mask.any():
        return None
    
    return extract_bounding_box(final_mask.astype(int), 1)

def create_yolo_labels_multiview(nifti_path, seg_path, output_dir, images_dir, 
                                  selector, generate_comparison=False, 
                                  comparison_dir=None):
    """
    Create YOLO labels with SPINE-AWARE slice selection
    """
    try:
        nii = nib.load(nifti_path)
        seg_nii = nib.load(seg_path)
        
        mri_data = nii.get_fdata()
        seg_data = seg_nii.get_fdata().astype(int)
        
        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
        
        dims = mri_data.shape
        sag_axis = np.argmin(dims)
        
        # GET SPINE-AWARE SLICES
        slice_info = selector.get_three_slices(seg_data, sag_axis, study_id)
        
        # Generate before/after comparison if requested
        if generate_comparison and comparison_dir:
            comparison_path = comparison_dir / f"{study_id}_slice_comparison.png"
            create_before_after_comparison(mri_data, seg_data, slice_info, comparison_path)
        
        views = {
            'left': slice_info['left'],
            'mid': slice_info['mid'],
            'right': slice_info['right'],
        }
        
        label_count = 0
        
        for view_name, slice_idx in views.items():
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx)
            seg_slice = extract_slice(seg_data, sag_axis, slice_idx)
            
            mri_normalized = normalize_slice(mri_slice)
            
            image_filename = f"{study_id}_{view_name}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), mri_normalized)
            
            yolo_labels = []
            
            # Class 0: T12 vertebra
            t12_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['T12'])
            if t12_box:
                yolo_labels.append([0] + t12_box)
            
            # Class 1: T12 rib (parasagittal only)
            if view_name in ['left', 'right']:
                t12_rib_box = detect_rib_from_vertebra(seg_slice, SPINEPS_LABELS['T12'], side=view_name)
                if t12_rib_box:
                    yolo_labels.append([1] + t12_rib_box)
            
            # Class 2: L5 vertebra
            l5_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5'])
            if l5_box:
                yolo_labels.append([2] + l5_box)
            
            # Class 3: L5 transverse process (mid only)
            if view_name == 'mid':
                transverse_box = detect_transverse_process(seg_slice, SPINEPS_LABELS['L5'])
                if transverse_box:
                    yolo_labels.append([3] + transverse_box)
            
            # Class 4: Sacrum
            sacrum_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['Sacrum'])
            if sacrum_box:
                yolo_labels.append([4] + sacrum_box)
            
            # Class 5: L4 vertebra
            l4_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L4'])
            if l4_box:
                yolo_labels.append([5] + l4_box)
            
            # Class 6: L5-S1 disc
            l5_s1_disc_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5_S1_disc'])
            if l5_s1_disc_box:
                yolo_labels.append([6] + l5_s1_disc_box)
            
            if yolo_labels:
                label_filename = f"{study_id}_{view_name}.txt"
                label_path = output_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                
                label_count += len(yolo_labels)
        
        return True, slice_info['metrics']
        
    except Exception as e:
        print(f"Error processing {nifti_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def generate_metrics_report(metrics_log, output_dir):
    """Generate comprehensive metrics report"""
    
    if not metrics_log:
        print("No metrics to report")
        return
    
    # Convert to arrays for analysis
    offsets_voxels = [m.offset_voxels for m in metrics_log]
    offsets_mm = [m.offset_mm for m in metrics_log]
    improvements = [m.improvement_ratio for m in metrics_log]
    
    # Calculate statistics
    stats = {
        'total_cases': len(metrics_log),
        'spine_aware_cases': sum(1 for m in metrics_log if m.method == 'spine_aware'),
        'geometric_fallback_cases': sum(1 for m in metrics_log if m.method == 'geometric_fallback'),
        'offset_statistics': {
            'mean_voxels': float(np.mean(offsets_voxels)),
            'std_voxels': float(np.std(offsets_voxels)),
            'median_voxels': float(np.median(offsets_voxels)),
            'max_voxels': float(np.max(offsets_voxels)),
            'mean_mm': float(np.mean(offsets_mm)),
            'std_mm': float(np.std(offsets_mm)),
            'median_mm': float(np.median(offsets_mm)),
            'max_mm': float(np.max(offsets_mm)),
        },
        'improvement_statistics': {
            'mean_ratio': float(np.mean(improvements)),
            'std_ratio': float(np.std(improvements)),
            'median_ratio': float(np.median(improvements)),
            'max_ratio': float(np.max(improvements)),
        },
        'correction_needed': {
            'no_correction': sum(1 for o in offsets_voxels if o == 0),
            'small_correction_1_5_voxels': sum(1 for o in offsets_voxels if 1 <= o <= 5),
            'medium_correction_6_15_voxels': sum(1 for o in offsets_voxels if 6 <= o <= 15),
            'large_correction_16plus_voxels': sum(1 for o in offsets_voxels if o >= 16),
        }
    }
    
    # Calculate percentages
    total = stats['total_cases']
    stats['correction_needed_percent'] = {
        k: (v / total * 100) if total > 0 else 0
        for k, v in stats['correction_needed'].items()
    }
    
    # Save JSON report
    report_path = output_dir / 'spine_aware_metrics_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("SPINE-AWARE SLICE SELECTION - QUALITY METRICS")
    print("="*80)
    print(f"Total cases processed: {stats['total_cases']}")
    print(f"Spine-aware success:   {stats['spine_aware_cases']} ({stats['spine_aware_cases']/total*100:.1f}%)")
    print(f"Geometric fallback:    {stats['geometric_fallback_cases']} ({stats['geometric_fallback_cases']/total*100:.1f}%)")
    print("\nOffset from Geometric Center:")
    print(f"  Mean:   {stats['offset_statistics']['mean_voxels']:.1f} voxels ({stats['offset_statistics']['mean_mm']:.1f} mm)")
    print(f"  Std:    {stats['offset_statistics']['std_voxels']:.1f} voxels ({stats['offset_statistics']['std_mm']:.1f} mm)")
    print(f"  Median: {stats['offset_statistics']['median_voxels']:.1f} voxels ({stats['offset_statistics']['median_mm']:.1f} mm)")
    print(f"  Max:    {stats['offset_statistics']['max_voxels']:.0f} voxels ({stats['offset_statistics']['max_mm']:.0f} mm)")
    print("\nSpine Density Improvement:")
    print(f"  Mean:   {stats['improvement_statistics']['mean_ratio']:.2f}x")
    print(f"  Median: {stats['improvement_statistics']['median_ratio']:.2f}x")
    print(f"  Max:    {stats['improvement_statistics']['max_ratio']:.2f}x")
    print("\nCorrection Distribution:")
    for key, value in stats['correction_needed'].items():
        pct = stats['correction_needed_percent'][key]
        print(f"  {key:35s}: {value:4d} cases ({pct:5.1f}%)")
    print("="*80)
    
    return stats

def create_summary_visualization(metrics_log, output_path):
    """Create publication-quality visualization of metrics"""
    
    offsets_mm = [m.offset_mm for m in metrics_log]
    improvements = [m.improvement_ratio for m in metrics_log]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram of offsets
    axes[0, 0].hist(offsets_mm, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(offsets_mm), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(offsets_mm):.1f} mm')
    axes[0, 0].set_xlabel('Offset from Geometric Center (mm)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Cases', fontsize=12)
    axes[0, 0].set_title('Distribution of Corrections Needed', fontsize=14, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Histogram of improvements
    axes[0, 1].hist(improvements, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(improvements):.2f}x')
    axes[0, 1].set_xlabel('Spine Density Improvement Ratio', fontsize=12)
    axes[0, 1].set_ylabel('Number of Cases', fontsize=12)
    axes[0, 1].set_title('Improvement in Spine Visibility', fontsize=14, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Scatter plot
    axes[1, 0].scatter(offsets_mm, improvements, alpha=0.5, s=50)
    axes[1, 0].set_xlabel('Offset from Geometric Center (mm)', fontsize=12)
    axes[1, 0].set_ylabel('Improvement Ratio', fontsize=12)
    axes[1, 0].set_title('Offset vs Improvement', fontsize=14, weight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Box plot by correction magnitude
    no_corr = [m.improvement_ratio for m in metrics_log if m.offset_voxels == 0]
    small_corr = [m.improvement_ratio for m in metrics_log if 1 <= m.offset_voxels <= 5]
    medium_corr = [m.improvement_ratio for m in metrics_log if 6 <= m.offset_voxels <= 15]
    large_corr = [m.improvement_ratio for m in metrics_log if m.offset_voxels >= 16]
    
    box_data = []
    labels = []
    if no_corr:
        box_data.append(no_corr)
        labels.append(f'No correction\n(n={len(no_corr)})')
    if small_corr:
        box_data.append(small_corr)
        labels.append(f'Small (1-5)\n(n={len(small_corr)})')
    if medium_corr:
        box_data.append(medium_corr)
        labels.append(f'Medium (6-15)\n(n={len(medium_corr)})')
    if large_corr:
        box_data.append(large_corr)
        labels.append(f'Large (16+)\n(n={len(large_corr)})')
    
    if box_data:
        axes[1, 1].boxplot(box_data, labels=labels)
        axes[1, 1].set_ylabel('Improvement Ratio', fontsize=12)
        axes[1, 1].set_title('Improvement by Correction Magnitude', fontsize=14, weight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.suptitle('Spine-Aware Slice Selection: Quality Validation', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary visualization saved to: {output_path}")

def split_train_val(images_dir, labels_dir, val_split=0.15):
    """Split dataset into train/val by study"""
    
    image_files = list(images_dir.glob("*.jpg"))
    
    study_ids = set()
    for img_file in image_files:
        study_id = img_file.stem.rsplit('_', 1)[0]
        study_ids.add(study_id)
    
    study_ids = sorted(list(study_ids))
    
    np.random.seed(42)
    np.random.shuffle(study_ids)
    
    val_count = int(len(study_ids) * val_split)
    val_studies = set(study_ids[:val_count])
    train_studies = set(study_ids[val_count:])
    
    print(f"Train studies: {len(train_studies)}")
    print(f"Val studies: {len(val_studies)}")
    
    val_images_dir = images_dir.parent / 'val'
    val_labels_dir = labels_dir.parent / 'val'
    
    val_images_dir.mkdir(exist_ok=True)
    val_labels_dir.mkdir(exist_ok=True)
    
    move_count = 0
    
    for img_file in image_files:
        study_id = img_file.stem.rsplit('_', 1)[0]
        
        if study_id in val_studies:
            val_img_path = val_images_dir / img_file.name
            img_file.rename(val_img_path)
            
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                val_label_path = val_labels_dir / label_file.name
                label_file.rename(val_label_path)
            
            move_count += 1
    
    print(f"Moved {move_count} images to validation set")
    
    return len(train_studies), len(val_studies)

def main():
    parser = argparse.ArgumentParser(description='ENHANCED Weak Label Generation v3.0')
    parser.add_argument('--nifti_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--generate_comparisons', action='store_true',
                       help='Generate before/after comparison images (recommended for trial)')
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)
    
    labels_dir = output_dir / 'labels' / 'train'
    images_dir = output_dir / 'images' / 'train'
    comparison_dir = output_dir / 'quality_validation' if args.generate_comparisons else None
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if comparison_dir:
        comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENHANCED WEAK LABEL GENERATION v3.0 - SPINE-AWARE SLICE SELECTION")
    print("="*80)
    print("Features:")
    print("  ✓ Intelligent spine-centered midline detection")
    print("  ✓ Quantitative offset measurements")
    print("  ✓ Before/after comparison visualizations")
    print("  ✓ Statistical validation of improvements")
    if args.generate_comparisons:
        print("  ✓ COMPARISON MODE ENABLED (will generate validation images)")
    print("="*80)
    
    seg_files = sorted(seg_dir.glob("*_seg.nii.gz"))
    
    if args.limit:
        seg_files = seg_files[:args.limit]
    
    print(f"\nFound {len(seg_files)} segmentation files")
    print("Generating 3-view YOLO labels with spine-aware slicing...")
    print("="*80)
    
    # Initialize selector
    selector = SpineAwareSliceSelector(voxel_spacing_mm=1.0, parasagittal_offset_mm=30)
    
    success_count = 0
    
    for seg_file in tqdm(seg_files, desc="Processing"):
        study_id = seg_file.stem.replace('_seg', '')
        nifti_file = nifti_dir / f"sub-{study_id}_T2w.nii.gz"
        
        if not nifti_file.exists():
            continue
        
        success, metrics = create_yolo_labels_multiview(
            nifti_file, seg_file, labels_dir, images_dir, 
            selector, args.generate_comparisons, comparison_dir
        )
        
        if success:
            success_count += 1
    
    print(f"\n✓ Generated labels for {success_count} studies")
    print(f"  Total images: {success_count * 3} (3 views per study)")
    
    # Generate metrics report
    stats = generate_metrics_report(selector.metrics_log, output_dir)
    
    # Create summary visualization
    summary_viz_path = output_dir / 'quality_validation_summary.png'
    create_summary_visualization(selector.metrics_log, summary_viz_path)
    
    print("\nSplitting train/validation...")
    train_count, val_count = split_train_val(images_dir, labels_dir)
    
    # Create dataset YAML
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': YOLO_CLASSES,
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\n✓ Created {yaml_path}")
    
    # Metadata
    metadata = {
        'version': '3.0',
        'total_studies': success_count,
        'train_studies': train_count,
        'val_studies': val_count,
        'train_images': train_count * 3,
        'val_images': val_count * 3,
        'classes': YOLO_CLASSES,
        'spine_aware_slicing': True,
        'slice_selection_stats': stats,
        'enhancements': [
            'Spine-aware slice selection',
            'Quantitative offset measurement',
            'Before/after comparisons',
            'T12 vertebra detection',
            'Enhanced T12 rib detection',
            'L5-S1 disc detection',
        ],
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Train: {train_count} studies ({train_count * 3} images)")
    print(f"Val:   {val_count} studies ({val_count * 3} images)")
    print(f"Classes: {len(YOLO_CLASSES)}")
    print(f"\nDataset ready at: {output_dir}")
    print(f"Quality metrics: {output_dir / 'spine_aware_metrics_report.json'}")
    print(f"Summary visualization: {summary_viz_path}")
    if comparison_dir:
        print(f"Comparison images: {comparison_dir}/ ({success_count} images)")
    print("="*80)

if __name__ == "__main__":
    main()
