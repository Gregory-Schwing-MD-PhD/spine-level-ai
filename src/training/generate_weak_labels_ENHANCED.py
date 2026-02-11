#!/usr/bin/env python3
"""
ENHANCED Weak Label Generation from SPINEPS
Extracts ALL available anatomical information:
- T12 vertebra (label 19)
- T12 rib attachment
- L4, L5 vertebrae
- L5 transverse processes
- Sacrum
- Intervertebral discs (for fusion detection)

Version 2.0 - Production Grade
"""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2
from scipy import ndimage
from scipy.ndimage import binary_erosion, label as scipy_label
import yaml

# SPINEPS Label Map (from official documentation)
SPINEPS_LABELS = {
    # Vertebrae
    'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
    'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13,
    'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
    'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25,
    'Sacrum': 26,
    # Discs (100+)
    'T12_L1_disc': 119,
    'L1_L2_disc': 120,
    'L2_L3_disc': 121,
    'L3_L4_disc': 122,
    'L4_L5_disc': 123,
    'L5_S1_disc': 124,
    'S1_S2_disc': 126,  # Sacralization indicator!
}

# YOLO Classes for Detection
YOLO_CLASSES = {
    0: 't12_vertebra',      # NEW! Critical for enumeration
    1: 't12_rib',           # Enhanced extraction
    2: 'l5_vertebra',
    3: 'l5_transverse_process',
    4: 'sacrum',
    5: 'l4_vertebra',
    6: 'l5_s1_disc',        # NEW! For fusion detection
}

def get_parasagittal_indices(data_shape, sag_axis):
    """Get slice indices for left, middle, right parasagittal views"""
    max_idx = data_shape[sag_axis]
    mid_idx = max_idx // 2
    offset = int(max_idx * 0.20)  # 20% offset for parasagittal
    
    left_idx = max(0, mid_idx - offset)
    right_idx = min(max_idx - 1, mid_idx + offset)
    
    return left_idx, mid_idx, right_idx

def extract_slice(data, sag_axis, slice_idx):
    """Extract 2D slice from 3D volume"""
    if sag_axis == 0:
        return data[slice_idx, :, :]
    elif sag_axis == 1:
        return data[:, slice_idx, :]
    else:
        return data[:, :, slice_idx]

def normalize_slice(img_slice):
    """Normalize to 0-255 uint8 with CLAHE enhancement"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) / 
                     (img_slice.max() - img_slice.min()) * 255)
        normalized = normalized.astype(np.uint8)
        
        # CLAHE enhancement for better visibility
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)
        
        return normalized
    return np.zeros_like(img_slice, dtype=np.uint8)

def extract_bounding_box(mask, label_id):
    """
    Extract YOLO format bounding box from segmentation mask
    Returns: [x_center, y_center, width, height] normalized to [0,1]
    """
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
    
    # YOLO format: x_center, y_center, width, height (all normalized)
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    
    # Sanity checks
    if box_width <= 0 or box_height <= 0:
        return None
    if box_width > 1 or box_height > 1:
        return None
    
    return [x_center, y_center, box_width, box_height]

def detect_rib_from_vertebra(seg_slice, vertebra_label, side='left'):
    """
    ENHANCED: Detect rib attachment from vertebra
    Ribs extend laterally from vertebral body
    
    Args:
        seg_slice: 2D segmentation
        vertebra_label: Vertebra label (e.g., 19 for T12)
        side: 'left' or 'right' parasagittal view
    """
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    # Find vertebra centroid
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    height, width = seg_slice.shape
    
    # Ribs extend LATERALLY from vertebra
    # On parasagittal views, ribs appear as extensions
    rib_search_mask = np.zeros_like(vert_mask)
    
    # Search region: lateral to vertebra center
    if side == 'left':
        # Rib extends to the left (lower x values)
        search_x_min = max(0, int(cx - width * 0.3))
        search_x_max = int(cx)
    else:  # right
        # Rib extends to the right (higher x values)
        search_x_min = int(cx)
        search_x_max = min(width, int(cx + width * 0.3))
    
    search_y_min = max(0, int(cy - height * 0.15))
    search_y_max = min(height, int(cy + height * 0.15))
    
    # In SPINEPS, ribs are part of the vertebra segmentation on parasagittal
    # We look for the lateral extension of the vertebra mask
    for y in range(search_y_min, search_y_max):
        for x in range(search_x_min, search_x_max):
            if vert_mask[y, x]:
                # Check if this is lateral extension (rib region)
                dist_from_center = abs(x - cx)
                if dist_from_center > width * 0.08:  # Beyond vertebral body
                    rib_search_mask[y, x] = True
    
    if not rib_search_mask.any():
        return None
    
    # Extract bounding box for rib region
    rib_mask_int = rib_search_mask.astype(int)
    return extract_bounding_box(rib_mask_int, 1)

def detect_transverse_process(seg_slice, vertebra_label):
    """
    ENHANCED: Detect transverse process using morphological analysis
    Transverse processes are lateral projections from vertebral body
    """
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    # Find vertebral body center
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    height, width = seg_slice.shape
    
    # Transverse processes: lateral projections, exclude central body
    transverse_mask = vert_mask.copy()
    
    # Define central body region (to exclude)
    central_width = width * 0.12  # ~12% of image width
    
    for y in range(height):
        for x in range(width):
            if transverse_mask[y, x]:
                dist_from_center = abs(x - cx)
                # If within central body region, exclude
                if dist_from_center < central_width:
                    transverse_mask[y, x] = False
    
    if not transverse_mask.any():
        return None
    
    # Additional morphological cleanup
    # Transverse processes should be connected components
    labeled, num_features = scipy_label(transverse_mask)
    
    if num_features == 0:
        return None
    
    # Take the two largest components (left and right transverse processes)
    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    if len(component_sizes) == 0:
        return None
    
    # Create mask of largest components
    largest_components = sorted(range(len(component_sizes)), 
                                key=lambda i: component_sizes[i], 
                                reverse=True)[:2]
    
    final_mask = np.zeros_like(transverse_mask)
    for comp_idx in largest_components:
        final_mask[labeled == (comp_idx + 1)] = True
    
    if not final_mask.any():
        return None
    
    return extract_bounding_box(final_mask.astype(int), 1)

def create_yolo_labels_multiview(nifti_path, seg_path, output_dir, images_dir):
    """
    ENHANCED: Create YOLO format labels from SPINEPS segmentation
    Now extracts ALL available anatomical information
    """
    try:
        nii = nib.load(nifti_path)
        seg_nii = nib.load(seg_path)
        
        mri_data = nii.get_fdata()
        seg_data = seg_nii.get_fdata().astype(int)
        
        dims = mri_data.shape
        sag_axis = np.argmin(dims)
        
        left_idx, mid_idx, right_idx = get_parasagittal_indices(dims, sag_axis)
        
        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
        
        views = {
            'left': left_idx,
            'mid': mid_idx,
            'right': right_idx,
        }
        
        label_count = 0
        
        for view_name, slice_idx in views.items():
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx)
            seg_slice = extract_slice(seg_data, sag_axis, slice_idx)
            
            mri_normalized = normalize_slice(mri_slice)
            
            # Save image
            image_filename = f"{study_id}_{view_name}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), mri_normalized)
            
            yolo_labels = []
            
            # Class 0: T12 vertebra (NEW!)
            t12_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['T12'])
            if t12_box:
                yolo_labels.append([0] + t12_box)
            
            # Class 1: T12 rib (ENHANCED - lateral views only)
            if view_name in ['left', 'right']:
                t12_rib_box = detect_rib_from_vertebra(seg_slice, SPINEPS_LABELS['T12'], side=view_name)
                if t12_rib_box:
                    yolo_labels.append([1] + t12_rib_box)
            
            # Class 2: L5 vertebra
            l5_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5'])
            if l5_box:
                yolo_labels.append([2] + l5_box)
            
            # Class 3: L5 transverse process (mid-sagittal only)
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
            
            # Class 6: L5-S1 disc (NEW! - fusion indicator)
            l5_s1_disc_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5_S1_disc'])
            if l5_s1_disc_box:
                yolo_labels.append([6] + l5_s1_disc_box)
            
            # Write YOLO format labels
            if yolo_labels:
                label_filename = f"{study_id}_{view_name}.txt"
                label_path = output_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                
                label_count += len(yolo_labels)
        
        return label_count > 0
        
    except Exception as e:
        print(f"Error processing {nifti_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def split_train_val(images_dir, labels_dir, val_split=0.15):
    """Split dataset into train/val by STUDY (not by image)"""
    
    image_files = list(images_dir.glob("*.jpg"))
    
    # Group by study ID
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
    
    # Create validation directories
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

def generate_quality_report(output_dir, nifti_dir, seg_dir):
    """Generate quality metrics for weak labels"""
    
    labels_train = list((output_dir / 'labels' / 'train').glob("*.txt"))
    labels_val = list((output_dir / 'labels' / 'val').glob("*.txt"))
    
    all_labels = labels_train + labels_val
    
    class_counts = {i: 0 for i in range(7)}
    total_boxes = 0
    
    for label_file in all_labels:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_boxes += 1
    
    report = {
        'total_images': len(all_labels),
        'total_boxes': total_boxes,
        'class_distribution': {
            YOLO_CLASSES[k]: v for k, v in class_counts.items()
        },
        'avg_boxes_per_image': total_boxes / len(all_labels) if all_labels else 0,
    }
    
    # Detection rates
    for class_id, class_name in YOLO_CLASSES.items():
        detection_rate = class_counts[class_id] / len(all_labels) if all_labels else 0
        report[f'{class_name}_detection_rate'] = detection_rate
    
    # Save report
    report_path = output_dir / 'weak_label_quality_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("WEAK LABEL QUALITY REPORT")
    print("="*60)
    print(f"Total images: {report['total_images']}")
    print(f"Total boxes: {report['total_boxes']}")
    print(f"Avg boxes/image: {report['avg_boxes_per_image']:.2f}")
    print("\nClass distribution:")
    for class_name, count in report['class_distribution'].items():
        rate = count / report['total_images'] if report['total_images'] > 0 else 0
        print(f"  {class_name:25s}: {count:5d} ({rate*100:5.1f}%)")
    print("="*60)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='ENHANCED Weak Label Generation')
    parser.add_argument('--nifti_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)
    
    labels_dir = output_dir / 'labels' / 'train'
    images_dir = output_dir / 'images' / 'train'
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ENHANCED WEAK LABEL GENERATION v2.0")
    print("="*60)
    print("New features:")
    print("  ✓ T12 vertebra detection")
    print("  ✓ Enhanced T12 rib detection")
    print("  ✓ L5-S1 disc detection (fusion indicator)")
    print("  ✓ Improved transverse process detection")
    print("  ✓ Quality metrics reporting")
    print("="*60)
    
    seg_files = sorted(seg_dir.glob("*_seg.nii.gz"))
    
    if args.limit:
        seg_files = seg_files[:args.limit]
    
    print(f"\nFound {len(seg_files)} segmentation files")
    print("Generating 3-view YOLO labels...")
    print("="*60)
    
    success_count = 0
    
    for seg_file in tqdm(seg_files, desc="Processing"):
        study_id = seg_file.stem.replace('_seg', '')
        nifti_file = nifti_dir / f"sub-{study_id}_T2w.nii.gz"
        
        if not nifti_file.exists():
            continue
        
        if create_yolo_labels_multiview(nifti_file, seg_file, labels_dir, images_dir):
            success_count += 1
    
    print(f"\n✓ Generated labels for {success_count} studies")
    print(f"  Total images: {success_count * 3} (3 views per study)")
    
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
    
    # Generate quality report
    quality_report = generate_quality_report(output_dir, nifti_dir, seg_dir)
    
    # Metadata
    metadata = {
        'version': '2.0',
        'total_studies': success_count,
        'train_studies': train_count,
        'val_studies': val_count,
        'train_images': train_count * 3,
        'val_images': val_count * 3,
        'classes': YOLO_CLASSES,
        'enhancements': [
            'T12 vertebra detection',
            'Enhanced T12 rib detection',
            'L5-S1 disc detection',
            'Improved transverse process detection',
        ],
        'quality_metrics': quality_report,
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train: {train_count} studies ({train_count * 3} images)")
    print(f"Val:   {val_count} studies ({val_count * 3} images)")
    print(f"Classes: {len(YOLO_CLASSES)}")
    print(f"\nDataset ready at: {output_dir}")
    print(f"Quality report: {output_dir / 'weak_label_quality_report.json'}")
    print("="*60)

if __name__ == "__main__":
    main()
