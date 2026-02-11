#!/usr/bin/env python3
"""
Generate YOLOv11 training labels from SPINEPS segmentations
Uses 3-slice approach: left-parasagittal, mid-sagittal, right-parasagittal
"""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2
from scipy import ndimage
import yaml

CLASSES = {
    0: 't12_rib',
    1: 'l5_vertebra',
    2: 'l5_transverse_process',
    3: 'sacrum',
    4: 'l4_vertebra',
}

def get_parasagittal_indices(data_shape, sag_axis):
    """Get slice indices for left, middle, right parasagittal views"""
    max_idx = data_shape[sag_axis]
    mid_idx = max_idx // 2
    offset = int(max_idx * 0.2)
    
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
    """Normalize to 0-255 uint8"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) / 
                     (img_slice.max() - img_slice.min()) * 255)
        return normalized.astype(np.uint8)
    return np.zeros_like(img_slice, dtype=np.uint8)

def extract_bounding_box(mask, label_id):
    """Extract YOLO format bounding box from segmentation mask"""
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

def detect_transverse_process(seg_slice, vertebra_label):
    """Detect transverse process region"""
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    y_coords, x_coords = np.where(vert_mask)
    cy, cx = y_coords.mean(), x_coords.mean()
    
    height, width = seg_slice.shape
    
    transverse_mask = vert_mask.copy()
    
    for y in range(height):
        for x in range(width):
            if transverse_mask[y, x]:
                dist_from_center = abs(x - cx)
                if dist_from_center < (width * 0.15):
                    transverse_mask[y, x] = False
    
    if not transverse_mask.any():
        return None
    
    return extract_bounding_box(transverse_mask.astype(int), 1)

def create_yolo_labels_multiview(nifti_path, seg_path, output_dir, images_dir):
    """Create YOLO format labels from SPINEPS segmentation"""
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
            
            image_filename = f"{study_id}_{view_name}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), mri_normalized)
            
            yolo_labels = []
            
            # Class 0: T12 rib (label 19 in SPINEPS)
            if view_name in ['left', 'right']:
                t12_box = extract_bounding_box(seg_slice, 19)
                if t12_box:
                    yolo_labels.append([0] + t12_box)
            
            # Class 1: L5 vertebra (label 24)
            l5_box = extract_bounding_box(seg_slice, 24)
            if l5_box:
                yolo_labels.append([1] + l5_box)
            
            # Class 2: L5 transverse process
            if view_name == 'mid':
                transverse_box = detect_transverse_process(seg_slice, 24)
                if transverse_box:
                    yolo_labels.append([2] + transverse_box)
            
            # Class 3: Sacrum (label 26)
            sacrum_box = extract_bounding_box(seg_slice, 26)
            if sacrum_box:
                yolo_labels.append([3] + sacrum_box)
            
            # Class 4: L4 vertebra (label 23)
            l4_box = extract_bounding_box(seg_slice, 23)
            if l4_box:
                yolo_labels.append([4] + l4_box)
            
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
        return False

def split_train_val(images_dir, labels_dir, val_split=0.15):
    """Split dataset into train/val"""
    
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
    parser = argparse.ArgumentParser(description='Generate YOLO weak labels from SPINEPS')
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
    print("WEAK LABEL GENERATION FROM SPINEPS")
    print("="*60)
    
    seg_files = sorted(seg_dir.glob("*_seg.nii.gz"))
    
    if args.limit:
        seg_files = seg_files[:args.limit]
    
    print(f"Found {len(seg_files)} segmentation files")
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
    
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 't12_rib',
            1: 'l5_vertebra',
            2: 'l5_transverse_process',
            3: 'sacrum',
            4: 'l4_vertebra',
        }
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\n✓ Created {yaml_path}")
    
    metadata = {
        'total_studies': success_count,
        'train_studies': train_count,
        'val_studies': val_count,
        'train_images': train_count * 3,
        'val_images': val_count * 3,
        'classes': dataset_config['names'],
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train: {train_count} studies ({train_count * 3} images)")
    print(f"Val:   {val_count} studies ({val_count * 3} images)")
    print(f"Classes: {len(dataset_config['names'])}")
    print(f"\nDataset ready at: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
