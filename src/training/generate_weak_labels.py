#!/usr/bin/env python3
"""
HYBRID: v4.0 Bulletproof Detection + v2.0 Quality Reporting
Best of both worlds!
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

# SPINEPS Label Map (from your v2.0)
SPINEPS_LABELS = {
    'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
    'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13,
    'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
    'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25,
    'Sacrum': 26,
    'T12_L1_disc': 119,
    'L1_L2_disc': 120,
    'L2_L3_disc': 121,
    'L3_L4_disc': 122,
    'L4_L5_disc': 123,
    'L5_S1_disc': 124,
    'S1_S2_disc': 126,
}

# YOLO Classes (from your v2.0)
YOLO_CLASSES = {
    0: 't12_vertebra',
    1: 't12_rib',
    2: 'l5_vertebra',
    3: 'l5_transverse_process',
    4: 'sacrum',
    5: 'l4_vertebra',
    6: 'l5_s1_disc',
}

# ============================================================================
# v4.0 BULLETPROOF FUNCTIONS (NEW)
# ============================================================================

class SpineAwareSliceSelector:
    """Intelligent slice selection using spine segmentation (v4.0)"""

    def __init__(self, voxel_spacing_mm=1.0, parasagittal_offset_mm=30):
        self.voxel_spacing_mm = voxel_spacing_mm
        self.parasagittal_offset_mm = parasagittal_offset_mm

    def find_sagittal_axis(self, data_shape):
        """Determine sagittal axis (smallest dimension)"""
        return np.argmin(data_shape)

    def find_optimal_midline(self, seg_data, sag_axis):
        """Find TRUE spinal midline using segmentation"""
        num_slices = seg_data.shape[sag_axis]
        geometric_mid = num_slices // 2

        # Get lumbar spine mask
        lumbar_labels = [SPINEPS_LABELS[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'Sacrum']]

        vertebra_mask = np.zeros_like(seg_data, dtype=bool)
        for label in lumbar_labels:
            if label in seg_data:
                vertebra_mask |= (seg_data == label)

        if not vertebra_mask.any():
            return geometric_mid, geometric_mid

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

        return geometric_mid, optimal_mid

    def get_three_slices(self, seg_data, sag_axis):
        """Get left, mid, right slice indices"""
        geometric_mid, optimal_mid = self.find_optimal_midline(seg_data, sag_axis)

        num_slices = seg_data.shape[sag_axis]
        offset_voxels = int(self.parasagittal_offset_mm / self.voxel_spacing_mm)

        left_idx = max(0, optimal_mid - offset_voxels)
        right_idx = min(num_slices - 1, optimal_mid + offset_voxels)

        return {
            'left': left_idx,
            'mid': optimal_mid,
            'right': right_idx,
        }


def extract_slice(data, sag_axis, slice_idx, thickness=1):
    """
    Extract 2D slice or Thick Slab MIP from 3D volume (v4.0)
    If thickness > 1, performs Maximum Intensity Projection (MIP).
    """
    if thickness <= 1:
        if sag_axis == 0:
            return data[slice_idx, :, :]
        elif sag_axis == 1:
            return data[:, slice_idx, :]
        else:
            return data[:, :, slice_idx]
    
    # MIP Logic
    half_thick = thickness // 2
    start = max(0, slice_idx - half_thick)
    end = min(data.shape[sag_axis], slice_idx + half_thick + 1)

    if sag_axis == 0:
        slab = data[start:end, :, :]
    elif sag_axis == 1:
        slab = data[:, start:end, :]
    else:
        slab = data[:, :, start:end]

    return np.max(slab, axis=sag_axis)


def detect_t12_rib_robust(seg_slice, vertebra_label, side='left'):
    """ROBUST T12 rib detection (v4.0)"""
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    # Get vertebra bounding box
    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min
    
    height, width = seg_slice.shape
    
    # Define search region
    if side == 'left':
        search_x_min = max(0, int(vert_x_min - vert_width * 0.8))
        search_x_max = int(vert_x_min)
    else:
        search_x_min = int(vert_x_max)
        search_x_max = min(width, int(vert_x_max + vert_width * 0.8))
    
    search_y_min = max(0, int(vert_y_min - vert_height * 0.5))
    search_y_max = min(height, int(vert_y_max + vert_height * 0.2))
    
    # Extract rib candidates
    rib_candidates = np.zeros_like(seg_slice)
    for y in range(search_y_min, search_y_max):
        for x in range(search_x_min, search_x_max):
            if seg_slice[y, x] > 0 and seg_slice[y, x] != vertebra_label:
                rib_candidates[y, x] = seg_slice[y, x]
    
    if not rib_candidates.any():
        return None
    
    # Connected component analysis
    labeled, num_features = scipy_label(rib_candidates > 0)
    
    if num_features == 0:
        return None
    
    # Size validation
    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    
    min_size = (vert_width * vert_height) * 0.15
    max_size = (search_y_max - search_y_min) * (search_x_max - search_x_min) * 0.7
    
    valid_components = [
        i + 1 for i, size in enumerate(component_sizes)
        if min_size <= size <= max_size
    ]
    
    if not valid_components:
        return None
    
    largest_comp = max(valid_components, key=lambda c: component_sizes[c - 1])
    rib_mask = (labeled == largest_comp)
    
    if not rib_mask.any():
        return None
    
    return extract_bounding_box(rib_mask.astype(int), 1)


def detect_l5_transverse_process_robust(seg_slice, vertebra_label):
    """ROBUST L5 TP detection with bilateral analysis (v4.0)"""
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min
    
    height, width = seg_slice.shape
    
    central_width = vert_width * 0.25
    
    tp_candidates = np.zeros_like(seg_slice)
    
    for y in range(int(vert_y_min - vert_height * 0.1), 
                   int(vert_y_max + vert_height * 0.1)):
        if y < 0 or y >= height:
            continue
        for x in range(width):
            if seg_slice[y, x] > 0 and seg_slice[y, x] != vertebra_label:
                dist_from_center_x = abs(x - cx)
                if dist_from_center_x > central_width:
                    tp_candidates[y, x] = seg_slice[y, x]
    
    if not tp_candidates.any():
        return None
    
    labeled, num_features = scipy_label(tp_candidates > 0)
    
    if num_features < 2:
        return None
    
    component_info = []
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled == comp_id)
        comp_size = comp_mask.sum()
        comp_coords = np.argwhere(comp_mask)
        
        if len(comp_coords) == 0:
            continue
        
        comp_y_mean = comp_coords[:, 0].mean()
        comp_x_mean = comp_coords[:, 1].mean()
        
        if comp_size < (vert_width * vert_height * 0.1):
            continue
        
        if abs(comp_x_mean - cx) < central_width:
            continue
        
        component_info.append({
            'id': comp_id,
            'size': comp_size,
            'x_mean': comp_x_mean,
        })
    
    if not component_info:
        return None
    
    component_info.sort(key=lambda c: c['size'], reverse=True)
    top_components = component_info[:2]
    
    if len(top_components) < 2:
        if top_components[0]['size'] >= (vert_width * vert_height * 0.2):
            final_mask = np.zeros_like(seg_slice, dtype=bool)
            final_mask[labeled == top_components[0]['id']] = True
            return extract_bounding_box(final_mask.astype(int), 1)
        return None
    
    left_comp = min(top_components, key=lambda c: c['x_mean'])
    right_comp = max(top_components, key=lambda c: c['x_mean'])
    
    size_ratio = max(left_comp['size'], right_comp['size']) / \
                 min(left_comp['size'], right_comp['size'])
    
    if size_ratio > 2.5:
        return extract_bounding_box((labeled == component_info[0]['id']).astype(int), 1)
    
    final_mask = np.zeros_like(seg_slice, dtype=bool)
    final_mask[labeled == left_comp['id']] = True
    final_mask[labeled == right_comp['id']] = True
    
    return extract_bounding_box(final_mask.astype(int), 1)


# ============================================================================
# v2.0 UTILITY FUNCTIONS (KEEP)
# ============================================================================

def normalize_slice(img_slice):
    """Normalize to 0-255 uint8 with CLAHE enhancement (v2.0)"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) /
                     (img_slice.max() - img_slice.min()) * 255)
        normalized = normalized.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)

        return normalized
    return np.zeros_like(img_slice, dtype=np.uint8)


def extract_bounding_box(mask, label_id):
    """Extract YOLO format bounding box (v2.0)"""
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

    if box_width <= 0 or box_height <= 0 or box_width > 1 or box_height > 1:
        return None

    return [x_center, y_center, box_width, box_height]


# ============================================================================
# MAIN LABEL GENERATION (HYBRID)
# ============================================================================

def create_yolo_labels_multiview(nifti_path, seg_path, output_dir, images_dir, 
                                  selector, use_mip=True):
    """
    HYBRID: v4.0 robust detection + v2.0 structure
    """
    try:
        nii = nib.load(nifti_path)
        seg_nii = nib.load(seg_path)

        mri_data = nii.get_fdata()
        seg_data = seg_nii.get_fdata().astype(int)

        dims = mri_data.shape
        sag_axis = np.argmin(dims)

        # SPINE-AWARE slice selection (NEW)
        slice_info = selector.get_three_slices(seg_data, sag_axis)
        
        views = {
            'left': slice_info['left'],
            'mid': slice_info['mid'],
            'right': slice_info['right'],
        }

        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')

        label_count = 0

        for view_name, slice_idx in views.items():
            # ADAPTIVE MIP THICKNESS (NEW)
            if use_mip:
                thickness = 15 if view_name in ['left', 'right'] else 5
            else:
                thickness = 1
            
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
            seg_slice = extract_slice(seg_data, sag_axis, slice_idx, thickness=thickness)

            mri_normalized = normalize_slice(mri_slice)

            # Save image
            image_filename = f"{study_id}_{view_name}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), mri_normalized)

            yolo_labels = []

            # Class 0: T12 vertebra
            t12_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['T12'])
            if t12_box:
                yolo_labels.append([0] + t12_box)

            # Class 1: T12 rib (ROBUST NEW)
            if view_name in ['left', 'right']:
                t12_rib_box = detect_t12_rib_robust(seg_slice, SPINEPS_LABELS['T12'], 
                                                      side=view_name)
                if t12_rib_box:
                    yolo_labels.append([1] + t12_rib_box)

            # Class 2: L5 vertebra
            l5_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5'])
            if l5_box:
                yolo_labels.append([2] + l5_box)

            # Class 3: L5 transverse process (ROBUST NEW)
            if view_name == 'mid':
                transverse_box = detect_l5_transverse_process_robust(seg_slice, 
                                                                       SPINEPS_LABELS['L5'])
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

            # Write labels
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
    """Split dataset (v2.0)"""
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


def generate_quality_report(output_dir, nifti_dir, seg_dir):
    """Generate quality metrics (v2.0)"""
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
    parser = argparse.ArgumentParser(description='HYBRID: v4.0 Detection + v2.0 Reporting')
    parser.add_argument('--nifti_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use_mip', action='store_true', default=True,
                       help='Use Thick Slab MIP (default: True)')
    parser.add_argument('--use_spine_aware', action='store_true', default=True,
                       help='Use spine-aware slice selection (default: True)')

    args = parser.parse_args()

    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)

    labels_dir = output_dir / 'labels' / 'train'
    images_dir = output_dir / 'images' / 'train'

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HYBRID WEAK LABEL GENERATION")
    print("v4.0 Robust Detection + v2.0 Quality Reporting")
    print("="*60)
    print("Features:")
    print("  ✓ Thick Slab MIP (v4.0)")
    print("  ✓ Spine-aware slice selection (v4.0)")
    print("  ✓ Robust T12 rib detection (v4.0)")
    print("  ✓ Robust L5 TP detection (v4.0)")
    print("  ✓ Quality reporting (v2.0)")
    print("="*60)

    seg_files = sorted(seg_dir.glob("*_seg.nii.gz"))

    if args.limit:
        seg_files = seg_files[:args.limit]

    print(f"\nFound {len(seg_files)} segmentation files")
    print("Generating 3-view YOLO labels...")
    print("="*60)

    selector = SpineAwareSliceSelector() if args.use_spine_aware else None

    success_count = 0

    for seg_file in tqdm(seg_files, desc="Processing"):
        study_id = seg_file.stem.replace('_seg', '')
        nifti_file = nifti_dir / f"sub-{study_id}_T2w.nii.gz"

        if not nifti_file.exists():
            continue

        if create_yolo_labels_multiview(nifti_file, seg_file, labels_dir, images_dir,
                                         selector, use_mip=args.use_mip):
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

    # Generate quality report (from v2.0)
    quality_report = generate_quality_report(output_dir, nifti_dir, seg_dir)

    # Metadata
    metadata = {
        'version': '3.0_HYBRID',
        'total_studies': success_count,
        'train_studies': train_count,
        'val_studies': val_count,
        'train_images': train_count * 3,
        'val_images': val_count * 3,
        'classes': YOLO_CLASSES,
        'features': {
            'v4_0_features': [
                'Thick Slab MIP (15mm ribs, 5mm midline)',
                'Spine-aware intelligent slice selection',
                'Robust T12 rib detection (morphological)',
                'Robust L5 TP detection (bilateral)',
            ],
            'v2_0_features': [
                'Quality reporting',
                'Class distribution analysis',
                'Detection rate metrics',
            ]
        },
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
    print("="*60)


if __name__ == "__main__":
    main()
