#!/usr/bin/env python3
"""
SPINEPS OUTPUT DIAGNOSTIC TOOL

Analyzes SPINEPS segmentation to determine:
1. What files are actually produced
2. What labels are present
3. Where the ribs/TPs are (if anywhere)
4. How to optimize parasagittal slices

Usage:
    python diagnose_spineps_output.py \
        --nifti_dir /path/to/nifti \
        --seg_dir /path/to/segmentations \
        --study_id 100206310
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import json
import matplotlib.pyplot as plt

# ============================================================================
# LABEL DEFINITIONS
# ============================================================================

SPINEPS_INSTANCE_LABELS = {
    'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
    'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13,
    'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
    'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25,
    'Sacrum': 26,
    'T12_L1_disc': 119, 'L1_L2_disc': 120, 'L2_L3_disc': 121,
    'L3_L4_disc': 122, 'L4_L5_disc': 123, 'L5_S1_disc': 124,
    'S1_S2_disc': 126,
}

SPINEPS_SEMANTIC_LABELS = {
    'spinal_cord': 1,
    'spinal_canal': 2,
    'vertebra_corpus': 3,
    'vertebra_disc': 4,
    'endplate': 5,
    'arcus_vertebrae': 6,
    'rib_left': 7,
    'rib_right': 8,
    'transverse_process_left': 9,
    'transverse_process_right': 10,
    'spinosus_process': 11,
    'articularis_superior_left': 12,
    'articularis_superior_right': 13,
    'articularis_inferior_left': 14,
    'articularis_inferior_right': 15,
}


def find_all_spineps_outputs(base_dir: Path, study_id: str):
    """Find ALL SPINEPS output files for a study"""
    
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR SPINEPS OUTPUTS: {study_id}")
    print(f"{'='*80}\n")
    
    # Possible locations
    search_dirs = [
        base_dir,
        base_dir / "derivatives_seg",
        base_dir.parent / "derivatives_seg",
        base_dir / study_id / "derivatives_seg",
    ]
    
    found_files = {}
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        print(f"Searching: {search_dir}")
        
        # Find all .nii.gz files
        for nifti_file in search_dir.rglob("*.nii.gz"):
            if study_id in str(nifti_file):
                file_type = "unknown"
                
                if "_seg-vert_msk" in nifti_file.name:
                    file_type = "instance"
                elif "_seg-spine_msk" in nifti_file.name:
                    file_type = "semantic"
                elif "_T2w.nii" in nifti_file.name:
                    file_type = "mri"
                
                found_files[file_type] = nifti_file
                print(f"  ✓ Found {file_type}: {nifti_file.name}")
    
    return found_files


def analyze_label_distribution(seg_path: Path, label_type: str):
    """Analyze what labels are actually present in the segmentation"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {label_type.upper()} LABELS: {seg_path.name}")
    print(f"{'='*80}\n")
    
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata().astype(int)
    
    unique_labels = np.unique(seg_data)
    
    print(f"Shape: {seg_data.shape}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Label range: {unique_labels.min()} - {unique_labels.max()}\n")
    
    # Determine which label set to use
    if label_type == "instance":
        label_names = {v: k for k, v in SPINEPS_INSTANCE_LABELS.items()}
    else:  # semantic
        label_names = {v: k for k, v in SPINEPS_SEMANTIC_LABELS.items()}
    
    print(f"{'Label ID':<10} {'Voxel Count':<15} {'Name'}")
    print(f"{'-'*60}")
    
    for label in unique_labels:
        if label == 0:
            continue
        
        count = (seg_data == label).sum()
        name = label_names.get(label, f"Unknown_{label}")
        
        print(f"{label:<10} {count:<15,} {name}")
    
    # Check for ribs specifically
    if label_type == "semantic":
        print(f"\n{'='*60}")
        print("RIB/TP ANALYSIS:")
        print(f"{'='*60}\n")
        
        rib_left = SPINEPS_SEMANTIC_LABELS.get('rib_left', 7)
        rib_right = SPINEPS_SEMANTIC_LABELS.get('rib_right', 8)
        tp_left = SPINEPS_SEMANTIC_LABELS.get('transverse_process_left', 9)
        tp_right = SPINEPS_SEMANTIC_LABELS.get('transverse_process_right', 10)
        
        rib_left_count = (seg_data == rib_left).sum()
        rib_right_count = (seg_data == rib_right).sum()
        tp_left_count = (seg_data == tp_left).sum()
        tp_right_count = (seg_data == tp_right).sum()
        
        print(f"Rib Left (label {rib_left}):  {rib_left_count:,} voxels")
        print(f"Rib Right (label {rib_right}): {rib_right_count:,} voxels")
        print(f"TP Left (label {tp_left}):   {tp_left_count:,} voxels")
        print(f"TP Right (label {tp_right}):  {tp_right_count:,} voxels")
        
        total_lateral = rib_left_count + rib_right_count + tp_left_count + tp_right_count
        
        if total_lateral > 0:
            print(f"\n✓ LATERAL STRUCTURES DETECTED! ({total_lateral:,} voxels total)")
        else:
            print(f"\n✗ NO LATERAL STRUCTURES DETECTED")
    
    return seg_data, unique_labels


def find_optimal_parasagittal_slices(seg_data: np.ndarray, label_id: int, label_name: str):
    """Find slices with maximum density of a specific structure"""
    
    print(f"\n{'='*80}")
    print(f"PARASAGITTAL OPTIMIZATION: {label_name}")
    print(f"{'='*80}\n")
    
    sag_axis = np.argmin(seg_data.shape)
    num_slices = seg_data.shape[sag_axis]
    
    print(f"Sagittal axis: {sag_axis}")
    print(f"Number of slices: {num_slices}")
    
    # Calculate density per slice
    densities = []
    
    for i in range(num_slices):
        if sag_axis == 0:
            slice_data = seg_data[i, :, :]
        elif sag_axis == 1:
            slice_data = seg_data[:, i, :]
        else:
            slice_data = seg_data[:, :, i]
        
        density = (slice_data == label_id).sum()
        densities.append(density)
    
    densities = np.array(densities)
    
    # Find peaks
    if densities.max() > 0:
        # Find all slices with >10% of max density
        threshold = 0.1 * densities.max()
        candidate_slices = np.where(densities >= threshold)[0]
        
        # Find left and right peaks
        if len(candidate_slices) > 0:
            left_slice = candidate_slices[0]
            right_slice = candidate_slices[-1]
            mid_slice = candidate_slices[len(candidate_slices)//2]
            
            print(f"\nOptimal slices for {label_name}:")
            print(f"  Left:  slice {left_slice:3d} (density: {densities[left_slice]:,})")
            print(f"  Mid:   slice {mid_slice:3d} (density: {densities[mid_slice]:,})")
            print(f"  Right: slice {right_slice:3d} (density: {densities[right_slice]:,})")
            
            # Compare to geometric center
            geometric_center = num_slices // 2
            print(f"\nGeometric center: slice {geometric_center}")
            print(f"  Offset from left:  {left_slice - geometric_center:+d} slices")
            print(f"  Offset from right: {right_slice - geometric_center:+d} slices")
            
            return {
                'left': left_slice,
                'mid': mid_slice,
                'right': right_slice,
                'densities': densities
            }
        else:
            print(f"\n✗ No slices with sufficient {label_name} density")
            return None
    else:
        print(f"\n✗ Structure not found: {label_name}")
        return None


def visualize_slice_selection(mri_data: np.ndarray, seg_data: np.ndarray, 
                              slice_info: dict, label_id: int, output_path: Path):
    """Create visualization showing optimal slice selection"""
    
    sag_axis = np.argmin(mri_data.shape)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (view_name, slice_idx) in enumerate([
        ('Left', slice_info['left']),
        ('Mid', slice_info['mid']),
        ('Right', slice_info['right'])
    ]):
        # Extract slices
        if sag_axis == 0:
            mri_slice = mri_data[slice_idx, :, :]
            seg_slice = seg_data[slice_idx, :, :]
        elif sag_axis == 1:
            mri_slice = mri_data[:, slice_idx, :]
            seg_slice = seg_data[:, slice_idx, :]
        else:
            mri_slice = mri_data[:, :, slice_idx]
            seg_slice = seg_data[:, :, slice_idx]
        
        # MRI
        axes[0, idx].imshow(mri_slice, cmap='gray')
        axes[0, idx].set_title(f'{view_name} (slice {slice_idx})', fontsize=14)
        axes[0, idx].axis('off')
        
        # Overlay
        mask = (seg_slice == label_id)
        overlay = np.zeros((*mri_slice.shape, 3), dtype=np.uint8)
        overlay[..., 0] = ((mri_slice - mri_slice.min()) / 
                          (mri_slice.max() - mri_slice.min()) * 255).astype(np.uint8)
        overlay[..., 1] = overlay[..., 0]
        overlay[..., 2] = overlay[..., 0]
        overlay[mask, 1] = 255  # Green overlay
        
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'With overlay (density: {mask.sum():,})', fontsize=14)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose SPINEPS segmentation outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--nifti_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--study_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./spineps_diagnostics')
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all SPINEPS outputs
    found_files = find_all_spineps_outputs(seg_dir, args.study_id)
    
    if not found_files:
        print(f"\n✗ No SPINEPS outputs found for {args.study_id}")
        print(f"\nSearched in:")
        print(f"  - {seg_dir}")
        print(f"  - {seg_dir / 'derivatives_seg'}")
        return
    
    # Analyze each file type
    results = {}
    
    if 'instance' in found_files:
        instance_data, instance_labels = analyze_label_distribution(
            found_files['instance'], 'instance'
        )
        results['instance'] = {
            'path': str(found_files['instance']),
            'labels': list(map(int, instance_labels)),
            'shape': list(instance_data.shape)
        }
    
    if 'semantic' in found_files:
        semantic_data, semantic_labels = analyze_label_distribution(
            found_files['semantic'], 'semantic'
        )
        results['semantic'] = {
            'path': str(found_files['semantic']),
            'labels': list(map(int, semantic_labels)),
            'shape': list(semantic_data.shape)
        }
        
        # Analyze rib distribution
        rib_left_id = SPINEPS_SEMANTIC_LABELS['rib_left']
        rib_info = find_optimal_parasagittal_slices(
            semantic_data, rib_left_id, "Left Rib"
        )
        
        if rib_info:
            results['rib_optimization'] = {
                'left': int(rib_info['left']),
                'mid': int(rib_info['mid']),
                'right': int(rib_info['right'])
            }
            
            # Visualize
            if 'mri' in found_files:
                mri_nii = nib.load(found_files['mri'])
                mri_data = mri_nii.get_fdata()
                
                vis_path = output_dir / f"{args.study_id}_rib_optimization.png"
                visualize_slice_selection(
                    mri_data, semantic_data, rib_info, rib_left_id, vis_path
                )
    
    # Save results
    results_path = output_dir / f"{args.study_id}_diagnostic_report.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved: {results_path}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if 'semantic' in found_files:
        print("✓ Semantic segmentation available!")
        print("  → Use semantic labels for rib/TP detection")
        print("  → Much more reliable than intensity-based")
        
        if 'rib_optimization' in results:
            print(f"\n✓ Optimal slices identified:")
            print(f"  → Use slices {results['rib_optimization']} for parasagittal views")
    else:
        print("✗ Semantic segmentation NOT available")
        print("  → Check SPINEPS output directory")
        print("  → May need to re-run SPINEPS with semantic model")
        print("  → Fallback to intensity-based detection")


if __name__ == "__main__":
    main()
