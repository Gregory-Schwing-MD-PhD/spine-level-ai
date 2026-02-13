#!/usr/bin/env python3
"""
SIMPLE DIAGNOSTIC: Show What Production Script Sees

NO reorientation. NO geometric middle. Just:
1. Convert DICOM → NIfTI (whatever dcm2niix gives us)
2. Run SPINEPS to get segmentation
3. Use SPINE-AWARE slicing to find actual midline
4. Show you that slice with correct aspect ratio

This is EXACTLY what lstv_screen_production_COMPLETE.py does.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
import subprocess
import sys
import os


def convert_dicom(dicom_dir, output_dir, series_id):
    """Convert DICOM - NO REORIENTATION"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        bids_base = f"series-{series_id}"

        cmd = [
            'dcm2niix', '-z', 'y', '-f', bids_base, '-o', str(output_dir),
            '-m', 'y', '-ba', 'n', '-i', 'n', '-x', 'n', '-p', 'n',
            str(dicom_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None

        nifti_files = list(output_dir.glob(f"{bids_base}*.nii.gz"))
        return nifti_files[0] if nifti_files else None

    except Exception as e:
        print(f"Conversion error: {e}")
        return None


def run_spineps(nifti_path):
    """Run SPINEPS to get instance segmentation"""
    try:
        env = os.environ.copy()
        env['SPINEPS_SEGMENTOR_MODELS'] = '/app/models'

        cmd = [
            'python', '-m', 'spineps.entrypoint', 'sample',
            '-i', str(nifti_path),
            '-model_instance', 'instance',
            '-override_instance'
        ]

        subprocess.run(cmd, capture_output=True, timeout=600, env=env)

        # Find instance mask
        derivatives_dir = nifti_path.parent / "derivatives_seg"
        if not derivatives_dir.exists():
            return None

        instance_files = list(derivatives_dir.glob("*_seg-vert_msk.nii.gz"))
        return instance_files[0] if instance_files else None

    except Exception as e:
        print(f"SPINEPS error: {e}")
        return None


def find_spine_midline(seg_data):
    """SPINE-AWARE: Find actual spine location"""
    lumbar_labels = [20, 21, 22, 23, 24, 26]  # L1-L5 + Sacrum
    
    vertebra_mask = np.zeros_like(seg_data, dtype=bool)
    for label in lumbar_labels:
        vertebra_mask |= (seg_data == label)

    if not vertebra_mask.any():
        return None

    # Find sagittal axis (smallest dimension)
    sag_axis = np.argmin(seg_data.shape)
    num_slices = seg_data.shape[sag_axis]

    # Calculate spine density per slice
    spine_density = np.zeros(num_slices)
    for i in range(num_slices):
        if sag_axis == 0:
            slice_mask = vertebra_mask[i, :, :]
        elif sag_axis == 1:
            slice_mask = vertebra_mask[:, i, :]
        else:
            slice_mask = vertebra_mask[:, :, i]
        spine_density[i] = slice_mask.sum()

    optimal_mid = int(np.argmax(spine_density))
    
    return {
        'midline_idx': optimal_mid,
        'sag_axis': sag_axis,
        'max_density': spine_density[optimal_mid]
    }


def extract_and_display(mri_data, seg_data, sag_axis, slice_idx, zooms, output_path):
    """Extract slice with correct aspect ratio and display"""
    
    # Extract 2D slice
    if sag_axis == 0:
        mri_slice = mri_data[slice_idx, :, :]
        seg_slice = seg_data[slice_idx, :, :]
        voxel_sizes = (zooms[1], zooms[2])
    elif sag_axis == 1:
        mri_slice = mri_data[:, slice_idx, :]
        seg_slice = seg_data[:, slice_idx, :]
        voxel_sizes = (zooms[0], zooms[2])
    else:
        mri_slice = mri_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
        voxel_sizes = (zooms[0], zooms[1])

    # Normalize MRI
    if mri_slice.max() > mri_slice.min():
        normalized = ((mri_slice - mri_slice.min()) / 
                     (mri_slice.max() - mri_slice.min()) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(mri_slice, dtype=np.uint8)

    # Correct aspect ratio
    h, w = normalized.shape
    physical_h = h * voxel_sizes[0]
    physical_w = w * voxel_sizes[1]
    aspect_ratio = physical_w / physical_h
    
    target_h = 500
    target_w = int(target_h * aspect_ratio)
    
    mri_resized = cv2.resize(normalized, (target_w, target_h))
    seg_resized = cv2.resize(seg_slice.astype(np.uint8), (target_w, target_h), 
                            interpolation=cv2.INTER_NEAREST)

    # Create RGB with labels
    rgb_img = cv2.cvtColor(mri_resized, cv2.COLOR_GRAY2RGB)
    
    # Add simple vertebra markers
    unique_labels = np.unique(seg_resized)
    lumbar_labels = [l for l in unique_labels if 20 <= l <= 26]
    
    for label in lumbar_labels:
        mask = (seg_resized == label)
        if mask.any():
            coords = np.argwhere(mask)
            cy, cx = int(coords[:, 0].mean()), int(coords[:, 1].mean())
            
            if label == 20:
                name = "L1"
            elif label == 21:
                name = "L2"
            elif label == 22:
                name = "L3"
            elif label == 23:
                name = "L4"
            elif label == 24:
                name = "L5"
            elif label == 25:
                name = "L6"
            elif label == 26:
                name = "Sacrum"
            else:
                name = str(label)
            
            color = (0, 255, 0) if label == 24 else (255, 255, 0)
            cv2.putText(rgb_img, name, (cx - 20, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.circle(rgb_img, (cx, cy), 5, color, -1)

    # Add info banner
    banner = np.zeros((80, rgb_img.shape[1], 3), dtype=np.uint8)
    banner[:] = (40, 40, 40)
    
    cv2.putText(banner, f"Slice {slice_idx} (axis {sag_axis})", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(banner, f"Shape: {mri_data.shape}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(banner, f"Voxel: {voxel_sizes[0]:.1f}x{voxel_sizes[1]:.1f}mm", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    final = np.vstack([banner, rgb_img])
    
    cv2.imwrite(str(output_path), final)
    return True


def main():
    parser = argparse.ArgumentParser(description='Simple diagnostic - shows what production sees')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--series_csv', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_studies', type=int, default=5)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_dir = output_dir / 'nifti_temp'
    images_dir = output_dir / 'diagnostic_images'
    nifti_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.series_csv)
    df['study_id'] = df['study_id'].astype(int)
    df['series_id'] = df['series_id'].astype(int)

    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])[:args.num_studies]

    print(f"\nProcessing {len(study_dirs)} studies...")
    print(f"This shows EXACTLY what the production script sees.\n")

    results = []

    for study_dir in study_dirs:
        study_id = int(study_dir.name)
        print(f"\n[Study {study_id}]")

        study_series = df[df['study_id'] == study_id]
        if len(study_series) == 0:
            print(f"  Not in CSV, skipping")
            continue

        # Find sagittal T2
        sagittal = study_series[
            study_series['series_description'].str.contains(
                'Sagittal T2', case=False, na=False
            )
        ]

        if len(sagittal) == 0:
            print(f"  No Sagittal T2 in CSV")
            continue

        series_id = int(sagittal.iloc[0]['series_id'])
        series_dir = study_dir / str(series_id)

        if not series_dir.exists():
            print(f"  Series {series_id} not on disk")
            continue

        csv_desc = sagittal.iloc[0]['series_description']
        print(f"  CSV: {csv_desc}")
        print(f"  Series: {series_id}")

        # Convert
        print(f"  Converting...")
        nifti_path = convert_dicom(series_dir, nifti_dir, series_id)
        if nifti_path is None:
            print(f"  ✗ Conversion failed")
            continue

        nii = nib.load(nifti_path)
        orientation = nib.aff2axcodes(nii.affine)
        zooms = nii.header.get_zooms()
        print(f"  Orientation: {orientation}")
        print(f"  Voxel spacing: {zooms}")

        # Run SPINEPS
        print(f"  Running SPINEPS...")
        seg_path = run_spineps(nifti_path)
        if seg_path is None:
            print(f"  ✗ SPINEPS failed")
            continue

        # Load data
        mri_data = nii.get_fdata()
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(int)

        # Find spine midline (SPINE-AWARE)
        print(f"  Finding spine midline...")
        midline_info = find_spine_midline(seg_data)
        if midline_info is None:
            print(f"  ✗ No spine found")
            continue

        midline_idx = midline_info['midline_idx']
        sag_axis = midline_info['sag_axis']
        density = midline_info['max_density']

        print(f"  Spine midline: slice {midline_idx} (axis {sag_axis}), density={density}")

        # Create image
        output_path = images_dir / f"study{study_id}_series{series_id}.jpg"
        success = extract_and_display(
            mri_data, seg_data, sag_axis, midline_idx, zooms, output_path
        )

        if success:
            print(f"  ✓ Saved: {output_path.name}")
            results.append({
                'study_id': study_id,
                'series_id': series_id,
                'csv_description': csv_desc,
                'orientation': str(orientation),
                'midline_slice': midline_idx,
                'sag_axis': sag_axis,
                'spine_density': density
            })

        # Cleanup
        nifti_path.unlink()

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'diagnostic_results.csv'
    results_df.to_csv(results_csv, index=False)

    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {results_csv}")
    print(f"Images: {images_dir}/")
    print(f"\nThese images show EXACTLY what the production script will see:")
    print(f"  - NO reorientation")
    print(f"  - SPINE-AWARE midline (not geometric middle)")
    print(f"  - Correct aspect ratios")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
