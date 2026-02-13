#!/usr/bin/env python3
"""
DIAGNOSTIC: Visualize Series Descriptions vs Actual Orientations

Creates preview images showing:
- What the CSV says the series is
- What the DICOM orientation actually is
- Middle slice of each series

This helps identify mislabeled series.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
import subprocess
import shutil
from tqdm import tqdm

def convert_dicom_to_nifti_diagnostic(dicom_dir, output_dir, series_id):
    """Convert DICOM and return path + orientation"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        bids_base = f"series-{series_id}"
        
        cmd = [
            'dcm2niix',
            '-z', 'y',
            '-f', bids_base,
            '-o', str(output_dir),
            '-m', 'y',
            '-ba', 'n',
            '-i', 'n',
            '-x', 'n',
            '-p', 'n',
            str(dicom_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return None, None
        
        # Find generated file
        nifti_files = list(output_dir.glob(f"{bids_base}*.nii.gz"))
        if not nifti_files:
            return None, None
        
        nifti_path = nifti_files[0]
        
        # Load and get orientation
        nii = nib.load(nifti_path)
        orientation = nib.aff2axcodes(nii.affine)
        
        return nifti_path, orientation
        
    except Exception as e:
        print(f"Error converting {series_id}: {e}")
        return None, None


def create_diagnostic_image(nifti_path, csv_desc, orientation, output_path):
    """Create annotated preview image"""
    try:
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        
        # Get middle slices from all 3 axes
        mid_ax0 = data[data.shape[0]//2, :, :]
        mid_ax1 = data[:, data.shape[1]//2, :]
        mid_ax2 = data[:, :, data.shape[2]//2]
        
        # Normalize each
        def norm(img):
            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min()) * 255
            return img.astype(np.uint8)
        
        mid_ax0 = norm(mid_ax0)
        mid_ax1 = norm(mid_ax1)
        mid_ax2 = norm(mid_ax2)
        
        # Resize to same height
        target_h = 300
        
        def resize_keep_aspect(img, target_h):
            h, w = img.shape
            aspect = w / h
            new_w = int(target_h * aspect)
            return cv2.resize(img, (new_w, target_h))
        
        mid_ax0 = resize_keep_aspect(mid_ax0, target_h)
        mid_ax1 = resize_keep_aspect(mid_ax1, target_h)
        mid_ax2 = resize_keep_aspect(mid_ax2, target_h)
        
        # Convert to RGB
        mid_ax0 = cv2.cvtColor(mid_ax0, cv2.COLOR_GRAY2RGB)
        mid_ax1 = cv2.cvtColor(mid_ax1, cv2.COLOR_GRAY2RGB)
        mid_ax2 = cv2.cvtColor(mid_ax2, cv2.COLOR_GRAY2RGB)
        
        # Stack horizontally
        combined = np.hstack([mid_ax0, mid_ax1, mid_ax2])
        
        # Add text banner
        banner_height = 120
        banner = np.zeros((banner_height, combined.shape[1], 3), dtype=np.uint8)
        banner[:] = (40, 40, 40)
        
        # CSV description
        csv_text = f"CSV: {csv_desc}"
        cv2.putText(banner, csv_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Actual orientation
        orient_text = f"DICOM Orientation: {orientation}"
        cv2.putText(banner, orient_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        # Orientation interpretation
        first_axis = orientation[0]
        if first_axis in ('R', 'L'):
            interpret = "SAGITTAL (first axis = L/R)"
            color = (0, 255, 0)  # Green
        elif first_axis in ('A', 'P'):
            interpret = "CORONAL (first axis = A/P)"
            color = (0, 165, 255)  # Orange
        else:
            interpret = "AXIAL (first axis = S/I)"
            color = (255, 0, 0)  # Red
        
        cv2.putText(banner, interpret, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add shape info
        shape_text = f"Shape: {data.shape}"
        cv2.putText(banner, shape_text, (10 + combined.shape[1]//2, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Axis labels
        cv2.putText(banner, "Axis 0 middle", (10 + combined.shape[1]//2, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(banner, "Axis 1 middle", (10 + combined.shape[1]//2, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(banner, "Axis 2 middle", (10 + combined.shape[1]//2, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Stack banner on top
        final = np.vstack([banner, combined])
        
        # Save
        cv2.imwrite(str(output_path), final)
        return True
        
    except Exception as e:
        print(f"Error creating image: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Diagnose series orientations vs CSV labels')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with study folders')
    parser.add_argument('--series_csv', type=str, required=True,
                       help='Series descriptions CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for diagnostic images')
    parser.add_argument('--num_studies', type=int, default=10,
                       help='Number of studies to check')
    
    args = parser.parse_args()
    
    print(f"Starting diagnostic with arguments:")
    print(f"  input_dir: {args.input_dir}")
    print(f"  series_csv: {args.series_csv}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  num_studies: {args.num_studies}")
    print()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Verify CSV exists
    csv_path = Path(args.series_csv)
    if not csv_path.exists():
        print(f"ERROR: Series CSV does not exist: {csv_path}")
        sys.exit(1)
    
    # Create subdirs
    nifti_dir = output_dir / 'nifti_temp'
    images_dir = output_dir / 'diagnostic_images'
    nifti_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Load CSV
    print(f"Loading CSV: {args.series_csv}")
    try:
        df = pd.read_csv(args.series_csv)
        df['study_id'] = df['study_id'].astype(int)
        df['series_id'] = df['series_id'].astype(int)
        print(f"Loaded {len(df)} series descriptions")
        print(f"\nSample descriptions:")
        for desc, count in df['series_description'].value_counts().head(5).items():
            print(f"  {desc}: {count}")
        print()
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Get study directories
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(study_dirs)} study directories in {input_dir}")
    
    if len(study_dirs) == 0:
        print(f"ERROR: No study directories found!")
        sys.exit(1)
    
    study_dirs = study_dirs[:args.num_studies]
    print(f"Processing first {len(study_dirs)} studies...\n")
    
    results = []
    
    for study_dir in tqdm(study_dirs):
        study_id = int(study_dir.name)
        print(f"\n[Study {study_id}]")
        
        # Get series for this study
        study_series = df[df['study_id'] == study_id]
        
        if len(study_series) == 0:
            print(f"  Not in CSV, skipping")
            continue
        
        print(f"  Found {len(study_series)} series in CSV")
        
        # Get available series directories
        series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
        print(f"  Found {len(series_dirs)} series directories on disk")
        
        if len(series_dirs) == 0:
            print(f"  No series directories, skipping")
            continue
        
        for series_dir in series_dirs:
            series_id = int(series_dir.name)
            
            # Look up in CSV
            series_info = study_series[study_series['series_id'] == series_id]
            
            if len(series_info) == 0:
                csv_desc = "NOT IN CSV"
            else:
                csv_desc = series_info.iloc[0]['series_description']
            
            print(f"  Processing series {series_id}: {csv_desc}")
            
            # Convert and check orientation
            nifti_path, orientation = convert_dicom_to_nifti_diagnostic(
                series_dir, nifti_dir, series_id
            )
            
            if nifti_path is None:
                print(f"    Failed to convert")
                continue
            
            print(f"    Orientation: {orientation}")
            
            # Determine actual type
            first_axis = orientation[0]
            if first_axis in ('R', 'L'):
                actual_type = "SAGITTAL"
            elif first_axis in ('A', 'P'):
                actual_type = "CORONAL"
            else:
                actual_type = "AXIAL"
            
            # Create diagnostic image
            output_path = images_dir / f"study{study_id}_series{series_id}.jpg"
            
            success = create_diagnostic_image(
                nifti_path, csv_desc, orientation, output_path
            )
            
            if success:
                print(f"    Created image: {output_path.name}")
                
                # Check for mismatch
                csv_lower = csv_desc.lower()
                mismatch = False
                
                if 'sagittal' in csv_lower and actual_type != "SAGITTAL":
                    mismatch = True
                elif 'coronal' in csv_lower and actual_type != "CORONAL":
                    mismatch = True
                elif 'axial' in csv_lower and actual_type != "AXIAL":
                    mismatch = True
                
                if mismatch:
                    print(f"    ⚠ MISMATCH: CSV says {csv_desc}, actually {actual_type}")
                
                results.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'csv_description': csv_desc,
                    'dicom_orientation': str(orientation),
                    'actual_type': actual_type,
                    'mismatch': mismatch,
                    'image_path': str(output_path)
                })
            else:
                print(f"    Failed to create image")
            
            # Clean up nifti
            try:
                nifti_path.unlink()
            except:
                pass
    
    # Save results
    if len(results) == 0:
        print("\n" + "="*80)
        print("ERROR: No results generated!")
        print("Check that:")
        print("  1. dcm2niix is available in the container")
        print("  2. DICOM files exist in the series directories")
        print("  3. The bind mounts are correct")
        sys.exit(1)
    
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'diagnostic_results.csv'
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")
    print(f"Total series checked: {len(results_df)}")
    print(f"Mismatches found: {results_df['mismatch'].sum()}")
    print()
    
    if results_df['mismatch'].sum() > 0:
        print("MISMATCHED SERIES:")
        mismatches = results_df[results_df['mismatch'] == True]
        for _, row in mismatches.iterrows():
            print(f"  Study {row['study_id']}, Series {row['series_id']}:")
            print(f"    CSV says: {row['csv_description']}")
            print(f"    Actually: {row['actual_type']} {row['dicom_orientation']}")
            print(f"    Image: {row['image_path']}")
            print()
    
    print(f"Results CSV: {results_csv}")
    print(f"Diagnostic images: {images_dir}/")
    print()
    
    # Summary by type
    print("CSV DESCRIPTIONS vs ACTUAL:")
    for csv_desc in results_df['csv_description'].unique():
        subset = results_df[results_df['csv_description'] == csv_desc]
        types = subset['actual_type'].value_counts()
        print(f"\n  CSV: '{csv_desc}'")
        for actual_type, count in types.items():
            pct = count / len(subset) * 100
            print(f"    → Actually {actual_type}: {count}/{len(subset)} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
