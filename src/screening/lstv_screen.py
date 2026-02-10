#!/usr/bin/env python3
"""
LSTV Screening Pipeline using SPINEPS
Screens lumbar MRI studies for LSTV candidates
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import pydicom
import nibabel as nib
from tqdm import tqdm
import subprocess
import shutil
import traceback

def load_series_descriptions(csv_path):
    """Load series descriptions to identify correct series"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Warning: Could not load series descriptions: {e}")
        return None

def select_best_series(study_dir, series_df=None, study_id=None):
    """
    Select the best series for LSTV screening
    Priority: Sagittal T2 > Sagittal T1 > Any Sagittal
    """
    series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    
    if not series_dirs:
        return None
    
    # If we have series descriptions, use them
    if series_df is not None and study_id is not None:
        study_series = series_df[series_df['study_id'] == int(study_id)]
        
        if len(study_series) > 0:
            # Priority order for LSTV screening
            priorities = [
                'Sagittal T2',
                'Sagittal T2/STIR',
                'SAG T2',
                'Sagittal T1',
                'SAG T1',
            ]
            
            for priority in priorities:
                matching = study_series[
                    study_series['series_description'].str.contains(priority, case=False, na=False)
                ]
                
                if len(matching) > 0:
                    series_id = str(matching.iloc[0]['series_id'])
                    series_path = study_dir / series_id
                    
                    if series_path.exists():
                        return series_path
    
    # Fallback: return first series
    return series_dirs[0]

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM series to NIfTI using dcm2niix"""
    try:
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract base name without any extensions
        # If output_path is "study.nii.gz", we want just "study"
        base_name = output_path.name.replace('.nii.gz', '').replace('.nii', '')

        # Use dcm2niix for conversion
        cmd = [
            'dcm2niix',
            '-z', 'y',  # Compress output
            '-f', base_name,  # Output filename (just the base, no extensions)
            '-o', str(output_path.parent),  # Output directory
            '-m', 'y',  # Merge 2D slices
            '-b', 'n',  # Don't create .json sidecar
            str(dicom_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"  ✗ dcm2niix failed: {result.stderr}")
            return None

        # Find the generated .nii.gz file (dcm2niix may add suffixes)
        nifti_files = sorted(output_path.parent.glob(f"{base_name}*.nii.gz"))

        if not nifti_files:
            print(f"  ✗ No NIfTI file generated")
            return None

        # Rename to our expected filename if needed
        final_file = nifti_files[0]
        if final_file != output_path:
            final_file.rename(output_path)
            final_file = output_path

        return final_file

    except subprocess.TimeoutExpired:
        print(f"  ✗ Conversion timeout (>120s)")
        return None
    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        traceback.print_exc()
        return None

def run_spineps_inference(nifti_path, output_dir):
    """
    Run SPINEPS segmentation on NIfTI volume using the CLI
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # SPINEPS creates a derivatives folder next to the input
        # We'll specify a custom derivative name
        derivative_name = "spineps_output"
        
        # Run SPINEPS CLI
        cmd = [
            'spineps',
            'sample',
            '-i', str(nifti_path),
            '-model_semantic', 't2w',  # For T2w sagittal images
            '-model_instance', 'instance',
            '-model_labeling', 't2w_labeling',  # VERIDAH labeling
            '-der_name', derivative_name,
            '-override_semantic',
            '-override_instance',
            '-override_ctd',
        ]
        
        print(f"    Running: {' '.join(cmd)}")
        sys.stdout.flush()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"  ✗ SPINEPS failed: {result.stderr}")
            return None
        
        # SPINEPS creates output in: <input_dir>/<derivative_name>/
        input_parent = nifti_path.parent
        seg_dir = input_parent / derivative_name
        
        # Find the instance segmentation output (seg-vert)
        seg_vert_files = list(seg_dir.glob("*_seg-vert_msk.nii.gz"))
        
        if not seg_vert_files:
            print(f"  ✗ SPINEPS output not found in {seg_dir}")
            # Try to list what's actually there
            if seg_dir.exists():
                print(f"    Contents: {list(seg_dir.glob('*'))}")
            return None
        
        # Copy to our output directory
        seg_output = output_dir / f"{nifti_path.stem}_seg.nii.gz"
        shutil.copy(seg_vert_files[0], seg_output)
        
        print(f"  ✓ Segmentation saved to {seg_output}")
        
        return seg_output
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ SPINEPS timeout (>10min)")
        return None
    except Exception as e:
        print(f"  ✗ SPINEPS error: {e}")
        traceback.print_exc()
        return None

def analyze_segmentation(seg_path):
    """
    Analyze SPINEPS segmentation for LSTV indicators
    
    SPINEPS vertebra instance labels:
    - Labels 1-25: Individual vertebrae (C1-L6)
    - Labels 100+X: IVD below vertebra X
    - Labels 200+X: Endplate of vertebra X
    - Label 26: Sacrum
    """
    try:
        # Load segmentation
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(int)
        
        # Get unique vertebra labels (1-25)
        unique_labels = np.unique(seg_data)
        vertebra_labels = [l for l in unique_labels if 1 <= l <= 25]
        
        # Count lumbar vertebrae (L1-L6 are labels 20-25 in SPINEPS)
        lumbar_labels = [l for l in vertebra_labels if 20 <= l <= 25]
        vertebra_count = len(lumbar_labels)
        
        # Check for sacrum
        has_sacrum = 26 in unique_labels
        
        # Check for fusion: look at spacing between adjacent vertebrae
        fusion_detected = False
        if len(lumbar_labels) >= 2:
            sorted_labels = sorted(lumbar_labels)
            
            for i in range(len(sorted_labels) - 1):
                curr_label = sorted_labels[i]
                next_label = sorted_labels[i + 1]
                
                curr_mask = (seg_data == curr_label)
                next_mask = (seg_data == next_label)
                
                if curr_mask.sum() > 100 and next_mask.sum() > 100:
                    # Calculate centroids
                    curr_centroid = np.array(np.where(curr_mask)).mean(axis=1)
                    next_centroid = np.array(np.where(next_mask)).mean(axis=1)
                    
                    # Calculate distance
                    distance = np.linalg.norm(curr_centroid - next_centroid)
                    
                    # Get voxel spacing
                    spacing = seg_nii.header.get_zooms()
                    physical_distance = distance * np.mean(spacing)
                    
                    # If distance is very small (<5mm), possible fusion
                    if physical_distance < 5:
                        fusion_detected = True
                        break
        
        # Flag if abnormal count or fusion
        # Normal: 5 lumbar vertebrae (L1-L5)
        # LSTV: 4 (sacralization) or 6 (lumbarization)
        is_lstv_candidate = (vertebra_count < 4 or vertebra_count > 6) or fusion_detected
        
        result = {
            'vertebra_count': vertebra_count,
            'has_sacrum': has_sacrum,
            'fusion_detected': fusion_detected,
            'is_lstv_candidate': is_lstv_candidate,
            'unique_labels': list(map(int, unique_labels)),
            'lumbar_labels': list(map(int, lumbar_labels)),
        }
        
        return result
        
    except Exception as e:
        print(f"  ✗ Analysis error: {e}")
        traceback.print_exc()
        return None

def extract_middle_slice(nifti_path, output_jpg):
    """Extract middle sagittal slice as JPG for Roboflow upload"""
    try:
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        
        # Determine orientation and extract sagittal slice
        # Sagittal is typically the smallest dimension
        dims = data.shape
        sag_axis = np.argmin(dims)
        
        mid_idx = dims[sag_axis] // 2
        
        if sag_axis == 0:
            slice_img = data[mid_idx, :, :]
        elif sag_axis == 1:
            slice_img = data[:, mid_idx, :]
        else:
            slice_img = data[:, :, mid_idx]
        
        # Normalize to 0-255
        if slice_img.max() > slice_img.min():
            slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        else:
            slice_img = np.zeros_like(slice_img, dtype=np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        slice_img = clahe.apply(slice_img)
        
        # Save as JPG
        output_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_jpg), slice_img)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Slice extraction error: {e}")
        traceback.print_exc()
        return False

def upload_to_roboflow(image_path, study_id, roboflow_key, workspace, project):
    """Upload image to Roboflow workspace"""
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=roboflow_key)
        workspace_obj = rf.workspace(workspace)
        project_obj = workspace_obj.project(project)
        
        # Upload image with metadata
        project_obj.upload(
            image_path=str(image_path),
            split="train",  # Added: organize uploads
            tag_names=["lstv-candidate", "automated-screening"],  # Added: useful tags
            num_retry_uploads=3
        )
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Roboflow upload error: {e}")
        return False

def load_progress(progress_file):
    """Load processing progress"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return {'processed': [], 'flagged': [], 'failed': []}
    return {'processed': [], 'flagged': [], 'failed': []}

def save_progress(progress_file, progress):
    """Save processing progress atomically"""
    try:
        temp_file = progress_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(progress_file)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def main():
    parser = argparse.ArgumentParser(description='LSTV Screening Pipeline')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with DICOM studies')
    parser.add_argument('--series_csv', type=str, default=None, help='Path to train_series_descriptions.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of studies to process')
    parser.add_argument('--roboflow_key', type=str, required=True, help='Roboflow API key')
    parser.add_argument('--roboflow_workspace', type=str, default='lstv-screening', help='Roboflow workspace name')
    parser.add_argument('--roboflow_project', type=str, default='lstv-candidates', help='Roboflow project name')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nifti_dir = output_dir / 'nifti'
    seg_dir = output_dir / 'segmentations'
    images_dir = output_dir / 'candidate_images'
    
    nifti_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Load series descriptions if available
    series_df = None
    if args.series_csv:
        series_csv = Path(args.series_csv)
        if series_csv.exists():
            series_df = load_series_descriptions(series_csv)
            print(f"Loaded series descriptions: {len(series_df)} entries")
        else:
            print(f"Warning: Series CSV not found: {args.series_csv}")
    
    # Load progress
    progress_file = output_dir / 'progress.json'
    progress = load_progress(progress_file)
    
    # Results CSV
    results_csv = output_dir / 'results.csv'
    
    # Find all study directories
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if args.limit:
        study_dirs = study_dirs[:args.limit]
    
    print("="*60)
    print("LSTV SCREENING PIPELINE")
    print("="*60)
    print(f"Total studies found: {len(study_dirs)}")
    print(f"Already processed: {len(progress['processed'])}")
    print(f"Already flagged: {len(progress['flagged'])}")
    print(f"Previous failures: {len(progress.get('failed', []))}")
    print("="*60)
    sys.stdout.flush()
    
    # Process studies
    for study_dir in tqdm(study_dirs, desc="Processing studies"):
        study_id = study_dir.name
        
        # Skip if already processed
        if study_id in progress['processed']:
            if args.verbose:
                print(f"  Skipping {study_id} (already processed)")
            continue
        
        print(f"\n[{study_id}]")
        sys.stdout.flush()
        
        try:
            # Select best series for LSTV screening
            series_dir = select_best_series(study_dir, series_df, study_id)
            
            if series_dir is None:
                print(f"  ✗ No series found")
                progress['failed'].append(study_id)
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue
            
            print(f"  Using series: {series_dir.name}")
            
            # Convert DICOM to NIfTI
            nifti_path = nifti_dir / f"{study_id}.nii.gz"
            
            if not nifti_path.exists():
                print(f"  Converting DICOM to NIfTI...")
                sys.stdout.flush()
                nifti_path = convert_dicom_to_nifti(series_dir, nifti_path)
                
                if nifti_path is None:
                    print(f"  ✗ Conversion failed")
                    progress['failed'].append(study_id)
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue
            
            # Run SPINEPS
            seg_path = seg_dir / f"{study_id}_seg.nii.gz"
            
            if not seg_path.exists():
                print(f"  Running SPINEPS segmentation...")
                sys.stdout.flush()
                seg_path = run_spineps_inference(nifti_path, seg_dir)
                
                if seg_path is None:
                    print(f"  ✗ Segmentation failed")
                    progress['failed'].append(study_id)
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue
            
            # Analyze segmentation
            print(f"  Analyzing segmentation...")
            sys.stdout.flush()
            analysis = analyze_segmentation(seg_path)
            
            if analysis is None:
                print(f"  ✗ Analysis failed")
                progress['failed'].append(study_id)
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue
            
            # Store results
            result = {
                'study_id': study_id,
                'series_id': series_dir.name,
                'vertebra_count': analysis['vertebra_count'],
                'has_sacrum': analysis['has_sacrum'],
                'fusion_detected': analysis['fusion_detected'],
                'is_lstv_candidate': analysis['is_lstv_candidate'],
                'unique_labels': str(analysis['unique_labels']),
                'lumbar_labels': str(analysis['lumbar_labels']),
            }
            
            # If LSTV candidate, upload to Roboflow
            if analysis['is_lstv_candidate']:
                print(f"  🚩 LSTV CANDIDATE DETECTED!")
                print(f"     Vertebra count: {analysis['vertebra_count']}")
                print(f"     Lumbar labels: {analysis['lumbar_labels']}")
                print(f"     Fusion: {analysis['fusion_detected']}")
                sys.stdout.flush()
                
                # Extract middle slice
                image_path = images_dir / f"{study_id}.jpg"
                
                if extract_middle_slice(nifti_path, image_path):
                    print(f"  Uploading to Roboflow...")
                    sys.stdout.flush()
                    
                    if upload_to_roboflow(
                        image_path,
                        study_id,
                        args.roboflow_key,
                        args.roboflow_workspace,
                        args.roboflow_project
                    ):
                        print(f"  ✓ Uploaded to Roboflow")
                        progress['flagged'].append(study_id)
                    else:
                        print(f"  ⚠ Roboflow upload failed (continuing)")
            else:
                print(f"  ✓ Normal (count={analysis['vertebra_count']}, labels={analysis['lumbar_labels']})")
            
            sys.stdout.flush()
            
            # Mark as processed
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
            
            # Append to results CSV
            df = pd.DataFrame([result])
            if results_csv.exists():
                df.to_csv(results_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(results_csv, mode='w', header=True, index=False)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted - progress saved!")
            save_progress(progress_file, progress)
            sys.exit(1)
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            traceback.print_exc()
            progress['failed'].append(study_id)
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("SCREENING COMPLETE")
    print("="*60)
    print(f"Total processed: {len(progress['processed'])}")
    print(f"LSTV candidates: {len(progress['flagged'])}")
    print(f"Failed: {len(progress.get('failed', []))}")
    print(f"Results: {results_csv}")
    print("="*60)

if __name__ == "__main__":
    main()
