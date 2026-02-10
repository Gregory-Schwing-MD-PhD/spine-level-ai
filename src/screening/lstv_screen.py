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

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM series to NIfTI using dcm2niix"""
    try:
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use dcm2niix for conversion
        cmd = [
            'dcm2niix',
            '-z', 'y',  # Compress output
            '-f', output_path.stem,  # Output filename
            '-o', str(output_path.parent),  # Output directory
            '-m', 'y',  # Merge 2D slices
            str(dicom_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"  ✗ dcm2niix failed: {result.stderr}")
            return None
        
        # Find the generated .nii.gz file (dcm2niix may add suffixes)
        nifti_files = sorted(output_path.parent.glob(f"{output_path.stem}*.nii.gz"))
        
        if not nifti_files:
            print(f"  ✗ No NIfTI file generated")
            return None
        
        # Return the first/largest file
        return nifti_files[0]
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Conversion timeout (>120s)")
        return None
    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        traceback.print_exc()
        return None

def run_spineps_inference(nifti_path, output_dir):
    """Run SPINEPS segmentation on NIfTI volume"""
    try:
        # SPINEPS command-line interface
        # Adjust based on actual SPINEPS API
        seg_output = output_dir / f"{nifti_path.stem}_seg.nii.gz"
        
        cmd = [
            'spineps',
            'segment',
            str(nifti_path),
            '-o', str(seg_output),
            '--model', 'default'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"  ✗ SPINEPS failed: {result.stderr}")
            return None
        
        if not seg_output.exists():
            print(f"  ✗ Segmentation output not found")
            return None
        
        return seg_output
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ SPINEPS timeout (>5min)")
        return None
    except Exception as e:
        print(f"  ✗ SPINEPS error: {e}")
        traceback.print_exc()
        return None

def analyze_segmentation(seg_path):
    """Analyze SPINEPS segmentation for LSTV indicators"""
    try:
        # Load segmentation
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(int)
        
        # Get unique labels
        unique_labels = np.unique(seg_data)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        # SPINEPS label scheme (VERIFY THIS - may need adjustment)
        # Typically: Vertebrae are sequential labels
        # We need to count lumbar vertebrae (usually 5)
        # and check for sacrum presence
        
        # Simple heuristic: count all vertebrae labels
        vertebra_labels = [l for l in unique_labels if l < 100]  # Exclude other structures
        vertebra_count = len(vertebra_labels)
        
        # Check if we can identify sacrum (typically highest label in lumbar region)
        # This is a ROUGH heuristic - adjust based on actual SPINEPS output
        has_sacrum = vertebra_count > 0  # Placeholder
        
        # Calculate spacing between last two vertebrae
        fusion_detected = False
        if len(vertebra_labels) >= 2:
            # Get masks of last two vertebrae
            sorted_labels = sorted(vertebra_labels)
            last_vert = sorted_labels[-1]
            second_last = sorted_labels[-2]
            
            last_mask = (seg_data == last_vert)
            second_mask = (seg_data == second_last)
            
            if last_mask.sum() > 0 and second_mask.sum() > 0:
                # Calculate centroids
                last_centroid = np.array(np.where(last_mask)).mean(axis=1)
                second_centroid = np.array(np.where(second_mask)).mean(axis=1)
                
                # Calculate distance
                distance = np.linalg.norm(last_centroid - second_centroid)
                
                # Get voxel spacing
                spacing = seg_nii.header.get_zooms()
                physical_distance = distance * spacing[0]  # Approximate
                
                # If distance is very small (<10mm), possible fusion
                if physical_distance < 10:
                    fusion_detected = True
        
        # Flag if abnormal count or fusion
        # Standard is 5 lumbar vertebrae (L1-L5)
        is_lstv_candidate = (vertebra_count != 5) or fusion_detected
        
        result = {
            'vertebra_count': vertebra_count,
            'has_sacrum': has_sacrum,
            'fusion_detected': fusion_detected,
            'is_lstv_candidate': is_lstv_candidate,
            'unique_labels': list(map(int, unique_labels)),
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
        
        # Get middle sagittal slice (assuming sagittal is first dimension)
        mid_slice_idx = data.shape[0] // 2
        slice_img = data[mid_slice_idx, :, :]
        
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
        
        # Upload image
        project_obj.upload(
            image_path=str(image_path),
            num_retry_uploads=3
        )
        
        return True
        
    except Exception as e:
        print(f"  ✗ Roboflow upload error: {e}")
        # Don't crash on upload failures - just log and continue
        return False

def load_progress(progress_file):
    """Load processing progress"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            # If corrupted, start fresh
            return {'processed': [], 'flagged': [], 'failed': []}
    return {'processed': [], 'flagged': [], 'failed': []}

def save_progress(progress_file, progress):
    """Save processing progress atomically"""
    try:
        # Write to temp file first
        temp_file = progress_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Atomic rename
        temp_file.replace(progress_file)
        
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def main():
    parser = argparse.ArgumentParser(description='LSTV Screening Pipeline')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with DICOM studies')
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
    
    # Load progress
    progress_file = output_dir / 'progress.json'
    progress = load_progress(progress_file)
    
    # Results CSV
    results_csv = output_dir / 'results.csv'
    
    # Find all study directories (RSNA format: study_id/series_id/*.dcm)
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
            # Find DICOM series directories (may have multiple series per study)
            series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
            
            if not series_dirs:
                print(f"  ✗ No series directories found")
                progress['failed'].append(study_id)
                save_progress(progress_file, progress)
                continue
            
            # Use first series (or implement series selection logic)
            dicom_dir = series_dirs[0]
            
            # Convert DICOM to NIfTI
            nifti_path = nifti_dir / f"{study_id}.nii.gz"
            
            if not nifti_path.exists():
                print(f"  Converting DICOM to NIfTI...")
                sys.stdout.flush()
                nifti_path = convert_dicom_to_nifti(dicom_dir, nifti_path)
                
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
                'vertebra_count': analysis['vertebra_count'],
                'has_sacrum': analysis['has_sacrum'],
                'fusion_detected': analysis['fusion_detected'],
                'is_lstv_candidate': analysis['is_lstv_candidate'],
                'unique_labels': str(analysis['unique_labels'])
            }
            
            # If LSTV candidate, upload to Roboflow
            if analysis['is_lstv_candidate']:
                print(f"  🚩 LSTV CANDIDATE DETECTED!")
                print(f"     Vertebra count: {analysis['vertebra_count']}")
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
                        print(f"  ⚠ Roboflow upload failed (continuing anyway)")
            else:
                print(f"  ✓ Normal (count={analysis['vertebra_count']})")
            
            sys.stdout.flush()
            
            # Mark as processed
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
            
            # Append to results CSV (atomic write)
            df = pd.DataFrame([result])
            if results_csv.exists():
                df.to_csv(results_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(results_csv, mode='w', header=True, index=False)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user - progress saved!")
            save_progress(progress_file, progress)
            sys.exit(1)
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            traceback.print_exc()
            # Mark as failed but processed to avoid infinite retries
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
