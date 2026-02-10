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
from roboflow import Roboflow
import subprocess
import shutil

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM series to NIfTI using dcm2niix"""
    try:
        # Use dcm2niix for conversion
        cmd = [
            'dcm2niix',
            '-z', 'y',  # Compress output
            '-f', output_path.stem,  # Output filename
            '-o', str(output_path.parent),  # Output directory
            str(dicom_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ✗ dcm2niix failed: {result.stderr}")
            return None
        
        # Find the generated .nii.gz file
        nifti_files = list(output_path.parent.glob(f"{output_path.stem}*.nii.gz"))
        
        if not nifti_files:
            print(f"  ✗ No NIfTI file generated")
            return None
        
        return nifti_files[0]
        
    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        return None

def run_spineps_inference(nifti_path, output_dir):
    """Run SPINEPS segmentation on NIfTI volume"""
    try:
        from spineps.seg_run import process_img_nii
        
        # Run SPINEPS
        seg_output = output_dir / f"{nifti_path.stem}_seg.nii.gz"
        
        # SPINEPS inference (adjust based on actual API)
        result = process_img_nii(
            img_path=str(nifti_path),
            derivative_name=str(seg_output),
            verbose=False
        )
        
        if not seg_output.exists():
            print(f"  ✗ SPINEPS segmentation failed")
            return None
        
        return seg_output
        
    except Exception as e:
        print(f"  ✗ SPINEPS error: {e}")
        return None

def analyze_segmentation(seg_path):
    """Analyze SPINEPS segmentation for LSTV indicators"""
    try:
        # Load segmentation
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata()
        
        # Get unique vertebrae labels (SPINEPS uses specific label scheme)
        # Assuming L1-L5 are labels 20-24, Sacrum is 26
        unique_labels = np.unique(seg_data)
        
        # Count lumbar vertebrae (adjust label range based on SPINEPS)
        lumbar_labels = [l for l in unique_labels if 20 <= l <= 24]
        vertebra_count = len(lumbar_labels)
        
        # Check for sacrum
        has_sacrum = 26 in unique_labels
        
        # Calculate L5-S1 distance (if both present)
        fusion_detected = False
        if 24 in unique_labels and has_sacrum:
            # Get centroids
            l5_mask = (seg_data == 24)
            sacrum_mask = (seg_data == 26)
            
            l5_centroid = np.array(np.where(l5_mask)).mean(axis=1)
            sacrum_centroid = np.array(np.where(sacrum_mask)).mean(axis=1)
            
            # Calculate distance (in voxels)
            distance = np.linalg.norm(l5_centroid - sacrum_centroid)
            
            # If distance is very small, likely fusion
            if distance < 5:  # Threshold in voxels
                fusion_detected = True
        
        # Flag if abnormal count or fusion
        is_lstv_candidate = (vertebra_count != 5) or fusion_detected
        
        return {
            'vertebra_count': vertebra_count,
            'has_sacrum': has_sacrum,
            'fusion_detected': fusion_detected,
            'is_lstv_candidate': is_lstv_candidate
        }
        
    except Exception as e:
        print(f"  ✗ Analysis error: {e}")
        return None

def extract_middle_slice(nifti_path, output_jpg):
    """Extract middle sagittal slice as JPG for Roboflow upload"""
    try:
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        
        # Get middle sagittal slice
        mid_slice = data.shape[0] // 2
        slice_img = data[mid_slice, :, :]
        
        # Normalize to 0-255
        slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        
        # Save as JPG
        cv2.imwrite(str(output_jpg), slice_img)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Slice extraction error: {e}")
        return False

def upload_to_roboflow(image_path, study_id, roboflow_key, workspace, project):
    """Upload image to Roboflow workspace"""
    try:
        rf = Roboflow(api_key=roboflow_key)
        project_obj = rf.workspace(workspace).project(project)
        
        project_obj.upload(
            image_path=str(image_path),
            annotation_name=f"{study_id}_lstv_candidate"
        )
        
        return True
        
    except Exception as e:
        print(f"  ✗ Roboflow upload error: {e}")
        return False

def load_progress(progress_file):
    """Load processing progress"""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'processed': [], 'flagged': []}

def save_progress(progress_file, progress):
    """Save processing progress"""
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

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
    
    # Find all study directories
    study_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if args.limit:
        study_dirs = study_dirs[:args.limit]
    
    print("="*60)
    print("LSTV SCREENING PIPELINE")
    print("="*60)
    print(f"Total studies: {len(study_dirs)}")
    print(f"Already processed: {len(progress['processed'])}")
    print(f"Already flagged: {len(progress['flagged'])}")
    print("="*60)
    
    # Process studies
    results = []
    
    for study_dir in tqdm(study_dirs, desc="Processing studies"):
        study_id = study_dir.name
        
        # Skip if already processed
        if study_id in progress['processed']:
            if args.verbose:
                print(f"  Skipping {study_id} (already processed)")
            continue
        
        print(f"\nProcessing: {study_id}")
        
        try:
            # Convert DICOM to NIfTI
            nifti_path = nifti_dir / f"{study_id}.nii.gz"
            
            if not nifti_path.exists():
                print(f"  Converting DICOM to NIfTI...")
                nifti_path = convert_dicom_to_nifti(study_dir, nifti_path)
                
                if nifti_path is None:
                    print(f"  ✗ Conversion failed, skipping")
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue
            
            # Run SPINEPS
            seg_path = seg_dir / f"{study_id}_seg.nii.gz"
            
            if not seg_path.exists():
                print(f"  Running SPINEPS segmentation...")
                seg_path = run_spineps_inference(nifti_path, seg_dir)
                
                if seg_path is None:
                    print(f"  ✗ Segmentation failed, skipping")
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue
            
            # Analyze segmentation
            print(f"  Analyzing segmentation...")
            analysis = analyze_segmentation(seg_path)
            
            if analysis is None:
                print(f"  ✗ Analysis failed, skipping")
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue
            
            # Store results
            result = {
                'study_id': study_id,
                'vertebra_count': analysis['vertebra_count'],
                'has_sacrum': analysis['has_sacrum'],
                'fusion_detected': analysis['fusion_detected'],
                'is_lstv_candidate': analysis['is_lstv_candidate']
            }
            
            results.append(result)
            
            # If LSTV candidate, upload to Roboflow
            if analysis['is_lstv_candidate']:
                print(f"  🚩 LSTV CANDIDATE DETECTED!")
                print(f"     Vertebra count: {analysis['vertebra_count']}")
                print(f"     Fusion detected: {analysis['fusion_detected']}")
                
                # Extract middle slice
                image_path = images_dir / f"{study_id}.jpg"
                
                if extract_middle_slice(nifti_path, image_path):
                    print(f"  Uploading to Roboflow...")
                    
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
                        print(f"  ✗ Roboflow upload failed")
            else:
                print(f"  ✓ Normal study (vertebra_count={analysis['vertebra_count']})")
            
            # Mark as processed
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
            
            # Append to results CSV
            df = pd.DataFrame([result])
            df.to_csv(results_csv, mode='a', header=not results_csv.exists(), index=False)
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            # Still mark as processed to avoid infinite retries
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("SCREENING COMPLETE")
    print("="*60)
    print(f"Total processed: {len(progress['processed'])}")
    print(f"LSTV candidates flagged: {len(progress['flagged'])}")
    print(f"Results saved to: {results_csv}")
    print("="*60)

if __name__ == "__main__":
    main()
