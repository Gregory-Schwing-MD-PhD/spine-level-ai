#!/usr/bin/env python3
"""
LSTV Screening Pipeline using SPINEPS
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
from tqdm import tqdm
import subprocess
import shutil
import traceback

def load_series_descriptions(csv_path):
    """Load series descriptions"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Warning: Could not load series descriptions: {e}")
        return None

def select_best_series(study_dir, series_df=None, study_id=None):
    """Select best series for LSTV screening"""
    series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    
    if not series_dirs:
        return None
    
    if series_df is not None and study_id is not None:
        study_series = series_df[series_df['study_id'] == int(study_id)]
        
        if len(study_series) > 0:
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
    
    return series_dirs[0]

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM to NIfTI with BIDS-compliant naming"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract study_id from output path
        # output_path should be: nifti/sub-100206310_T2w.nii.gz
        study_id = output_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
        
        # SPINEPS expects BIDS format: sub-{id}_T2w
        # BUT dcm2niix adds .nii automatically, so we give it: sub-{id}_T2w
        # Result will be: sub-{id}_T2w.nii.gz (perfect!)
        bids_base = f"sub-{study_id}_T2w"  # NO .nii extension here!
        
        cmd = [
            'dcm2niix',
            '-z', 'y',  # Compress
            '-f', bids_base,  # Filename WITHOUT .nii extension
            '-o', str(output_path.parent),
            '-m', 'y',  # Merge
            '-b', 'n',  # No JSON
            str(dicom_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"  ✗ dcm2niix failed: {result.stderr[:200]}")
            return None
        
        # dcm2niix creates: sub-{id}_T2w.nii.gz (exactly what we want!)
        expected_output = output_path.parent / f"{bids_base}.nii.gz"
        
        if not expected_output.exists():
            # Try to find with suffixes
            nifti_files = sorted(output_path.parent.glob(f"{bids_base}*.nii.gz"))
            
            if not nifti_files:
                print(f"  ✗ No NIfTI generated")
                print(f"    Expected: {expected_output}")
                return None
            
            # Take first file and rename if needed
            generated_file = nifti_files[0]
            if generated_file != expected_output:
                if expected_output.exists():
                    expected_output.unlink()
                shutil.move(str(generated_file), str(expected_output))
        
        return expected_output
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Conversion timeout")
        return None
    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        return None

def run_spineps_inference(nifti_path, output_dir):
    """Run SPINEPS segmentation"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use wrapper script
        cmd = [
            'bash', '/work/src/screening/spineps_wrapper.sh',
            'sample',
            '-i', str(nifti_path),
            '-model_semantic', 't2w',
            '-model_instance', 'instance',
            '-model_labeling', 't2w_labeling',
            '-override_semantic',
            '-override_instance',
            '-override_ctd',
        ]
        
        print(f"    Running SPINEPS...")
        sys.stdout.flush()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"  ✗ SPINEPS failed (code {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return None
        
        # SPINEPS creates output NEXT TO the input file
        # Structure: /data/output/nifti/derivatives_seg/
        input_parent = nifti_path.parent
        derivatives_base = input_parent / "derivatives_seg"
        
        if not derivatives_base.exists():
            print(f"  ✗ derivatives_seg not found")
            return None
        
        # Extract study_id from input
        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
        
        # SPINEPS creates files directly in derivatives_seg (flat structure)
        # File pattern: sub-{id}_mod-T2w_seg-vert_msk.nii.gz
        seg_pattern = f"sub-{study_id}_mod-T2w_seg-vert_msk.nii.gz"
        seg_file = derivatives_base / seg_pattern
        
        if not seg_file.exists():
            # Try to find any seg-vert file
            seg_files = list(derivatives_base.glob("*_seg-vert_msk.nii.gz"))
            
            if not seg_files:
                print(f"  ✗ No segmentation file found")
                print(f"    Expected: {seg_pattern}")
                print(f"    Contents:")
                for item in sorted(derivatives_base.iterdir())[:10]:
                    print(f"      - {item.name}")
                return None
            
            seg_file = seg_files[0]
        
        # Copy to output directory
        seg_output = output_dir / f"{study_id}_seg.nii.gz"
        shutil.copy(seg_file, seg_output)
        
        print(f"  ✓ Saved: {seg_output.name}")
        return seg_output
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return None

def analyze_segmentation(seg_path):
    """Analyze segmentation for LSTV"""
    try:
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(int)
        
        unique_labels = np.unique(seg_data)
        vertebra_labels = [l for l in unique_labels if 1 <= l <= 25]
        lumbar_labels = [l for l in vertebra_labels if 20 <= l <= 25]
        
        vertebra_count = len(lumbar_labels)
        has_sacrum = 26 in unique_labels
        has_l6 = 25 in lumbar_labels
        s1_s2_disc = 126 in unique_labels
        
        is_lstv = (vertebra_count != 5 or s1_s2_disc or has_l6)
        
        lstv_type = "normal"
        if vertebra_count < 5:
            lstv_type = "sacralization"
        elif vertebra_count > 5 or has_l6:
            lstv_type = "lumbarization"
        elif s1_s2_disc:
            lstv_type = "s1_s2_disc"
        
        return {
            'vertebra_count': vertebra_count,
            'has_sacrum': has_sacrum,
            'has_l6': has_l6,
            's1_s2_disc': s1_s2_disc,
            'is_lstv_candidate': is_lstv,
            'lstv_type': lstv_type,
            'unique_labels': list(map(int, unique_labels)),
            'lumbar_labels': list(map(int, lumbar_labels)),
        }
        
    except Exception as e:
        print(f"  ✗ Analysis error: {e}")
        return None

def extract_middle_slice(nifti_path, output_jpg):
    """Extract middle slice"""
    try:
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        
        dims = data.shape
        sag_axis = np.argmin(dims)
        mid_idx = dims[sag_axis] // 2
        
        if sag_axis == 0:
            slice_img = data[mid_idx, :, :]
        elif sag_axis == 1:
            slice_img = data[:, mid_idx, :]
        else:
            slice_img = data[:, :, mid_idx]
        
        if slice_img.max() > slice_img.min():
            slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        else:
            slice_img = np.zeros_like(slice_img, dtype=np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        slice_img = clahe.apply(slice_img)
        
        output_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_jpg), slice_img)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Slice error: {e}")
        return False

def upload_to_roboflow(image_path, study_id, roboflow_key, workspace, project):
    """Upload to Roboflow"""
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=roboflow_key)
        workspace_obj = rf.workspace(workspace)
        project_obj = workspace_obj.project(project)
        
        project_obj.upload(
            image_path=str(image_path),
            split="train",
            tag_names=["lstv-candidate", "automated"],
            num_retry_uploads=3
        )
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Upload error: {e}")
        return False

def load_progress(progress_file):
    """Load progress"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            return {'processed': [], 'flagged': [], 'failed': []}
    return {'processed': [], 'flagged': [], 'failed': []}

def save_progress(progress_file, progress):
    """Save progress"""
    try:
        temp_file = progress_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(progress_file)
    except Exception as e:
        print(f"Warning: Save failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='LSTV Screening')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--series_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--roboflow_key', type=str, required=True)
    parser.add_argument('--roboflow_workspace', type=str, default='lstv-screening')
    parser.add_argument('--roboflow_project', type=str, default='lstv-candidates')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nifti_dir = output_dir / 'nifti'
    seg_dir = output_dir / 'segmentations'
    images_dir = output_dir / 'candidate_images'
    
    nifti_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Load series
    series_df = None
    if args.series_csv:
        series_csv = Path(args.series_csv)
        if series_csv.exists():
            series_df = load_series_descriptions(series_csv)
            print(f"Loaded series: {len(series_df)} entries")
    
    # Progress
    progress_file = output_dir / 'progress.json'
    progress = load_progress(progress_file)
    results_csv = output_dir / 'results.csv'
    
    # Find studies
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if args.limit:
        study_dirs = study_dirs[:args.limit]
    
    print("="*60)
    print("LSTV SCREENING")
    print("="*60)
    print(f"Studies: {len(study_dirs)}")
    print(f"Processed: {len(progress['processed'])}")
    print("="*60)
    sys.stdout.flush()
    
    # Process
    for study_dir in tqdm(study_dirs, desc="Processing"):
        study_id = study_dir.name
        
        if study_id in progress['processed']:
            continue
        
        print(f"\n[{study_id}]")
        sys.stdout.flush()
        
        try:
            # Select series
            series_dir = select_best_series(study_dir, series_df, study_id)
            if series_dir is None:
                print(f"  ✗ No series")
                progress['failed'].append(study_id)
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue
            
            print(f"  Series: {series_dir.name}")
            
            # Convert - use BIDS naming convention for SPINEPS
            nifti_path = nifti_dir / f"sub-{study_id}_T2w.nii.gz"
            if not nifti_path.exists():
                print(f"  Converting...")
                nifti_path = convert_dicom_to_nifti(series_dir, nifti_path)
                if nifti_path is None:
                    progress['failed'].append(study_id)
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue
            
            # Segment
            seg_path = seg_dir / f"{study_id}_seg.nii.gz"
            if not seg_path.exists():
                print(f"  Segmenting...")
                seg_path = run_spineps_inference(nifti_path, seg_dir)
                if seg_path is None:
                    progress['failed'].append(study_id)
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue
            
            # Analyze
            print(f"  Analyzing...")
            analysis = analyze_segmentation(seg_path)
            if analysis is None:
                progress['failed'].append(study_id)
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue
            
            # Save
            result = {
                'study_id': study_id,
                'series_id': series_dir.name,
                'vertebra_count': analysis['vertebra_count'],
                'is_lstv_candidate': analysis['is_lstv_candidate'],
                'lstv_type': analysis['lstv_type'],
                'lumbar_labels': str(analysis['lumbar_labels']),
            }
            
            # Upload if LSTV
            if analysis['is_lstv_candidate']:
                print(f"  🚩 LSTV! Count={analysis['vertebra_count']}, Type={analysis['lstv_type']}")
                
                image_path = images_dir / f"{study_id}.jpg"
                if extract_middle_slice(nifti_path, image_path):
                    if upload_to_roboflow(image_path, study_id, args.roboflow_key,
                                         args.roboflow_workspace, args.roboflow_project):
                        print(f"  ✓ Uploaded")
                        progress['flagged'].append(study_id)
            else:
                print(f"  ✓ Normal ({analysis['vertebra_count']} lumbar)")
            
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
            
            # CSV
            df = pd.DataFrame([result])
            if results_csv.exists():
                df.to_csv(results_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(results_csv, mode='w', header=True, index=False)
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted!")
            save_progress(progress_file, progress)
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            progress['failed'].append(study_id)
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Processed: {len(progress['processed'])}")
    print(f"LSTV: {len(progress['flagged'])}")
    print(f"Failed: {len(progress.get('failed', []))}")
    print("="*60)

if __name__ == "__main__":
    main()
