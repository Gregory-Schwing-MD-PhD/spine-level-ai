#!/usr/bin/env python3
"""
LSTV Screening Pipeline - FIXED v1.1
Aligns slice selection with training pipeline

KEY FIX: Now generates spine-aware + parasagittal slices matching training
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


# ============================================================================
# SPINE-AWARE SLICE SELECTOR (Copied from generate_weak_labels.py)
# ============================================================================

class SpineAwareSliceSelector:
    """Intelligent slice selection using spine segmentation"""

    def __init__(self, voxel_spacing_mm=1.0, parasagittal_offset_mm=30):
        self.voxel_spacing_mm = voxel_spacing_mm
        self.parasagittal_offset_mm = parasagittal_offset_mm

    def find_sagittal_axis(self, data_shape):
        """Determine sagittal axis (smallest dimension)"""
        return np.argmin(data_shape)

    def calculate_spine_density(self, seg_data, sag_axis, slice_idx):
        """Calculate spine content in a slice"""
        # Lumbar vertebrae labels from SPINEPS
        lumbar_labels = [20, 21, 22, 23, 24, 26]  # L1-L5 + Sacrum

        vertebra_mask = np.zeros_like(seg_data, dtype=bool)
        for label in lumbar_labels:
            vertebra_mask |= (seg_data == label)

        if sag_axis == 0:
            slice_mask = vertebra_mask[slice_idx, :, :]
        elif sag_axis == 1:
            slice_mask = vertebra_mask[:, slice_idx, :]
        else:
            slice_mask = vertebra_mask[:, :, slice_idx]

        return slice_mask.sum()

    def find_optimal_midline(self, seg_data, sag_axis):
        """Find TRUE spinal midline using segmentation"""
        num_slices = seg_data.shape[sag_axis]
        geometric_mid = num_slices // 2

        lumbar_labels = [20, 21, 22, 23, 24, 26]

        vertebra_mask = np.zeros_like(seg_data, dtype=bool)
        for label in lumbar_labels:
            if label in seg_data:
                vertebra_mask |= (seg_data == label)

        if not vertebra_mask.any():
            return geometric_mid

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
        return optimal_mid

    def get_three_slices(self, seg_data, sag_axis):
        """Get left, mid, right slice indices"""
        optimal_mid = self.find_optimal_midline(seg_data, sag_axis)

        num_slices = seg_data.shape[sag_axis]
        offset_voxels = int(self.parasagittal_offset_mm / self.voxel_spacing_mm)

        left_idx = max(0, optimal_mid - offset_voxels)
        right_idx = min(num_slices - 1, optimal_mid + offset_voxels)

        return {
            'left': left_idx,
            'mid': optimal_mid,
            'right': right_idx,
            'sag_axis': sag_axis,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_series_descriptions(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: {e}")
        return None


def select_best_series(study_dir, series_df=None, study_id=None):
    """Select best T2 sagittal series"""
    series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    if not series_dirs:
        return None

    if series_df is not None and study_id is not None:
        study_series = series_df[series_df['study_id'] == int(study_id)]
        if len(study_series) > 0:
            priorities = ['Sagittal T2', 'Sagittal T2/STIR', 'SAG T2', 'Sagittal T1', 'SAG T1']
            for priority in priorities:
                matching = study_series[study_series['series_description'].str.contains(
                    priority, case=False, na=False)]
                if len(matching) > 0:
                    series_id = str(matching.iloc[0]['series_id'])
                    series_path = study_dir / series_id
                    if series_path.exists():
                        return series_path
    return series_dirs[0]


def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM to NIfTI using dcm2niix"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        study_id = output_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
        bids_base = f"sub-{study_id}_T2w"

        cmd = ['dcm2niix', '-z', 'y', '-f', bids_base, '-o', str(output_path.parent),
               '-m', 'y', '-b', 'n', str(dicom_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"  ✗ dcm2niix failed")
            return None

        expected_output = output_path.parent / f"{bids_base}.nii.gz"
        if not expected_output.exists():
            nifti_files = sorted(output_path.parent.glob(f"{bids_base}*.nii.gz"))
            if not nifti_files:
                return None
            generated_file = nifti_files[0]
            if generated_file != expected_output:
                if expected_output.exists():
                    expected_output.unlink()
                shutil.move(str(generated_file), str(expected_output))
        return expected_output
    except:
        return None


def run_spineps_inference(nifti_path, output_dir):
    """Run SPINEPS segmentation"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = ['bash', '/work/src/screening/spineps_wrapper.sh', 'sample',
               '-i', str(nifti_path),
               '-model_semantic', 't2w',
               '-model_instance', 'instance',
               '-model_labeling', 't2w_labeling',
               '-override_semantic', '-override_instance', '-override_ctd']

        print(f"    Running SPINEPS...")
        sys.stdout.flush()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            return None

        input_parent = nifti_path.parent
        derivatives_base = input_parent / "derivatives_seg"
        if not derivatives_base.exists():
            return None

        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
        seg_pattern = f"sub-{study_id}_mod-T2w_seg-vert_msk.nii.gz"
        seg_file = derivatives_base / seg_pattern

        if not seg_file.exists():
            seg_files = list(derivatives_base.glob("*_seg-vert_msk.nii.gz"))
            if not seg_files:
                return None
            seg_file = seg_files[0]

        seg_output = output_dir / f"{study_id}_seg.nii.gz"
        shutil.copy(seg_file, seg_output)
        print(f"  ✓ Saved: {seg_output.name}")
        return seg_output
    except:
        return None


def analyze_segmentation(seg_path):
    """Analyze segmentation for LSTV candidates"""
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
    except:
        return None


def extract_slice(data, sag_axis, slice_idx, thickness=1):
    """Extract 2D slice from 3D volume"""
    if thickness <= 1:
        if sag_axis == 0:
            return data[slice_idx, :, :]
        elif sag_axis == 1:
            return data[:, slice_idx, :]
        else:
            return data[:, :, slice_idx]

    # Thick slab MIP
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


def normalize_slice(img_slice):
    """Normalize to 0-255 uint8 with CLAHE enhancement"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) /
                     (img_slice.max() - img_slice.min()) * 255)
        normalized = normalized.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)
        return normalized
    return np.zeros_like(img_slice, dtype=np.uint8)


def extract_three_view_images(nifti_path, seg_path, output_dir, study_id, selector):
    """
    FIXED: Extract 3 views matching training pipeline
    - LEFT parasagittal (ribs visible)
    - MID spine-aware (TPs visible)
    - RIGHT parasagittal (ribs visible)
    """
    try:
        nii = nib.load(nifti_path)
        seg_nii = nib.load(seg_path)

        mri_data = nii.get_fdata()
        seg_data = seg_nii.get_fdata().astype(int)

        # Get spine-aware slice indices
        dims = mri_data.shape
        sag_axis = np.argmin(dims)
        slice_info = selector.get_three_slices(seg_data, sag_axis)

        views = {
            'left': slice_info['left'],
            'mid': slice_info['mid'],
            'right': slice_info['right'],
        }

        output_paths = {}

        for view_name, slice_idx in views.items():
            # Use thick slab MIP for ribs, thin for midline
            thickness = 15 if view_name in ['left', 'right'] else 5
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
            normalized = normalize_slice(mri_slice)

            output_path = output_dir / f"{study_id}_{view_name}.jpg"
            cv2.imwrite(str(output_path), normalized)
            output_paths[view_name] = output_path

        return output_paths

    except Exception as e:
        print(f"  ✗ Image extraction failed: {e}")
        return None


def upload_to_roboflow(image_path, study_id, roboflow_key, workspace, project):
    """Upload image to Roboflow"""
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
    except:
        return False


def load_progress(progress_file):
    """Load progress from JSON file"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'processed': [], 'flagged': [], 'failed': []}


def save_progress(progress_file, progress):
    """Save progress to JSON file"""
    try:
        temp_file = progress_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(progress_file)
    except:
        pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LSTV Screening - FIXED v1.1 (Aligned slice selection)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CRITICAL FIX:
  Now generates spine-aware + parasagittal slices matching training pipeline
  - LEFT: parasagittal (mid-30mm) for T12 ribs
  - MID: spine-aware optimal for L5 TPs
  - RIGHT: parasagittal (mid+30mm) for T12 ribs

Examples:
  # Run screening with fixed slice selection
  python lstv_screen_FIXED.py \\
    --input_dir /data/dicom \\
    --output_dir /data/lstv_screening \\
    --roboflow_key YOUR_KEY \\
    --generate_three_views
        """
    )
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--series_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--roboflow_key', type=str, required=True)
    parser.add_argument('--roboflow_workspace', type=str, default='lstv-screening')
    parser.add_argument('--roboflow_project', type=str, default='lstv-candidates')
    parser.add_argument('--generate_three_views', action='store_true',
                       help='Generate 3-view images matching training (RECOMMENDED)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_dir = output_dir / 'nifti'
    seg_dir = output_dir / 'segmentations'
    images_dir = output_dir / 'candidate_images'
    for d in [nifti_dir, seg_dir, images_dir]:
        d.mkdir(exist_ok=True)

    # Load series descriptions if available
    series_df = None
    if args.series_csv:
        series_csv = Path(args.series_csv)
        if series_csv.exists():
            series_df = load_series_descriptions(series_csv)
            if series_df is not None:
                print(f"Loaded series: {len(series_df)} entries")

    # Initialize spine-aware slice selector
    selector = SpineAwareSliceSelector()

    # Progress tracking
    progress_file = output_dir / 'progress.json'
    progress = load_progress(progress_file)
    results_csv = output_dir / 'results.csv'

    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if args.limit:
        study_dirs = study_dirs[:args.limit]

    print("="*60)
    print("LSTV SCREENING - FIXED v1.1")
    print("="*60)
    print(f"Studies: {len(study_dirs)}")
    print(f"Processed: {len(progress['processed'])}")
    if args.generate_three_views:
        print("Mode: 3-VIEW SPINE-AWARE (matches training)")
    else:
        print("Mode: SINGLE MIDLINE VIEW (legacy)")
    print("="*60)
    sys.stdout.flush()

    for study_dir in tqdm(study_dirs, desc="Processing"):
        study_id = study_dir.name
        if study_id in progress['processed']:
            continue

        print(f"\n[{study_id}]")
        sys.stdout.flush()

        try:
            # Select best series
            series_dir = select_best_series(study_dir, series_df, study_id)
            if series_dir is None:
                print(f"  ✗ No series")
                progress['failed'].append(study_id)
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue

            print(f"  Series: {series_dir.name}")

            # Convert to NIfTI
            nifti_path = nifti_dir / f"sub-{study_id}_T2w.nii.gz"
            if not nifti_path.exists():
                print(f"  Converting...")
                nifti_path = convert_dicom_to_nifti(series_dir, nifti_path)
                if nifti_path is None:
                    progress['failed'].append(study_id)
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue

            # Run SPINEPS segmentation
            seg_path = seg_dir / f"{study_id}_seg.nii.gz"
            if not seg_path.exists():
                print(f"  Segmenting...")
                seg_path = run_spineps_inference(nifti_path, seg_dir)
                if seg_path is None:
                    progress['failed'].append(study_id)
                    progress['processed'].append(study_id)
                    save_progress(progress_file, progress)
                    continue

            # Analyze segmentation
            print(f"  Analyzing...")
            analysis = analyze_segmentation(seg_path)
            if analysis is None:
                progress['failed'].append(study_id)
                progress['processed'].append(study_id)
                save_progress(progress_file, progress)
                continue

            result = {
                'study_id': study_id,
                'series_id': series_dir.name,
                'vertebra_count': analysis['vertebra_count'],
                'is_lstv_candidate': analysis['is_lstv_candidate'],
                'lstv_type': analysis['lstv_type'],
                'lumbar_labels': str(analysis['lumbar_labels']),
            }

            # If LSTV candidate, extract and upload images
            if analysis['is_lstv_candidate']:
                print(f"  🚩 LSTV! Count={analysis['vertebra_count']}, "
                      f"Type={analysis['lstv_type']}")

                if args.generate_three_views:
                    # FIXED: Generate 3 views matching training
                    image_paths = extract_three_view_images(
                        nifti_path, seg_path, images_dir, study_id, selector)
                    
                    if image_paths:
                        # Upload all 3 views
                        upload_success = 0
                        for view_name, image_path in image_paths.items():
                            if upload_to_roboflow(
                                image_path, f"{study_id}_{view_name}",
                                args.roboflow_key, args.roboflow_workspace,
                                args.roboflow_project
                            ):
                                upload_success += 1
                        
                        print(f"  ✓ Uploaded {upload_success}/3 views")
                        progress['flagged'].append(study_id)
                else:
                    # Legacy: single midline view (NOT RECOMMENDED)
                    image_path = images_dir / f"{study_id}.jpg"
                    # (old extract_middle_slice code would go here)
                    print(f"  ⚠ Legacy mode - use --generate_three_views instead")
            else:
                print(f"  ✓ Normal ({analysis['vertebra_count']} lumbar)")

            progress['processed'].append(study_id)
            save_progress(progress_file, progress)

            # Save result to CSV
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

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Processed: {len(progress['processed'])}")
    print(f"LSTV: {len(progress['flagged'])}")
    print(f"Failed: {len(progress.get('failed', []))}")
    print("="*60)


if __name__ == "__main__":
    main()
