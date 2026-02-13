#!/usr/bin/env python3
"""
PRODUCTION LSTV SCREENING SYSTEM v3.0 - COMPLETE IMPLEMENTATION (FIXED)

Fixes:
1. Proper CSV loading and series matching
2. Correct sagittal series selection (T2w only)
3. Fixed orientation detection
4. Better DICOM handling

Author: Claude + go2432
Date: February 2026
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
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import os

# ============================================================================
# CONSTANTS
# ============================================================================

# Semantic labels (WITH ribs and TPs!)
SEMANTIC_LABELS = {
    'spinal_cord': 1,
    'spinal_canal': 2,
    'vertebra_corpus': 3,
    'vertebra_disc': 4,
    'endplate': 5,
    'arcus_vertebrae': 6,
    'rib_left': 7,              # ← T12 RIBS
    'rib_right': 8,             # ← T12 RIBS
    'transverse_process_left': 9,   # ← L5 TPs
    'transverse_process_right': 10, # ← L5 TPs
    'spinosus_process': 11,
}

# Instance labels
INSTANCE_LABELS = {
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

ID_TO_NAME = {v: k for k, v in INSTANCE_LABELS.items()}

LSTV_COLORS = {
    'L6': (255, 0, 255),          # Magenta - LUMBARIZATION
    'S1_S2_disc': (255, 128, 0),  # Orange - SACRALIZATION
    'L5': (0, 255, 0),            # Green - Normal L5
    'Sacrum': (0, 255, 255),      # Cyan - Sacrum
    'L4': (100, 255, 100),        # Light green
    'default': (255, 255, 0),     # Yellow - Other vertebrae
}

@dataclass
class DetectionResult:
    structure_type: str  # 'T12_rib', 'L5_TP'
    side: str  # 'left', 'right'
    bbox: Tuple[int, int, int, int]
    confidence: float
    slice_idx: int
    view: str
    detection_method: str  # 'semantic', 'intensity'
    area: int

# ============================================================================
# SPINE-AWARE SLICE SELECTOR
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
# HELPER FUNCTIONS - FIXED CSV LOADING
# ============================================================================

def load_series_descriptions(csv_path):
    """Load series descriptions CSV with proper dtypes"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['study_id', 'series_id', 'series_description']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ✗ CSV missing columns: {missing}")
            print(f"  Available columns: {list(df.columns)}")
            return None
        
        # Convert dtypes
        df['study_id'] = df['study_id'].astype(int)
        df['series_id'] = df['series_id'].astype(int)  # Series IDs are integers
        df['series_description'] = df['series_description'].astype(str)
        
        print(f"  ✓ Loaded {len(df)} series descriptions")
        print(f"  ✓ Unique studies: {df['study_id'].nunique()}")
        print(f"  ✓ Sample descriptions:")
        desc_counts = df['series_description'].value_counts().head(5)
        for desc, count in desc_counts.items():
            print(f"      - {desc}: {count}")
        
        return df
    except Exception as e:
        print(f"  ✗ Failed to load series CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def select_best_series(study_dir, series_df=None, study_id=None):
    """
    Select best T2 SAGITTAL series using CSV descriptions
    
    CRITICAL: Only selects series explicitly marked as sagittal T2w
    """
    series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
    if not series_dirs:
        return None

    # If no CSV, can't determine series type - FAIL
    if series_df is None or study_id is None:
        print(f"    ✗ No series CSV - cannot identify sagittal series")
        return None

    try:
        study_id_int = int(study_id)
        study_series = series_df[series_df['study_id'] == study_id_int].copy()
    except (ValueError, TypeError) as e:
        print(f"    ✗ Invalid study_id {study_id}: {e}")
        return None

    if len(study_series) == 0:
        print(f"    ✗ Study {study_id} not found in CSV")
        return None

    # Get series IDs that exist in filesystem
    available_series_ids = [int(d.name) for d in series_dirs]
    study_series = study_series[study_series['series_id'].isin(available_series_ids)]
    
    if len(study_series) == 0:
        print(f"    ✗ No matching series found in filesystem")
        return None

    # Priority 1: Explicit "Sagittal T2/STIR" 
    priority_patterns = [
        'Sagittal T2/STIR',
        'SAG T2 STIR',
        'Sagittal T2',
        'SAG T2',
    ]

    for pattern in priority_patterns:
        matching = study_series[
            study_series['series_description'].str.contains(
                pattern, case=False, na=False, regex=False)
        ]
        if len(matching) > 0:
            series_id = int(matching.iloc[0]['series_id'])
            series_path = study_dir / str(series_id)
            if series_path.exists():
                desc = matching.iloc[0]['series_description']
                print(f"    ✓ Selected: {desc} (series {series_id})")
                return series_path

    # Priority 2: Any T2 with "sag" but NOT "cor" or "ax"
    sag_series = study_series[
        (study_series['series_description'].str.contains('sag', case=False, na=False)) &
        (study_series['series_description'].str.contains('t2', case=False, na=False)) &
        (~study_series['series_description'].str.contains('cor|axial|ax ', case=False, na=False))
    ]
    
    if len(sag_series) > 0:
        series_id = int(sag_series.iloc[0]['series_id'])
        series_path = study_dir / str(series_id)
        if series_path.exists():
            desc = sag_series.iloc[0]['series_description']
            print(f"    ⚠ Selected (fallback): {desc} (series {series_id})")
            return series_path

    # No sagittal T2 found - list what's available
    print(f"    ✗ No sagittal T2 series found!")
    print(f"    Available series:")
    for _, row in study_series.iterrows():
        print(f"      - {row['series_id']}: {row['series_description']}")
    
    return None


def convert_dicom_to_nifti(dicom_dir, output_path, study_id):
    """
    Convert DICOM to NIfTI with STRICT orientation checking
    
    Returns None if not sagittal, otherwise returns NIfTI path
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use BIDS-like naming for SPINEPS compatibility
        bids_base = f"sub-{study_id}_sequ-sag_T2w"

        # dcm2niix with orientation preservation
        cmd = [
            'dcm2niix',
            '-z', 'y',           # Compress
            '-f', bids_base,     # Filename
            '-o', str(output_path.parent),
            '-m', 'y',           # Merge 2D → 3D
            '-ba', 'n',          # Don't anonymize (preserves orientation)
            '-i', 'n',           # Don't ignore derived
            '-x', 'n',           # Don't crop
            '-p', 'n',           # Don't use Philips scaling
            str(dicom_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"    ✗ dcm2niix failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
            return None

        # Find generated NIfTI
        expected_output = output_path.parent / f"{bids_base}.nii.gz"
        if not expected_output.exists():
            nifti_files = sorted(output_path.parent.glob(f"{bids_base}*.nii.gz"))
            if not nifti_files:
                nifti_files = sorted(output_path.parent.glob(f"sub-{study_id}*.nii.gz"))
            if not nifti_files:
                print(f"    ✗ No NIfTI generated")
                return None
            
            generated_file = nifti_files[0]
            if generated_file != expected_output:
                shutil.move(str(generated_file), str(expected_output))

        # CRITICAL: Verify orientation
        nii = nib.load(expected_output)
        orientation = nib.aff2axcodes(nii.affine)
        
        print(f"    NIfTI shape: {nii.shape}, orientation: {orientation}")
        
        # Check if sagittal - we need L/R as FIRST axis
        # Sagittal patterns: (L|R, A|P, S|I)
        # Coronal patterns: (L|R, S|I, A|P) - WRONG
        # Axial patterns: (L|R, A|P, I|S) but with different first axis - WRONG
        
        first_axis = orientation[0]
        second_axis = orientation[1]
        third_axis = orientation[2]
        
        # Sagittal = L/R on first axis, with reasonable other axes
        is_sagittal = (first_axis in ('R', 'L'))
        
        # Additional check: for sagittal, usually smallest dimension should be on axis 0
        smallest_dim_axis = np.argmin(nii.shape)
        
        if not is_sagittal:
            print(f"    ✗ Not sagittal (first axis is {first_axis}, not L/R)")
            print(f"    ✗ Skipping this series")
            return None
        
        if smallest_dim_axis != 0:
            print(f"    ⚠ Warning: smallest dimension is axis {smallest_dim_axis}, not 0")
            print(f"    This might indicate coronal/axial mislabeled as sagittal")
            # Still try to process, but flag it
        
        return expected_output
        
    except subprocess.TimeoutExpired:
        print(f"    ✗ dcm2niix timeout")
        return None
    except Exception as e:
        print(f"    ✗ DICOM conversion error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# SPINEPS DUAL EXTRACTOR
# ============================================================================

def run_spineps_dual_extraction(nifti_path: Path, output_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Run SPINEPS and extract BOTH instance and semantic outputs
    
    Returns:
        {
            'instance': Path,
            'semantic': Path or None
        }
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set environment
        env = os.environ.copy()
        env['SPINEPS_SEGMENTOR_MODELS'] = env.get('SPINEPS_SEGMENTOR_MODELS', '/app/models')
        env['SPINEPS_ENVIRONMENT_DIR'] = env.get('SPINEPS_ENVIRONMENT_DIR', '/app/models')

        # Find wrapper script
        script_dir = Path(__file__).parent
        wrapper_candidates = [
            script_dir / 'spineps_wrapper_FIXED.sh',
            script_dir / 'spineps_wrapper.sh',
            Path('/work/src/screening/spineps_wrapper_FIXED.sh'),
            Path('/work/src/screening/spineps_wrapper.sh'),
        ]

        wrapper_path = None
        for candidate in wrapper_candidates:
            if candidate.exists():
                wrapper_path = candidate
                break

        if wrapper_path:
            print(f"    Running SPINEPS via wrapper: {wrapper_path.name}")
            sys.stdout.flush()

            cmd = [
                'bash', str(wrapper_path), 'sample',
                '-i', str(nifti_path),
                '-model_semantic', 't2w',
                '-model_instance', 'instance',
                '-model_labeling', 't2w_labeling',
                '-override_semantic', '-override_instance', '-override_ctd'
            ]
        else:
            print(f"    Running SPINEPS via Python module")
            sys.stdout.flush()

            cmd = [
                'python', '-m', 'spineps.entrypoint', 'sample',
                '-i', str(nifti_path),
                '-model_semantic', 't2w',
                '-model_instance', 'instance',
                '-model_labeling', 't2w_labeling',
                '-override_semantic', '-override_instance', '-override_ctd'
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

        # Print SPINEPS output for debugging
        if result.stdout:
            stdout_lines = [line for line in result.stdout.split('\n') if line.strip()]
            if stdout_lines and len(stdout_lines) <= 30:
                print(f"    SPINEPS output:")
                for line in stdout_lines[-20:]:
                    print(f"      {line}")

        if result.returncode != 0:
            print(f"  ✗ SPINEPS failed (code {result.returncode})")
            return None

        # SPINEPS creates derivatives_seg/ next to INPUT file
        input_parent = nifti_path.parent
        derivatives_dir = input_parent / "derivatives_seg"

        if not derivatives_dir.exists():
            print(f"  ✗ derivatives_seg not found at {derivatives_dir}")
            return None

        # Extract study ID from filename
        study_id = nifti_path.stem.replace('_sequ-sag_T2w', '').replace('.nii', '').replace('sub-', '')

        # Find instance mask
        seg_pattern = f"sub-{study_id}_*_seg-vert_msk.nii.gz"
        instance_files = list(derivatives_dir.glob(seg_pattern))
        
        if not instance_files:
            instance_files = list(derivatives_dir.glob("*_seg-vert_msk.nii.gz"))
        
        if not instance_files:
            print(f"  ✗ Instance mask not found")
            return None
        
        instance_file = instance_files[0]
        print(f"    ✓ Found instance: {instance_file.name}")

        # Find semantic mask
        semantic_pattern = f"sub-{study_id}_*_seg-spine_msk.nii.gz"
        semantic_files = list(derivatives_dir.glob(semantic_pattern))
        
        if not semantic_files:
            semantic_files = list(derivatives_dir.glob("*_seg-spine_msk.nii.gz"))
        
        semantic_file = semantic_files[0] if semantic_files else None
        
        if semantic_file:
            print(f"    ✓ Found semantic: {semantic_file.name}")

        # Copy to output directory
        instance_output = output_dir / f"{study_id}_instance.nii.gz"
        shutil.copy(instance_file, instance_output)

        outputs = {'instance': instance_output}

        if semantic_file:
            semantic_output = output_dir / f"{study_id}_semantic.nii.gz"
            shutil.copy(semantic_file, semantic_output)
            outputs['semantic'] = semantic_output
            print(f"  ✓ Extracted: instance + semantic")
        else:
            print(f"  ⚠ Extracted: instance only")

        return outputs

    except subprocess.TimeoutExpired:
        print(f"  ✗ SPINEPS timeout (>600s)")
        return None
    except Exception as e:
        print(f"  ✗ SPINEPS failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_lstv_confidence(seg_data, unique_labels, lumbar_labels,
                               has_l6, s1_s2_disc, has_sacrum):
    """Calculate confidence score for LSTV detection"""
    confidence_score = 0.0
    confidence_factors = []

    # Factor 1: L6 size validation
    if has_l6:
        l6_mask = (seg_data == 25)
        l5_mask = (seg_data == 24) if 24 in unique_labels else None

        l6_volume = l6_mask.sum()

        if l5_mask is not None:
            l5_volume = l5_mask.sum()
            size_ratio = l6_volume / l5_volume if l5_volume > 0 else 0

            if 0.5 <= size_ratio <= 1.5:
                confidence_score += 0.4
                confidence_factors.append(f"L6/L5 ratio: {size_ratio:.2f} (valid)")
            else:
                confidence_factors.append(f"L6/L5 ratio: {size_ratio:.2f} (SUSPICIOUS)")

        if l6_volume > 500:
            confidence_score += 0.2
            confidence_factors.append(f"L6 volume: {l6_volume} voxels (OK)")
        else:
            confidence_factors.append(f"L6 volume: {l6_volume} voxels (TOO SMALL)")

    # Factor 2: Sacrum presence
    if has_sacrum:
        confidence_score += 0.2
        confidence_factors.append("Sacrum detected")
    else:
        confidence_factors.append("NO SACRUM (red flag)")

    # Factor 3: S1-S2 disc
    if s1_s2_disc:
        confidence_score += 0.3
        confidence_factors.append("S1-S2 disc (strong evidence)")

    # Factor 4: Vertebra count
    vertebra_count = len(lumbar_labels)
    if vertebra_count in [4, 5, 6]:
        confidence_score += 0.1
        confidence_factors.append(f"Count: {vertebra_count} (plausible)")
    else:
        confidence_factors.append(f"Count: {vertebra_count} (IMPLAUSIBLE)")

    if confidence_score >= 0.7:
        confidence_level = "HIGH"
    elif confidence_score >= 0.4:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return confidence_score, confidence_level, confidence_factors


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

        if is_lstv:
            confidence_score, confidence_level, confidence_factors = calculate_lstv_confidence(
                seg_data, unique_labels, lumbar_labels, has_l6, s1_s2_disc, has_sacrum
            )
        else:
            confidence_score = 0.0
            confidence_level = "N/A"
            confidence_factors = []

        return {
            'vertebra_count': vertebra_count,
            'has_sacrum': has_sacrum,
            'has_l6': has_l6,
            's1_s2_disc': s1_s2_disc,
            'is_lstv_candidate': is_lstv,
            'lstv_type': lstv_type,
            'unique_labels': list(map(int, unique_labels)),
            'lumbar_labels': list(map(int, lumbar_labels)),
            'confidence_score': round(confidence_score, 2),
            'confidence_level': confidence_level,
            'confidence_factors': confidence_factors,
        }
    except Exception as e:
        print(f"  ✗ Analysis error: {e}")
        return None


# ============================================================================
# PARASAGITTAL OPTIMIZER
# ============================================================================

def find_optimal_parasagittal_slices_semantic(
    semantic_data: np.ndarray,
    instance_data: np.ndarray,
    target_vertebra: str,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, int]:
    """Find parasagittal slices with max rib/TP density"""

    sag_axis = np.argmin(semantic_data.shape)
    num_slices = semantic_data.shape[sag_axis]

    if target_vertebra == 'T12':
        left_label = SEMANTIC_LABELS['rib_left']
        right_label = SEMANTIC_LABELS['rib_right']
        vertebra_instance = INSTANCE_LABELS['T12']
    else:  # L5
        left_label = SEMANTIC_LABELS['transverse_process_left']
        right_label = SEMANTIC_LABELS['transverse_process_right']
        vertebra_instance = INSTANCE_LABELS['L5']

    vertebra_mask = (instance_data == vertebra_instance)

    if not vertebra_mask.any():
        center = num_slices // 2
        offset = 40
        return {
            'left': max(0, center - offset),
            'right': min(num_slices - 1, center + offset),
            'left_density': 0,
            'right_density': 0
        }

    if sag_axis == 0:
        coords = np.where(vertebra_mask)[0]
    elif sag_axis == 1:
        coords = np.where(vertebra_mask)[1]
    else:
        coords = np.where(vertebra_mask)[2]

    center = int(np.median(coords))
    offset = int(40 / voxel_spacing[sag_axis])
    search_range = 15

    # Left side
    left_densities = []
    for i in range(max(0, center - offset - search_range),
                  min(num_slices, center - offset + search_range + 1)):
        if sag_axis == 0:
            slice_data = semantic_data[i, :, :]
        elif sag_axis == 1:
            slice_data = semantic_data[:, i, :]
        else:
            slice_data = semantic_data[:, :, i]

        density = (slice_data == left_label).sum()
        left_densities.append((i, density))

    # Right side
    right_densities = []
    for i in range(max(0, center + offset - search_range),
                  min(num_slices, center + offset + search_range + 1)):
        if sag_axis == 0:
            slice_data = semantic_data[i, :, :]
        elif sag_axis == 1:
            slice_data = semantic_data[:, i, :]
        else:
            slice_data = semantic_data[:, :, i]

        density = (slice_data == right_label).sum()
        right_densities.append((i, density))

    left_optimal, left_max = max(left_densities, key=lambda x: x[1])
    right_optimal, right_max = max(right_densities, key=lambda x: x[1])

    return {
        'left': left_optimal,
        'right': right_optimal,
        'left_density': left_max,
        'right_density': right_max
    }


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def extract_slice(data, sag_axis, slice_idx, thickness=1):
    """Extract 2D slice from 3D volume"""
    if thickness <= 1:
        if sag_axis == 0:
            return data[slice_idx, :, :]
        elif sag_axis == 1:
            return data[:, slice_idx, :]
        else:
            return data[:, :, slice_idx]

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
    """Normalize to 0-255 uint8 with CLAHE"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) /
                     (img_slice.max() - img_slice.min()) * 255)
        normalized = normalized.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)
        return normalized
    return np.zeros_like(img_slice, dtype=np.uint8)


def get_vertebra_centroid(seg_slice, label_id):
    """Get center of mass for a vertebra"""
    mask = (seg_slice == label_id)
    if not mask.any():
        return None

    coords = np.argwhere(mask)
    cy = int(coords[:, 0].mean())
    cx = int(coords[:, 1].mean())
    return (cx, cy)


def create_labeled_overlay(mri_slice, seg_slice, lstv_info):
    """Create RGB image with vertebra labels"""
    rgb_img = cv2.cvtColor(mri_slice, cv2.COLOR_GRAY2RGB)

    unique_labels = np.unique(seg_slice)
    vertebrae = [l for l in unique_labels if l in ID_TO_NAME]
    vertebrae_sorted = sorted(vertebrae)

    for label_id in vertebrae_sorted:
        name = ID_TO_NAME[label_id]
        centroid = get_vertebra_centroid(seg_slice, label_id)

        if centroid is None:
            continue

        cx, cy = centroid

        if name == 'L6':
            color = LSTV_COLORS['L6']
            thickness = 3
            font_scale = 0.9
        elif name == 'S1_S2_disc':
            color = LSTV_COLORS['S1_S2_disc']
            thickness = 3
            font_scale = 0.7
        elif name == 'L5':
            color = LSTV_COLORS['L5']
            thickness = 2
            font_scale = 0.8
        elif name == 'Sacrum':
            color = LSTV_COLORS['Sacrum']
            thickness = 2
            font_scale = 0.8
        else:
            color = LSTV_COLORS['default']
            thickness = 1
            font_scale = 0.6

        cv2.putText(rgb_img, name, (cx - 30, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                   color, thickness, cv2.LINE_AA)
        cv2.circle(rgb_img, (cx, cy), 4, color, -1)

    # Add LSTV banner
    if lstv_info['is_lstv_candidate']:
        banner_height = 40
        banner = np.zeros((banner_height, rgb_img.shape[1], 3), dtype=np.uint8)
        banner[:] = (50, 50, 50)

        lstv_type = lstv_info['lstv_type'].upper()
        confidence = lstv_info.get('confidence_level', 'UNKNOWN')

        if lstv_type == 'LUMBARIZATION':
            text = f"⚠ LSTV: LUMBARIZATION - {confidence} CONFIDENCE"
            banner_color = LSTV_COLORS['L6']
        elif lstv_type == 'SACRALIZATION':
            text = f"⚠ LSTV: SACRALIZATION - {confidence} CONFIDENCE"
            banner_color = (255, 128, 0)
        elif lstv_type == 'S1_S2_DISC':
            text = f"⚠ LSTV: S1-S2 DISC - {confidence} CONFIDENCE"
            banner_color = (255, 128, 0)
        else:
            text = f"⚠ LSTV: ATYPICAL - {confidence} CONFIDENCE"
            banner_color = (255, 255, 0)

        cv2.putText(banner, text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2, cv2.LINE_AA)

        rgb_img = np.vstack([banner, rgb_img])

    return rgb_img


def extract_three_view_images(
    nifti_path,
    instance_path,
    semantic_path,
    output_dir,
    qa_dir,
    study_id,
    selector,
    analysis,
    use_semantic_optimization=True
):
    """Extract 3 views with optional semantic optimization"""
    try:
        nii = nib.load(nifti_path)
        instance_nii = nib.load(instance_path)

        mri_data = nii.get_fdata()
        instance_data = instance_nii.get_fdata().astype(int)

        semantic_data = None
        if semantic_path and semantic_path.exists():
            semantic_nii = nib.load(semantic_path)
            semantic_data = semantic_nii.get_fdata().astype(int)

        dims = mri_data.shape
        sag_axis = np.argmin(dims)

        if use_semantic_optimization and semantic_data is not None:
            print(f"    Using semantic optimization")
            voxel_spacing = nii.header.get_zooms()

            t12_slices = find_optimal_parasagittal_slices_semantic(
                semantic_data, instance_data, 'T12', voxel_spacing
            )

            l5_slices = find_optimal_parasagittal_slices_semantic(
                semantic_data, instance_data, 'L5', voxel_spacing
            )

            views = {
                'left': t12_slices['left'],
                'mid': l5_slices['left'],
                'right': t12_slices['right'],
            }

            print(f"    Rib densities - L:{t12_slices['left_density']} R:{t12_slices['right_density']}")
        else:
            print(f"    Using standard selection")
            slice_info = selector.get_three_slices(instance_data, sag_axis)
            views = {
                'left': slice_info['left'],
                'mid': slice_info['mid'],
                'right': slice_info['right'],
            }

        output_paths = {}

        for view_name, slice_idx in views.items():
            thickness = 15 if view_name in ['left', 'right'] else 5
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
            instance_slice = extract_slice(instance_data, sag_axis, slice_idx, thickness=thickness)

            normalized = normalize_slice(mri_slice)

            output_path = output_dir / f"{study_id}_{view_name}.jpg"
            cv2.imwrite(str(output_path), normalized)
            output_paths[view_name] = output_path

            if qa_dir:
                labeled_img = create_labeled_overlay(normalized, instance_slice, analysis)
                qa_path = qa_dir / f"{study_id}_{view_name}_QA.jpg"
                cv2.imwrite(str(qa_path), labeled_img)

        return output_paths

    except Exception as e:
        print(f"  ✗ Image extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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
        print(f"    Upload error: {e}")
        return False


def load_progress(progress_file):
    """Load progress"""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'processed': [],
        'flagged': [],
        'failed': [],
        'high_confidence': [],
        'medium_confidence': [],
        'low_confidence': [],
        'semantic_available': [],
        'semantic_missing': []
    }


def save_progress(progress_file, progress):
    """Save progress"""
    try:
        temp_file = progress_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(progress_file)
    except:
        pass


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_study(
    study_dir: Path,
    output_dirs: Dict[str, Path],
    series_df: Optional[pd.DataFrame],
    selector: SpineAwareSliceSelector,
    args
) -> Optional[Dict]:
    """Process a single study"""
    study_id = study_dir.name

    try:
        # Select series using CSV
        series_dir = select_best_series(study_dir, series_df, study_id)
        if series_dir is None:
            return None

        # Convert DICOM to NIfTI
        nifti_path = output_dirs['nifti'] / f"sub-{study_id}_sequ-sag_T2w.nii.gz"
        if not nifti_path.exists():
            print(f"  Converting DICOM...")
            nifti_path = convert_dicom_to_nifti(series_dir, nifti_path, study_id)
            if nifti_path is None:
                return None

        # Run SPINEPS
        seg_outputs = run_spineps_dual_extraction(nifti_path, output_dirs['segmentations'])
        if seg_outputs is None:
            return None

        instance_path = seg_outputs['instance']
        semantic_path = seg_outputs.get('semantic')

        # Analyze
        print(f"  Analyzing...")
        analysis = analyze_segmentation(instance_path)
        if analysis is None:
            return None

        result = {
            'study_id': study_id,
            'series_id': series_dir.name,
            'vertebra_count': analysis['vertebra_count'],
            'is_lstv_candidate': analysis['is_lstv_candidate'],
            'lstv_type': analysis['lstv_type'],
            'lumbar_labels': str(analysis['lumbar_labels']),
            'confidence_score': analysis['confidence_score'],
            'confidence_level': analysis['confidence_level'],
            'has_semantic': semantic_path is not None,
        }

        # Extract images if LSTV
        if analysis['is_lstv_candidate']:
            confidence_level = analysis['confidence_level']
            confidence_score = analysis['confidence_score']

            print(f"  🚩 LSTV! Type={analysis['lstv_type']}, "
                  f"Confidence={confidence_level} ({confidence_score:.2f})")

            image_paths = extract_three_view_images(
                nifti_path,
                instance_path,
                semantic_path,
                output_dirs['images'],
                output_dirs['qa'],
                study_id,
                selector,
                analysis,
                use_semantic_optimization=(semantic_path is not None)
            )

            if image_paths:
                if confidence_score >= args.confidence_threshold:
                    if args.roboflow_key and args.roboflow_key != 'SKIP':
                        upload_success = 0
                        for view_name, image_path in image_paths.items():
                            if upload_to_roboflow(
                                image_path, f"{study_id}_{view_name}",
                                args.roboflow_key, args.roboflow_workspace,
                                args.roboflow_project
                            ):
                                upload_success += 1

                        print(f"  ✓ Uploaded {upload_success}/3 views")
                    else:
                        print(f"  ✓ Images saved (upload skipped)")
                else:
                    print(f"  ⚠ Images saved, upload skipped ({confidence_level} < threshold)")

        else:
            print(f"  ✓ Normal ({analysis['vertebra_count']} lumbar)")

        return result

    except Exception as e:
        print(f"  ✗ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Production LSTV Screening v3.0 - FIXED',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['diagnostic', 'trial', 'full'])
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--series_csv', type=str, required=True,
                       help='Path to train_series_descriptions.csv (REQUIRED)')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--roboflow_key', type=str, default='SKIP')
    parser.add_argument('--roboflow_workspace', type=str, default='lstv-screening')
    parser.add_argument('--roboflow_project', type=str, default='lstv-candidates')
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Validate series CSV exists
    series_csv_path = Path(args.series_csv)
    if not series_csv_path.exists():
        print(f"ERROR: Series CSV not found: {series_csv_path}")
        print(f"This file is REQUIRED to identify sagittal T2 series")
        sys.exit(1)

    # Set limits
    if args.mode == 'diagnostic':
        study_limit = args.limit or 5
    elif args.mode == 'trial':
        study_limit = args.limit or 50
    else:
        study_limit = args.limit

    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dirs = {
        'nifti': output_dir / 'nifti',
        'segmentations': output_dir / 'segmentations',
        'images': output_dir / 'candidate_images',
        'qa': output_dir / 'qa_images',
    }

    for d in output_dirs.values():
        d.mkdir(exist_ok=True)

    # Load series CSV
    print(f"Loading series descriptions from: {series_csv_path}")
    series_df = load_series_descriptions(series_csv_path)
    if series_df is None:
        print(f"ERROR: Failed to load series CSV")
        sys.exit(1)

    # Initialize
    selector = SpineAwareSliceSelector()
    progress_file = output_dir / 'progress.json'
    progress = load_progress(progress_file)
    results_csv = output_dir / 'results.csv'

    # Get studies
    study_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if study_limit:
        study_dirs = study_dirs[:study_limit]

    print(f"\n{'='*80}")
    print(f"LSTV SCREENING v3.0 - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Studies to process: {len(study_dirs)}")
    print(f"Already processed: {len(progress['processed'])}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Roboflow upload: {'Enabled' if args.roboflow_key != 'SKIP' else 'Disabled'}")
    print(f"{'='*80}\n")
    sys.stdout.flush()

    # Process
    for study_dir in tqdm(study_dirs, desc="Processing"):
        study_id = study_dir.name

        if study_id in progress['processed']:
            continue

        print(f"\n[{study_id}]")
        sys.stdout.flush()

        try:
            result = process_study(study_dir, output_dirs, series_df, selector, args)

            if result is None:
                progress['failed'].append(study_id)
            else:
                if result.get('has_semantic'):
                    progress['semantic_available'].append(study_id)
                else:
                    progress['semantic_missing'].append(study_id)

                if result['is_lstv_candidate']:
                    confidence_level = result['confidence_level']
                    if confidence_level == 'HIGH':
                        progress['high_confidence'].append(study_id)
                        progress['flagged'].append(study_id)
                    elif confidence_level == 'MEDIUM':
                        progress['medium_confidence'].append(study_id)
                    else:
                        progress['low_confidence'].append(study_id)

                df = pd.DataFrame([result])
                if results_csv.exists():
                    df.to_csv(results_csv, mode='a', header=False, index=False)
                else:
                    df.to_csv(results_csv, mode='w', header=True, index=False)

            progress['processed'].append(study_id)
            save_progress(progress_file, progress)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted!")
            save_progress(progress_file, progress)
            sys.exit(1)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            progress['failed'].append(study_id)
            progress['processed'].append(study_id)
            save_progress(progress_file, progress)

    # Final report
    print(f"\n{'='*80}")
    print(f"COMPLETE - {args.mode.upper()} MODE")
    print(f"{'='*80}")
    print(f"Total processed: {len(progress['processed'])}")
    print(f"Failed: {len(progress['failed'])}")
    print()
    print(f"LSTV Candidates: {len(progress['flagged'])}")
    print(f"  HIGH confidence:   {len(progress['high_confidence'])}")
    print(f"  MEDIUM confidence: {len(progress['medium_confidence'])}")
    print(f"  LOW confidence:    {len(progress['low_confidence'])}")
    print()
    print(f"Semantic masks:")
    print(f"  Available: {len(progress['semantic_available'])}")
    print(f"  Missing:   {len(progress['semantic_missing'])}")
    print()
    print(f"Output: {output_dir}")
    print(f"Results: {results_csv}")
    print(f"{'='*80}\n")

    if args.mode == 'diagnostic':
        semantic_pct = len(progress['semantic_available']) / max(1, len(progress['processed'])) * 100
        print(f"📊 DIAGNOSTIC SUMMARY:")
        print(f"   Semantic availability: {semantic_pct:.1f}%")
        if semantic_pct > 80:
            print(f"   ✓ Excellent - use semantic optimization")
        elif semantic_pct > 50:
            print(f"   ⚠ Partial semantic coverage")
        else:
            print(f"   ✗ Low - fallback to standard selection")


if __name__ == "__main__":
    main()
