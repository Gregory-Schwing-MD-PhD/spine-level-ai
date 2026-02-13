#!/usr/bin/env python3
"""
PRODUCTION LSTV SCREENING SYSTEM v3.0 - COMPLETE IMPLEMENTATION

Merges robust SPINEPS execution with advanced rib/TP detection logic.
Combines the best of lstv_screen.py (robust execution) and 
lstv_screen_enhanced.py (confidence scoring + QA).

KEY FEATURES:
- Extracts BOTH instance AND semantic SPINEPS outputs
- Detects ribs from semantic labels (when available)
- Optimizes parasagittal slices using rib density
- Confidence scoring (HIGH/MEDIUM/LOW)
- QA images with vertebra labels
- Smart Roboflow upload filtering
- Three operational modes: diagnostic, trial, full

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
# HELPER FUNCTIONS
# ============================================================================

def load_series_descriptions(csv_path):
    """Load series descriptions CSV"""
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
    except Exception as e:
        print(f"  ✗ DICOM conversion error: {e}")
        return None


# ============================================================================
# SPINEPS DUAL EXTRACTOR (FIXED)
# ============================================================================

def run_spineps_dual_extraction(nifti_path: Path, output_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Run SPINEPS and extract BOTH instance and semantic outputs
    
    Uses the wrapper script to properly set environment variables.
    
    Returns:
        {
            'instance': Path,
            'semantic': Path or None
        }
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find wrapper script dynamically
        script_dir = Path(__file__).parent
        wrapper_path = script_dir / 'spineps_wrapper.sh'
        
        # Fallback to hardcoded path if not found locally
        if not wrapper_path.exists():
            wrapper_path = Path('/work/src/screening/spineps_wrapper.sh')
        
        if not wrapper_path.exists():
            print(f"  ✗ Wrapper not found at {wrapper_path}")
            return None

        cmd = [
            'bash', str(wrapper_path), 'sample',
            '-i', str(nifti_path),
            '-model_semantic', 't2w',
            '-model_instance', 'instance',
            '-model_labeling', 't2w_labeling',
            '-override_semantic', '-override_instance', '-override_ctd'
        ]

        print(f"    Running SPINEPS...")
        sys.stdout.flush()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  ✗ SPINEPS failed with return code {result.returncode}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return None

        # Find derivatives directory
        input_parent = nifti_path.parent
        derivatives_dir = input_parent / "derivatives_seg"
        if not derivatives_dir.exists():
            print(f"  ✗ derivatives_seg directory not found")
            return None

        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')

        # Find instance mask (vertebrae)
        instance_pattern = f"*{study_id}*_seg-vert_msk.nii.gz"
        instance_files = list(derivatives_dir.glob(instance_pattern))
        
        if not instance_files:
            # Try alternative pattern
            instance_files = list(derivatives_dir.glob("*_seg-vert_msk.nii.gz"))
        
        if not instance_files:
            print(f"  ✗ Instance mask not found")
            return None

        instance_file = instance_files[0]

        # Find semantic mask (ribs, TPs, etc)
        semantic_pattern = f"*{study_id}*_seg-spine_msk.nii.gz"
        semantic_files = list(derivatives_dir.glob(semantic_pattern))
        
        if not semantic_files:
            # Try alternative pattern
            semantic_files = list(derivatives_dir.glob("*_seg-spine_msk.nii.gz"))

        semantic_file = semantic_files[0] if semantic_files else None

        # Copy outputs to final location
        instance_output = output_dir / f"{study_id}_instance.nii.gz"
        shutil.copy(instance_file, instance_output)

        outputs = {'instance': instance_output}

        if semantic_file:
            semantic_output = output_dir / f"{study_id}_semantic.nii.gz"
            shutil.copy(semantic_file, semantic_output)
            outputs['semantic'] = semantic_output
            print(f"  ✓ Saved: instance + semantic masks")
        else:
            print(f"  ⚠ Saved: instance only (semantic not available)")

        return outputs

    except subprocess.TimeoutExpired:
        print(f"  ✗ SPINEPS timeout (>600s)")
        return None
    except Exception as e:
        print(f"  ✗ SPINEPS failed: {e}")
        return None


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

def calculate_lstv_confidence(seg_data, unique_labels, lumbar_labels,
                               has_l6, s1_s2_disc, has_sacrum):
    """
    Calculate confidence score for LSTV detection

    Returns:
        confidence_score (float): 0.0-1.0
        confidence_level (str): LOW/MEDIUM/HIGH
        confidence_factors (list): Human-readable factors
    """
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

            # L6 should be similar size to L5 (0.5-1.5x)
            if 0.5 <= size_ratio <= 1.5:
                confidence_score += 0.4
                confidence_factors.append(f"L6/L5 ratio: {size_ratio:.2f} (valid)")
            else:
                confidence_factors.append(f"L6/L5 ratio: {size_ratio:.2f} (SUSPICIOUS)")

        # Minimum absolute size check
        if l6_volume > 500:
            confidence_score += 0.2
            confidence_factors.append(f"L6 volume: {l6_volume} voxels (OK)")
        else:
            confidence_factors.append(f"L6 volume: {l6_volume} voxels (TOO SMALL)")

    # Factor 2: Sacrum must be present
    if has_sacrum:
        confidence_score += 0.2
        confidence_factors.append("Sacrum detected")
    else:
        confidence_factors.append("NO SACRUM (red flag)")

    # Factor 3: S1-S2 disc is strong sacralization indicator
    if s1_s2_disc:
        confidence_score += 0.3
        confidence_factors.append("S1-S2 disc (strong evidence)")

    # Factor 4: Vertebra count consistency
    vertebra_count = len(lumbar_labels)
    if vertebra_count in [4, 5, 6]:
        confidence_score += 0.1
        confidence_factors.append(f"Count: {vertebra_count} (plausible)")
    else:
        confidence_factors.append(f"Count: {vertebra_count} (IMPLAUSIBLE)")

    # Determine confidence level
    if confidence_score >= 0.7:
        confidence_level = "HIGH"
    elif confidence_score >= 0.4:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return confidence_score, confidence_level, confidence_factors


def analyze_segmentation(seg_path):
    """Analyze segmentation for LSTV candidates with confidence scoring"""
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

        # Calculate confidence
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
# PARASAGITTAL OPTIMIZER (RIB-DENSITY BASED)
# ============================================================================

def find_optimal_parasagittal_slices_semantic(
    semantic_data: np.ndarray,
    instance_data: np.ndarray,
    target_vertebra: str,  # 'T12' or 'L5'
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, int]:
    """
    Find parasagittal slices with MAXIMUM rib/TP density using semantic labels

    This is MUCH better than generic "vertebra density" optimization!
    """

    sag_axis = np.argmin(semantic_data.shape)
    num_slices = semantic_data.shape[sag_axis]

    # Get target labels
    if target_vertebra == 'T12':
        left_label = SEMANTIC_LABELS['rib_left']
        right_label = SEMANTIC_LABELS['rib_right']
        vertebra_instance = INSTANCE_LABELS['T12']
    else:  # L5
        left_label = SEMANTIC_LABELS['transverse_process_left']
        right_label = SEMANTIC_LABELS['transverse_process_right']
        vertebra_instance = INSTANCE_LABELS['L5']

    # Find vertebra center from instance labels
    vertebra_mask = (instance_data == vertebra_instance)

    if not vertebra_mask.any():
        # Fallback to geometric center
        center = num_slices // 2
        offset = 40
        return {
            'left': max(0, center - offset), 
            'right': min(num_slices - 1, center + offset),
            'left_density': 0,
            'right_density': 0
        }

    # Get vertebra center
    if sag_axis == 0:
        coords = np.where(vertebra_mask)[0]
    elif sag_axis == 1:
        coords = np.where(vertebra_mask)[1]
    else:
        coords = np.where(vertebra_mask)[2]

    center = int(np.median(coords))
    offset = int(40 / voxel_spacing[sag_axis])

    # OPTIMIZATION: Search for slices with MAX rib/TP density
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
    """Create RGB image with vertebra labels overlaid"""
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

        # Choose color based on LSTV significance
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

        # Draw text label
        cv2.putText(rgb_img, name, (cx - 30, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                   color, thickness, cv2.LINE_AA)

        cv2.circle(rgb_img, (cx, cy), 4, color, -1)

    # Add LSTV banner if applicable
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
    """
    Extract 3 views with optional semantic-based optimization and QA labels
    
    Args:
        use_semantic_optimization: If True and semantic data available, 
                                   optimize parasagittal slices for rib density
    """
    try:
        nii = nib.load(nifti_path)
        instance_nii = nib.load(instance_path)

        mri_data = nii.get_fdata()
        instance_data = instance_nii.get_fdata().astype(int)

        # Try to load semantic if available
        semantic_data = None
        if semantic_path and semantic_path.exists():
            semantic_nii = nib.load(semantic_path)
            semantic_data = semantic_nii.get_fdata().astype(int)

        dims = mri_data.shape
        sag_axis = np.argmin(dims)
        
        # Determine slice selection method
        if use_semantic_optimization and semantic_data is not None:
            # NEW: Use rib-density optimization
            print(f"    Using semantic rib-density optimization")
            voxel_spacing = nii.header.get_zooms()
            
            # Get T12-optimized slices (for ribs)
            t12_slices = find_optimal_parasagittal_slices_semantic(
                semantic_data, instance_data, 'T12', voxel_spacing
            )
            
            # Get L5-optimized slice (for TPs)
            l5_slices = find_optimal_parasagittal_slices_semantic(
                semantic_data, instance_data, 'L5', voxel_spacing
            )
            
            views = {
                'left': t12_slices['left'],
                'mid': l5_slices['left'],  # Use L5 center for midline
                'right': t12_slices['right'],
            }
            
            print(f"    Rib densities - L:{t12_slices['left_density']} R:{t12_slices['right_density']}")
        else:
            # FALLBACK: Use standard spine-aware selection
            print(f"    Using standard spine-aware selection")
            slice_info = selector.get_three_slices(instance_data, sag_axis)
            views = {
                'left': slice_info['left'],
                'mid': slice_info['mid'],
                'right': slice_info['right'],
            }

        output_paths = {}

        for view_name, slice_idx in views.items():
            # Use thick slab MIP for parasagittal, thin for midline
            thickness = 15 if view_name in ['left', 'right'] else 5
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
            instance_slice = extract_slice(instance_data, sag_axis, slice_idx, thickness=thickness)

            normalized = normalize_slice(mri_slice)

            # Save standard image (for Roboflow)
            output_path = output_dir / f"{study_id}_{view_name}.jpg"
            cv2.imwrite(str(output_path), normalized)
            output_paths[view_name] = output_path

            # Save QA image with labels (for manual review)
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
    except Exception as e:
        print(f"    Upload error: {e}")
        return False


def load_progress(progress_file):
    """Load progress from JSON file"""
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
    """Save progress to JSON file"""
    try:
        temp_file = progress_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(progress_file)
    except:
        pass


# ============================================================================
# MAIN SCREENING LOGIC
# ============================================================================

def process_study(
    study_dir: Path,
    output_dirs: Dict[str, Path],
    series_df: Optional[pd.DataFrame],
    selector: SpineAwareSliceSelector,
    args
) -> Optional[Dict]:
    """
    Process a single study - COMPLETE IMPLEMENTATION
    
    Returns result dict or None on failure
    """
    study_id = study_dir.name

    try:
        # 1. Select best series
        series_dir = select_best_series(study_dir, series_df, study_id)
        if series_dir is None:
            print(f"  ✗ No series found")
            return None

        print(f"  Series: {series_dir.name}")

        # 2. Convert DICOM to NIfTI
        nifti_path = output_dirs['nifti'] / f"sub-{study_id}_T2w.nii.gz"
        if not nifti_path.exists():
            print(f"  Converting DICOM...")
            nifti_path = convert_dicom_to_nifti(series_dir, nifti_path)
            if nifti_path is None:
                return None

        # 3. Run SPINEPS dual extraction
        seg_outputs = run_spineps_dual_extraction(nifti_path, output_dirs['segmentations'])
        if seg_outputs is None:
            return None

        instance_path = seg_outputs['instance']
        semantic_path = seg_outputs.get('semantic')

        # 4. Analyze instance segmentation
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

        # 5. If LSTV candidate, extract images
        if analysis['is_lstv_candidate']:
            confidence_level = analysis['confidence_level']
            confidence_score = analysis['confidence_score']

            print(f"  🚩 LSTV! Type={analysis['lstv_type']}, "
                  f"Confidence={confidence_level} ({confidence_score:.2f})")

            # Extract 3-view images
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
                # Smart upload based on confidence threshold
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
        description='Production LSTV Screening v3.0 - COMPLETE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OPERATIONAL MODES:
  diagnostic  → Test on 5 studies to check semantic label availability
  trial       → Validate on 50 studies before full run
  full        → Process all studies (~2700 → ~500 LSTV candidates)

FEATURES:
  ✓ Dual SPINEPS extraction (instance + semantic)
  ✓ Rib-density parasagittal optimization (when semantic available)
  ✓ Confidence scoring (HIGH/MEDIUM/LOW)
  ✓ QA images with vertebra labels
  ✓ Smart Roboflow filtering

EXAMPLES:
  # Diagnostic mode (test semantic availability)
  python lstv_screen_production_COMPLETE.py --mode diagnostic \\
    --input_dir /data/dicom --output_dir /out \\
    --roboflow_key SKIP

  # Trial mode (50 studies)
  python lstv_screen_production_COMPLETE.py --mode trial \\
    --input_dir /data/dicom --output_dir /out --limit 50 \\
    --roboflow_key YOUR_KEY --confidence_threshold 0.7

  # Full production run
  python lstv_screen_production_COMPLETE.py --mode full \\
    --input_dir /data/dicom --output_dir /out \\
    --roboflow_key YOUR_KEY --confidence_threshold 0.7
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                       choices=['diagnostic', 'trial', 'full'],
                       help='Operation mode')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input DICOM directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--series_csv', type=str, default=None,
                       help='Series descriptions CSV')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of studies')
    parser.add_argument('--roboflow_key', type=str, default='SKIP',
                       help='Roboflow API key (use SKIP to disable upload)')
    parser.add_argument('--roboflow_workspace', type=str, default='lstv-screening')
    parser.add_argument('--roboflow_project', type=str, default='lstv-candidates')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='Minimum confidence for auto-upload (0.0-1.0)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Set study limits by mode
    if args.mode == 'diagnostic':
        study_limit = args.limit or 5
    elif args.mode == 'trial':
        study_limit = args.limit or 50
    else:  # full
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

    # Load series descriptions if available
    series_df = None
    if args.series_csv:
        series_csv = Path(args.series_csv)
        if series_csv.exists():
            series_df = load_series_descriptions(series_csv)
            if series_df is not None:
                print(f"✓ Loaded series descriptions: {len(series_df)} entries")

    # Initialize selector
    selector = SpineAwareSliceSelector()

    # Progress tracking
    progress_file = output_dir / 'progress.json'
    progress = load_progress(progress_file)
    results_csv = output_dir / 'results.csv'

    # Get study list
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

    # Process studies
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
                # Track semantic availability
                if result.get('has_semantic'):
                    progress['semantic_available'].append(study_id)
                else:
                    progress['semantic_missing'].append(study_id)

                # Track confidence levels
                if result['is_lstv_candidate']:
                    confidence_level = result['confidence_level']
                    if confidence_level == 'HIGH':
                        progress['high_confidence'].append(study_id)
                        progress['flagged'].append(study_id)
                    elif confidence_level == 'MEDIUM':
                        progress['medium_confidence'].append(study_id)
                    else:
                        progress['low_confidence'].append(study_id)

                # Save result to CSV
                df = pd.DataFrame([result])
                if results_csv.exists():
                    df.to_csv(results_csv, mode='a', header=False, index=False)
                else:
                    df.to_csv(results_csv, mode='w', header=True, index=False)

            progress['processed'].append(study_id)
            save_progress(progress_file, progress)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user!")
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
    print(f"  HIGH confidence:   {len(progress['high_confidence'])} → Uploaded")
    print(f"  MEDIUM confidence: {len(progress['medium_confidence'])} → Manual review")
    print(f"  LOW confidence:    {len(progress['low_confidence'])} → Flagged only")
    print()
    print(f"Semantic masks:")
    print(f"  Available: {len(progress['semantic_available'])}")
    print(f"  Missing:   {len(progress['semantic_missing'])}")
    print()
    print(f"Output directory: {output_dir}")
    print(f"QA images: {output_dirs['qa']}")
    print(f"Results CSV: {results_csv}")
    print(f"{'='*80}\n")

    if args.mode == 'diagnostic':
        semantic_pct = len(progress['semantic_available']) / max(1, len(progress['processed'])) * 100
        print(f"📊 DIAGNOSTIC SUMMARY:")
        print(f"   Semantic availability: {semantic_pct:.1f}%")
        if semantic_pct > 80:
            print(f"   ✓ Excellent! Proceed with semantic-based optimization")
        elif semantic_pct > 50:
            print(f"   ⚠ Partial - semantic optimization will work for some cases")
        else:
            print(f"   ✗ Low - consider intensity-based fallback for ribs/TPs")


if __name__ == "__main__":
    main()
