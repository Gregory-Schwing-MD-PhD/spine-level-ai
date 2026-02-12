#!/usr/bin/env python3
"""
PRODUCTION WEAK LABEL GENERATION v6.1
Intensity-Based Detection + Comprehensive Validation + TUNED PARAMETERS

NEW IN v6.1 (PARAMETER TUNING):
- Relaxed intensity-based detection constraints (v6.0 was too strict - 0% success)
- Multi-scale edge detection (catches more subtle structures)
- Multi-threshold intensity detection (handles variable contrast)
- Wider search regions for ribs and TPs
- Smaller minimum size thresholds
- Single-sided TP detection allowed
- Expected improvement: 0% → 40-70% intensity-based success

Features:
- Thick Slab MIP for enhanced visibility
- Spine-aware intelligent slice selection
- IMPROVED Intensity-based T12 rib detection (v6.1 tuned)
- IMPROVED Intensity-based L5 TP detection (v6.1 tuned)
- Comprehensive quality metrics and validation
- SPINEPS label quality assessment
- Method comparison visualizations

Version 6.1 - PRODUCTION GRADE WITH TUNED INTENSITY-BASED DETECTION
"""

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2
from scipy import ndimage
from scipy.ndimage import label as scipy_label
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

# ============================================================================
# LABEL DEFINITIONS
# ============================================================================

SPINEPS_LABELS = {
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

YOLO_CLASSES = {
    0: 't12_vertebra',
    1: 't12_rib',
    2: 'l5_vertebra',
    3: 'l5_transverse_process',
    4: 'sacrum',
    5: 'l4_vertebra',
    6: 'l5_s1_disc',
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SliceMetrics:
    """Comprehensive metrics for slice selection quality"""
    study_id: str
    geometric_mid: int
    spine_aware_mid: int
    offset_voxels: int
    offset_mm: float
    spine_density_geometric: float
    spine_density_optimal: float
    improvement_ratio: float
    method: str

    def to_dict(self):
        return asdict(self)

@dataclass
class DetectionMetrics:
    """Metrics comparing detection methods"""
    study_id: str
    view: str
    structure: str
    label_based_success: bool
    intensity_based_success: bool
    method_used: str
    
    def to_dict(self):
        return asdict(self)

# ============================================================================
# SPINE-AWARE SLICE SELECTOR
# ============================================================================

class SpineAwareSliceSelector:
    """Intelligent slice selection using spine segmentation with comprehensive metrics"""

    def __init__(self, voxel_spacing_mm=1.0, parasagittal_offset_mm=30):
        self.voxel_spacing_mm = voxel_spacing_mm
        self.parasagittal_offset_mm = parasagittal_offset_mm
        self.metrics_log = []

    def find_sagittal_axis(self, data_shape):
        """Determine sagittal axis (smallest dimension)"""
        return np.argmin(data_shape)

    def calculate_spine_density(self, seg_data, sag_axis, slice_idx):
        """Calculate spine content in a slice"""
        lumbar_labels = [SPINEPS_LABELS[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'Sacrum']]

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

    def find_optimal_midline(self, seg_data, sag_axis, study_id=None):
        """Find TRUE spinal midline using segmentation"""
        num_slices = seg_data.shape[sag_axis]
        geometric_mid = num_slices // 2

        lumbar_labels = [SPINEPS_LABELS[k] for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'Sacrum']]

        vertebra_mask = np.zeros_like(seg_data, dtype=bool)
        for label in lumbar_labels:
            if label in seg_data:
                vertebra_mask |= (seg_data == label)

        if not vertebra_mask.any():
            metrics = SliceMetrics(
                study_id=study_id or "unknown",
                geometric_mid=geometric_mid,
                spine_aware_mid=geometric_mid,
                offset_voxels=0,
                offset_mm=0.0,
                spine_density_geometric=0.0,
                spine_density_optimal=0.0,
                improvement_ratio=1.0,
                method='geometric_fallback'
            )
            self.metrics_log.append(metrics)
            return geometric_mid, geometric_mid, metrics

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

        offset_voxels = abs(optimal_mid - geometric_mid)
        offset_mm = offset_voxels * self.voxel_spacing_mm

        density_geometric = spine_density[geometric_mid]
        density_optimal = spine_density[optimal_mid]

        improvement_ratio = density_optimal / density_geometric if density_geometric > 0 else 1.0

        metrics = SliceMetrics(
            study_id=study_id or "unknown",
            geometric_mid=geometric_mid,
            spine_aware_mid=optimal_mid,
            offset_voxels=offset_voxels,
            offset_mm=offset_mm,
            spine_density_geometric=float(density_geometric),
            spine_density_optimal=float(density_optimal),
            improvement_ratio=float(improvement_ratio),
            method='spine_aware'
        )

        self.metrics_log.append(metrics)

        return geometric_mid, optimal_mid, metrics

    def get_three_slices(self, seg_data, sag_axis, study_id=None):
        """Get left, mid, right slice indices with metrics"""
        geometric_mid, optimal_mid, metrics = self.find_optimal_midline(seg_data, sag_axis, study_id)

        num_slices = seg_data.shape[sag_axis]
        offset_voxels = int(self.parasagittal_offset_mm / self.voxel_spacing_mm)

        left_idx = max(0, optimal_mid - offset_voxels)
        right_idx = min(num_slices - 1, optimal_mid + offset_voxels)

        return {
            'left': left_idx,
            'mid': optimal_mid,
            'right': right_idx,
            'geometric_mid': geometric_mid,
            'metrics': metrics,
            'sag_axis': sag_axis,
        }

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def extract_slice(data, sag_axis, slice_idx, thickness=1):
    """Extract 2D slice or Thick Slab MIP from 3D volume"""
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
    """Normalize to 0-255 uint8 with CLAHE enhancement"""
    if img_slice.max() > img_slice.min():
        normalized = ((img_slice - img_slice.min()) /
                     (img_slice.max() - img_slice.min()) * 255)
        normalized = normalized.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(normalized)

        return normalized
    return np.zeros_like(img_slice, dtype=np.uint8)


def extract_bounding_box(mask, label_id):
    """Extract YOLO format bounding box"""
    label_mask = (mask == label_id)

    if not label_mask.any():
        return None

    coords = np.argwhere(label_mask)
    if len(coords) == 0:
        return None

    y_coords = coords[:, 0]
    x_coords = coords[:, 1]

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    height, width = mask.shape

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    if box_width <= 0 or box_height <= 0 or box_width > 1 or box_height > 1:
        return None

    return [x_center, y_center, box_width, box_height]

# ============================================================================
# LABEL-BASED DETECTION (Original SPINEPS-based)
# ============================================================================

def detect_t12_rib_label_based(seg_slice, vertebra_label, side='left'):
    """ORIGINAL: T12 rib detection using SPINEPS labels"""
    vert_mask = (seg_slice == vertebra_label)

    if not vert_mask.any():
        return None

    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()

    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min

    height, width = seg_slice.shape

    if side == 'left':
        search_x_min = max(0, int(vert_x_min - vert_width * 0.8))
        search_x_max = int(vert_x_min)
    else:
        search_x_min = int(vert_x_max)
        search_x_max = min(width, int(vert_x_max + vert_width * 0.8))

    search_y_min = max(0, int(vert_y_min - vert_height * 0.5))
    search_y_max = min(height, int(vert_y_max + vert_height * 0.2))

    rib_candidates = np.zeros_like(seg_slice)
    for y in range(search_y_min, search_y_max):
        for x in range(search_x_min, search_x_max):
            if seg_slice[y, x] > 0 and seg_slice[y, x] != vertebra_label:
                rib_candidates[y, x] = seg_slice[y, x]

    if not rib_candidates.any():
        return None

    labeled, num_features = scipy_label(rib_candidates > 0)

    if num_features == 0:
        return None

    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]

    min_size = (vert_width * vert_height) * 0.15
    max_size = (search_y_max - search_y_min) * (search_x_max - search_x_min) * 0.7

    valid_components = [
        i + 1 for i, size in enumerate(component_sizes)
        if min_size <= size <= max_size
    ]

    if not valid_components:
        return None

    largest_comp = max(valid_components, key=lambda c: component_sizes[c - 1])
    rib_mask = (labeled == largest_comp)

    if not rib_mask.any():
        return None

    return extract_bounding_box(rib_mask.astype(int), 1)


def detect_l5_tp_label_based(seg_slice, vertebra_label):
    """ORIGINAL: L5 TP detection using SPINEPS labels"""
    vert_mask = (seg_slice == vertebra_label)

    if not vert_mask.any():
        return None

    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()

    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min

    height, width = seg_slice.shape

    central_width = vert_width * 0.25

    tp_candidates = np.zeros_like(seg_slice)

    for y in range(int(vert_y_min - vert_height * 0.1),
                   int(vert_y_max + vert_height * 0.1)):
        if y < 0 or y >= height:
            continue
        for x in range(width):
            if seg_slice[y, x] > 0 and seg_slice[y, x] != vertebra_label:
                dist_from_center_x = abs(x - cx)
                if dist_from_center_x > central_width:
                    tp_candidates[y, x] = seg_slice[y, x]

    if not tp_candidates.any():
        return None

    labeled, num_features = scipy_label(tp_candidates > 0)

    if num_features < 2:
        return None

    component_info = []
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled == comp_id)
        comp_size = comp_mask.sum()
        comp_coords = np.argwhere(comp_mask)

        if len(comp_coords) == 0:
            continue

        comp_y_mean = comp_coords[:, 0].mean()
        comp_x_mean = comp_coords[:, 1].mean()

        if comp_size < (vert_width * vert_height * 0.1):
            continue

        if abs(comp_x_mean - cx) < central_width:
            continue

        component_info.append({
            'id': comp_id,
            'size': comp_size,
            'x_mean': comp_x_mean,
        })

    if not component_info:
        return None

    component_info.sort(key=lambda c: c['size'], reverse=True)
    top_components = component_info[:2]

    if len(top_components) < 2:
        if top_components[0]['size'] >= (vert_width * vert_height * 0.2):
            final_mask = np.zeros_like(seg_slice, dtype=bool)
            final_mask[labeled == top_components[0]['id']] = True
            return extract_bounding_box(final_mask.astype(int), 1)
        return None

    left_comp = min(top_components, key=lambda c: c['x_mean'])
    right_comp = max(top_components, key=lambda c: c['x_mean'])

    size_ratio = max(left_comp['size'], right_comp['size']) / \
                 min(left_comp['size'], right_comp['size'])

    if size_ratio > 2.5:
        return extract_bounding_box((labeled == component_info[0]['id']).astype(int), 1)

    final_mask = np.zeros_like(seg_slice, dtype=bool)
    final_mask[labeled == left_comp['id']] = True
    final_mask[labeled == right_comp['id']] = True

    return extract_bounding_box(final_mask.astype(int), 1)

# ============================================================================
# INTENSITY-BASED DETECTION (NEW v6.0)
# ============================================================================

def detect_t12_rib_intensity_based(mri_slice, seg_slice, vertebra_label, side='left'):
    """
    v6.1 IMPROVED: Detect T12 ribs using MRI intensity patterns with RELAXED parameters
    Does NOT rely on SPINEPS labels for ribs
    
    Key improvements from v6.0:
    - Wider search regions (1.2x -> 1.5x width, -30%/+10% -> -50%/+30% height)
    - Multi-scale edge detection (two Canny thresholds)
    - Multi-threshold intensity detection (Otsu + Adaptive)
    - Relaxed shape constraints (aspect 1.5->1.2, solidity 0.5-0.95->0.4-0.98)
    - Smaller minimum size (10% -> 5%)
    - Gentler preprocessing (CLAHE 3.0 -> 2.0)
    """
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min
    
    height, width = mri_slice.shape
    
    # WIDER search region (v6.1)
    if side == 'left':
        search_x_min = max(0, int(vert_x_min - vert_width * 1.5))  # Was 1.2
        search_x_max = int(vert_x_min + vert_width * 0.2)  # Allow overlap
    else:
        search_x_min = int(vert_x_max - vert_width * 0.2)
        search_x_max = min(width, int(vert_x_max + vert_width * 1.5))
    
    # TALLER search region (v6.1)
    search_y_min = max(0, int(vert_y_min - vert_height * 0.5))  # Was 0.3
    search_y_max = min(height, int(vert_y_max + vert_height * 0.3))  # Was 0.1
    
    # Extract search region
    search_roi = mri_slice[search_y_min:search_y_max, search_x_min:search_x_max].copy()
    
    if search_roi.size == 0:
        return None
    
    # Normalize
    search_roi = cv2.normalize(search_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # GENTLER CLAHE (v6.1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))  # Was 3.0
    search_roi_enhanced = clahe.apply(search_roi)
    
    # MULTI-SCALE EDGE DETECTION (v6.1)
    edges1 = cv2.Canny(search_roi_enhanced, 30, 100)  # Relaxed (was 50/150)
    edges2 = cv2.Canny(search_roi_enhanced, 20, 80)   # Even more relaxed
    edges = cv2.bitwise_or(edges1, edges2)
    
    # SMALLER dilation kernel (v6.1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Was (3,3)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # MULTI-THRESHOLD intensity detection (v6.1)
    _, thresh_otsu = cv2.threshold(search_roi_enhanced, 0, 255, 
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    thresh_adaptive = cv2.adaptiveThreshold(search_roi_enhanced, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    thresh_combined = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    
    # Combine edges and intensity
    combined = cv2.bitwise_or(edges_dilated, thresh_combined)
    
    # GENTLER cleanup (v6.1)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))  # Was (5,5)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Connected components
    labeled, num_features = scipy_label(combined > 0)
    
    if num_features == 0:
        return None
    
    # Analyze components with RELAXED constraints (v6.1)
    best_component = None
    best_score = 0
    
    # SMALLER minimum size (v6.1)
    min_size = (vert_width * vert_height) * 0.05  # Was 0.1
    max_size = (search_y_max - search_y_min) * (search_x_max - search_x_min) * 0.6  # Was 0.5
    
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled == comp_id)
        comp_size = comp_mask.sum()
        
        if comp_size < min_size or comp_size > max_size:
            continue
        
        comp_coords = np.argwhere(comp_mask)
        comp_y = comp_coords[:, 0]
        comp_x = comp_coords[:, 1]
        
        comp_y_min, comp_y_max = comp_y.min(), comp_y.max()
        comp_x_min, comp_x_max = comp_x.min(), comp_x.max()
        comp_height = comp_y_max - comp_y_min + 1
        comp_width = comp_x_max - comp_x_min + 1
        
        if comp_width < 1 or comp_height < 1:
            continue
        
        aspect_ratio = comp_width / comp_height
        
        # RELAXED aspect ratio (v6.1)
        if aspect_ratio < 1.2:  # Was 1.5
            continue
        
        # Position validation
        comp_center_y = (comp_y_min + comp_y_max) / 2 + search_y_min
        vert_center_y = (vert_y_min + vert_y_max) / 2
        y_distance = abs(comp_center_y - vert_center_y)
        
        # RELAXED position (v6.1)
        if y_distance > vert_height * 0.8:  # Was 0.6
            continue
        
        # Curvature check
        hull = cv2.convexHull(comp_coords[:, [1, 0]])
        hull_area = cv2.contourArea(hull)
        solidity = comp_size / hull_area if hull_area > 0 else 0
        
        # RELAXED solidity (v6.1)
        if solidity < 0.4 or solidity > 0.98:  # Was 0.5-0.95
            continue
        
        # Score
        size_score = min(comp_size / max_size, 1.0)
        aspect_score = min(aspect_ratio / 3.0, 1.0)
        position_score = 1.0 - (y_distance / (vert_height * 0.8))
        shape_score = solidity
        
        score = (
            size_score * 0.3 +
            aspect_score * 0.25 +
            position_score * 0.25 +
            shape_score * 0.2
        )
        
        if score > best_score:
            best_score = score
            best_component = comp_id
    
    if best_component is None:
        return None
    
    # Create full-size mask
    rib_mask_roi = (labeled == best_component)
    rib_mask = np.zeros_like(mri_slice, dtype=bool)
    rib_mask[search_y_min:search_y_max, search_x_min:search_x_max] = rib_mask_roi
    
    return extract_bounding_box(rib_mask.astype(int), 1)


def detect_l5_tp_intensity_based(mri_slice, seg_slice, vertebra_label):
    """
    v6.1 IMPROVED: Detect L5 transverse processes using MRI intensity patterns with RELAXED parameters
    Does NOT rely on SPINEPS labels for TPs
    
    Key improvements from v6.0:
    - Wider lateral search (1.5x -> 2.0x width)
    - Added vertical margin (0% -> ±20%)
    - Multi-scale edge detection
    - Multi-threshold intensity detection
    - Relaxed size constraints (5% -> 3%)
    - Single-sided detection allowed (was bilateral required)
    - Relaxed position constraints
    """
    vert_mask = (seg_slice == vertebra_label)
    
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min
    
    height, width = mri_slice.shape
    
    # WIDER bilateral search with VERTICAL MARGIN (v6.1)
    search_y_min = max(0, int(vert_y_min - vert_height * 0.2))  # Was 0
    search_y_max = min(height, int(vert_y_max + vert_height * 0.2))  # Was 0
    
    left_x_min = max(0, int(cx - vert_width * 2.0))  # Was 1.5
    left_x_max = int(cx - vert_width * 0.15)  # Was 0.2
    
    right_x_min = int(cx + vert_width * 0.15)
    right_x_max = min(width, int(cx + vert_width * 2.0))
    
    # Detect both sides
    left_tp = _detect_single_tp(
        mri_slice, left_x_min, left_x_max,
        search_y_min, search_y_max,
        cx, cy, vert_width, vert_height
    )
    
    right_tp = _detect_single_tp(
        mri_slice, right_x_min, right_x_max,
        search_y_min, search_y_max,
        cx, cy, vert_width, vert_height
    )
    
    # RELAXED: Accept single-sided (v6.1)
    if left_tp is None and right_tp is None:
        return None
    
    combined_mask = np.zeros_like(mri_slice, dtype=bool)
    
    if left_tp is not None:
        combined_mask = np.logical_or(combined_mask, left_tp)
    
    if right_tp is not None:
        combined_mask = np.logical_or(combined_mask, right_tp)
    
    return extract_bounding_box(combined_mask.astype(int), 1)


def _detect_single_tp(mri_slice, x_min, x_max, y_min, y_max, cx, cy, vert_width, vert_height):
    """
    v6.1 IMPROVED: Detect single TP with RELAXED constraints
    
    Key improvements:
    - Multi-scale edge detection
    - Multi-threshold intensity
    - Gentler CLAHE (2.0)
    - Smaller min size (5% -> 3%)
    - Relaxed lateral distance (30% -> 20%)
    - Relaxed vertical alignment (30% -> 50%)
    """
    
    if x_max <= x_min or y_max <= y_min:
        return None
    
    roi = mri_slice[y_min:y_max, x_min:x_max].copy()
    
    if roi.size == 0:
        return None
    
    # Normalize
    roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # GENTLER CLAHE (v6.1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    roi = clahe.apply(roi)
    
    # MULTI-SCALE edge detection (v6.1)
    edges1 = cv2.Canny(roi, 30, 90)  # Relaxed
    edges2 = cv2.Canny(roi, 20, 70)  # More relaxed
    edges = cv2.bitwise_or(edges1, edges2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # MULTI-THRESHOLD (v6.1)
    _, thresh_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(roi, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    
    # Combine
    combined = cv2.bitwise_or(edges, thresh)
    
    # Gentle cleanup
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    
    # Connected components
    labeled, num_features = scipy_label(combined > 0)
    
    if num_features == 0:
        return None
    
    # RELAXED size constraints (v6.1)
    min_size = (vert_width * vert_height) * 0.03  # Was 0.05
    max_size = (y_max - y_min) * (x_max - x_min) * 0.5  # Was 0.4
    
    best_comp = None
    best_score = 0
    
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled == comp_id)
        comp_size = comp_mask.sum()
        
        if comp_size < min_size or comp_size > max_size:
            continue
        
        comp_coords = np.argwhere(comp_mask)
        comp_y_mean = comp_coords[:, 0].mean() + y_min
        comp_x_mean = comp_coords[:, 1].mean() + x_min
        
        # RELAXED lateral distance (v6.1)
        x_distance = abs(comp_x_mean - cx)
        if x_distance < vert_width * 0.2:  # Was 0.3
            continue
        
        # RELAXED vertical alignment (v6.1)
        y_distance = abs(comp_y_mean - cy)
        if y_distance > vert_height * 0.5:  # Was 0.3
            continue
        
        # Score
        size_score = min(comp_size / max_size, 1.0)
        lateral_score = min(x_distance / (vert_width * 1.0), 1.0)
        vertical_score = 1.0 - (y_distance / (vert_height * 0.5))
        
        score = (
            size_score * 0.4 +
            lateral_score * 0.3 +
            vertical_score * 0.3
        )
        
        if score > best_score:
            best_score = score
            best_comp = comp_id
    
    if best_comp is None:
        return None
    
    tp_mask_roi = (labeled == best_comp)
    tp_mask = np.zeros_like(mri_slice, dtype=bool)
    tp_mask[y_min:y_max, x_min:x_max] = tp_mask_roi
    
    return tp_mask

# ============================================================================
# HYBRID DETECTION WITH FALLBACK
# ============================================================================

def detect_t12_rib_hybrid(mri_slice, seg_slice, vertebra_label, side='left', detection_metrics=None, study_id='', view=''):
    """
    Try intensity-based first, fallback to label-based
    Track which method succeeded
    """
    # Try intensity-based (NEW)
    intensity_box = detect_t12_rib_intensity_based(mri_slice, seg_slice, vertebra_label, side)
    
    # Try label-based (ORIGINAL)
    label_box = detect_t12_rib_label_based(seg_slice, vertebra_label, side)
    
    # Track metrics
    if detection_metrics is not None:
        detection_metrics.append(DetectionMetrics(
            study_id=study_id,
            view=view,
            structure='t12_rib',
            label_based_success=(label_box is not None),
            intensity_based_success=(intensity_box is not None),
            method_used='intensity' if intensity_box is not None else ('label' if label_box is not None else 'none')
        ))
    
    # Prefer intensity-based
    if intensity_box is not None:
        return intensity_box
    
    return label_box


def detect_l5_tp_hybrid(mri_slice, seg_slice, vertebra_label, detection_metrics=None, study_id='', view=''):
    """
    Try intensity-based first, fallback to label-based
    Track which method succeeded
    """
    # Try intensity-based (NEW)
    intensity_box = detect_l5_tp_intensity_based(mri_slice, seg_slice, vertebra_label)
    
    # Try label-based (ORIGINAL)
    label_box = detect_l5_tp_label_based(seg_slice, vertebra_label)
    
    # Track metrics
    if detection_metrics is not None:
        detection_metrics.append(DetectionMetrics(
            study_id=study_id,
            view=view,
            structure='l5_tp',
            label_based_success=(label_box is not None),
            intensity_based_success=(intensity_box is not None),
            method_used='intensity' if intensity_box is not None else ('label' if label_box is not None else 'none')
        ))
    
    # Prefer intensity-based
    if intensity_box is not None:
        return intensity_box
    
    return label_box

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_before_after_comparison(mri_data, seg_data, slice_info, output_path):
    """Before/after spine-aware slice comparison"""
    sag_axis = slice_info['sag_axis']
    geometric_mid = slice_info['geometric_mid']
    optimal_mid = slice_info['mid']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Geometric
    geometric_views = {
        'left': max(0, geometric_mid - 30),
        'mid': geometric_mid,
        'right': min(mri_data.shape[sag_axis] - 1, geometric_mid + 30),
    }

    for i, (view_name, slice_idx) in enumerate(geometric_views.items()):
        mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=1)
        seg_slice = extract_slice(seg_data, sag_axis, slice_idx, thickness=1)

        normalized = normalize_slice(mri_slice)

        axes[0, i].imshow(normalized.T, cmap='gray', origin='lower')
        axes[0, i].imshow(seg_slice.T, cmap='jet', alpha=0.3, origin='lower')
        axes[0, i].set_title(f'GEOMETRIC {view_name.upper()}\nSlice {slice_idx}',
                            fontsize=14, color='red', weight='bold')
        axes[0, i].axis('off')

    # Row 2: Spine-aware
    spine_views = {
        'left': slice_info['left'],
        'mid': optimal_mid,
        'right': slice_info['right'],
    }

    for i, (view_name, slice_idx) in enumerate(spine_views.items()):
        mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=1)
        seg_slice = extract_slice(seg_data, sag_axis, slice_idx, thickness=1)

        normalized = normalize_slice(mri_slice)

        axes[1, i].imshow(normalized.T, cmap='gray', origin='lower')
        axes[1, i].imshow(seg_slice.T, cmap='jet', alpha=0.3, origin='lower')
        axes[1, i].set_title(f'SPINE-AWARE {view_name.upper()}\nSlice {slice_idx}',
                            fontsize=14, color='green', weight='bold')
        axes[1, i].axis('off')

    # Add metrics
    metrics = slice_info['metrics']
    fig.text(0.5, 0.02,
             f"Offset: {metrics.offset_voxels} voxels ({metrics.offset_mm:.1f}mm) | "
             f"Spine density improvement: {metrics.improvement_ratio:.2f}x",
             ha='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.suptitle('BEFORE (Geometric) vs AFTER (Spine-Aware) Slice Selection',
                 fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_detection_method_comparison(detection_metrics, output_dir):
    """NEW v6.0: Visualize intensity vs label-based detection performance"""
    
    if not detection_metrics:
        return
    
    # Aggregate by structure
    rib_metrics = [m for m in detection_metrics if m.structure == 't12_rib']
    tp_metrics = [m for m in detection_metrics if m.structure == 'l5_tp']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # T12 Rib Detection Rates
    if rib_metrics:
        total_rib = len(rib_metrics)
        intensity_success = sum(m.intensity_based_success for m in rib_metrics)
        label_success = sum(m.label_based_success for m in rib_metrics)
        both_success = sum(m.intensity_based_success and m.label_based_success for m in rib_metrics)
        only_intensity = sum(m.intensity_based_success and not m.label_based_success for m in rib_metrics)
        only_label = sum(not m.intensity_based_success and m.label_based_success for m in rib_metrics)
        both_failed = sum(not m.intensity_based_success and not m.label_based_success for m in rib_metrics)
        
        categories = ['Intensity\nBased', 'Label\nBased', 'Both\nMethods', 'Only\nIntensity', 'Only\nLabel', 'Both\nFailed']
        values = [
            intensity_success/total_rib*100,
            label_success/total_rib*100,
            both_success/total_rib*100,
            only_intensity/total_rib*100,
            only_label/total_rib*100,
            both_failed/total_rib*100
        ]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#95a5a6']
        
        bars = axes[0, 0].bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 0].set_ylabel('Detection Rate (%)', fontsize=12, weight='bold')
        axes[0, 0].set_title('T12 Rib Detection Performance', fontsize=14, weight='bold')
        axes[0, 0].grid(alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 105])
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.1f}%\n({int(val*total_rib/100)})',
                          ha='center', va='bottom', fontsize=10, weight='bold')
    
    # L5 TP Detection Rates
    if tp_metrics:
        total_tp = len(tp_metrics)
        intensity_success = sum(m.intensity_based_success for m in tp_metrics)
        label_success = sum(m.label_based_success for m in tp_metrics)
        both_success = sum(m.intensity_based_success and m.label_based_success for m in tp_metrics)
        only_intensity = sum(m.intensity_based_success and not m.label_based_success for m in tp_metrics)
        only_label = sum(not m.intensity_based_success and m.label_based_success for m in tp_metrics)
        both_failed = sum(not m.intensity_based_success and not m.label_based_success for m in tp_metrics)
        
        values_tp = [
            intensity_success/total_tp*100,
            label_success/total_tp*100,
            both_success/total_tp*100,
            only_intensity/total_tp*100,
            only_label/total_tp*100,
            both_failed/total_tp*100
        ]
        
        bars = axes[0, 1].bar(categories, values_tp, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 1].set_ylabel('Detection Rate (%)', fontsize=12, weight='bold')
        axes[0, 1].set_title('L5 Transverse Process Detection Performance', fontsize=14, weight='bold')
        axes[0, 1].grid(alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 105])
        
        for bar, val in zip(bars, values_tp):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.1f}%\n({int(val*total_tp/100)})',
                          ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Method usage pie chart - Ribs
    if rib_metrics:
        method_counts = {}
        for m in rib_metrics:
            method = m.method_used
            method_counts[method] = method_counts.get(method, 0) + 1
        
        labels = []
        sizes = []
        colors_pie = []
        
        if 'intensity' in method_counts:
            labels.append(f'Intensity\n({method_counts["intensity"]})')
            sizes.append(method_counts['intensity'])
            colors_pie.append('#2ecc71')
        if 'label' in method_counts:
            labels.append(f'Label Fallback\n({method_counts["label"]})')
            sizes.append(method_counts['label'])
            colors_pie.append('#3498db')
        if 'none' in method_counts:
            labels.append(f'Failed\n({method_counts["none"]})')
            sizes.append(method_counts['none'])
            colors_pie.append('#e74c3c')
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                      startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        axes[1, 0].set_title('T12 Rib: Final Method Used', fontsize=14, weight='bold')
    
    # Method usage pie chart - TPs
    if tp_metrics:
        method_counts = {}
        for m in tp_metrics:
            method = m.method_used
            method_counts[method] = method_counts.get(method, 0) + 1
        
        labels = []
        sizes = []
        colors_pie = []
        
        if 'intensity' in method_counts:
            labels.append(f'Intensity\n({method_counts["intensity"]})')
            sizes.append(method_counts['intensity'])
            colors_pie.append('#2ecc71')
        if 'label' in method_counts:
            labels.append(f'Label Fallback\n({method_counts["label"]})')
            sizes.append(method_counts['label'])
            colors_pie.append('#3498db')
        if 'none' in method_counts:
            labels.append(f'Failed\n({method_counts["none"]})')
            sizes.append(method_counts['none'])
            colors_pie.append('#e74c3c')
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                      startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        axes[1, 1].set_title('L5 TP: Final Method Used', fontsize=14, weight='bold')
    
    plt.suptitle('Intensity-Based vs Label-Based Detection Comparison (v6.0)', 
                 fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = output_dir / 'detection_method_comparison.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detection method comparison: {output_path}")


def create_summary_visualization(metrics_log, output_path):
    """Spine-aware slice selection summary"""
    
    offsets_mm = [m.offset_mm for m in metrics_log]
    improvements = [m.improvement_ratio for m in metrics_log]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram of offsets
    axes[0, 0].hist(offsets_mm, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(offsets_mm), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(offsets_mm):.1f} mm')
    axes[0, 0].set_xlabel('Offset from Geometric Center (mm)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Cases', fontsize=12)
    axes[0, 0].set_title('Distribution of Corrections Needed', fontsize=14, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Histogram of improvements
    axes[0, 1].hist(improvements, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(improvements):.2f}x')
    axes[0, 1].set_xlabel('Spine Density Improvement Ratio', fontsize=12)
    axes[0, 1].set_ylabel('Number of Cases', fontsize=12)
    axes[0, 1].set_title('Improvement in Spine Visibility', fontsize=14, weight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Scatter plot
    axes[1, 0].scatter(offsets_mm, improvements, alpha=0.5, s=50, color='purple')
    axes[1, 0].set_xlabel('Offset from Geometric Center (mm)', fontsize=12)
    axes[1, 0].set_ylabel('Improvement Ratio', fontsize=12)
    axes[1, 0].set_title('Offset vs Improvement', fontsize=14, weight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Box plot
    no_corr = [m.improvement_ratio for m in metrics_log if m.offset_voxels == 0]
    small_corr = [m.improvement_ratio for m in metrics_log if 1 <= m.offset_voxels <= 5]
    medium_corr = [m.improvement_ratio for m in metrics_log if 6 <= m.offset_voxels <= 15]
    large_corr = [m.improvement_ratio for m in metrics_log if m.offset_voxels >= 16]

    box_data = []
    labels = []
    if no_corr:
        box_data.append(no_corr)
        labels.append(f'No correction\n(n={len(no_corr)})')
    if small_corr:
        box_data.append(small_corr)
        labels.append(f'Small (1-5)\n(n={len(small_corr)})')
    if medium_corr:
        box_data.append(medium_corr)
        labels.append(f'Medium (6-15)\n(n={len(medium_corr)})')
    if large_corr:
        box_data.append(large_corr)
        labels.append(f'Large (16+)\n(n={len(large_corr)})')

    if box_data:
        bp = axes[1, 1].boxplot(box_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1, 1].set_ylabel('Improvement Ratio', fontsize=12)
        axes[1, 1].set_title('Improvement by Correction Magnitude', fontsize=14, weight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')

    plt.suptitle('Spine-Aware Slice Selection: Quality Validation', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN LABEL GENERATION
# ============================================================================

def create_yolo_labels_multiview(nifti_path, seg_path, output_dir, images_dir,
                                  selector, use_mip=True, generate_comparison=False,
                                  comparison_dir=None, detection_metrics=None):
    """Generate YOLO labels with hybrid detection"""
    try:
        nii = nib.load(nifti_path)
        seg_nii = nib.load(seg_path)

        mri_data = nii.get_fdata()
        seg_data = seg_nii.get_fdata().astype(int)

        study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')

        dims = mri_data.shape
        sag_axis = np.argmin(dims)

        # GET SPINE-AWARE SLICES
        slice_info = selector.get_three_slices(seg_data, sag_axis, study_id)

        # Generate comparison if requested
        if generate_comparison and comparison_dir:
            comparison_path = comparison_dir / f"{study_id}_slice_comparison.png"
            create_before_after_comparison(mri_data, seg_data, slice_info, comparison_path)

        views = {
            'left': slice_info['left'],
            'mid': slice_info['mid'],
            'right': slice_info['right'],
        }

        label_count = 0

        for view_name, slice_idx in views.items():
            # MIP thickness
            if use_mip:
                thickness = 15 if view_name in ['left', 'right'] else 5
            else:
                thickness = 1

            mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
            seg_slice = extract_slice(seg_data, sag_axis, slice_idx, thickness=thickness)

            mri_normalized = normalize_slice(mri_slice)

            # Save image
            image_filename = f"{study_id}_{view_name}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), mri_normalized)

            yolo_labels = []

            # Class 0: T12 vertebra
            t12_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['T12'])
            if t12_box:
                yolo_labels.append([0] + t12_box)

            # Class 1: T12 rib (HYBRID)
            if view_name in ['left', 'right']:
                t12_rib_box = detect_t12_rib_hybrid(
                    mri_slice, seg_slice, SPINEPS_LABELS['T12'],
                    side=view_name, 
                    detection_metrics=detection_metrics,
                    study_id=study_id,
                    view=view_name
                )
                if t12_rib_box:
                    yolo_labels.append([1] + t12_rib_box)

            # Class 2: L5 vertebra
            l5_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5'])
            if l5_box:
                yolo_labels.append([2] + l5_box)

            # Class 3: L5 TP (HYBRID)
            if view_name == 'mid':
                transverse_box = detect_l5_tp_hybrid(
                    mri_slice, seg_slice, SPINEPS_LABELS['L5'],
                    detection_metrics=detection_metrics,
                    study_id=study_id,
                    view=view_name
                )
                if transverse_box:
                    yolo_labels.append([3] + transverse_box)

            # Class 4: Sacrum
            sacrum_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['Sacrum'])
            if sacrum_box:
                yolo_labels.append([4] + sacrum_box)

            # Class 5: L4 vertebra
            l4_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L4'])
            if l4_box:
                yolo_labels.append([5] + l4_box)

            # Class 6: L5-S1 disc
            l5_s1_disc_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['L5_S1_disc'])
            if l5_s1_disc_box:
                yolo_labels.append([6] + l5_s1_disc_box)

            # Write labels
            if yolo_labels:
                label_filename = f"{study_id}_{view_name}.txt"
                label_path = output_dir / label_filename

                with open(label_path, 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

                label_count += len(yolo_labels)

        return True, slice_info['metrics']

    except Exception as e:
        print(f"Error processing {nifti_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# ============================================================================
# REPORTING
# ============================================================================

def generate_metrics_report(metrics_log, output_dir):
    """Generate spine-aware metrics report"""

    if not metrics_log:
        print("No metrics to report")
        return None

    offsets_voxels = [m.offset_voxels for m in metrics_log]
    offsets_mm = [m.offset_mm for m in metrics_log]
    improvements = [m.improvement_ratio for m in metrics_log]

    stats = {
        'total_cases': len(metrics_log),
        'spine_aware_cases': sum(1 for m in metrics_log if m.method == 'spine_aware'),
        'geometric_fallback_cases': sum(1 for m in metrics_log if m.method == 'geometric_fallback'),
        'offset_statistics': {
            'mean_voxels': float(np.mean(offsets_voxels)),
            'std_voxels': float(np.std(offsets_voxels)),
            'median_voxels': float(np.median(offsets_voxels)),
            'max_voxels': float(np.max(offsets_voxels)),
            'mean_mm': float(np.mean(offsets_mm)),
            'std_mm': float(np.std(offsets_mm)),
            'median_mm': float(np.median(offsets_mm)),
            'max_mm': float(np.max(offsets_mm)),
        },
        'improvement_statistics': {
            'mean_ratio': float(np.mean(improvements)),
            'std_ratio': float(np.std(improvements)),
            'median_ratio': float(np.median(improvements)),
            'max_ratio': float(np.max(improvements)),
        },
        'correction_needed': {
            'no_correction': sum(1 for o in offsets_voxels if o == 0),
            'small_correction_1_5_voxels': sum(1 for o in offsets_voxels if 1 <= o <= 5),
            'medium_correction_6_15_voxels': sum(1 for o in offsets_voxels if 6 <= o <= 15),
            'large_correction_16plus_voxels': sum(1 for o in offsets_voxels if o >= 16),
        }
    }

    total = stats['total_cases']
    stats['correction_needed_percent'] = {
        k: (v / total * 100) if total > 0 else 0
        for k, v in stats['correction_needed'].items()
    }

    report_path = output_dir / 'spine_aware_metrics_report.json'
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*80)
    print("SPINE-AWARE SLICE SELECTION - QUALITY METRICS")
    print("="*80)
    print(f"Total cases processed: {stats['total_cases']}")
    print(f"Spine-aware success:   {stats['spine_aware_cases']} ({stats['spine_aware_cases']/total*100:.1f}%)")
    print(f"Geometric fallback:    {stats['geometric_fallback_cases']} ({stats['geometric_fallback_cases']/total*100:.1f}%)")
    print("\nOffset from Geometric Center:")
    print(f"  Mean:   {stats['offset_statistics']['mean_voxels']:.1f} voxels ({stats['offset_statistics']['mean_mm']:.1f} mm)")
    print(f"  Median: {stats['offset_statistics']['median_voxels']:.1f} voxels ({stats['offset_statistics']['median_mm']:.1f} mm)")
    print("\nSpine Density Improvement:")
    print(f"  Mean:   {stats['improvement_statistics']['mean_ratio']:.2f}x")
    print(f"  Median: {stats['improvement_statistics']['median_ratio']:.2f}x")
    print("\nCorrection Distribution:")
    for key, value in stats['correction_needed'].items():
        pct = stats['correction_needed_percent'][key]
        print(f"  {key:35s}: {value:4d} cases ({pct:5.1f}%)")
    print("="*80)

    return stats


def generate_detection_metrics_report(detection_metrics, output_dir):
    """NEW v6.0: Generate detection method comparison report"""
    
    if not detection_metrics:
        print("No detection metrics to report")
        return None
    
    rib_metrics = [m for m in detection_metrics if m.structure == 't12_rib']
    tp_metrics = [m for m in detection_metrics if m.structure == 'l5_tp']
    
    report = {
        't12_rib': {
            'total_attempts': len(rib_metrics),
            'intensity_success': sum(m.intensity_based_success for m in rib_metrics),
            'label_success': sum(m.label_based_success for m in rib_metrics),
            'both_success': sum(m.intensity_based_success and m.label_based_success for m in rib_metrics),
            'only_intensity': sum(m.intensity_based_success and not m.label_based_success for m in rib_metrics),
            'only_label': sum(not m.intensity_based_success and m.label_based_success for m in rib_metrics),
            'both_failed': sum(not m.intensity_based_success and not m.label_based_success for m in rib_metrics),
        },
        'l5_tp': {
            'total_attempts': len(tp_metrics),
            'intensity_success': sum(m.intensity_based_success for m in tp_metrics),
            'label_success': sum(m.label_based_success for m in tp_metrics),
            'both_success': sum(m.intensity_based_success and m.label_based_success for m in tp_metrics),
            'only_intensity': sum(m.intensity_based_success and not m.label_based_success for m in tp_metrics),
            'only_label': sum(not m.intensity_based_success and m.label_based_success for m in tp_metrics),
            'both_failed': sum(not m.intensity_based_success and not m.label_based_success for m in tp_metrics),
        }
    }
    
    # Calculate rates
    for structure in ['t12_rib', 'l5_tp']:
        total = report[structure]['total_attempts']
        if total > 0:
            report[structure]['intensity_rate'] = report[structure]['intensity_success'] / total * 100
            report[structure]['label_rate'] = report[structure]['label_success'] / total * 100
            report[structure]['improvement'] = report[structure]['only_intensity'] / total * 100
    
    report_path = output_dir / 'detection_method_comparison.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("DETECTION METHOD COMPARISON (v6.0)")
    print("="*80)
    
    for structure in ['t12_rib', 'l5_tp']:
        data = report[structure]
        total = data['total_attempts']
        if total == 0:
            continue
        
        print(f"\n{structure.upper().replace('_', ' ')}:")
        print(f"  Total attempts:        {total}")
        print(f"  Intensity success:     {data['intensity_success']} ({data['intensity_rate']:.1f}%)")
        print(f"  Label success:         {data['label_success']} ({data['label_rate']:.1f}%)")
        print(f"  Both methods:          {data['both_success']}")
        print(f"  Only intensity:        {data['only_intensity']} ({data['improvement']:.1f}%)")
        print(f"  Only label:            {data['only_label']}")
        print(f"  Both failed:           {data['both_failed']}")
    
    print("="*80)
    
    return report


def generate_quality_report(output_dir):
    """Generate class distribution report"""
    labels_train = list((output_dir / 'labels' / 'train').glob("*.txt"))
    labels_val = list((output_dir / 'labels' / 'val').glob("*.txt"))

    all_labels = labels_train + labels_val

    class_counts = {i: 0 for i in range(7)}
    total_boxes = 0

    for label_file in all_labels:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_boxes += 1

    report = {
        'total_images': len(all_labels),
        'total_boxes': total_boxes,
        'class_distribution': {
            YOLO_CLASSES[k]: v for k, v in class_counts.items()
        },
        'avg_boxes_per_image': total_boxes / len(all_labels) if all_labels else 0,
    }

    for class_id, class_name in YOLO_CLASSES.items():
        detection_rate = class_counts[class_id] / len(all_labels) if all_labels else 0
        report[f'{class_name}_detection_rate'] = detection_rate

    report_path = output_dir / 'weak_label_quality_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print("WEAK LABEL QUALITY REPORT")
    print("="*80)
    print(f"Total images: {report['total_images']}")
    print(f"Total boxes: {report['total_boxes']}")
    print(f"Avg boxes/image: {report['avg_boxes_per_image']:.2f}")
    print("\nClass Distribution:")
    for class_name, count in report['class_distribution'].items():
        rate = count / report['total_images'] if report['total_images'] > 0 else 0
        print(f"  {class_name:25s}: {count:5d} ({rate*100:5.1f}%)")
    print("="*80)

    return report


def split_train_val(images_dir, labels_dir, val_split=0.15):
    """Split dataset into train/val"""
    image_files = list(images_dir.glob("*.jpg"))

    study_ids = set()
    for img_file in image_files:
        study_id = img_file.stem.rsplit('_', 1)[0]
        study_ids.add(study_id)

    study_ids = sorted(list(study_ids))

    np.random.seed(42)
    np.random.shuffle(study_ids)

    val_count = int(len(study_ids) * val_split)
    val_studies = set(study_ids[:val_count])
    train_studies = set(study_ids[val_count:])

    print(f"\nTrain studies: {len(train_studies)}")
    print(f"Val studies: {len(val_studies)}")

    val_images_dir = images_dir.parent / 'val'
    val_labels_dir = labels_dir.parent / 'val'

    val_images_dir.mkdir(exist_ok=True)
    val_labels_dir.mkdir(exist_ok=True)

    move_count = 0

    for img_file in image_files:
        study_id = img_file.stem.rsplit('_', 1)[0]

        if study_id in val_studies:
            val_img_path = val_images_dir / img_file.name
            img_file.rename(val_img_path)

            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                val_label_path = val_labels_dir / label_file.name
                label_file.rename(val_label_path)

            move_count += 1

    print(f"Moved {move_count} images to validation set")

    return len(train_studies), len(val_studies)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Production Weak Label Generation v6.1 - Intensity-Based Detection (TUNED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NEW IN v6.1 (PARAMETER TUNING):
  Relaxed intensity-based constraints - v6.0 was too strict (0% detection)
  - Multi-scale edge detection (2 Canny thresholds)
  - Multi-threshold intensity (Otsu + Adaptive)
  - Wider search regions (ribs: 1.2x->1.5x, TPs: 1.5x->2.0x)
  - Relaxed shape constraints (aspect, solidity, size)
  - Single-sided TP detection allowed
  - Expected: 0% → 40-70% intensity-based success

Examples:
  # Standard with improved intensity-based detection
  python generate_weak_labels.py --nifti_dir /data/nifti --seg_dir /data/seg --output_dir /data/output

  # With full validation
  python generate_weak_labels.py --nifti_dir /data/nifti --seg_dir /data/seg --output_dir /data/output --generate_comparisons --validate_spineps
        """
    )
    parser.add_argument('--nifti_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use_mip', action='store_true', default=True)
    parser.add_argument('--use_spine_aware', action='store_true', default=True)
    parser.add_argument('--use_intensity_based', action='store_true', default=True,
                       help='Use intensity-based detection (NEW v6.0)')
    parser.add_argument('--generate_comparisons', action='store_true',
                       help='Generate validation visualizations')
    parser.add_argument('--validate_spineps', action='store_true',
                       help='Validate SPINEPS label quality')

    args = parser.parse_args()

    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)

    labels_dir = output_dir / 'labels' / 'train'
    images_dir = output_dir / 'images' / 'train'
    comparison_dir = output_dir / 'quality_validation' if args.generate_comparisons else None

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if comparison_dir:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PRODUCTION WEAK LABEL GENERATION v6.1")
    print("="*80)
    print("Features:")
    print("  ✓ Thick Slab MIP (15mm ribs, 5mm midline)")
    print("  ✓ Spine-aware intelligent slice selection")
    if args.use_intensity_based:
        print("  ✓ Intensity-based T12 rib detection (v6.1 TUNED)")
        print("  ✓ Intensity-based L5 TP detection (v6.1 TUNED)")
        print("  ✓ Hybrid detection with label-based fallback")
        print("  ✓ Multi-scale edge detection (NEW in v6.1)")
        print("  ✓ Multi-threshold intensity detection (NEW in v6.1)")
    else:
        print("  ✓ Label-based detection (original)")
    if args.generate_comparisons:
        print("  ✓ VALIDATION: Before/after comparisons")
        print("  ✓ VALIDATION: Detection method comparison")
    if args.validate_spineps:
        print("  ✓ SPINEPS label quality validation")
    print("="*80)

    seg_files = sorted(seg_dir.glob("*_seg.nii.gz"))

    if args.limit:
        seg_files = seg_files[:args.limit]

    print(f"\nFound {len(seg_files)} segmentation files")
    print("Generating 3-view YOLO labels...")
    print("="*80)

    selector = SpineAwareSliceSelector() if args.use_spine_aware else None
    detection_metrics = [] if args.validate_spineps else None

    success_count = 0

    for seg_file in tqdm(seg_files, desc="Processing"):
        study_id = seg_file.stem.replace('_seg', '')
        nifti_file = nifti_dir / f"sub-{study_id}_T2w.nii.gz"

        if not nifti_file.exists():
            continue

        success, metrics = create_yolo_labels_multiview(
            nifti_file, seg_file, labels_dir, images_dir,
            selector, use_mip=args.use_mip,
            generate_comparison=args.generate_comparisons,
            comparison_dir=comparison_dir,
            detection_metrics=detection_metrics
        )

        if success:
            success_count += 1

    print(f"\n✓ Generated labels for {success_count} studies")

    # Generate reports
    if args.generate_comparisons and selector:
        print("\nGenerating spine-aware metrics...")
        spine_stats = generate_metrics_report(selector.metrics_log, output_dir)

        summary_viz_path = output_dir / 'quality_validation_summary.png'
        create_summary_visualization(selector.metrics_log, summary_viz_path)
        print(f"✓ Summary visualization: {summary_viz_path}")

    if args.validate_spineps and detection_metrics:
        print("\nGenerating detection method comparison...")
        detection_report = generate_detection_metrics_report(detection_metrics, output_dir)
        create_detection_method_comparison(detection_metrics, output_dir)

    # Split train/val
    print("\nSplitting train/validation...")
    train_count, val_count = split_train_val(images_dir, labels_dir)

    # Quality report
    quality_report = generate_quality_report(output_dir)

    # Dataset YAML
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': YOLO_CLASSES,
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"\n✓ Created {yaml_path}")

    # Metadata
    metadata = {
        'version': '6.1',
        'improvements_from_6.0': 'Tuned intensity-based parameters - relaxed constraints for better detection',
        'total_studies': success_count,
        'train_studies': train_count,
        'val_studies': val_count,
        'features': {
            'thick_slab_mip': args.use_mip,
            'spine_aware_slicing': args.use_spine_aware,
            'intensity_based_detection': args.use_intensity_based,
            'multi_scale_edges': args.use_intensity_based,
            'multi_threshold_intensity': args.use_intensity_based,
            'validation_enabled': args.generate_comparisons,
            'spineps_validation': args.validate_spineps,
        },
        'quality_metrics': quality_report,
    }

    if args.generate_comparisons and selector and 'spine_stats' in locals():
        metadata['spine_aware_metrics'] = spine_stats

    if args.validate_spineps and detection_metrics and 'detection_report' in locals():
        metadata['detection_comparison'] = detection_report

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Version: 6.1 (Intensity-Based Detection - TUNED PARAMETERS)")
    print(f"Train: {train_count} studies ({train_count * 3} images)")
    print(f"Val:   {val_count} studies ({val_count * 3} images)")
    print(f"\nDataset ready at: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
