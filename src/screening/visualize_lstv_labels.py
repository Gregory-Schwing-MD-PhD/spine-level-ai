#!/usr/bin/env python3
"""
LSTV Label Verification & QA Visualization
Overlays vertebra labels on MRI slices for manual auditing

Usage:
    python visualize_lstv_labels.py \
        --nifti_dir /data/nifti \
        --seg_dir /data/seg \
        --output_dir /data/qa_reports \
        --limit 5
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import json

# ============================================================================
# LABEL DEFINITIONS
# ============================================================================

SPINEPS_LABELS = {
    'T11': 18, 'T12': 19,
    'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'L6': 25,
    'Sacrum': 26,
    'T12_L1_disc': 119, 'L1_L2_disc': 120, 'L2_L3_disc': 121,
    'L3_L4_disc': 122, 'L4_L5_disc': 123, 'L5_S1_disc': 124,
    'S1_S2_disc': 126,
}

ID_TO_NAME = {v: k for k, v in SPINEPS_LABELS.items()}

LSTV_COLORS = {
    'L6': (255, 0, 255),          # Magenta - LUMBARIZATION
    'S1_S2_disc': (255, 128, 0),  # Orange - SACRALIZATION
    'L5': (0, 255, 0),            # Green - Normal L5
    'Sacrum': (0, 255, 255),      # Cyan - Sacrum
    'L4': (100, 255, 100),        # Light green
    'default': (255, 255, 0),     # Yellow - Other vertebrae
}

# ============================================================================
# SPINE-AWARE SLICE SELECTOR (minimal version)
# ============================================================================

class SpineAwareSliceSelector:
    """Minimal spine-aware slice selection for QA"""
    
    def __init__(self, voxel_spacing_mm=1.0, parasagittal_offset_mm=30):
        self.voxel_spacing_mm = voxel_spacing_mm
        self.parasagittal_offset_mm = parasagittal_offset_mm
    
    def find_optimal_midline(self, seg_data, sag_axis):
        """Find TRUE spinal midline using segmentation"""
        num_slices = seg_data.shape[sag_axis]
        geometric_mid = num_slices // 2
        
        lumbar_labels = [20, 21, 22, 23, 24, 26]  # L1-L5 + Sacrum
        
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
    
    def get_three_slices(self, seg_data, sag_axis, study_id=None):
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
# IMAGE PROCESSING
# ============================================================================

def extract_slice(data, sag_axis, slice_idx, thickness=1):
    """Extract 2D slice with optional MIP"""
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
    """Normalize to 0-255 with CLAHE"""
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
                confidence_factors.append(f"✓ L6/L5 size ratio: {size_ratio:.2f} (valid)")
            else:
                confidence_factors.append(f"⚠ L6/L5 size ratio: {size_ratio:.2f} (SUSPICIOUS)")
        
        # Minimum absolute size check (empirical threshold)
        if l6_volume > 500:
            confidence_score += 0.2
            confidence_factors.append(f"✓ L6 volume: {l6_volume} voxels (sufficient)")
        else:
            confidence_factors.append(f"⚠ L6 volume: {l6_volume} voxels (TOO SMALL)")
    
    # Factor 2: Sacrum must be present
    if has_sacrum:
        confidence_score += 0.2
        confidence_factors.append("✓ Sacrum detected")
    else:
        confidence_factors.append("⚠ NO SACRUM (red flag)")
    
    # Factor 3: S1-S2 disc is strong sacralization indicator
    if s1_s2_disc:
        confidence_score += 0.3
        confidence_factors.append("✓ S1-S2 disc visible (strong evidence)")
    
    # Factor 4: Vertebra count consistency
    vertebra_count = len(lumbar_labels)
    if vertebra_count in [4, 5, 6]:
        confidence_score += 0.1
        confidence_factors.append(f"✓ Vertebra count: {vertebra_count} (plausible)")
    else:
        confidence_factors.append(f"⚠ Vertebra count: {vertebra_count} (IMPLAUSIBLE)")
    
    # Determine confidence level
    if confidence_score >= 0.7:
        confidence_level = "HIGH"
    elif confidence_score >= 0.4:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    return confidence_score, confidence_level, confidence_factors

# ============================================================================
# LABELED OVERLAY CREATION
# ============================================================================

def create_labeled_overlay(mri_slice, seg_slice, lstv_info):
    """
    Create RGB image with vertebra labels overlaid
    
    Args:
        mri_slice: Normalized MRI image (grayscale)
        seg_slice: Segmentation mask with label IDs
        lstv_info: Dict with vertebra counts and LSTV flags
    
    Returns:
        RGB image with text labels and color-coded borders
    """
    # Convert grayscale to RGB
    rgb_img = cv2.cvtColor(mri_slice, cv2.COLOR_GRAY2RGB)
    
    # Find all vertebrae in this slice
    unique_labels = np.unique(seg_slice)
    vertebrae = [l for l in unique_labels if l in ID_TO_NAME]
    
    # Sort by anatomical position (superior to inferior)
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
        elif name == 'L4':
            color = LSTV_COLORS['L4']
            thickness = 2
            font_scale = 0.7
        else:
            color = LSTV_COLORS['default']
            thickness = 1
            font_scale = 0.6
        
        # Draw text label
        cv2.putText(rgb_img, name, (cx - 30, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   color, thickness, cv2.LINE_AA)
        
        # Draw small marker at centroid
        cv2.circle(rgb_img, (cx, cy), 4, color, -1)
    
    # Add LSTV warning banner if applicable
    if lstv_info['is_lstv_candidate']:
        banner_height = 50
        banner = np.zeros((banner_height, rgb_img.shape[1], 3), dtype=np.uint8)
        banner[:] = (50, 50, 50)  # Dark gray
        
        lstv_type = lstv_info['lstv_type'].upper()
        vert_count = lstv_info['vertebra_count']
        confidence = lstv_info.get('confidence_level', 'UNKNOWN')
        
        if lstv_type == 'LUMBARIZATION':
            banner_color = LSTV_COLORS['L6']
            text = f"⚠ LSTV: LUMBARIZATION ({vert_count} lumbar, L6 present) - {confidence} CONFIDENCE"
        elif lstv_type == 'SACRALIZATION':
            banner_color = (255, 128, 0)
            text = f"⚠ LSTV: SACRALIZATION ({vert_count} lumbar) - {confidence} CONFIDENCE"
        elif lstv_type == 'S1_S2_DISC':
            banner_color = (255, 128, 0)
            text = f"⚠ LSTV: S1-S2 disc visible (sacralization) - {confidence} CONFIDENCE"
        else:
            banner_color = (255, 255, 0)
            text = f"⚠ LSTV: Atypical ({vert_count} lumbar) - {confidence} CONFIDENCE"
        
        cv2.putText(banner, text, (10, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2, cv2.LINE_AA)
        
        rgb_img = np.vstack([banner, rgb_img])
    
    return rgb_img

# ============================================================================
# QA REPORT GENERATION
# ============================================================================

def generate_qa_report(nifti_path, seg_path, output_path, selector):
    """
    Generate comprehensive QA report with labeled overlays
    
    Args:
        nifti_path: Path to MRI NIfTI
        seg_path: Path to SPINEPS segmentation
        output_path: Output PDF path
        selector: SpineAwareSliceSelector instance
    
    Returns:
        lstv_info dict with analysis results
    """
    # Load data
    nii = nib.load(nifti_path)
    seg_nii = nib.load(seg_path)
    
    mri_data = nii.get_fdata()
    seg_data = seg_nii.get_fdata().astype(int)
    
    study_id = nifti_path.stem.replace('_T2w', '').replace('.nii', '').replace('sub-', '')
    
    # Analyze segmentation
    unique_labels = np.unique(seg_data)
    vertebra_labels = [l for l in unique_labels if 1 <= l <= 25]
    lumbar_labels = [l for l in vertebra_labels if 20 <= l <= 25]
    
    vertebra_count = len(lumbar_labels)
    has_sacrum = 26 in unique_labels
    has_l6 = 25 in lumbar_labels
    s1_s2_disc = 126 in unique_labels
    is_lstv = (vertebra_count != 5 or s1_s2_disc or has_l6)
    
    # Calculate confidence
    confidence_score, confidence_level, confidence_factors = calculate_lstv_confidence(
        seg_data, unique_labels, lumbar_labels, has_l6, s1_s2_disc, has_sacrum
    )
    
    lstv_type = "normal"
    if vertebra_count < 5:
        lstv_type = "sacralization"
    elif vertebra_count > 5 or has_l6:
        lstv_type = "lumbarization"
    elif s1_s2_disc:
        lstv_type = "s1_s2_disc"
    
    lstv_info = {
        'study_id': study_id,
        'vertebra_count': vertebra_count,
        'has_sacrum': has_sacrum,
        'has_l6': has_l6,
        's1_s2_disc': s1_s2_disc,
        'is_lstv_candidate': is_lstv,
        'lstv_type': lstv_type,
        'lumbar_labels': [ID_TO_NAME.get(l, f'ID_{l}') for l in lumbar_labels],
        'confidence_score': round(confidence_score, 2),
        'confidence_level': confidence_level,
        'confidence_factors': confidence_factors,
    }
    
    # Get spine-aware slices
    sag_axis = np.argmin(mri_data.shape)
    slice_info = selector.get_three_slices(seg_data, sag_axis, study_id)
    
    views = {
        'Left Parasagittal': slice_info['left'],
        'Midline (Spine-Aware)': slice_info['mid'],
        'Right Parasagittal': slice_info['right'],
    }
    
    # Create PDF report
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(20, 14))
        
        # Title page with detailed info
        title_ax = plt.subplot(3, 3, 1)
        title_ax.axis('off')
        
        title_text = f"LSTV QA REPORT\n\nStudy ID: {study_id}\n\n"
        
        if is_lstv:
            title_text += f"⚠ LSTV CANDIDATE ⚠\n"
            title_text += f"Type: {lstv_type.upper()}\n"
            title_text += f"Lumbar count: {vertebra_count}\n"
            title_text += f"Confidence: {confidence_level} ({confidence_score:.2f})\n\n"
        else:
            title_text += f"✓ NORMAL ANATOMY\n"
            title_text += f"Lumbar count: {vertebra_count}\n\n"
        
        title_text += f"Vertebrae detected:\n"
        for name in lstv_info['lumbar_labels']:
            title_text += f"  • {name}\n"
        
        title_text += f"\nConfidence factors:\n"
        for factor in confidence_factors:
            title_text += f"  {factor}\n"
        
        if has_l6:
            title_text += "\n🔴 L6 PRESENT (Lumbarization)"
        if s1_s2_disc:
            title_text += "\n🔴 S1-S2 disc visible"
        if not has_sacrum:
            title_text += "\n⚠ Sacrum not detected"
        
        # Color code based on confidence
        if is_lstv:
            if confidence_level == "HIGH":
                box_color = 'lightgreen'
            elif confidence_level == "MEDIUM":
                box_color = 'wheat'
            else:
                box_color = 'lightcoral'
        else:
            box_color = 'lightblue'
        
        title_ax.text(0.1, 0.9, title_text, 
                     fontsize=12, verticalalignment='top',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7))
        
        # Generate labeled overlays for each view
        for idx, (view_name, slice_idx) in enumerate(views.items(), start=2):
            thickness = 15 if 'Parasagittal' in view_name else 5
            
            mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
            seg_slice = extract_slice(seg_data, sag_axis, slice_idx, thickness=thickness)
            
            normalized = normalize_slice(mri_slice)
            
            # Create labeled overlay
            labeled_img = create_labeled_overlay(normalized, seg_slice, lstv_info)
            
            ax = plt.subplot(3, 3, idx)
            ax.imshow(labeled_img)
            ax.set_title(f'{view_name}\nSlice {slice_idx}', fontsize=14, weight='bold')
            ax.axis('off')
        
        # Legend
        legend_ax = plt.subplot(3, 3, 5)
        legend_ax.axis('off')
        
        legend_patches = [
            mpatches.Patch(color=np.array(LSTV_COLORS['L6'])/255, label='L6 (Lumbarization)'),
            mpatches.Patch(color=np.array(LSTV_COLORS['S1_S2_disc'])/255, label='S1-S2 disc'),
            mpatches.Patch(color=np.array(LSTV_COLORS['L5'])/255, label='L5 (Normal)'),
            mpatches.Patch(color=np.array(LSTV_COLORS['Sacrum'])/255, label='Sacrum'),
            mpatches.Patch(color=np.array(LSTV_COLORS['default'])/255, label='Other vertebrae'),
        ]
        
        legend_ax.legend(handles=legend_patches, loc='center', fontsize=14, frameon=True, 
                        title='Color Legend', title_fontsize=16)
        
        # Add recommendation box
        rec_ax = plt.subplot(3, 3, 8)
        rec_ax.axis('off')
        
        rec_text = "ROBOFLOW UPLOAD RECOMMENDATION:\n\n"
        
        if is_lstv:
            if confidence_level == "HIGH":
                rec_text += "✅ UPLOAD TO ROBOFLOW\n"
                rec_text += "High confidence LSTV case\n"
                rec_text += "Good training data"
            elif confidence_level == "MEDIUM":
                rec_text += "⚠ MANUAL REVIEW RECOMMENDED\n"
                rec_text += "Moderate confidence\n"
                rec_text += "Review before upload"
            else:
                rec_text += "❌ DO NOT UPLOAD\n"
                rec_text += "Low confidence\n"
                rec_text += "Likely false positive"
        else:
            rec_text += "ℹ Normal anatomy\n"
            rec_text += "Use as negative example"
        
        if confidence_level == "HIGH":
            rec_box_color = 'lightgreen'
        elif confidence_level == "MEDIUM":
            rec_box_color = 'wheat'
        else:
            rec_box_color = 'lightcoral'
        
        rec_ax.text(0.1, 0.9, rec_text,
                   fontsize=13, verticalalignment='top',
                   family='monospace', weight='bold',
                   bbox=dict(boxstyle='round', facecolor=rec_box_color, alpha=0.7))
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()
    
    return lstv_info

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate LSTV QA visualizations with confidence scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate QA reports for trial data
    python visualize_lstv_labels.py \\
        --nifti_dir /data/nifti \\
        --seg_dir /data/seg \\
        --output_dir /data/qa_reports \\
        --limit 5

    # Process all studies
    python visualize_lstv_labels.py \\
        --nifti_dir /data/nifti \\
        --seg_dir /data/seg \\
        --output_dir /data/qa_reports
        """
    )
    parser.add_argument('--nifti_dir', type=str, required=True,
                       help='Directory with NIfTI files')
    parser.add_argument('--seg_dir', type=str, required=True,
                       help='Directory with SPINEPS segmentations')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for QA reports')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of studies to process')
    
    args = parser.parse_args()
    
    nifti_dir = Path(args.nifti_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selector = SpineAwareSliceSelector()
    
    seg_files = sorted(seg_dir.glob("*_seg.nii.gz"))
    if args.limit:
        seg_files = seg_files[:args.limit]
    
    print("="*80)
    print("LSTV QA REPORT GENERATION")
    print("="*80)
    print(f"Studies to process: {len(seg_files)}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    results = []
    lstv_count = 0
    high_conf_count = 0
    medium_conf_count = 0
    low_conf_count = 0
    
    for seg_file in tqdm(seg_files, desc="Generating QA reports"):
        study_id = seg_file.stem.replace('_seg', '')
        nifti_file = nifti_dir / f"sub-{study_id}_T2w.nii.gz"
        
        if not nifti_file.exists():
            print(f"⚠ Skipping {study_id}: NIfTI not found")
            continue
        
        output_pdf = output_dir / f"{study_id}_QA_report.pdf"
        
        try:
            lstv_info = generate_qa_report(nifti_file, seg_file, output_pdf, selector)
            results.append(lstv_info)
            
            if lstv_info['is_lstv_candidate']:
                lstv_count += 1
                
                if lstv_info['confidence_level'] == 'HIGH':
                    high_conf_count += 1
                    print(f"✓ {study_id}: LSTV {lstv_info['lstv_type']} (HIGH confidence)")
                elif lstv_info['confidence_level'] == 'MEDIUM':
                    medium_conf_count += 1
                    print(f"⚠ {study_id}: LSTV {lstv_info['lstv_type']} (MEDIUM confidence)")
                else:
                    low_conf_count += 1
                    print(f"⚠ {study_id}: LSTV {lstv_info['lstv_type']} (LOW confidence)")
            else:
                print(f"✓ {study_id}: Normal")
        
        except Exception as e:
            print(f"✗ {study_id}: Error - {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary JSON
    summary = {
        'total_studies': len(results),
        'lstv_candidates': lstv_count,
        'high_confidence': high_conf_count,
        'medium_confidence': medium_conf_count,
        'low_confidence': low_conf_count,
        'upload_recommended': high_conf_count,
        'manual_review_needed': medium_conf_count,
        'reject': low_conf_count,
        'studies': results,
    }
    
    summary_path = output_dir / 'qa_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("QA REPORT SUMMARY")
    print("="*80)
    print(f"Total studies:       {len(results)}")
    print(f"LSTV candidates:     {lstv_count}")
    print()
    print("Confidence breakdown:")
    print(f"  HIGH:   {high_conf_count} → ✅ UPLOAD TO ROBOFLOW")
    print(f"  MEDIUM: {medium_conf_count} → ⚠ MANUAL REVIEW")
    print(f"  LOW:    {low_conf_count} → ❌ REJECT")
    print()
    print(f"PDF reports: {output_dir}")
    print(f"Summary JSON: {summary_path}")
    print("="*80)
    
    print("\nNext steps:")
    print(f"  1. Review HIGH confidence PDFs: ls {output_dir}/*_QA_report.pdf")
    print(f"  2. Upload HIGH confidence to Roboflow: {high_conf_count} cases")
    print(f"  3. Manually review MEDIUM confidence: {medium_conf_count} cases")
    print(f"  4. Ignore LOW confidence: {low_conf_count} cases")


if __name__ == "__main__":
    main()
