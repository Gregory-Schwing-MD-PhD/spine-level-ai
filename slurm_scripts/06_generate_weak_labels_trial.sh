#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=weak_labels_v6.1_trial
#SBATCH -o logs/weak_labels_v6.1_%j.out
#SBATCH -e logs/weak_labels_v6.1_%j.err

set -euo pipefail

# ============================================================================
# ANSI COLOR CODES
# ============================================================================
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
PURPLE='\033[0;35m'
NC='\033[0m'

BOX_H="═"
BOX_V="║"
BOX_TL="╔"
BOX_TR="╗"
BOX_BL="╚"
BOX_BR="╝"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    local text="$1"
    local width=80
    echo -e "${BOLD}${CYAN}"
    printf "${BOX_TL}"; printf "${BOX_H}%.0s" $(seq 1 $((width-2))); printf "${BOX_TR}\n"
    printf "${BOX_V}%-$((width-2))s${BOX_V}\n" " $text"
    printf "${BOX_BL}"; printf "${BOX_H}%.0s" $(seq 1 $((width-2))); printf "${BOX_BR}\n"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${BOLD}${BLUE}▶ $1${NC}"
    echo -e "${BLUE}$(printf '─%.0s' {1..80})${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_stat() {
    local label="$1"
    local value="$2"
    printf "${BOLD}%-35s${NC}: ${GREEN}%s${NC}\n" "$label" "$value"
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

print_header "WEAK LABEL GENERATION v6.1 - TUNED INTENSITY-BASED DETECTION"
echo -e "${CYAN}Job ID:${NC} $SLURM_JOB_ID"
echo -e "${CYAN}Start:${NC}  $(date)"
echo ""

print_section "Environment Setup"

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

if ! command -v singularity &> /dev/null; then
    print_error "Singularity not found!"
    exit 1
fi
print_success "Singularity: $(which singularity)"

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

print_section "Path Configuration"

PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/lstv_screening/trial/nifti"
SEG_DIR="${PROJECT_DIR}/results/lstv_screening/trial/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_v6_trial"

IMG_PATH="${HOME}/singularity_cache/spine-level-ai-yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    print_error "YOLOv11 container not found: $IMG_PATH"
    exit 1
fi

if [[ ! -d "$NIFTI_DIR" ]]; then
    print_error "NIfTI directory not found: $NIFTI_DIR"
    exit 1
fi

NIFTI_COUNT=$(find $NIFTI_DIR -name "*.nii.gz" 2>/dev/null | wc -l)
SEG_COUNT=$(find $SEG_DIR -name "*_seg.nii.gz" 2>/dev/null | wc -l)

print_success "Container: $IMG_PATH"
print_stat "NIfTI files" "$NIFTI_COUNT"
print_stat "Segmentation files" "$SEG_COUNT"

mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# ============================================================================
# FEATURE SHOWCASE
# ============================================================================

print_section "v6.0 Features Enabled"

echo ""
echo -e "${BOLD}${PURPLE}┌─────────────────────────────────────────────────────────────┐${NC}"
echo -e "${BOLD}${PURPLE}│                    NEW IN v6.1                              │${NC}"
echo -e "${BOLD}${PURPLE}├─────────────────────────────────────────────────────────────┤${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${GREEN}●${NC} ${BOLD}TUNED Intensity-Based Rib Detection${NC}                   ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Relaxed constraints (v6.0 was TOO STRICT - 0%)       ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Multi-scale edges (Canny 30/100 + 20/80)             ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Multi-threshold intensity (Otsu + Adaptive)          ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Wider search (1.2x→1.5x), smaller min (10%→5%)       ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}                                                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${GREEN}●${NC} ${BOLD}TUNED L5 TP Detection${NC}                                 ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Wider lateral search (1.5x→2.0x width)               ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Vertical margins added (±20% height)                 ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Single-sided detection OK (was bilateral required)   ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Smaller minimum size (5%→3% vertebra area)           ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}                                                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${YELLOW}●${NC} ${BOLD}Expected Improvement${NC}                                 ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → T12 ribs: 0% → ${GREEN}40-70%${NC} intensity detection            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → L5 TPs:   0% → ${GREEN}50-80%${NC} intensity detection            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Combined: ${GREEN}+10-30${NC} additional detections/100 studies    ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}                                                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${CYAN}●${NC} ${BOLD}VALIDATION MODE${NC}                                      ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Before/after spine-aware slice comparison            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Detection method performance analysis                ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → SPINEPS label quality validation                     ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}└─────────────────────────────────────────────────────────────┘${NC}"
echo ""

# ============================================================================
# LABEL GENERATION
# ============================================================================

print_section "Running Label Generation v6.1 (TUNED PARAMETERS)"

echo ""
echo -e "${BOLD}Command:${NC}"
echo -e "  ${CYAN}python generate_weak_labels.py${NC}"
echo -e "    ${YELLOW}--use_intensity_based${NC}    (v6.1 TUNED: relaxed constraints)"
echo -e "    ${YELLOW}--generate_comparisons${NC}   (Validation visualizations)"
echo -e "    ${YELLOW}--validate_spineps${NC}       (Label quality assessment)"
echo ""

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $NIFTI_DIR:/data/nifti \
    --bind $SEG_DIR:/data/seg \
    --bind $OUTPUT_DIR:/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/generate_weak_labels.py \
        --nifti_dir /data/nifti \
        --seg_dir /data/seg \
        --output_dir /data/output \
        --use_intensity_based \
        --generate_comparisons \
        --validate_spineps \
        --use_mip \
        --use_spine_aware

exit_code=$?

if [ $exit_code -ne 0 ]; then
    print_error "Label generation failed (exit code: $exit_code)"
    exit $exit_code
fi

print_success "Label generation complete!"

# ============================================================================
# VALIDATION ANALYSIS
# ============================================================================

print_header "VALIDATION ANALYSIS"

# ============================================================================
# 1. SPINE-AWARE SLICE SELECTION METRICS
# ============================================================================

if [[ -f "$OUTPUT_DIR/spine_aware_metrics_report.json" ]]; then
    print_section "1. Spine-Aware Slice Selection Performance"

    python3 << PYEOF
import json
import sys
import os

BOLD = '\033[1m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
RED = '\033[0;31m'
NC = '\033[0m'

try:
    output_dir = '${OUTPUT_DIR}'
    with open(f'{output_dir}/spine_aware_metrics_report.json') as f:
        stats = json.load(f)

    total = stats['total_cases']

    print(f"\n{BOLD}{CYAN}Overview:{NC}")
    print(f"  Total cases:          {GREEN}{total}{NC}")
    print(f"  Spine-aware success:  {GREEN}{stats['spine_aware_cases']}{NC} ({stats['spine_aware_cases']/total*100:.1f}%)")
    print(f"  Geometric fallback:   {YELLOW}{stats['geometric_fallback_cases']}{NC} ({stats['geometric_fallback_cases']/total*100:.1f}%)")

    print(f"\n{BOLD}{CYAN}Offset from Geometric Center:{NC}")
    print(f"  Mean:   {stats['offset_statistics']['mean_mm']:.1f} ± {stats['offset_statistics']['std_mm']:.1f} mm")
    print(f"  Median: {stats['offset_statistics']['median_mm']:.1f} mm")
    print(f"  Max:    {stats['offset_statistics']['max_mm']:.1f} mm")

    print(f"\n{BOLD}{CYAN}Spine Density Improvement:{NC}")
    print(f"  Mean:   {GREEN}{stats['improvement_statistics']['mean_ratio']:.2f}x{NC}")
    print(f"  Median: {GREEN}{stats['improvement_statistics']['median_ratio']:.2f}x{NC}")
    print(f"  Max:    {GREEN}{stats['improvement_statistics']['max_ratio']:.2f}x{NC}")

    print(f"\n{BOLD}{CYAN}Correction Distribution:{NC}")

    max_width = 50
    for key, value in stats['correction_needed'].items():
        pct = stats['correction_needed_percent'][key]
        bar_width = int((pct / 100) * max_width)
        bar = '█' * bar_width

        if 'no_correction' in key:
            color = GREEN
        elif 'small' in key:
            color = CYAN
        elif 'medium' in key:
            color = YELLOW
        else:
            color = RED

        label = key.replace('_', ' ').title()
        print(f"  {label:30s} {color}{bar}{NC} {value:3d} ({pct:5.1f}%)")

    needs_correction = (stats['correction_needed']['small_correction_1_5_voxels'] +
                       stats['correction_needed']['medium_correction_6_15_voxels'] +
                       stats['correction_needed']['large_correction_16plus_voxels'])
    pct_needs_correction = (needs_correction / total * 100) if total > 0 else 0

    print(f"\n{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{MAGENTA}SPINE-AWARE JUSTIFICATION:{NC}")
    print(f"{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{GREEN}{needs_correction} cases ({pct_needs_correction:.1f}%){NC} needed correction")

    if pct_needs_correction > 50:
        print(f"{BOLD}{GREEN}✓ STRONG EVIDENCE{NC} for spine-aware slicing (>50% benefit)")
    elif pct_needs_correction > 30:
        print(f"{BOLD}{GREEN}✓ GOOD EVIDENCE{NC} for spine-aware slicing (>30% benefit)")
    else:
        print(f"{BOLD}{YELLOW}⚠ MODERATE EVIDENCE{NC} (<30% benefit)")

    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

else
    print_warning "Spine-aware metrics not found"
fi

# ============================================================================
# 2. DETECTION METHOD COMPARISON (NEW v6.0)
# ============================================================================

if [[ -f "$OUTPUT_DIR/detection_method_comparison.json" ]]; then
    print_section "2. Detection Method Comparison (v6.0)"

    python3 << PYEOF
import json
import sys

BOLD = '\033[1m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
PURPLE = '\033[0;35m'
NC = '\033[0m'

try:
    output_dir = '${OUTPUT_DIR}'
    with open(f'{output_dir}/detection_method_comparison.json') as f:
        report = json.load(f)

    print(f"\n{BOLD}{PURPLE}┌─────────────────────────────────────────────────────────────┐{NC}")
    print(f"{BOLD}{PURPLE}│     INTENSITY (v6.1 TUNED) vs LABEL-BASED DETECTION         │{NC}")
    print(f"{BOLD}{PURPLE}└─────────────────────────────────────────────────────────────┘{NC}")

    for structure_key, display_name in [('t12_rib', 'T12 RIB'), ('l5_tp', 'L5 TRANSVERSE PROCESS')]:
        data = report[structure_key]
        total = data['total_attempts']
        
        if total == 0:
            continue

        print(f"\n{BOLD}{CYAN}{display_name}:{NC}")
        print(f"  Total attempts:        {total}")
        print()
        
        # Method success rates
        intensity_rate = data['intensity_success'] / total * 100
        label_rate = data['label_success'] / total * 100
        
        print(f"  {BOLD}Method Success Rates:{NC}")
        
        # Intensity bar
        int_bar_width = int(intensity_rate / 2)
        int_bar = '█' * int_bar_width
        print(f"    Intensity-based: {GREEN}{int_bar}{NC} {data['intensity_success']:3d} ({intensity_rate:5.1f}%)")
        
        # Label bar
        lbl_bar_width = int(label_rate / 2)
        lbl_bar = '█' * lbl_bar_width
        print(f"    Label-based:     {BLUE}{lbl_bar}{NC} {data['label_success']:3d} ({label_rate:5.1f}%)")
        
        print()
        
        # Breakdown
        print(f"  {BOLD}Detailed Breakdown:{NC}")
        print(f"    Both methods succeeded:  {data['both_success']:3d}")
        print(f"    {GREEN}Only intensity (NEW!):   {data['only_intensity']:3d}{NC} → {GREEN}+{data['improvement']:.1f}% gain{NC}")
        print(f"    Only label (fallback):   {data['only_label']:3d}")
        print(f"    {RED}Both failed:             {data['both_failed']:3d}{NC}")
        
        # Impact assessment
        improvement = data['only_intensity']
        if improvement > 0:
            print()
            print(f"  {BOLD}{GREEN}✓ INTENSITY-BASED IMPROVEMENT: {improvement} additional detections ({data['improvement']:.1f}%){NC}")

    print(f"\n{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{PURPLE}RECOMMENDATION:{NC}")
    
    total_improvement = (report['t12_rib']['only_intensity'] + report['l5_tp']['only_intensity'])
    
    if total_improvement > 0:
        print(f"{BOLD}{GREEN}✓ INTENSITY-BASED DETECTION IS WORKING!{NC}")
        print(f"  Detected {GREEN}{total_improvement} additional structures{NC} that labels missed")
        print(f"  {GREEN}→ Use intensity-based for full dataset{NC}")
    else:
        print(f"{BOLD}{YELLOW}⚠ No improvement over label-based{NC}")
        print(f"  Consider parameter tuning or fallback to labels only")
    
    print(f"{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")
    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

else
    print_warning "Detection method comparison not found"
fi

# ============================================================================
# 3. CLASS DISTRIBUTION
# ============================================================================

if [[ -f "$OUTPUT_DIR/weak_label_quality_report.json" ]]; then
    print_section "3. Class Distribution & Quality Metrics"

    python3 << PYEOF
import json
import sys

BOLD = '\033[1m'
GREEN = '\033[0;32m'
CYAN = '\033[0;36m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

try:
    output_dir = '${OUTPUT_DIR}'
    with open(f'{output_dir}/weak_label_quality_report.json') as f:
        report = json.load(f)

    print(f"\n{BOLD}{CYAN}Dataset Statistics:{NC}")
    print(f"  Total images:     {GREEN}{report['total_images']}{NC}")
    print(f"  Total boxes:      {GREEN}{report['total_boxes']}{NC}")
    print(f"  Avg boxes/image:  {GREEN}{report['avg_boxes_per_image']:.2f}{NC}")

    print(f"\n{BOLD}{CYAN}Class Distribution:{NC}")

    max_width = 40
    max_count = max(report['class_distribution'].values()) if report['class_distribution'] else 1

    # Sort by count descending
    sorted_classes = sorted(report['class_distribution'].items(),
                           key=lambda x: x[1], reverse=True)

    for class_name, count in sorted_classes:
        rate = count / report['total_images'] if report['total_images'] > 0 else 0
        bar_width = int((count / max_count) * max_width)
        bar = '█' * bar_width

        # Color code critical classes
        if 't12_rib' in class_name:
            color = YELLOW
            marker = '◆'
        elif 'l5_transverse' in class_name:
            color = YELLOW
            marker = '◆'
        else:
            color = GREEN
            marker = '●'

        print(f"  {marker} {class_name:25s} {color}{bar}{NC} {count:5d} ({rate*100:5.1f}%)")

    # Critical class analysis
    rib_rate = report.get('t12_rib_detection_rate', 0) * 100
    tp_rate = report.get('l5_transverse_process_detection_rate', 0) * 100

    print(f"\n{BOLD}{CYAN}Critical Structures (Lateral):{NC}")
    
    rib_color = GREEN if rib_rate > 70 else (YELLOW if rib_rate > 50 else RED)
    tp_color = GREEN if tp_rate > 70 else (YELLOW if tp_rate > 50 else RED)
    
    print(f"  T12 Rib:          {rib_color}{rib_rate:5.1f}%{NC}")
    print(f"  L5 Trans Process: {tp_color}{tp_rate:5.1f}%{NC}")
    
    if rib_rate > 70 and tp_rate > 70:
        print(f"\n  {BOLD}{GREEN}✓ EXCELLENT detection rates for lateral structures!{NC}")
    elif rib_rate > 50 or tp_rate > 50:
        print(f"\n  {BOLD}{YELLOW}⚠ MODERATE detection rates - room for improvement{NC}")
    else:
        print(f"\n  {BOLD}{RED}✗ LOW detection rates - needs investigation{NC}")

    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

else
    print_warning "Class distribution report not found"
fi

# ============================================================================
# OUTPUT FILE SUMMARY
# ============================================================================

print_section "Output Files"

COMPARISON_COUNT=$(find $OUTPUT_DIR/quality_validation -name "*_slice_comparison.png" 2>/dev/null | wc -l || echo 0)

print_stat "Dataset directory" "$OUTPUT_DIR"
print_stat "Dataset YAML" "$OUTPUT_DIR/dataset.yaml"
print_stat "Metadata" "$OUTPUT_DIR/metadata.json"

echo ""
echo -e "${BOLD}${CYAN}Validation Outputs:${NC}"

if [[ -f "$OUTPUT_DIR/spine_aware_metrics_report.json" ]]; then
    print_stat "  Spine-aware metrics" "spine_aware_metrics_report.json"
fi

if [[ -f "$OUTPUT_DIR/detection_method_comparison.json" ]]; then
    print_stat "  Detection comparison" "detection_method_comparison.json"
fi

if [[ -f "$OUTPUT_DIR/weak_label_quality_report.json" ]]; then
    print_stat "  Quality report" "weak_label_quality_report.json"
fi

echo ""
echo -e "${BOLD}${CYAN}Visualizations:${NC}"

if [[ -f "$OUTPUT_DIR/quality_validation_summary.png" ]]; then
    print_stat "  Spine-aware summary" "quality_validation_summary.png"
fi

if [[ -f "$OUTPUT_DIR/detection_method_comparison.png" ]]; then
    print_stat "  Detection comparison" "detection_method_comparison.png"
fi

if [[ $COMPARISON_COUNT -gt 0 ]]; then
    print_stat "  Comparison images" "$COMPARISON_COUNT in quality_validation/"
fi

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print_header "FINAL RECOMMENDATIONS"

OUTPUT_DIR_FOR_PYTHON="${OUTPUT_DIR}"

python3 << PYEOF
import json
import sys
import os
from pathlib import Path

BOLD = '\033[1m'
GREEN = '\033[0;32m'
CYAN = '\033[0;36m'
YELLOW = '\033[1;33m'
PURPLE = '\033[0;35m'
NC = '\033[0m'

output_dir = Path('${OUTPUT_DIR}')

try:
    # Load all reports
    spine_metrics = None
    detection_metrics = None
    quality_metrics = None
    
    if (output_dir / 'spine_aware_metrics_report.json').exists():
        with open(output_dir / 'spine_aware_metrics_report.json') as f:
            spine_metrics = json.load(f)
    
    if (output_dir / 'detection_method_comparison.json').exists():
        with open(output_dir / 'detection_method_comparison.json') as f:
            detection_metrics = json.load(f)
    
    if (output_dir / 'weak_label_quality_report.json').exists():
        with open(output_dir / 'weak_label_quality_report.json') as f:
            quality_metrics = json.load(f)
    
    print(f"{BOLD}Trial Validation Results:{NC}\n")
    
    # Spine-aware analysis
    if spine_metrics:
        total = spine_metrics['total_cases']
        needs_correction = (
            spine_metrics['correction_needed']['small_correction_1_5_voxels'] +
            spine_metrics['correction_needed']['medium_correction_6_15_voxels'] +
            spine_metrics['correction_needed']['large_correction_16plus_voxels']
        )
        pct_needs = (needs_correction / total * 100) if total > 0 else 0
        
        print(f"{BOLD}1. Spine-Aware Slicing:{NC}")
        print(f"   • {pct_needs:.1f}% of cases needed correction")
        print(f"   • Mean improvement: {spine_metrics['improvement_statistics']['mean_ratio']:.2f}x")
        
        if pct_needs > 50:
            print(f"   {GREEN}✓ STRONGLY RECOMMENDED for full dataset{NC}")
        elif pct_needs > 30:
            print(f"   {GREEN}✓ RECOMMENDED for full dataset{NC}")
        else:
            print(f"   {YELLOW}⚠ Optional - moderate benefit{NC}")
        print()
    
    # Detection method analysis
    if detection_metrics:
        rib_improvement = detection_metrics['t12_rib']['only_intensity']
        tp_improvement = detection_metrics['l5_tp']['only_intensity']
        total_improvement = rib_improvement + tp_improvement
        
        print(f"{BOLD}2. Intensity-Based Detection (NEW v6.0):{NC}")
        print(f"   • T12 ribs: +{rib_improvement} additional detections")
        print(f"   • L5 TPs:   +{tp_improvement} additional detections")
        print(f"   • Total:    {GREEN}+{total_improvement} structures detected{NC}")
        
        if total_improvement > 0:
            print(f"   {GREEN}✓ INTENSITY-BASED DETECTION WORKING!{NC}")
            print(f"   {GREEN}✓ USE FOR FULL DATASET{NC}")
        else:
            print(f"   {YELLOW}⚠ No improvement - consider label-based only{NC}")
        print()
    
    # Quality analysis
    if quality_metrics:
        rib_rate = quality_metrics.get('t12_rib_detection_rate', 0) * 100
        tp_rate = quality_metrics.get('l5_transverse_process_detection_rate', 0) * 100
        
        print(f"{BOLD}3. Overall Detection Quality:{NC}")
        print(f"   • T12 rib detection:  {rib_rate:.1f}%")
        print(f"   • L5 TP detection:    {tp_rate:.1f}%")
        
        if rib_rate > 70 and tp_rate > 70:
            print(f"   {GREEN}✓ EXCELLENT - Ready for training{NC}")
        elif rib_rate > 50 or tp_rate > 50:
            print(f"   {YELLOW}⚠ MODERATE - Acceptable for baseline{NC}")
        else:
            print(f"   {YELLOW}⚠ LOW - May need parameter tuning{NC}")
        print()
    
    # Next steps
    print(f"{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{PURPLE}NEXT STEPS:{NC}")
    print(f"{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")
    print()
    print(f"{BOLD}1. Review Visualizations:{NC}")
    print(f"   {CYAN}cd {output_dir}{NC}")
    print(f"   {CYAN}# Check: quality_validation_summary.png{NC}")
    print(f"   {CYAN}# Check: detection_method_comparison.png{NC}")
    print(f"   {CYAN}# Browse: quality_validation/*.png{NC}")
    print()
    print(f"{BOLD}2. Run Full Dataset:{NC}")
    print(f"   {CYAN}# Remove trial data and run on full set{NC}")
    print(f"   {CYAN}sbatch slurm_scripts/06_generate_weak_labels_full.sh{NC}")
    print()
    print(f"{BOLD}3. Train Baseline Model:{NC}")
    print(f"   {CYAN}sbatch slurm_scripts/07_train_yolo_baseline.sh{NC}")
    print()

except Exception as e:
    print(f"Error generating recommendations: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
PYEOF

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
print_header "JOB COMPLETE"
echo -e "${CYAN}End:${NC}    $(date)"
echo -e "${CYAN}Output:${NC} ${GREEN}$OUTPUT_DIR${NC}"
echo ""
echo -e "${BOLD}${CYAN}Review outputs and proceed to full dataset if results look good!${NC}"
echo ""
