#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=weak_labels_v5_trial
#SBATCH -o logs/weak_labels_v5_%j.out
#SBATCH -e logs/weak_labels_v5_%j.err

set -euo pipefail

# ============================================================================
# ANSI COLOR CODES FOR BEAUTIFUL OUTPUT
# ============================================================================
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Box drawing characters
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

print_header "WEAK LABEL GENERATION v5.0 - VALIDATION TRIAL"
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
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_v5_trial"

IMG_PATH="${HOME}/singularity_cache/spine-level-ai-yolo.sif"

# Validation
if [[ ! -f "$IMG_PATH" ]]; then
    print_error "YOLOv11 container not found: $IMG_PATH"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

if [[ ! -d "$NIFTI_DIR" ]]; then
    print_error "NIfTI directory not found: $NIFTI_DIR"
    echo "Run trial screening first: sbatch slurm_scripts/04_lstv_screen_trial.sh"
    exit 1
fi

NIFTI_COUNT=$(find $NIFTI_DIR -name "*.nii.gz" | wc -l)
SEG_COUNT=$(find $SEG_DIR -name "*_seg.nii.gz" | wc -l)

print_success "Container: $IMG_PATH"
print_stat "NIfTI files" "$NIFTI_COUNT"
print_stat "Segmentation files" "$SEG_COUNT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# ============================================================================
# LABEL GENERATION
# ============================================================================

print_section "Label Generation (v5.0 with Validation)"

echo ""
echo -e "${BOLD}${MAGENTA}Features Enabled:${NC}"
echo -e "  ${GREEN}✓${NC} Thick Slab MIP (15mm ribs, 5mm midline)"
echo -e "  ${GREEN}✓${NC} Spine-aware intelligent slice selection"
echo -e "  ${GREEN}✓${NC} Robust T12 rib detection (morphological)"
echo -e "  ${GREEN}✓${NC} Robust L5 TP detection (bilateral)"
echo -e "  ${GREEN}✓${NC} ${YELLOW}VALIDATION MODE${NC}: Before/after comparisons"
echo -e "  ${GREEN}✓${NC} ${YELLOW}VALIDATION MODE${NC}: Quantitative metrics"
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
        --generate_comparisons \
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

# Function to parse and display JSON metrics
display_json_report() {
    local json_file="$1"
    local report_title="$2"
    
    if [[ ! -f "$json_file" ]]; then
        print_warning "Report not found: $json_file"
        return 1
    fi
    
    print_section "$report_title"
    
    python3 << PYEOF
import json
import sys

try:
    with open('$json_file') as f:
        data = json.load(f)
    
    # Pretty print with colors (will be colored by bash)
    print(json.dumps(data, indent=2))
    
except Exception as e:
    print(f"Error parsing JSON: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ============================================================================
# SPINE-AWARE METRICS VISUALIZATION
# ============================================================================

if [[ -f "$OUTPUT_DIR/spine_aware_metrics_report.json" ]]; then
    print_section "Spine-Aware Slice Selection Metrics"
    
    python3 << 'PYEOF'
import json
import sys

# ANSI colors
BOLD = '\033[1m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
MAGENTA = '\033[0;35m'
RED = '\033[0;31m'
NC = '\033[0m'

try:
    with open('$OUTPUT_DIR/spine_aware_metrics_report.json') as f:
        stats = json.load(f)

    total = stats['total_cases']
    
    # Summary statistics
    print(f"\n{BOLD}{CYAN}Overview:{NC}")
    print(f"  Total cases:          {GREEN}{total}{NC}")
    print(f"  Spine-aware success:  {GREEN}{stats['spine_aware_cases']}{NC} ({stats['spine_aware_cases']/total*100:.1f}%)")
    print(f"  Geometric fallback:   {YELLOW}{stats['geometric_fallback_cases']}{NC} ({stats['geometric_fallback_cases']/total*100:.1f}%)")
    
    # Offset statistics
    print(f"\n{BOLD}{CYAN}Offset from Geometric Center:{NC}")
    print(f"  Mean:   {stats['offset_statistics']['mean_mm']:.1f} ± {stats['offset_statistics']['std_mm']:.1f} mm")
    print(f"  Median: {stats['offset_statistics']['median_mm']:.1f} mm")
    print(f"  Max:    {stats['offset_statistics']['max_mm']:.1f} mm")
    
    # Improvement statistics
    print(f"\n{BOLD}{CYAN}Spine Density Improvement:{NC}")
    print(f"  Mean:   {GREEN}{stats['improvement_statistics']['mean_ratio']:.2f}x{NC}")
    print(f"  Median: {GREEN}{stats['improvement_statistics']['median_ratio']:.2f}x{NC}")
    print(f"  Max:    {GREEN}{stats['improvement_statistics']['max_ratio']:.2f}x{NC}")
    
    # Correction distribution with visual bars
    print(f"\n{BOLD}{CYAN}Correction Distribution:{NC}")
    
    max_width = 50
    for key, value in stats['correction_needed'].items():
        pct = stats['correction_needed_percent'][key]
        bar_width = int((pct / 100) * max_width)
        bar = '█' * bar_width
        
        # Color code by severity
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
    
    # Justification analysis
    needs_correction = (stats['correction_needed']['small_correction_1_5_voxels'] + 
                       stats['correction_needed']['medium_correction_6_15_voxels'] + 
                       stats['correction_needed']['large_correction_16plus_voxels'])
    pct_needs_correction = (needs_correction / total * 100) if total > 0 else 0
    
    print(f"\n{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{MAGENTA}JUSTIFICATION FOR SPINE-AWARE SLICING:{NC}")
    print(f"{BOLD}{MAGENTA}═══════════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{GREEN}{needs_correction} cases ({pct_needs_correction:.1f}%){NC} needed correction!")
    
    if pct_needs_correction > 50:
        print(f"{BOLD}{GREEN}✓ STRONG JUSTIFICATION:{NC} >50% of cases benefit from correction")
        print(f"  Spine-aware slicing is {BOLD}ESSENTIAL{NC} for full dataset!")
    elif pct_needs_correction > 30:
        print(f"{BOLD}{YELLOW}✓ GOOD JUSTIFICATION:{NC} >30% of cases benefit from correction")
        print(f"  Spine-aware slicing is {BOLD}RECOMMENDED{NC} for full dataset")
    else:
        print(f"{BOLD}{YELLOW}⚠ MODERATE JUSTIFICATION:{NC} <30% of cases need correction")
        print(f"  Benefits present but less pronounced")
    
    # Expected impact
    mean_improvement = stats['improvement_statistics']['mean_ratio']
    improvement_pct = (mean_improvement - 1.0) * 100
    
    print(f"\n{BOLD}{CYAN}Expected Impact on Detection:{NC}")
    print(f"  Spine visibility improvement: {GREEN}+{improvement_pct:.1f}%{NC}")
    print(f"  Estimated T12 rib AP gain:   {GREEN}+{improvement_pct*0.5:.1f}% to +{improvement_pct:.1f}%{NC}")
    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

else
    print_warning "Spine-aware metrics not found (validation mode not enabled)"
fi

# ============================================================================
# CLASS DISTRIBUTION METRICS
# ============================================================================

if [[ -f "$OUTPUT_DIR/weak_label_quality_report.json" ]]; then
    print_section "Class Distribution Metrics"
    
    python3 << 'PYEOF'
import json
import sys

BOLD = '\033[1m'
GREEN = '\033[0;32m'
CYAN = '\033[0;36m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

try:
    with open('$OUTPUT_DIR/weak_label_quality_report.json') as f:
        report = json.load(f)
    
    print(f"\n{BOLD}{CYAN}Dataset Statistics:{NC}")
    print(f"  Total images:     {GREEN}{report['total_images']}{NC}")
    print(f"  Total boxes:      {GREEN}{report['total_boxes']}{NC}")
    print(f"  Avg boxes/image:  {GREEN}{report['avg_boxes_per_image']:.2f}{NC}")
    
    print(f"\n{BOLD}{CYAN}Class Distribution:{NC}")
    
    max_width = 40
    max_count = max(report['class_distribution'].values()) if report['class_distribution'] else 1
    
    for class_name, count in sorted(report['class_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
        rate = count / report['total_images'] if report['total_images'] > 0 else 0
        bar_width = int((count / max_count) * max_width)
        bar = '█' * bar_width
        
        # Color code critical classes
        if 't12_rib' in class_name:
            color = YELLOW
        else:
            color = GREEN
        
        print(f"  {class_name:25s} {color}{bar}{NC} {count:5d} ({rate*100:5.1f}%)")
    
    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

else
    print_warning "Class distribution report not found"
fi

# ============================================================================
# OUTPUT SUMMARY
# ============================================================================

print_section "Output Files"

COMPARISON_COUNT=$(find $OUTPUT_DIR/quality_validation -name "*_slice_comparison.png" 2>/dev/null | wc -l || echo 0)

print_stat "Dataset directory" "$OUTPUT_DIR"
print_stat "Dataset YAML" "$OUTPUT_DIR/dataset.yaml"
print_stat "Metadata" "$OUTPUT_DIR/metadata.json"

if [[ -f "$OUTPUT_DIR/spine_aware_metrics_report.json" ]]; then
    print_stat "Spine-aware metrics" "$OUTPUT_DIR/spine_aware_metrics_report.json"
fi

if [[ -f "$OUTPUT_DIR/weak_label_quality_report.json" ]]; then
    print_stat "Quality report" "$OUTPUT_DIR/weak_label_quality_report.json"
fi

if [[ -f "$OUTPUT_DIR/quality_validation_summary.png" ]]; then
    print_stat "Summary visualization" "$OUTPUT_DIR/quality_validation_summary.png"
fi

if [[ $COMPARISON_COUNT -gt 0 ]]; then
    print_stat "Comparison images" "$COMPARISON_COUNT in $OUTPUT_DIR/quality_validation/"
fi

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print_header "RECOMMENDATIONS"

python3 << 'PYEOF'
import json
import sys

BOLD = '\033[1m'
GREEN = '\033[0;32m'
CYAN = '\033[0;36m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

spine_metrics_file = '$OUTPUT_DIR/spine_aware_metrics_report.json'

try:
    with open(spine_metrics_file) as f:
        stats = json.load(f)
    
    total = stats['total_cases']
    needs_correction = (stats['correction_needed']['small_correction_1_5_voxels'] + 
                       stats['correction_needed']['medium_correction_6_15_voxels'] + 
                       stats['correction_needed']['large_correction_16plus_voxels'])
    pct_needs_correction = (needs_correction / total * 100) if total > 0 else 0
    mean_improvement = stats['improvement_statistics']['mean_ratio']
    
    print(f"{BOLD}Trial Results ({total} cases):{NC}")
    print(f"  • {pct_needs_correction:.1f}% of cases needed correction")
    print(f"  • Mean spine visibility improved {GREEN}{mean_improvement:.2f}x{NC}")
    print(f"  • Median offset: {stats['offset_statistics']['median_mm']:.1f}mm")
    print()
    
    if pct_needs_correction > 50:
        print(f"{BOLD}{GREEN}✓ PROCEED WITH FULL RUN{NC}")
        print(f"  {GREEN}Strong evidence{NC} for spine-aware slicing")
        print(f"  Expected to significantly improve T12 rib detection")
    elif pct_needs_correction > 30:
        print(f"{BOLD}{GREEN}✓ PROCEED WITH FULL RUN{NC}")
        print(f"  {GREEN}Good evidence{NC} for spine-aware slicing")
        print(f"  Moderate expected improvement in detection")
    else:
        print(f"{BOLD}{YELLOW}⚠ EVALUATE CAREFULLY{NC}")
        print(f"  {YELLOW}Moderate evidence{NC} - benefits present but less pronounced")
        print(f"  Consider cost/benefit of additional computation")
    
    print(f"\n{BOLD}Next Steps:{NC}")
    print(f"  1. Review comparison images: {CYAN}$OUTPUT_DIR/quality_validation/{NC}")
    print(f"  2. Check summary plot: {CYAN}$OUTPUT_DIR/quality_validation_summary.png{NC}")
    print(f"  3. Run full dataset with: {YELLOW}--generate_comparisons{NC} flag removed")
    print(f"  4. Train baseline: {CYAN}sbatch slurm_scripts/07_train_yolo_baseline.sh{NC}")
    
except FileNotFoundError:
    print(f"{YELLOW}⚠ Validation metrics not available{NC}")
    print(f"Re-run with --generate_comparisons flag for detailed analysis")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
PYEOF

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
print_header "JOB COMPLETE"
echo -e "${CYAN}End:${NC}    $(date)"
echo -e "${CYAN}Output:${NC} ${GREEN}$OUTPUT_DIR${NC}"
echo ""
