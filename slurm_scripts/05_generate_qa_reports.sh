#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=lstv_qa_reports
#SBATCH -o logs/lstv_qa_reports_%j.out
#SBATCH -e logs/lstv_qa_reports_%j.err

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

print_header "LSTV QA REPORT GENERATION"
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
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/trial/qa_reports"

IMG_PATH="${HOME}/singularity_cache/spine-level-ai-spineps.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    print_error "SPINEPS container not found: $IMG_PATH"
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
# QA REPORT GENERATION
# ============================================================================

print_section "Generating QA Reports with Confidence Scoring"

echo ""
echo -e "${BOLD}Features:${NC}"
echo -e "  ${GREEN}●${NC} Vertebra label overlays (L1-L6, Sacrum)"
echo -e "  ${GREEN}●${NC} LSTV confidence scoring (HIGH/MEDIUM/LOW)"
echo -e "  ${GREEN}●${NC} Upload recommendations"
echo -e "  ${GREEN}●${NC} Multi-view visualization (3 slices)"
echo -e "  ${GREEN}●${NC} PDF reports for manual review"
echo ""

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $NIFTI_DIR:/data/nifti \
    --bind $SEG_DIR:/data/seg \
    --bind $OUTPUT_DIR:/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/screening/visualize_lstv_labels.py \
        --nifti_dir /data/nifti \
        --seg_dir /data/seg \
        --output_dir /data/output \
        --limit 5

exit_code=$?

if [ $exit_code -ne 0 ]; then
    print_error "QA report generation failed (exit code: $exit_code)"
    exit $exit_code
fi

print_success "QA reports generated!"

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

print_header "QA SUMMARY ANALYSIS"

if [[ -f "$OUTPUT_DIR/qa_summary.json" ]]; then
    print_section "Confidence Breakdown"

    python3 << PYEOF
import json
import sys

BOLD = '\033[1m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
RED = '\033[0;31m'
PURPLE = '\033[0;35m'
NC = '\033[0m'

try:
    output_dir = '${OUTPUT_DIR}'
    with open(f'{output_dir}/qa_summary.json') as f:
        summary = json.load(f)

    total = summary['total_studies']
    lstv = summary['lstv_candidates']
    high = summary['high_confidence']
    medium = summary['medium_confidence']
    low = summary['low_confidence']

    print(f"\n{BOLD}{CYAN}Overview:{NC}")
    print(f"  Total studies:    {total}")
    print(f"  LSTV candidates:  {lstv} ({lstv/total*100:.1f}%)")

    print(f"\n{BOLD}{PURPLE}┌─────────────────────────────────────────────────────────────┐{NC}")
    print(f"{BOLD}{PURPLE}│               CONFIDENCE SCORING RESULTS                    │{NC}")
    print(f"{BOLD}{PURPLE}└─────────────────────────────────────────────────────────────┘{NC}")

    print(f"\n{BOLD}{GREEN}HIGH CONFIDENCE ({high} cases):{NC}")
    print(f"  ✅ UPLOAD TO ROBOFLOW")
    print(f"  Strong evidence for LSTV")
    print(f"  Good training data quality")

    print(f"\n{BOLD}{YELLOW}MEDIUM CONFIDENCE ({medium} cases):{NC}")
    print(f"  ⚠ MANUAL REVIEW REQUIRED")
    print(f"  Moderate evidence")
    print(f"  Check PDF reports before upload")

    print(f"\n{BOLD}{RED}LOW CONFIDENCE ({low} cases):{NC}")
    print(f"  ❌ DO NOT UPLOAD")
    print(f"  Likely false positives")
    print(f"  Reject from training set")

    print(f"\n{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{PURPLE}ROBOFLOW UPLOAD STRATEGY:{NC}")
    print(f"{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")

    print(f"\n{BOLD}Phase 1 - Immediate Upload ({high} cases):{NC}")
    print(f"  Upload HIGH confidence cases immediately")
    print(f"  Expected precision: >90%")

    print(f"\n{BOLD}Phase 2 - Manual Review ({medium} cases):{NC}")
    print(f"  Review PDF reports in: {output_dir}")
    print(f"  Upload verified cases")
    print(f"  Expected precision: 70-90% → filter to >90%")

    print(f"\n{BOLD}Phase 3 - Active Learning:{NC}")
    print(f"  Train baseline on HIGH confidence")
    print(f"  Use YOLO to verify MEDIUM confidence")
    print(f"  Re-label disagreements")

    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

else
    print_warning "Summary JSON not found"
fi

# ============================================================================
# OUTPUT FILES
# ============================================================================

print_section "Output Files"

PDF_COUNT=$(find $OUTPUT_DIR -name "*_QA_report.pdf" 2>/dev/null | wc -l || echo 0)

print_stat "QA reports (PDF)" "$PDF_COUNT"
print_stat "Summary JSON" "qa_summary.json"

echo ""
echo -e "${BOLD}${CYAN}View reports:${NC}"
echo -e "  ${CYAN}ls -lh ${OUTPUT_DIR}/*.pdf${NC}"
echo -e "  ${CYAN}cat ${OUTPUT_DIR}/qa_summary.json${NC}"

echo ""
echo -e "${BOLD}${CYAN}Example PDFs to review:${NC}"
find $OUTPUT_DIR -name "*_QA_report.pdf" 2>/dev/null | head -3 | while read pdf; do
    echo -e "  ${CYAN}$(basename $pdf)${NC}"
done

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
print_header "JOB COMPLETE"
echo -e "${CYAN}End:${NC}    $(date)"
echo -e "${CYAN}Output:${NC} ${GREEN}$OUTPUT_DIR${NC}"
echo ""
echo -e "${BOLD}${CYAN}Next steps:${NC}"
echo -e "  1. Review HIGH confidence PDFs"
echo -e "  2. Upload to Roboflow (filter by confidence)"
echo -e "  3. Train baseline model"
echo ""
