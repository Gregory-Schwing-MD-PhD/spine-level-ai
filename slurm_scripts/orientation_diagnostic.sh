#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=2:00:00
#SBATCH --job-name=orientation_diagnostic
#SBATCH -o logs/orientation_diag_%j.out
#SBATCH -e logs/orientation_diag_%j.err

#===============================================================================
# ORIENTATION DIAGNOSTIC - Check CSV Labels vs Actual DICOM Orientations
#
# This script:
#   1. Converts 10 studies worth of series to NIfTI
#   2. Checks actual DICOM orientation
#   3. Compares to CSV labels
#   4. Creates diagnostic images showing mismatches
#
# Expected time: ~10-15 minutes
#===============================================================================

set -euo pipefail

# ============================================================================
# COLOR CODES & HELPER FUNCTIONS
# ============================================================================

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    local text="$1"
    echo -e "\n${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║ ${text}${NC}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${CYAN}ℹ${NC} $1"; }

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

print_header "ORIENTATION DIAGNOSTIC - CSV vs DICOM Check"

echo -e "${CYAN}Job Information:${NC}"
echo -e "  Job ID:    ${SLURM_JOB_ID}"
echo -e "  Start:     $(date)"
echo -e "  Node:      $(hostname)"
echo -e "  GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
echo ""

# Critical: Set up Singularity environment
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export SINGULARITY_CACHEDIR="${HOME}/singularity_cache"

mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

# Python environment
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH

# Verify Singularity
if ! command -v singularity &> /dev/null; then
    print_error "Singularity not found!"
    exit 1
fi
print_success "Singularity: $(which singularity)"

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"

# Validate inputs
if [[ ! -d "$DATA_DIR" ]]; then
    print_error "Data directory not found: ${DATA_DIR}"
    exit 1
fi

if [[ ! -f "$SERIES_CSV" ]]; then
    print_error "Series CSV not found: ${SERIES_CSV}"
    exit 1
fi

print_success "Data directory: ${DATA_DIR}"
print_success "Series CSV: ${SERIES_CSV}"

# Output directory
OUTPUT_DIR="${PROJECT_DIR}/results/orientation_diagnostic"
mkdir -p "$OUTPUT_DIR"

print_info "Output directory: ${OUTPUT_DIR}"

# Container
CONTAINER_NAME="spine-level-ai-spineps"
IMG_PATH="${SINGULARITY_CACHEDIR}/${CONTAINER_NAME}.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    print_warning "Container not found, pulling..."
    singularity pull "$IMG_PATH" "docker://go2432/${CONTAINER_NAME}:latest"
fi
print_success "Container: ${CONTAINER_NAME}.sif"

echo ""
echo -e "${BOLD}Configuration:${NC}"
echo -e "  Studies to check:    10"
echo -e "  Input directory:     ${DATA_DIR}"
echo -e "  Series CSV:          ${SERIES_CSV}"
echo -e "  Output directory:    ${OUTPUT_DIR}"
echo ""

# ============================================================================
# RUN DIAGNOSTIC
# ============================================================================

print_header "Running Orientation Diagnostic"

singularity exec --nv \
    --bind "${PROJECT_DIR}:/work" \
    --bind "${DATA_DIR}:/data/input:ro" \
    --bind "${OUTPUT_DIR}:/data/output:rw" \
    --bind "$(dirname $SERIES_CSV):/data/raw:ro" \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/screening/diagnose_series_orientation.py \
        --input_dir /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --num_studies 10

if [ $? -eq 0 ]; then
    print_success "Diagnostic complete!"
else
    print_error "Diagnostic failed!"
    exit 1
fi

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print_header "Diagnostic Results"

# Check if results exist
RESULTS_CSV="${OUTPUT_DIR}/diagnostic_results.csv"
IMAGES_DIR="${OUTPUT_DIR}/diagnostic_images"

if [[ -f "$RESULTS_CSV" ]]; then
    echo ""
    echo -e "${BOLD}Analysis:${NC}"
    
    # Count mismatches using Python
    python3 << 'PYEOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('${RESULTS_CSV}')
    
    total = len(df)
    mismatches = df['mismatch'].sum()
    mismatch_pct = mismatches / total * 100 if total > 0 else 0
    
    print(f"  Total series checked:  {total}")
    print(f"  Mismatches found:      {mismatches} ({mismatch_pct:.1f}%)")
    print()
    
    if mismatches > 0:
        print("CSV LABEL ACCURACY:")
        for csv_desc in df['csv_description'].unique():
            subset = df[df['csv_description'] == csv_desc]
            subset_mismatches = subset['mismatch'].sum()
            subset_pct = (len(subset) - subset_mismatches) / len(subset) * 100 if len(subset) > 0 else 0
            print(f"  '{csv_desc}':")
            print(f"    Accuracy: {subset_pct:.1f}% ({len(subset) - subset_mismatches}/{len(subset)} correct)")
            
            # Show what they actually are
            if subset_mismatches > 0:
                actual_types = subset['actual_type'].value_counts()
                print(f"    Actually: {dict(actual_types)}")
    else:
        print("✓ All CSV labels match DICOM orientations!")
    
except Exception as e:
    print(f"Error reading results: {e}")
    sys.exit(1)
PYEOF
    
    echo ""
    echo -e "${BOLD}Output Files:${NC}"
    echo "  Results CSV:        ${RESULTS_CSV}"
    echo "  Diagnostic images:  ${IMAGES_DIR}/"
    echo ""
    
    # Count images
    NUM_IMAGES=$(ls -1 "${IMAGES_DIR}"/*.jpg 2>/dev/null | wc -l)
    echo "  Images created:     ${NUM_IMAGES}"
    echo ""
    
    print_info "Review images in: ${IMAGES_DIR}/"
    print_info "Each image shows CSV label vs actual DICOM orientation"
    
else
    print_warning "Results CSV not found: ${RESULTS_CSV}"
fi

echo ""
echo -e "${BOLD}Next Steps:${NC}"
echo ""
echo "  1. REVIEW DIAGNOSTIC IMAGES"
echo "     cd ${IMAGES_DIR}"
echo "     # View images to visually confirm mismatches"
echo ""
echo "  2. If CSV labels are WRONG:"
echo "     → Update lstv_screen_production_COMPLETE.py to try all series"
echo "     → Already done in latest version - it tries multiple series"
echo ""
echo "  3. If CSV labels are CORRECT but orientation detection is wrong:"
echo "     → May need to adjust orientation detection logic"
echo "     → Check DICOM tags for acquisition plane"
echo ""

echo -e "${BOLD}End time:${NC} $(date)"
ELAPSED=$((SECONDS / 60))
echo -e "${BOLD}Total time:${NC} ${ELAPSED} minutes"
echo ""

print_success "Diagnostic complete - review images to verify!"
