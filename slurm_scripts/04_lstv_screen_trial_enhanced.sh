#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=02:00:00
#SBATCH --job-name=lstv_trial_enhanced
#SBATCH -o logs/lstv_trial_enhanced_%j.out
#SBATCH -e logs/lstv_trial_enhanced_%j.err

set -euo pipefail
set -x

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

print_stat() {
    local label="$1"
    local value="$2"
    printf "${BOLD}%-35s${NC}: ${GREEN}%s${NC}\n" "$label" "$value"
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

print_header "LSTV SCREENING - ENHANCED v2.0 (TRIAL)"
echo -e "${CYAN}Job ID:${NC} $SLURM_JOB_ID"
echo -e "${CYAN}Start:${NC}  $(date)"
echo -e "${CYAN}GPU:${NC}    $CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi

print_section "Environment Setup"

# Singularity temp setup
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

# Cleanup on exit
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

if ! command -v singularity &> /dev/null; then
    print_error "Singularity not found!"
    exit 1
fi
print_success "Singularity: $(which singularity)"

export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

print_section "Path Configuration"

PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/trial_enhanced"
NIFTI_DIR="${OUTPUT_DIR}/nifti"
SCRIPT_PATH="${PROJECT_DIR}/src/screening/lstv_screen_enhanced.py"
MODELS_CACHE="${PROJECT_DIR}/spineps_models"

mkdir -p $MODELS_CACHE
mkdir -p $OUTPUT_DIR/logs
mkdir -p $NIFTI_DIR

DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-spineps:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spine-level-ai-spineps.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling SPINEPS container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

print_success "Container: $IMG_PATH"
print_stat "Data directory" "$DATA_DIR"
print_stat "Output directory" "$OUTPUT_DIR"

ROBOFLOW_KEY="izolWNqCVveKyMrACYzN"
ROBOFLOW_WORKSPACE="spinelevelai"
ROBOFLOW_PROJECT="lstv-candidates"

# ============================================================================
# FEATURE SHOWCASE
# ============================================================================

print_section "Enhanced v2.0 Features"

echo ""
echo -e "${BOLD}${PURPLE}┌─────────────────────────────────────────────────────────────┐${NC}"
echo -e "${BOLD}${PURPLE}│                    NEW IN v2.0                              │${NC}"
echo -e "${BOLD}${PURPLE}├─────────────────────────────────────────────────────────────┤${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${GREEN}●${NC} ${BOLD}Integrated Confidence Scoring${NC}                        ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → HIGH/MEDIUM/LOW classification                      ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → L6 size validation (0.5-1.5x L5)                    ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Sacrum presence check                               ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → S1-S2 disc detection                                ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}                                                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${GREEN}●${NC} ${BOLD}QA Images with Labels${NC}                                ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Vertebra labels overlaid (L1-L6, Sacrum)            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Color-coded by significance                         ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Confidence banner on each image                     ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}                                                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${YELLOW}●${NC} ${BOLD}Smart Upload Filtering${NC}                               ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Auto-upload HIGH confidence (≥0.7)                  ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Skip MEDIUM/LOW (manual review)                     ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Prevents false positives in training data          ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}                                                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}  ${CYAN}●${NC} ${BOLD}Comprehensive Reporting${NC}                               ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Confidence breakdown in progress.json               ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Per-case confidence factors                         ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}│${NC}    → Upload decision tracking                            ${BOLD}${PURPLE}│${NC}"
echo -e "${BOLD}${PURPLE}└─────────────────────────────────────────────────────────────┘${NC}"
echo ""

# ============================================================================
# SCREENING
# ============================================================================

print_section "Running Enhanced LSTV Screening"

echo ""
echo -e "${BOLD}Command:${NC}"
echo -e "  ${CYAN}python lstv_screen_enhanced.py${NC}"
echo -e "    ${YELLOW}--confidence_threshold 0.7${NC}  (Auto-upload HIGH only)"
echo -e "    ${YELLOW}--generate_three_views${NC}      (3-view with QA labels)"
echo ""

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/input \
    --bind $OUTPUT_DIR:/data/output \
    --bind $NIFTI_DIR:/data/output/nifti \
    --bind $MODELS_CACHE:/app/models \
    --bind $(dirname $SERIES_CSV):/data/raw \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/screening/lstv_screen_enhanced.py \
        --input_dir /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --limit 5 \
        --roboflow_key "$ROBOFLOW_KEY" \
        --roboflow_workspace "$ROBOFLOW_WORKSPACE" \
        --roboflow_project "$ROBOFLOW_PROJECT" \
        --generate_three_views \
        --confidence_threshold 0.7 \
        --verbose

exit_code=$?

if [ $exit_code -ne 0 ]; then
    print_error "Screening failed (exit code: $exit_code)"
    exit $exit_code
fi

print_success "Screening complete!"

# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

print_header "CONFIDENCE ANALYSIS"

if [[ -f "$OUTPUT_DIR/progress.json" ]]; then
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
    with open(f'{output_dir}/progress.json') as f:
        progress = json.load(f)

    total_processed = len(progress['processed'])
    high = len(progress.get('high_confidence', []))
    medium = len(progress.get('medium_confidence', []))
    low = len(progress.get('low_confidence', []))
    flagged = len(progress.get('flagged', []))

    print(f"\n{BOLD}{CYAN}Overview:{NC}")
    print(f"  Total processed:  {total_processed}")
    print(f"  LSTV candidates:  {high + medium + low}")
    print(f"  Uploaded:         {flagged}")

    print(f"\n{BOLD}{PURPLE}┌─────────────────────────────────────────────────────────────┐{NC}")
    print(f"{BOLD}{PURPLE}│               CONFIDENCE DISTRIBUTION                       │{NC}")
    print(f"{BOLD}{PURPLE}└─────────────────────────────────────────────────────────────┘{NC}")

    print(f"\n{BOLD}{GREEN}HIGH CONFIDENCE ({high}):{NC}")
    print(f"  ✅ AUTO-UPLOADED TO ROBOFLOW")
    print(f"  Strong evidence for LSTV")
    print(f"  Expected precision: >90%")

    print(f"\n{BOLD}{YELLOW}MEDIUM CONFIDENCE ({medium}):{NC}")
    print(f"  ⚠ SAVED FOR MANUAL REVIEW")
    print(f"  Check QA images before upload")
    print(f"  Expected precision: 70-90%")

    print(f"\n{BOLD}{RED}LOW CONFIDENCE ({low}):{NC}")
    print(f"  ❌ REJECTED (not uploaded)")
    print(f"  Likely false positives")
    print(f"  Expected precision: <50%")

    print(f"\n{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")
    print(f"{BOLD}{PURPLE}ROBOFLOW QUALITY CONTROL:{NC}")
    print(f"{BOLD}{PURPLE}═══════════════════════════════════════════════════════════{NC}")

    if high > 0:
        print(f"\n{BOLD}{GREEN}✓ UPLOADED {high} HIGH CONFIDENCE CASES{NC}")
        print(f"  Training data precision: Expected >90%")
        print(f"  No manual review needed for these cases")
    else:
        print(f"\n{BOLD}{YELLOW}⚠ NO HIGH CONFIDENCE CASES IN TRIAL{NC}")
        print(f"  This is OK for trial (only 5 studies)")

    if medium > 0:
        print(f"\n{BOLD}NEXT STEPS - Manual Review:{NC}")
        print(f"  1. View QA images: ls ${output_dir}/qa_images/")
        print(f"  2. Review {medium} MEDIUM confidence cases")
        print(f"  3. Upload verified cases to Roboflow")
    
    print()

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

else
    print_warning "Progress file not found"
fi

# ============================================================================
# OUTPUT FILES
# ============================================================================

print_section "Output Files"

QA_COUNT=$(find $OUTPUT_DIR/qa_images -name "*_QA.jpg" 2>/dev/null | wc -l || echo 0)
IMG_COUNT=$(find $OUTPUT_DIR/candidate_images -name "*.jpg" 2>/dev/null | wc -l || echo 0)

print_stat "Candidate images" "$IMG_COUNT"
print_stat "QA images (labeled)" "$QA_COUNT"
print_stat "Progress tracking" "progress.json"

echo ""
echo -e "${BOLD}${CYAN}View QA images:${NC}"
echo -e "  ${CYAN}ls -lh ${OUTPUT_DIR}/qa_images/*_QA.jpg${NC}"

echo ""
echo -e "${BOLD}${CYAN}Example QA images:${NC}"
find $OUTPUT_DIR/qa_images -name "*_QA.jpg" 2>/dev/null | head -3 | while read img; do
    echo -e "  ${CYAN}$(basename $img)${NC}"
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
echo -e "  1. Review QA images with labels"
echo -e "  2. Proceed to weak label generation"
echo -e "  3. Train baseline model"
echo ""
