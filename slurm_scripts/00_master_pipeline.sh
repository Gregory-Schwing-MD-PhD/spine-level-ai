#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --job-name=lstv_master_pipeline
#SBATCH -o logs/lstv_master_%j.out
#SBATCH -e logs/lstv_master_%j.err

#===============================================================================
# LSTV DETECTION MASTER PIPELINE v3.0 - UPDATED
#
# This script orchestrates the complete LSTV detection workflow:
#   1. Diagnostic    → Check SPINEPS semantic output availability (5 studies)
#   2. Trial         → Validate rib detection approach (50 studies)
#   3. Full Screen   → Process all studies (~2700 → ~500 LSTV candidates)
#   4. Weak Labels   → Generate training labels (300 LSTV candidates)
#   5. QA Reports    → Create review PDFs for all detections
#   6. Roboflow Prep → Upload high-confidence cases for human refinement
#
# Updated to use: lstv_screen_production_COMPLETE.py
# Expected timeline: ~8-12 hours total
#===============================================================================

set -euo pipefail

# ============================================================================
# COLOR CODES & HELPER FUNCTIONS
# ============================================================================

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() {
    local text="$1"
    echo -e "\n${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║ ${text}${NC}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_phase() {
    local phase="$1"
    local desc="$2"
    echo -e "\n${BOLD}${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${PURPLE}  PHASE ${phase}: ${desc}${NC}"
    echo -e "${BOLD}${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${CYAN}ℹ${NC} $1"; }

checkpoint() {
    local phase="$1"
    local status="$2"
    echo "$phase:$status:$(date +%s)" >> "${CHECKPOINT_FILE}"
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

print_header "LSTV DETECTION MASTER PIPELINE - v3.0 (UPDATED)"

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

# Pipeline stage directories
STAGE_BASE="${PROJECT_DIR}/results/lstv_pipeline_v3"
DIAGNOSTIC_DIR="${STAGE_BASE}/01_diagnostic"
TRIAL_DIR="${STAGE_BASE}/02_trial"
FULL_DIR="${STAGE_BASE}/03_full_screening"
WEAK_LABELS_DIR="${STAGE_BASE}/04_weak_labels"
QA_DIR="${STAGE_BASE}/05_qa_reports"
ROBOFLOW_DIR="${STAGE_BASE}/06_roboflow_upload"

for dir in "$STAGE_BASE" "$DIAGNOSTIC_DIR" "$TRIAL_DIR" "$FULL_DIR" \
           "$WEAK_LABELS_DIR" "$QA_DIR" "$ROBOFLOW_DIR"; do
    mkdir -p "$dir"
done

CHECKPOINT_FILE="${STAGE_BASE}/pipeline_checkpoint.txt"
MODELS_CACHE="${PROJECT_DIR}/spineps_models"
mkdir -p "$MODELS_CACHE"

# Container
CONTAINER_NAME="spine-level-ai-spineps"
IMG_PATH="${SINGULARITY_CACHEDIR}/${CONTAINER_NAME}.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    print_warning "Container not found, pulling..."
    singularity pull "$IMG_PATH" "docker://go2432/${CONTAINER_NAME}:latest"
fi
print_success "Container: ${CONTAINER_NAME}.sif"

# Roboflow credentials
ROBOFLOW_KEY="${ROBOFLOW_API_KEY:-izolWNqCVveKyMrACYzN}"
ROBOFLOW_WORKSPACE="spinelevelai"
ROBOFLOW_PROJECT="lstv-candidates"

echo ""
echo -e "${BOLD}Pipeline Configuration:${NC}"
echo -e "  Base directory:    ${STAGE_BASE}"
echo -e "  Data directory:    ${DATA_DIR}"
echo -e "  Series CSV:        ${SERIES_CSV}"
echo -e "  Checkpoint file:   ${CHECKPOINT_FILE}"
echo -e "  Script:            lstv_screen_production_COMPLETE.py"
echo ""

# ============================================================================
# PHASE 1: DIAGNOSTIC (5 studies)
# ============================================================================

print_phase "1" "DIAGNOSTIC - Verify SPINEPS Semantic Output"

if grep -q "diagnostic:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "Diagnostic already complete, skipping..."
else
    echo "Running diagnostic on 5 studies to check for semantic segmentation..."
    echo ""

    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${DATA_DIR}:/data/input" \
        --bind "${DIAGNOSTIC_DIR}:/data/output" \
        --bind "${MODELS_CACHE}:/app/models" \
        --env SPINEPS_SEGMENTOR_MODELS=/app/models \
        --pwd /work \
        "$IMG_PATH" \
        python /work/src/screening/lstv_screen_production_COMPLETE.py \
            --mode diagnostic \
            --input_dir /data/input \
            --output_dir /data/output \
            --roboflow_key SKIP \
            --confidence_threshold 0.5

    if [ $? -eq 0 ]; then
        checkpoint "diagnostic" "success"
        print_success "Diagnostic complete!"
    else
        print_error "Diagnostic failed!"
        exit 1
    fi
fi

# Check diagnostic results
PROGRESS_FILE="${DIAGNOSTIC_DIR}/progress.json"

if [[ -f "$PROGRESS_FILE" ]]; then
    echo ""
    echo -e "${BOLD}Diagnostic Results:${NC}"
    
    SEMANTIC_COUNT=$(python3 -c "import json; p=json.load(open('${PROGRESS_FILE}')); print(len(p.get('semantic_available', [])))" 2>/dev/null || echo "0")
    SEMANTIC_MISSING=$(python3 -c "import json; p=json.load(open('${PROGRESS_FILE}')); print(len(p.get('semantic_missing', [])))" 2>/dev/null || echo "0")
    TOTAL=$(python3 -c "import json; p=json.load(open('${PROGRESS_FILE}')); print(len(p.get('processed', [])))" 2>/dev/null || echo "0")
    
    if [ "$TOTAL" -gt 0 ]; then
        SEMANTIC_PCT=$(python3 -c "print(int(${SEMANTIC_COUNT} / ${TOTAL} * 100))")
        
        echo -e "  Total processed:       ${TOTAL}"
        echo -e "  Semantic available:    ${SEMANTIC_COUNT}"
        echo -e "  Semantic missing:      ${SEMANTIC_MISSING}"
        echo -e "  Availability:          ${SEMANTIC_PCT}%"
        echo ""
        
        if [ "$SEMANTIC_PCT" -ge 80 ]; then
            print_success "Excellent! Semantic labels available (${SEMANTIC_PCT}%)."
            print_info "Will use semantic rib-density optimization for parasagittal slices."
            USE_SEMANTIC_OPT="yes"
            EXPECTED_RIB_DETECTION="85-90%"
        elif [ "$SEMANTIC_PCT" -ge 50 ]; then
            print_warning "Partial semantic availability (${SEMANTIC_PCT}%)."
            print_info "Will use semantic optimization when available, fallback otherwise."
            USE_SEMANTIC_OPT="yes"
            EXPECTED_RIB_DETECTION="70-80%"
        else
            print_warning "Low semantic availability (${SEMANTIC_PCT}%)."
            print_info "Will use standard spine-aware selection (no rib optimization)."
            USE_SEMANTIC_OPT="no"
            EXPECTED_RIB_DETECTION="60-70%"
        fi
        
        echo -e "  Expected rib detection: ${EXPECTED_RIB_DETECTION}"
        echo ""
    else
        print_error "No studies processed in diagnostic phase!"
        exit 1
    fi
else
    print_error "Progress file not found: ${PROGRESS_FILE}"
    exit 1
fi

# ============================================================================
# PHASE 2: TRIAL RUN (50 studies)
# ============================================================================

print_phase "2" "TRIAL - Validate Detection on 50 Studies"

if grep -q "trial:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "Trial already complete, skipping..."
else
    echo "Running trial on 50 studies to validate detection quality..."
    echo "  Semantic optimization: ${USE_SEMANTIC_OPT}"
    echo "  Output: ${TRIAL_DIR}"
    echo ""

    # Add series CSV if available
    SERIES_CSV_ARG=""
    if [[ -f "$SERIES_CSV" ]]; then
        SERIES_CSV_ARG="--series_csv /data/raw/train_series_descriptions.csv"
    fi

    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${DATA_DIR}:/data/input" \
        --bind "${TRIAL_DIR}:/data/output" \
        --bind "${MODELS_CACHE}:/app/models" \
        --bind "$(dirname $SERIES_CSV):/data/raw" \
        --env SPINEPS_SEGMENTOR_MODELS=/app/models \
        --pwd /work \
        "$IMG_PATH" \
        python /work/src/screening/lstv_screen_production_COMPLETE.py \
            --mode trial \
            --input_dir /data/input \
            ${SERIES_CSV_ARG} \
            --output_dir /data/output \
            --limit 50 \
            --roboflow_key SKIP \
            --confidence_threshold 0.7

    if [ $? -eq 0 ]; then
        checkpoint "trial" "success"
        print_success "Trial complete!"
    else
        print_error "Trial failed!"
        exit 1
    fi
fi

# Analyze trial results
TRIAL_PROGRESS="${TRIAL_DIR}/progress.json"
TRIAL_RESULTS="${TRIAL_DIR}/results.csv"

if [[ -f "$TRIAL_PROGRESS" ]] && [[ -f "$TRIAL_RESULTS" ]]; then
    echo ""
    echo -e "${BOLD}Trial Results:${NC}"

    python3 << 'PYEOF'
import json
import pandas as pd
import sys

try:
    with open('${TRIAL_PROGRESS}') as f:
        progress = json.load(f)
    
    df = pd.read_csv('${TRIAL_RESULTS}')
    
    total = len(progress.get('processed', []))
    lstv_candidates = len(progress.get('flagged', []))
    high_conf = len(progress.get('high_confidence', []))
    medium_conf = len(progress.get('medium_confidence', []))
    low_conf = len(progress.get('low_confidence', []))
    
    print(f"  Studies processed:     {total}")
    print(f"  LSTV candidates:       {lstv_candidates} ({lstv_candidates/max(total,1)*100:.1f}%)")
    print(f"  High confidence:       {high_conf}")
    print(f"  Medium confidence:     {medium_conf}")
    print(f"  Low confidence:        {low_conf}")
    
    if lstv_candidates > 0:
        avg_conf = df[df['is_lstv_candidate'] == True]['confidence_score'].mean()
        print(f"  Average confidence:    {avg_conf:.2f}")
    
    # Check if we should proceed
    if lstv_candidates >= 5:  # At least 5 LSTV candidates in 50 studies
        print("\n✓ Trial results acceptable")
        sys.exit(0)
    else:
        print("\n✗ Insufficient LSTV candidates detected")
        sys.exit(1)

except Exception as e:
    print(f"Error analyzing results: {e}")
    sys.exit(1)
PYEOF

    PROCEED=$?
    echo ""

    if [ $PROCEED -eq 0 ]; then
        print_success "Trial validation passed. Proceeding to full screening."
    else
        print_error "Trial validation failed. Review output before continuing."
        echo ""
        echo "Review directories:"
        echo "  Images: ${TRIAL_DIR}/candidate_images/"
        echo "  QA:     ${TRIAL_DIR}/qa_images/"
        echo ""
        echo "To continue anyway, delete checkpoint and re-run:"
        echo "  rm ${CHECKPOINT_FILE}"
        exit 1
    fi
else
    print_warning "Trial results not found, proceeding cautiously..."
fi

# ============================================================================
# PHASE 3: FULL SCREENING (~2700 studies → ~500 LSTV candidates)
# ============================================================================

print_phase "3" "FULL SCREENING - Process All Studies"

if grep -q "full_screening:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "Full screening already complete, skipping..."
else
    echo "Running full LSTV screening on entire dataset..."
    echo "  Expected LSTV candidates: ~500 (15-20% of 2700)"
    echo "  Confidence threshold: 0.7 (only HIGH confidence uploaded to Roboflow)"
    echo "  Estimated time: 4-6 hours"
    echo ""

    # Add series CSV if available
    SERIES_CSV_ARG=""
    if [[ -f "$SERIES_CSV" ]]; then
        SERIES_CSV_ARG="--series_csv /data/raw/train_series_descriptions.csv"
    fi

    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${DATA_DIR}:/data/input" \
        --bind "${FULL_DIR}:/data/output" \
        --bind "${MODELS_CACHE}:/app/models" \
        --bind "$(dirname $SERIES_CSV):/data/raw" \
        --env SPINEPS_SEGMENTOR_MODELS=/app/models \
        --pwd /work \
        "$IMG_PATH" \
        python /work/src/screening/lstv_screen_production_COMPLETE.py \
            --mode full \
            --input_dir /data/input \
            ${SERIES_CSV_ARG} \
            --output_dir /data/output \
            --roboflow_key "${ROBOFLOW_KEY}" \
            --roboflow_workspace "${ROBOFLOW_WORKSPACE}" \
            --roboflow_project "${ROBOFLOW_PROJECT}" \
            --confidence_threshold 0.7

    if [ $? -eq 0 ]; then
        checkpoint "full_screening" "success"
        print_success "Full screening complete!"
    else
        print_error "Full screening failed!"
        exit 1
    fi
fi

# Analyze full screening results
FULL_PROGRESS="${FULL_DIR}/progress.json"
FULL_RESULTS="${FULL_DIR}/results.csv"

if [[ -f "$FULL_PROGRESS" ]] && [[ -f "$FULL_RESULTS" ]]; then
    echo ""
    echo -e "${BOLD}Full Screening Results:${NC}"

    python3 << 'PYEOF'
import json
import pandas as pd

try:
    with open('${FULL_PROGRESS}') as f:
        progress = json.load(f)
    
    df = pd.read_csv('${FULL_RESULTS}')
    
    total = len(progress.get('processed', []))
    failed = len(progress.get('failed', []))
    lstv_total = df['is_lstv_candidate'].sum()
    high_conf = len(progress.get('high_confidence', []))
    medium_conf = len(progress.get('medium_confidence', []))
    low_conf = len(progress.get('low_confidence', []))
    uploaded = len(progress.get('flagged', []))
    
    semantic_avail = len(progress.get('semantic_available', []))
    semantic_pct = semantic_avail / max(total, 1) * 100
    
    print(f"  Studies processed:     {total}")
    print(f"  Failed:                {failed}")
    print(f"  LSTV candidates:       {lstv_total} ({lstv_total/max(total,1)*100:.1f}%)")
    print(f"")
    print(f"  Confidence breakdown:")
    print(f"    HIGH:                {high_conf} → Uploaded to Roboflow")
    print(f"    MEDIUM:              {medium_conf} → For manual review")
    print(f"    LOW:                 {low_conf} → Flagged only")
    print(f"")
    print(f"  Uploaded to Roboflow:  {uploaded}")
    print(f"  Semantic available:    {semantic_avail} ({semantic_pct:.1f}%)")
    
except Exception as e:
    print(f"Error: {e}")
PYEOF
    echo ""
fi

# ============================================================================
# PHASE 4: WEAK LABEL GENERATION
# ============================================================================

print_phase "4" "WEAK LABEL GENERATION - Create YOLO Training Labels"

print_info "Weak label generation using existing weak label script..."
print_info "This phase uses the separate weak label generation pipeline."
print_info "Skipping in this master script - run manually if needed."
checkpoint "weak_labels" "skipped"

# ============================================================================
# PHASE 5: QA REPORT GENERATION
# ============================================================================

print_phase "5" "QA REPORTS - Already Generated During Screening"

print_info "QA images were generated during the screening phases:"
echo "  Diagnostic QA: ${DIAGNOSTIC_DIR}/qa_images/"
echo "  Trial QA:      ${TRIAL_DIR}/qa_images/"
echo "  Full QA:       ${FULL_DIR}/qa_images/"
echo ""
print_success "QA images available for manual review."
checkpoint "qa_reports" "complete"

# ============================================================================
# PHASE 6: ROBOFLOW UPLOAD STATUS
# ============================================================================

print_phase "6" "ROBOFLOW UPLOAD - Already Completed During Full Screening"

if [[ -f "$FULL_PROGRESS" ]]; then
    UPLOADED=$(python3 -c "import json; p=json.load(open('${FULL_PROGRESS}')); print(len(p.get('flagged', [])))" 2>/dev/null || echo "0")
    
    echo "  Images uploaded:       ${UPLOADED}"
    echo "  Workspace:             ${ROBOFLOW_WORKSPACE}"
    echo "  Project:               ${ROBOFLOW_PROJECT}"
    echo "  URL:                   https://app.roboflow.com/${ROBOFLOW_WORKSPACE}/${ROBOFLOW_PROJECT}"
    echo ""
    
    if [ "$UPLOADED" -gt 0 ]; then
        print_success "High-confidence LSTV cases uploaded to Roboflow."
    else
        print_warning "No images uploaded (none met confidence threshold)."
    fi
fi

checkpoint "roboflow_prep" "complete"

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================

print_header "PIPELINE COMPLETE - READY FOR NEXT STEPS"

echo -e "${BOLD}Results Summary:${NC}"
echo ""

# Count outputs
LSTV_IMAGES=$(find "${FULL_DIR}/candidate_images" -name "*.jpg" 2>/dev/null | wc -l)
QA_IMAGES=$(find "${FULL_DIR}/qa_images" -name "*_QA.jpg" 2>/dev/null | wc -l)

echo "  LSTV candidate images:  ${LSTV_IMAGES}"
echo "  QA images:              ${QA_IMAGES}"
echo ""

echo -e "${BOLD}${GREEN}Next Steps:${NC}"
echo ""
echo "  1. REVIEW QA IMAGES"
echo "     Location: ${FULL_DIR}/qa_images/"
echo "     → Check detection quality and confidence scoring"
echo "     → Focus on HIGH and MEDIUM confidence cases"
echo ""
echo "  2. ROBOFLOW ANNOTATION (Medical Students)"
echo "     URL: https://app.roboflow.com/${ROBOFLOW_WORKSPACE}/${ROBOFLOW_PROJECT}"
echo "     → Review uploaded HIGH confidence cases"
echo "     → Refine bounding boxes for T12 ribs and L5 TPs"
echo "     → Expected time: ~3 min per image"
echo ""
echo "  3. GENERATE WEAK LABELS (If needed)"
echo "     Run: python src/training/generate_weak_labels_v7.py"
echo "     → Creates YOLO-format training labels"
echo "     → Input: LSTV candidates from full screening"
echo ""
echo "  4. TRAIN DETECTION MODEL"
echo "     → Baseline: Use weak labels only (mAP@50: 0.70-0.75)"
echo "     → Refined: Fuse weak + human labels (mAP@50: 0.85-0.90)"
echo ""

echo -e "${BOLD}${CYAN}Output Directories:${NC}"
echo "  Pipeline base:    ${STAGE_BASE}/"
echo "  Diagnostic:       ${DIAGNOSTIC_DIR}/"
echo "  Trial:            ${TRIAL_DIR}/"
echo "  Full screening:   ${FULL_DIR}/"
echo "    ├─ nifti/            (converted NIfTI files)"
echo "    ├─ segmentations/    (SPINEPS instance + semantic masks)"
echo "    ├─ candidate_images/ (LSTV candidate slices)"
echo "    ├─ qa_images/        (labeled QA images)"
echo "    └─ results.csv       (detection results)"
echo ""

echo -e "${BOLD}Key Files:${NC}"
echo "  Progress:         ${FULL_DIR}/progress.json"
echo "  Results CSV:      ${FULL_DIR}/results.csv"
echo "  Checkpoint:       ${CHECKPOINT_FILE}"
echo ""

echo -e "${BOLD}End time:${NC} $(date)"
ELAPSED=$((SECONDS / 60))
echo -e "${BOLD}Total time:${NC} ${ELAPSED} minutes"
echo ""

print_success "Master pipeline complete!"
print_info "Review QA images and proceed to annotation phase."
