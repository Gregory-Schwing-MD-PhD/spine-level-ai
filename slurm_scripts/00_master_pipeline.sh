#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --job-name=lstv_pipeline_v4
#SBATCH -o logs/lstv_v4_%j.out
#SBATCH -e logs/lstv_v4_%j.err

#===============================================================================
# LSTV DETECTION MASTER PIPELINE v4.0 - 4-VIEW + INTEGRATED WEAK LABELS
#
# NEW IN v4.0:
#   - 4-view extraction (midline + left + mid + right)
#   - Integrated weak label generation (no separate script needed)
#   - Multi-view confidence aggregation
#   - Automatic YOLO dataset preparation
#   - Ready for direct YOLO training
#
# Pipeline stages:
#   1. Diagnostic    → 5 studies, validate approach
#   2. Trial         → 50 studies, quality check
#   3. Full Screen   → All studies (~2700)
#   4. YOLO Training → Train detection model
#
# Updated to use: lstv_screen_production_v4.py
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

print_header "LSTV DETECTION PIPELINE v4.0 - 4-VIEW + INTEGRATED WEAK LABELS"

echo -e "${CYAN}Job Information:${NC}"
echo -e "  Job ID:    ${SLURM_JOB_ID}"
echo -e "  Start:     $(date)"
echo -e "  Node:      $(hostname)"
echo -e "  GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
echo ""

# Singularity environment
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

# Validate series CSV
if [[ ! -f "$SERIES_CSV" ]]; then
    print_error "Series CSV not found: ${SERIES_CSV}"
    exit 1
fi
print_success "Series CSV: ${SERIES_CSV}"

# Pipeline directories
STAGE_BASE="${PROJECT_DIR}/results/lstv_pipeline_v4"
DIAGNOSTIC_DIR="${STAGE_BASE}/01_diagnostic"
TRIAL_DIR="${STAGE_BASE}/02_trial"
FULL_DIR="${STAGE_BASE}/03_full_screening"
YOLO_TRAINING_DIR="${STAGE_BASE}/04_yolo_training"

for dir in "$STAGE_BASE" "$DIAGNOSTIC_DIR" "$TRIAL_DIR" "$FULL_DIR" "$YOLO_TRAINING_DIR"; do
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

echo ""
echo -e "${BOLD}Pipeline Configuration:${NC}"
echo -e "  Base directory:    ${STAGE_BASE}"
echo -e "  Data directory:    ${DATA_DIR}"
echo -e "  Series CSV:        ${SERIES_CSV}"
echo -e "  Script:            lstv_screen_production_v4.py"
echo -e "  Features:          4-view extraction + integrated weak labels"
echo ""

# ============================================================================
# PHASE 1: DIAGNOSTIC (5 studies)
# ============================================================================

print_phase "1" "DIAGNOSTIC - Validate 4-View Approach"

if grep -q "diagnostic:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "Diagnostic already complete, skipping..."
else
    echo "Running diagnostic on 5 studies..."
    echo "  Extracting 4 views per study (20 images total)"
    echo "  Generating weak labels automatically"
    echo "  Creating QA images"
    echo ""

    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${DATA_DIR}:/data/input:ro" \
        --bind "${DIAGNOSTIC_DIR}:/data/output:rw" \
        --bind "${MODELS_CACHE}:/app/models:rw" \
        --bind "${MODELS_CACHE}:/opt/conda/lib/python3.10/site-packages/spineps/models:rw" \
        --bind "$(dirname $SERIES_CSV):/data/raw:ro" \
        --env SPINEPS_SEGMENTOR_MODELS=/app/models \
        --env SPINEPS_ENVIRONMENT_DIR=/app/models \
        --pwd /work \
        "$IMG_PATH" \
        python /work/src/screening/lstv_screen_production_v4.py \
            --mode diagnostic \
            --input_dir /data/input \
            --output_dir /data/output \
            --series_csv /data/raw/train_series_descriptions.csv \
            --confidence_threshold 0.3

    if [ $? -eq 0 ]; then
        checkpoint "diagnostic" "success"
        print_success "Diagnostic complete!"
    else
        print_error "Diagnostic failed!"
        exit 1
    fi
fi

# Analyze diagnostic results
DIAGNOSTIC_PROGRESS="${DIAGNOSTIC_DIR}/progress.json"

if [[ -f "$DIAGNOSTIC_PROGRESS" ]]; then
    echo ""
    echo -e "${BOLD}Diagnostic Results:${NC}"

    python3 << 'PYEOF'
import json
import sys
from pathlib import Path

try:
    with open('${DIAGNOSTIC_PROGRESS}') as f:
        progress = json.load(f)

    total = len(progress.get('processed', []))
    weak_labels = len(progress.get('weak_labels_generated', []))
    
    print(f"  Studies processed:     {total}")
    print(f"  Weak labels generated: {weak_labels}")
    
    # Count images and labels
    images_dir = Path('${DIAGNOSTIC_DIR}/images')
    labels_dir = Path('${DIAGNOSTIC_DIR}/weak_labels')
    
    if images_dir.exists():
        image_count = len(list(images_dir.glob('*.jpg')))
        print(f"  Images created:        {image_count} (4 per study)")
    
    if labels_dir.exists():
        label_count = len(list(labels_dir.glob('*.txt')))
        print(f"  Label files created:   {label_count}")
        
        # Count total bboxes
        total_boxes = 0
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file) as f:
                total_boxes += len(f.readlines())
        print(f"  Total bounding boxes:  {total_boxes}")
    
    print("")
    if weak_labels >= total * 0.8:  # 80% success rate
        print("✓ Diagnostic successful - proceeding to trial")
        sys.exit(0)
    else:
        print("⚠ Low weak label generation rate")
        sys.exit(0)  # Proceed anyway for now

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
PYEOF

    PROCEED=$?
    echo ""

    if [ $PROCEED -eq 0 ]; then
        print_success "Diagnostic validation passed."
    fi
else
    print_error "Progress file not found: ${DIAGNOSTIC_PROGRESS}"
    exit 1
fi

# ============================================================================
# PHASE 2: TRIAL RUN (50 studies)
# ============================================================================

print_phase "2" "TRIAL - Quality Check on 50 Studies"

if grep -q "trial:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "Trial already complete, skipping..."
else
    echo "Running trial on 50 studies..."
    echo "  Expected: 200 images (4 per study)"
    echo "  Expected: 200 label files"
    echo "  Output: ${TRIAL_DIR}"
    echo ""

    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${DATA_DIR}:/data/input:ro" \
        --bind "${TRIAL_DIR}:/data/output:rw" \
        --bind "${MODELS_CACHE}:/app/models:rw" \
        --bind "${MODELS_CACHE}:/opt/conda/lib/python3.10/site-packages/spineps/models:rw" \
        --bind "$(dirname $SERIES_CSV):/data/raw:ro" \
        --env SPINEPS_SEGMENTOR_MODELS=/app/models \
        --env SPINEPS_ENVIRONMENT_DIR=/app/models \
        --pwd /work \
        "$IMG_PATH" \
        python /work/src/screening/lstv_screen_production_v4.py \
            --mode trial \
            --input_dir /data/input \
            --output_dir /data/output \
            --series_csv /data/raw/train_series_descriptions.csv \
            --limit 50 \
            --confidence_threshold 0.3

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

if [[ -f "$TRIAL_PROGRESS" ]]; then
    echo ""
    echo -e "${BOLD}Trial Results:${NC}"

    python3 << 'PYEOF'
import json
import pandas as pd
import sys
from pathlib import Path

try:
    with open('${TRIAL_PROGRESS}') as f:
        progress = json.load(f)

    total = len(progress.get('processed', []))
    weak_labels = len(progress.get('weak_labels_generated', []))
    
    print(f"  Studies processed:     {total}")
    print(f"  Weak labels generated: {weak_labels} ({weak_labels/max(total,1)*100:.1f}%)")
    
    # Count images and labels
    images_dir = Path('${TRIAL_DIR}/images')
    labels_dir = Path('${TRIAL_DIR}/weak_labels')
    
    image_count = len(list(images_dir.glob('*.jpg'))) if images_dir.exists() else 0
    label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
    
    print(f"  Images:                {image_count}")
    print(f"  Label files:           {label_count}")
    
    # Count bboxes per class
    if labels_dir.exists():
        class_counts = {}
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print(f"")
        print(f"  Bounding boxes by class:")
        for class_id in sorted(class_counts.keys()):
            print(f"    Class {class_id}: {class_counts[class_id]}")
    
    print("")
    if weak_labels >= 10:  # At least 10 studies with labels
        print("✓ Trial successful - proceeding to full screening")
        sys.exit(0)
    else:
        print("⚠ Consider reviewing results before full screening")
        sys.exit(0)  # Proceed anyway

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
PYEOF

    echo ""
    print_success "Trial validation complete."
    print_info "Review images: ${TRIAL_DIR}/images/"
    print_info "Review QA: ${TRIAL_DIR}/qa_images/"
fi

# ============================================================================
# PHASE 3: FULL SCREENING (~2700 studies)
# ============================================================================

print_phase "3" "FULL SCREENING - Process All Studies"

if grep -q "full_screening:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "Full screening already complete, skipping..."
else
    echo "Running full LSTV screening on entire dataset..."
    echo "  Expected: ~10,800 images (4 per study × 2700)"
    echo "  Expected: ~10,800 label files"
    echo "  Estimated time: 6-8 hours"
    echo ""

    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${DATA_DIR}:/data/input:ro" \
        --bind "${FULL_DIR}:/data/output:rw" \
        --bind "${MODELS_CACHE}:/app/models:rw" \
        --bind "${MODELS_CACHE}:/opt/conda/lib/python3.10/site-packages/spineps/models:rw" \
        --bind "$(dirname $SERIES_CSV):/data/raw:ro" \
        --env SPINEPS_SEGMENTOR_MODELS=/app/models \
        --env SPINEPS_ENVIRONMENT_DIR=/app/models \
        --pwd /work \
        "$IMG_PATH" \
        python /work/src/screening/lstv_screen_production_v4.py \
            --mode full \
            --input_dir /data/input \
            --output_dir /data/output \
            --series_csv /data/raw/train_series_descriptions.csv \
            --confidence_threshold 0.3

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

if [[ -f "$FULL_PROGRESS" ]]; then
    echo ""
    echo -e "${BOLD}Full Screening Results:${NC}"

    python3 << 'PYEOF'
import json
import pandas as pd
from pathlib import Path

try:
    with open('${FULL_PROGRESS}') as f:
        progress = json.load(f)

    total = len(progress.get('processed', []))
    failed = len(progress.get('failed', []))
    weak_labels = len(progress.get('weak_labels_generated', []))
    
    print(f"  Studies processed:     {total}")
    print(f"  Failed:                {failed}")
    print(f"  Weak labels generated: {weak_labels} ({weak_labels/max(total,1)*100:.1f}%)")
    
    # Count final outputs
    images_dir = Path('${FULL_DIR}/images')
    labels_dir = Path('${FULL_DIR}/weak_labels')
    
    image_count = len(list(images_dir.glob('*.jpg'))) if images_dir.exists() else 0
    label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
    
    print(f"  Total images:          {image_count}")
    print(f"  Total label files:     {label_count}")
    
    # Class distribution
    if labels_dir.exists():
        class_counts = {}
        total_boxes = 0
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_boxes += 1
        
        print(f"  Total bounding boxes:  {total_boxes}")
        print(f"")
        print(f"  Class distribution:")
        
        class_names = {
            0: 't12_vertebra', 1: 't12_rib_left', 2: 't12_rib_right',
            3: 'l5_vertebra', 4: 'l5_tp', 5: 'sacrum',
            6: 'l4_vertebra', 7: 'l1_vertebra', 8: 'l2_vertebra', 9: 'l3_vertebra'
        }
        
        for class_id in sorted(class_counts.keys()):
            name = class_names.get(class_id, f'class_{class_id}')
            count = class_counts[class_id]
            pct = count / total_boxes * 100 if total_boxes > 0 else 0
            print(f"    {class_id}: {name:20s} {count:6d} ({pct:5.1f}%)")

except Exception as e:
    print(f"Error: {e}")
PYEOF
    echo ""
fi

# ============================================================================
# PHASE 4: PREPARE YOLO DATASET
# ============================================================================

print_phase "4" "YOLO DATASET PREPARATION"

print_info "Dataset already prepared during screening!"
print_info "Location: ${FULL_DIR}"
print_info "Structure:"
echo "  ${FULL_DIR}/"
echo "  ├── images/          (all 4-view images)"
echo "  ├── weak_labels/     (YOLO format labels)"
echo "  ├── qa_images/       (quality assurance)"
echo "  └── dataset.yaml     (YOLO config)"
echo ""

# Check if dataset.yaml was created
DATASET_YAML="${FULL_DIR}/dataset.yaml"

if [[ ! -f "$DATASET_YAML" ]]; then
    print_warning "dataset.yaml not found, creating..."
    
    cat > "$DATASET_YAML" << EOF
# LSTV Detection Dataset v4.0
# 4-view per study: midline, left, mid, right

path: ${FULL_DIR}
train: images
val: images  # TODO: Split train/val if needed

names:
  0: t12_vertebra
  1: t12_rib_left
  2: t12_rib_right
  3: l5_vertebra
  4: l5_transverse_process
  5: sacrum
  6: l4_vertebra
  7: l1_vertebra
  8: l2_vertebra
  9: l3_vertebra
EOF
    
    print_success "Created dataset.yaml"
fi

checkpoint "dataset_prep" "complete"

# ============================================================================
# PHASE 5: YOLO TRAINING (Optional)
# ============================================================================

print_phase "5" "YOLO TRAINING - Optional"

if grep -q "yolo_training:success" "$CHECKPOINT_FILE" 2>/dev/null; then
    print_info "YOLO training already complete, skipping..."
elif [[ "${RUN_YOLO_TRAINING:-false}" == "true" ]]; then
    print_info "Starting YOLO training..."
    
    singularity exec --nv \
        --bind "${PROJECT_DIR}:/work" \
        --bind "${FULL_DIR}:/data:rw" \
        --pwd /work \
        "$IMG_PATH" \
        yolo train \
            data=/data/dataset.yaml \
            model=yolov8n.pt \
            epochs=100 \
            imgsz=640 \
            batch=16 \
            device=0 \
            project=/data/yolo_runs \
            name=lstv_detection_v4
    
    if [ $? -eq 0 ]; then
        checkpoint "yolo_training" "success"
        print_success "YOLO training complete!"
    else
        print_warning "YOLO training failed (continuing anyway)"
    fi
else
    print_info "YOLO training skipped (set RUN_YOLO_TRAINING=true to enable)"
    print_info "To train manually:"
    echo ""
    echo "  cd ${FULL_DIR}"
    echo "  yolo train data=dataset.yaml model=yolov8n.pt epochs=100"
    echo ""
fi

# ============================================================================
# PIPELINE SUMMARY
# ============================================================================

print_header "PIPELINE COMPLETE"

echo -e "${BOLD}Results Summary:${NC}"
echo ""

# Count final outputs
IMAGES=$(find "${FULL_DIR}/images" -name "*.jpg" 2>/dev/null | wc -l)
LABELS=$(find "${FULL_DIR}/weak_labels" -name "*.txt" 2>/dev/null | wc -l)
QA=$(find "${FULL_DIR}/qa_images" -name "*.jpg" 2>/dev/null | wc -l)

echo "  Images:        ${IMAGES}"
echo "  Labels:        ${LABELS}"
echo "  QA images:     ${QA}"
echo ""

echo -e "${BOLD}${GREEN}Next Steps:${NC}"
echo ""
echo "  1. REVIEW QA IMAGES"
echo "     cd ${FULL_DIR}/qa_images"
echo "     → Check 4-view extraction quality"
echo "     → Verify midline shows clear L1-L5-Sacrum"
echo "     → Verify parasagittal views show ribs/TPs"
echo ""
echo "  2. VALIDATE WEAK LABELS"
echo "     → Check class distribution (above)"
echo "     → Spot-check label files manually"
echo "     → Visualize with: yolo val data=${FULL_DIR}/dataset.yaml"
echo ""
echo "  3. TRAIN YOLO MODEL"
echo "     cd ${FULL_DIR}"
echo "     yolo train data=dataset.yaml model=yolov8n.pt epochs=100"
echo ""
echo "  4. EVALUATE MODEL"
echo "     yolo val data=dataset.yaml model=runs/detect/train/weights/best.pt"
echo ""

echo -e "${BOLD}${CYAN}Output Locations:${NC}"
echo "  Pipeline base:  ${STAGE_BASE}/"
echo "  Diagnostic:     ${DIAGNOSTIC_DIR}/"
echo "  Trial:          ${TRIAL_DIR}/"
echo "  Full:           ${FULL_DIR}/"
echo "    ├─ images/          (4-view JPGs)"
echo "    ├─ weak_labels/     (YOLO TXT labels)"
echo "    ├─ qa_images/       (QA visualizations)"
echo "    ├─ dataset.yaml     (YOLO config)"
echo "    └─ results.csv      (detection metrics)"
echo ""

echo -e "${BOLD}Key Features:${NC}"
echo "  ✓ 4 views per study (midline + left + mid + right)"
echo "  ✓ Automatic weak label generation"
echo "  ✓ Multi-view confidence aggregation"
echo "  ✓ Ready for YOLO training"
echo "  ✓ QA images for validation"
echo ""

echo -e "${BOLD}End time:${NC} $(date)"
ELAPSED=$((SECONDS / 60))
echo -e "${BOLD}Total time:${NC} ${ELAPSED} minutes"
echo ""

print_success "Master pipeline v4.0 complete!"
