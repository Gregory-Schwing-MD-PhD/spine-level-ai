#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=train_trial_baseline
#SBATCH -o logs/train_trial_baseline_%j.out
#SBATCH -e logs/train_trial_baseline_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "YOLOv11 TRAINING - TRIAL BASELINE"
echo "Dataset: Trial (5 studies), Weak labels only"
echo "Purpose: Validate pipeline and spine-aware effectiveness"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

nvidia-smi

# Environment setup
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

which singularity

export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR

export NXF_SINGULARITY_HOME_MOUNT=true

# Clean environment
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset R_LIBS
unset R_LIBS_USER
unset R_LIBS_SITE

PROJECT_DIR="$(pwd)"
DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_trial/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/trial_baseline"  # ← CLEAR NAME

# WandB setup
export WANDB_API_KEY="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"
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_PROJECT="lstv-detection"
export WANDB_NAME="trial_baseline"  # ← Track in WandB
mkdir -p $WANDB_DIR

# Container
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spineps-lstv-yolo:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling YOLOv11 container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

if [[ ! -f "$DATA_YAML" ]]; then
    echo "ERROR: Dataset not found at $DATA_YAML"
    exit 1
fi

echo "================================================================"
echo "EXPERIMENT: trial_baseline"
echo "Dataset:    Trial (5 studies, weak labels)"
echo "Purpose:    Validate spine-aware slice selection"
echo "Output:     $OUTPUT_DIR"
echo "WandB:      lstv-detection/trial_baseline"
echo "================================================================"

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=/work/wandb \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_NAME=$WANDB_NAME \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/train_yolo.py \
        --data /work/data/training/lstv_yolo_trial/dataset.yaml \
        --model n \
        --epochs 50 \
        --batch 8 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name trial_baseline

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "================================================================"

# Display results
if [[ -f "$OUTPUT_DIR/final_metrics.json" ]]; then
    echo ""
    echo "TRIAL BASELINE PERFORMANCE:"
    python3 << 'PYEOF'
import json

with open('runs/lstv/trial_baseline/final_metrics.json') as f:
    metrics = json.load(f)

print(f"  mAP@50:    {metrics.get('map50', 0):.4f}")
print(f"  mAP@50-95: {metrics.get('map50_95', 0):.4f}")

if 'per_class_ap' in metrics and 't12_rib' in metrics['per_class_ap']:
    t12_ap = metrics['per_class_ap']['t12_rib'].get('ap50', 0)
    print(f"  T12 rib:   {t12_ap:.4f}")
PYEOF
    echo ""
fi

echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "================================================================"
echo "EXPERIMENT TRACKING"
echo "================================================================"
echo ""
echo "This is: trial_baseline (5 studies, weak labels)"
echo ""
echo "Next experiments in pipeline:"
echo "  1. full_baseline    - 500 studies, weak labels"
echo "  2. trial_refined    - 50 studies, weak + human (optional test)"
echo "  3. full_refined     - 500 studies, weak + human (FINAL)"
echo ""
echo "Compare results:"
echo "  cat runs/lstv/trial_baseline/final_metrics.json"
echo "  cat runs/lstv/full_baseline/final_metrics.json  (after full run)"
echo "  cat runs/lstv/full_refined/final_metrics.json   (after refinement)"
echo ""
echo "================================================================"
