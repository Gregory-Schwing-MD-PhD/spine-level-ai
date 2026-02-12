#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=yolo_baseline_trial
#SBATCH -o logs/yolo_baseline_trial_%j.out
#SBATCH -e logs/yolo_baseline_trial_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "YOLOv11 TRAINING - BASELINE TRIAL (Weak Labels Only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

# Environment setup (matching your format)
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
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/baseline_trial"

# WandB setup
export WANDB_API_KEY="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"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p $WANDB_DIR

# Container
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-yolo:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spine-level-ai-yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling YOLOv11 container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

if [[ ! -f "$DATA_YAML" ]]; then
    echo "ERROR: Dataset not found at $DATA_YAML"
    echo "Weak labels not generated yet!"
    exit 1
fi

echo "================================================================"
echo "Training YOLOv11n on TRIAL dataset (BASELINE - Weak Labels Only)"
echo "Data YAML: $DATA_YAML"
echo "Output:    $OUTPUT_DIR"
echo "Container: $IMG_PATH"
echo "================================================================"

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=/work/wandb \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/train_yolo.py \
        --data /work/data/training/lstv_yolo_trial/dataset.yaml \
        --model n \
        --epochs 50 \
        --batch 8 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name baseline_trial

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

with open('runs/lstv/baseline_trial/final_metrics.json') as f:
    metrics = json.load(f)

print(f"  mAP@50:    {metrics.get('map50', 0):.4f}")
print(f"  mAP@50-95: {metrics.get('map50_95', 0):.4f}")
print(f"  Precision: {metrics.get('precision', 0):.4f}")
print(f"  Recall:    {metrics.get('recall', 0):.4f}")

if 'per_class_ap' in metrics and 't12_rib' in metrics['per_class_ap']:
    t12_ap = metrics['per_class_ap']['t12_rib'].get('ap50', 0)
    print(f"\n  T12 rib AP@50: {t12_ap:.4f}")
    
    if t12_ap > 0.70:
        print("  ✅ EXCELLENT T12 detection!")
    elif t12_ap > 0.60:
        print("  ✓ GOOD T12 detection")
    else:
        print("  ⚠ T12 detection needs improvement")
PYEOF
    echo ""
fi

echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "================================================================"
echo "NEXT STEP: Review spine-aware validation metrics!"
echo "================================================================"
echo ""
echo "View validation results:"
echo "  cat data/training/lstv_yolo_trial/spine_aware_metrics_report.json"
echo "  xdg-open data/training/lstv_yolo_trial/quality_validation_summary.png"
echo ""
echo "If validation shows strong justification (mean offset >5mm):"
echo "  sbatch slurm_scripts/06_generate_weak_labels_full.sh"
echo ""
echo "================================================================"
