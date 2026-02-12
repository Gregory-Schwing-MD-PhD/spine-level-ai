#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=06:00:00
#SBATCH --job-name=train_full_baseline
#SBATCH -o logs/train_full_baseline_%j.out
#SBATCH -e logs/train_full_baseline_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "YOLOv11 TRAINING - FULL BASELINE"
echo "Dataset: Full (500 studies), Weak labels only"
echo "Purpose: Production baseline for comparison with refined model"
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

unset LD_LIBRARY_PATH
unset PYTHONPATH
unset R_LIBS
unset R_LIBS_USER
unset R_LIBS_SITE

PROJECT_DIR="$(pwd)"
DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_full/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/full_baseline"  # ← CLEAR NAME

# WandB setup
export WANDB_API_KEY="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"
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_PROJECT="lstv-detection"
export WANDB_NAME="full_baseline"  # ← Track in WandB
mkdir -p $WANDB_DIR

# Container
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spineps-lstv-yolo:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling YOLOv11 container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

if [[ ! -f "$DATA_YAML" ]]; then
    echo "ERROR: Dataset not found at $DATA_YAML"
    echo "Run: sbatch slurm_scripts/06_generate_weak_labels_full.sh"
    exit 1
fi

echo "================================================================"
echo "EXPERIMENT: full_baseline"
echo "Dataset:    Full (500 studies, weak labels with spine-aware)"
echo "Purpose:    Production baseline for comparison"
echo "Output:     $OUTPUT_DIR"
echo "WandB:      lstv-detection/full_baseline"
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
        --data /work/data/training/lstv_yolo_full/dataset.yaml \
        --model n \
        --epochs 100 \
        --batch 16 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name full_baseline

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "================================================================"

# Display results
if [[ -f "$OUTPUT_DIR/final_metrics.json" ]]; then
    echo ""
    echo "FULL BASELINE PERFORMANCE:"
    python3 << 'PYEOF'
import json

with open('runs/lstv/full_baseline/final_metrics.json') as f:
    metrics = json.load(f)

print(f"  mAP@50:    {metrics.get('map50', 0):.4f}")
print(f"  mAP@50-95: {metrics.get('map50_95', 0):.4f}")

if 'per_class_ap' in metrics:
    print("\nPer-class AP@50:")
    for cls in ['t12_vertebra', 't12_rib', 'l5_vertebra', 'sacrum']:
        if cls in metrics['per_class_ap']:
            ap = metrics['per_class_ap'][cls].get('ap50', 0)
            print(f"  {cls:20s}: {ap:.4f}")
    
    if 't12_rib' in metrics['per_class_ap']:
        t12_ap = metrics['per_class_ap']['t12_rib'].get('ap50', 0)
        print(f"\n🎯 T12 RIB (critical): {t12_ap:.4f}")
        
        if t12_ap >= 0.75:
            print("   ✅ EXCELLENT - Clinical threshold met!")
        elif t12_ap >= 0.65:
            print("   ✓ GOOD - Will improve with human refinement")
        else:
            print("   ⚠ NEEDS IMPROVEMENT - Human refinement critical")
PYEOF
    echo ""
fi

echo "Results: $OUTPUT_DIR"
echo ""
echo "================================================================"
echo "NEXT STEP: Human Refinement"
echo "================================================================"
echo ""
echo "This baseline will be compared against:"
echo "  full_refined (500 studies, weak + human labels)"
echo ""
echo "To improve this baseline:"
echo "  1. Med students annotate 200 images"
echo "  2. Fuse weak + human labels"
echo "  3. Train full_refined model"
echo "  4. Compare: full_baseline vs full_refined"
echo ""
echo "Expected improvement with human refinement:"
echo "  T12 rib: +10-20% absolute"
echo "  Overall mAP@50: +15-25% absolute"
echo ""
echo "================================================================"
