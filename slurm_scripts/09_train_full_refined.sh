#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=06:00:00
#SBATCH --job-name=train_full_refined
#SBATCH -o logs/train_full_refined_%j.out
#SBATCH -e logs/train_full_refined_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "YOLOv11 TRAINING - FULL REFINED"
echo "Dataset: Full (500 studies), Weak + Human labels"
echo "Purpose: FINAL production model with human-in-the-loop"
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
DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_refined/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/full_refined"  # ← FINAL MODEL

# WandB setup
export WANDB_API_KEY="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"
export WANDB_DIR="${PROJECT_DIR}/wandb"
export WANDB_PROJECT="lstv-detection"
export WANDB_NAME="full_refined"  # ← Track in WandB
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
    echo "ERROR: Refined dataset not found at $DATA_YAML"
    echo "Run label fusion first!"
    exit 1
fi

echo "================================================================"
echo "EXPERIMENT: full_refined (FINAL MODEL)"
echo "Dataset:    Full (500 studies, weak + human labels)"
echo "Purpose:    Final production model for clinical deployment"
echo "Output:     $OUTPUT_DIR"
echo "WandB:      lstv-detection/full_refined"
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
        --data /work/data/training/lstv_yolo_refined/dataset.yaml \
        --model n \
        --epochs 100 \
        --batch 16 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name full_refined

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "================================================================"

# Display results and comparison
if [[ -f "$OUTPUT_DIR/final_metrics.json" ]]; then
    echo ""
    echo "FULL REFINED PERFORMANCE:"
    python3 << 'PYEOF'
import json
from pathlib import Path

refined_file = Path('runs/lstv/full_refined/final_metrics.json')
baseline_file = Path('runs/lstv/full_baseline/final_metrics.json')

with open(refined_file) as f:
    refined = json.load(f)

print(f"  mAP@50:    {refined.get('map50', 0):.4f}")
print(f"  mAP@50-95: {refined.get('map50_95', 0):.4f}")

if 'per_class_ap' in refined:
    print("\nPer-class AP@50:")
    for cls in ['t12_vertebra', 't12_rib', 'l5_vertebra', 'sacrum']:
        if cls in refined['per_class_ap']:
            ap = refined['per_class_ap'][cls].get('ap50', 0)
            print(f"  {cls:20s}: {ap:.4f}")
    
    if 't12_rib' in refined['per_class_ap']:
        t12_ap = refined['per_class_ap']['t12_rib'].get('ap50', 0)
        print(f"\n🎯 T12 RIB (critical): {t12_ap:.4f}")
        
        if t12_ap >= 0.80:
            print("   🔥 EXCELLENT - Clinical deployment ready!")
        elif t12_ap >= 0.75:
            print("   ✅ VERY GOOD - Exceeds clinical threshold!")
        elif t12_ap >= 0.70:
            print("   ✓ GOOD - Acceptable for clinical use")
        else:
            print("   ⚠ Below clinical threshold (75%)")

# Compare with baseline
if baseline_file.exists():
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    print("\n" + "="*60)
    print("COMPARISON: Baseline vs Refined")
    print("="*60)
    
    metrics = ['map50', 'map50_95', 'precision', 'recall']
    
    for metric in metrics:
        b_val = baseline.get(metric, 0)
        r_val = refined.get(metric, 0)
        improvement = ((r_val - b_val) / b_val * 100) if b_val > 0 else 0
        print(f"{metric:15s}: {b_val:.4f} → {r_val:.4f} ({improvement:+.1f}%)")
    
    # T12 rib comparison
    if 't12_rib' in baseline.get('per_class_ap', {}) and 't12_rib' in refined.get('per_class_ap', {}):
        b_t12 = baseline['per_class_ap']['t12_rib'].get('ap50', 0)
        r_t12 = refined['per_class_ap']['t12_rib'].get('ap50', 0)
        t12_improvement = ((r_t12 - b_t12) / b_t12 * 100) if b_t12 > 0 else 0
        
        print(f"\n🎯 T12 RIB:      {b_t12:.4f} → {r_t12:.4f} ({t12_improvement:+.1f}%)")
        
        if t12_improvement > 15:
            print("   🔥 MAJOR IMPROVEMENT from human refinement!")
        elif t12_improvement > 5:
            print("   ✅ GOOD IMPROVEMENT from human refinement")
        else:
            print("   ⚠ Modest improvement - review annotation quality")
    
    print("="*60)
else:
    print("\nBaseline model not found - cannot compare")

PYEOF
    echo ""
fi

echo "Results: $OUTPUT_DIR"
echo ""
echo "================================================================"
echo "FINAL MODEL READY"
echo "================================================================"
echo ""
echo "This is the production model for clinical deployment!"
echo ""
echo "Model location: runs/lstv/full_refined/weights/best.pt"
echo ""
echo "Experiment comparison available in WandB:"
echo "  https://wandb.ai/your-username/lstv-detection"
echo ""
echo "All experiments:"
echo "  - trial_baseline:  Validation run (5 studies)"
echo "  - full_baseline:   Weak labels only (500 studies)"
echo "  - full_refined:    Weak + Human labels (500 studies) ← THIS ONE"
echo ""
echo "================================================================"
