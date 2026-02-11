#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=06:00:00
#SBATCH --job-name=yolo_refined
#SBATCH -o logs/yolo_refined_%j.out
#SBATCH -e logs/yolo_refined_%j.err

set -euo pipefail

echo "================================================================"
echo "YOLOv11 TRAINING - REFINED LABELS (Human-in-the-Loop)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

PROJECT_DIR="$(pwd)"
REFINED_DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_refined/dataset.yaml"
WEAK_DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_trial/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/refined"

export WANDB_API_KEY="your_wandb_key_here"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p $WANDB_DIR

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

if [[ ! -f "$REFINED_DATA_YAML" ]]; then
    echo "ERROR: Refined dataset not found!"
    echo "Run label fusion first"
    exit 1
fi

echo "Training YOLOv11n on REFINED labels..."
echo "Data YAML: $REFINED_DATA_YAML"
echo "Output:    $OUTPUT_DIR"
echo "Container: $IMG_PATH"
echo "================================================================"

# Train on refined labels
singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=/work/wandb \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/train_yolo.py \
        --data /work/data/training/lstv_yolo_refined/dataset.yaml \
        --model n \
        --epochs 100 \
        --batch 16 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name refined

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "================================================================"

# Compare with weak-only baseline
echo ""
echo "Comparing refined vs weak-only..."

WEAK_METRICS="${PROJECT_DIR}/runs/lstv/trial/final_metrics.json"
REFINED_METRICS="${OUTPUT_DIR}/final_metrics.json"

if [[ -f "$WEAK_METRICS" ]] && [[ -f "$REFINED_METRICS" ]]; then
    python3 << PYEOF
import json

with open('$WEAK_METRICS') as f:
    weak = json.load(f)

with open('$REFINED_METRICS') as f:
    refined = json.load(f)

print("\\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"{'Metric':<20s} {'Weak-only':<12s} {'Refined':<12s} {'Improvement':<12s}")
print("-"*60)

metrics_to_compare = ['map50', 'map50_95', 'precision', 'recall']

for metric in metrics_to_compare:
    w_val = weak.get(metric, 0)
    r_val = refined.get(metric, 0)
    improvement = ((r_val - w_val) / w_val * 100) if w_val > 0 else 0
    
    print(f"{metric:<20s} {w_val:<12.4f} {r_val:<12.4f} {improvement:+11.1f}%")

print("="*60)

# Save comparison
comparison = {
    'weak_only': weak,
    'refined': refined,
    'improvements': {
        m: ((refined.get(m, 0) - weak.get(m, 0)) / weak.get(m, 1) * 100) if weak.get(m, 1) > 0 else 0
        for m in metrics_to_compare
    }
}

with open('$PROJECT_DIR/runs/lstv/refined/comparison_report.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("\\n✓ Comparison saved to: runs/lstv/refined/comparison_report.json")
PYEOF
fi

echo ""
echo "Results saved to:"
echo "  Weights: ${OUTPUT_DIR}/weights/best.pt"
echo "  Metrics: ${OUTPUT_DIR}/final_metrics.json"
echo "  Comparison: ${OUTPUT_DIR}/comparison_report.json"
echo ""
echo "================================================================"
