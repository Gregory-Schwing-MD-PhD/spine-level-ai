#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=yolo_trial
#SBATCH -o logs/yolo_trial_%j.out
#SBATCH -e logs/yolo_trial_%j.err

set -euo pipefail

echo "================================================================"
echo "YOLOv11 TRAINING - TRIAL"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

PROJECT_DIR="$(pwd)"
DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_trial/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/trial"

export WANDB_API_KEY="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"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p $WANDB_DIR

IMG_PATH="${HOME}/singularity_cache/spine-level-ai-yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

echo "Training YOLOv11n on trial dataset..."
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
        --batch 16 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name trial

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "================================================================"
echo ""
echo "Results saved to:"
echo "  Weights: ${OUTPUT_DIR}/weights/best.pt"
echo "  Metrics: ${OUTPUT_DIR}/final_metrics.json"
echo ""
