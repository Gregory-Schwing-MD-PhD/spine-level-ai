#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --job-name=yolo_full
#SBATCH -o logs/yolo_full_%j.out
#SBATCH -e logs/yolo_full_%j.err

set -euo pipefail

echo "================================================================"
echo "YOLOv11 TRAINING - FULL DATASET (500 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

PROJECT_DIR="$(pwd)"
DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_full/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/runs/lstv/full"

export WANDB_API_KEY="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"
export WANDB_DIR="${PROJECT_DIR}/wandb"
mkdir -p $WANDB_DIR

IMG_PATH="${HOME}/singularity_cache/spine-level-ai-yolo.sif"

echo "Training YOLOv11m on full dataset..."
echo "Data YAML: $DATA_YAML"
echo "Output:    $OUTPUT_DIR"
echo "================================================================"

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_DIR=/work/wandb \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/train_yolo.py \
        --data /work/data/training/lstv_yolo_full/dataset.yaml \
        --model m \
        --epochs 200 \
        --batch 32 \
        --imgsz 640 \
        --project /work/runs/lstv \
        --name full

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "================================================================"
