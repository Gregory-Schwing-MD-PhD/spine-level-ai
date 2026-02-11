#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_yolo
#SBATCH -o logs/eval_yolo_%j.out
#SBATCH -e logs/eval_yolo_%j.err

set -euo pipefail

echo "================================================================"
echo "YOLOv11 MODEL EVALUATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
WEIGHTS="${PROJECT_DIR}/runs/lstv/trial/weights/best.pt"
DATA_YAML="${PROJECT_DIR}/data/training/lstv_yolo_trial/dataset.yaml"
OUTPUT_DIR="${PROJECT_DIR}/results/evaluation/trial"
TEST_IMAGES="${PROJECT_DIR}/results/lstv_screening/trial/candidate_images"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

echo "Evaluating model..."
echo "Weights: $WEIGHTS"
echo "Data: $DATA_YAML"
echo "Output: $OUTPUT_DIR"
echo "================================================================"

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/evaluate_model.py \
        --weights /work/runs/lstv/trial/weights/best.pt \
        --data /work/data/training/lstv_yolo_trial/dataset.yaml \
        --output /work/results/evaluation/trial \
        --conf 0.25 \
        --test-images /work/results/lstv_screening/trial/candidate_images

echo "================================================================"
echo "Evaluation complete!"
echo "End time: $(date)"
echo "================================================================"
echo ""
echo "Results location:"
echo "  Report:  ${OUTPUT_DIR}/EVALUATION_REPORT.md"
echo "  Metrics: ${OUTPUT_DIR}/evaluation_results.json"
echo "  Plots:   ${OUTPUT_DIR}/plots/"
echo ""
