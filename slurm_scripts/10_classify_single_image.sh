#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=classify_single
#SBATCH -o logs/classify_single_%j.out
#SBATCH -e logs/classify_single_%j.err

set -euo pipefail

echo "================================================================"
echo "LSTV CLASSIFICATION - Single Image Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
WEIGHTS="${PROJECT_DIR}/runs/lstv/trial/weights/best.pt"
TEST_IMAGE="${PROJECT_DIR}/results/lstv_screening/trial/candidate_images/100206310.jpg"
OUTPUT="${PROJECT_DIR}/results/inference/single_test.json"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Model weights not found at $WEIGHTS"
    echo "Run training first: sbatch slurm_scripts/07_train_yolo_trial.sh"
    exit 1
fi

if [[ ! -f "$TEST_IMAGE" ]]; then
    echo "ERROR: Test image not found at $TEST_IMAGE"
    echo "Using first available image instead..."
    TEST_IMAGE=$(find ${PROJECT_DIR}/results/lstv_screening/trial/candidate_images -name "*.jpg" | head -1)
    
    if [[ -z "$TEST_IMAGE" ]]; then
        echo "ERROR: No test images found!"
        exit 1
    fi
fi

echo "Testing classification..."
echo "Model:  $WEIGHTS"
echo "Image:  $TEST_IMAGE"
echo "Output: $OUTPUT"
echo "================================================================"

mkdir -p $(dirname $OUTPUT)

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/inference/lstv_classifier.py \
        --weights /work/runs/lstv/trial/weights/best.pt \
        --image "$TEST_IMAGE" \
        --output /work/results/inference/single_test.json \
        --conf 0.25

echo "================================================================"
echo "Classification complete!"
echo "End time: $(date)"
echo "================================================================"

if [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "Result:"
    cat $OUTPUT
    echo ""
fi
