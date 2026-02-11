#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=classify_trial
#SBATCH -o logs/classify_trial_%j.out
#SBATCH -e logs/classify_trial_%j.err

set -euo pipefail

echo "================================================================"
echo "LSTV CLASSIFICATION - Trial Batch"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
WEIGHTS="${PROJECT_DIR}/runs/lstv/trial/weights/best.pt"
IMAGE_DIR="${PROJECT_DIR}/results/lstv_screening/trial/candidate_images"
OUTPUT="${PROJECT_DIR}/results/inference/trial_classifications.json"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Model weights not found at $WEIGHTS"
    exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "ERROR: Image directory not found at $IMAGE_DIR"
    exit 1
fi

NUM_IMAGES=$(find $IMAGE_DIR -name "*.jpg" | wc -l)
echo "Classifying $NUM_IMAGES images..."
echo "Model:      $WEIGHTS"
echo "Images:     $IMAGE_DIR"
echo "Output:     $OUTPUT"
echo "================================================================"

mkdir -p $(dirname $OUTPUT)

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/inference/lstv_classifier.py \
        --weights /work/runs/lstv/trial/weights/best.pt \
        --image-dir /work/results/lstv_screening/trial/candidate_images \
        --output /work/results/inference/trial_classifications.json \
        --conf 0.25

echo "================================================================"
echo "Classification complete!"
echo "End time: $(date)"
echo "================================================================"

if [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "Summary statistics:"
    python3 << PYEOF
import json
with open('$OUTPUT') as f:
    data = json.load(f)
    
classes = {}
for result in data:
    cls = result['classification']
    classes[cls] = classes.get(cls, 0) + 1

print("\nClassifications:")
for cls, count in sorted(classes.items()):
    print(f"  {cls:20s}: {count:4d} ({count/len(data)*100:.1f}%)")
print(f"\nTotal: {len(data)} images")
PYEOF
fi
