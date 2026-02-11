#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=classify_full
#SBATCH -o logs/classify_full_%j.out
#SBATCH -e logs/classify_full_%j.err

set -euo pipefail

echo "================================================================"
echo "LSTV CLASSIFICATION - Full Production Batch"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
WEIGHTS="${PROJECT_DIR}/runs/lstv/full/weights/best.pt"
IMAGE_DIR="${PROJECT_DIR}/results/lstv_screening/full/candidate_images"
OUTPUT="${PROJECT_DIR}/results/inference/full_classifications.json"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: Model weights not found at $WEIGHTS"
    echo "Run full training first: sbatch slurm_scripts/08_train_yolo_full.sh"
    exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "ERROR: Image directory not found at $IMAGE_DIR"
    echo "Run full screening first: sbatch slurm_scripts/05_lstv_screen_full.sh"
    exit 1
fi

NUM_IMAGES=$(find $IMAGE_DIR -name "*.jpg" | wc -l)
echo "Classifying $NUM_IMAGES LSTV candidates..."
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
        --weights /work/runs/lstv/full/weights/best.pt \
        --image-dir /work/results/lstv_screening/full/candidate_images \
        --output /work/results/inference/full_classifications.json \
        --conf 0.25

echo "================================================================"
echo "Classification complete!"
echo "End time: $(date)"
echo "================================================================"

if [[ -f "$OUTPUT" ]]; then
    echo ""
    echo "Final Statistics:"
    python3 << PYEOF
import json
with open('$OUTPUT') as f:
    data = json.load(f)

classes = {}
high_conf = {}
for result in data:
    cls = result['classification']
    conf = result['confidence']
    classes[cls] = classes.get(cls, 0) + 1
    if conf > 0.7:
        high_conf[cls] = high_conf.get(cls, 0) + 1

print("\nAll Classifications:")
for cls, count in sorted(classes.items()):
    print(f"  {cls:20s}: {count:4d} ({count/len(data)*100:.1f}%)")

print("\nHigh Confidence (>0.7):")
for cls, count in sorted(high_conf.items()):
    print(f"  {cls:20s}: {count:4d}")

print(f"\nTotal: {len(data)} images")

# Count clinical recommendations
needs_review = sum(1 for r in data if 'MANUAL REVIEW' in r['clinical_recommendation'])
lstv_detected = sum(1 for r in data if 'LSTV DETECTED' in r['clinical_recommendation'])

print(f"\nClinical Summary:")
print(f"  LSTV Detected:      {lstv_detected}")
print(f"  Needs Manual Review: {needs_review}")
print(f"  Normal:             {classes.get('NORMAL', 0)}")
PYEOF
fi

echo ""
echo "Results saved to: $OUTPUT"
echo "================================================================"
