#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=weak_labels_trial
#SBATCH -o logs/weak_labels_trial_%j.out
#SBATCH -e logs/weak_labels_trial_%j.err

set -euo pipefail

echo "================================================================"
echo "WEAK LABEL GENERATION - TRIAL (50 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/lstv_screening/trial/nifti"
SEG_DIR="${PROJECT_DIR}/results/lstv_screening/trial/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_trial"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

echo "Generating weak labels from SPINEPS segmentations..."
echo "Input NIfTI: $NIFTI_DIR"
echo "Input Segs:  $SEG_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Container:   $IMG_PATH"
echo "================================================================"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $NIFTI_DIR:/data/nifti \
    --bind $SEG_DIR:/data/seg \
    --bind $OUTPUT_DIR:/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/generate_weak_labels.py \
        --nifti_dir /data/nifti \
        --seg_dir /data/seg \
        --output_dir /data/output \
        --limit 50

echo "================================================================"
echo "Complete!"
echo "End time: $(date)"
echo "================================================================"
