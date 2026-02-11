#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=weak_labels_full
#SBATCH -o logs/weak_labels_full_%j.out
#SBATCH -e logs/weak_labels_full_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "WEAK LABEL GENERATION - FULL (500 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/lstv_screening/full/nifti"
SEG_DIR="${PROJECT_DIR}/results/lstv_screening/full/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_full"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

if [[ ! -d "$NIFTI_DIR" ]]; then
    echo "ERROR: NIfTI directory not found at $NIFTI_DIR"
    echo "Run full screening first: sbatch slurm_scripts/05_lstv_screen_full.sh"
    exit 1
fi

if [[ ! -d "$SEG_DIR" ]]; then
    echo "ERROR: Segmentation directory not found at $SEG_DIR"
    echo "Run full screening first: sbatch slurm_scripts/05_lstv_screen_full.sh"
    exit 1
fi

NIFTI_COUNT=$(find $NIFTI_DIR -name "*.nii.gz" | wc -l)
SEG_COUNT=$(find $SEG_DIR -name "*_seg.nii.gz" | wc -l)

echo "Generating weak labels from SPINEPS segmentations..."
echo "Input NIfTI:   $NIFTI_DIR ($NIFTI_COUNT files)"
echo "Input Segs:    $SEG_DIR ($SEG_COUNT files)"
echo "Output:        $OUTPUT_DIR"
echo "Limit:         500 studies (for training)"
echo "Container:     $IMG_PATH"
echo "================================================================"

if [[ $SEG_COUNT -lt 100 ]]; then
    echo "WARNING: Only $SEG_COUNT segmentations found"
    echo "Expected ~500 LSTV candidates from full screening"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

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
        --limit 500

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Label generation failed"
    exit $exit_code
fi

echo "================================================================"
echo "Complete!"
echo "End time: $(date)"
echo "================================================================"

if [[ -f "$OUTPUT_DIR/metadata.json" ]]; then
    echo ""
    echo "Dataset Summary:"
    cat $OUTPUT_DIR/metadata.json
    echo ""
fi

echo "Output location: $OUTPUT_DIR"
echo ""
echo "Next step:"
echo "  sbatch slurm_scripts/08_train_yolo_full.sh"
echo "================================================================"
