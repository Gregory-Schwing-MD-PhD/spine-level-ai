#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=fuse_labels
#SBATCH -o logs/fuse_labels_%j.out
#SBATCH -e logs/fuse_labels_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "LABEL FUSION - Merging Weak + Human Labels"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

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
WEAK_LABELS_DIR="${PROJECT_DIR}/data/training/lstv_yolo_full/labels/train"
HUMAN_LABELS_DIR="${PROJECT_DIR}/data/training/human_refined/labels"
WEAK_IMAGES_DIR="${PROJECT_DIR}/data/training/lstv_yolo_full/images/train"
HUMAN_IMAGES_DIR="${PROJECT_DIR}/data/training/human_refined/images"
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_refined"

# Container
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spineps-lstv-yolo:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling YOLOv11 container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

# Check inputs exist
if [[ ! -d "$WEAK_LABELS_DIR" ]]; then
    echo "ERROR: Weak labels not found at $WEAK_LABELS_DIR"
    echo "Run: sbatch slurm_scripts/06_generate_weak_labels_full.sh"
    exit 1
fi

if [[ ! -d "$HUMAN_LABELS_DIR" ]]; then
    echo "ERROR: Human labels not found at $HUMAN_LABELS_DIR"
    echo "Expected location: data/training/human_refined/labels/"
    echo ""
    echo "Did you:"
    echo "  1. Med students annotate 200 images?"
    echo "  2. Export from Roboflow to this directory?"
    exit 1
fi

HUMAN_COUNT=$(find $HUMAN_LABELS_DIR -name "*.txt" | wc -l)

if [[ $HUMAN_COUNT -lt 50 ]]; then
    echo "WARNING: Only $HUMAN_COUNT human labels found"
    echo "Expected ~200 labels"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "================================================================"
echo "Fusing labels..."
echo "Weak labels:  $WEAK_LABELS_DIR ($(find $WEAK_LABELS_DIR -name "*.txt" | wc -l) files)"
echo "Human labels: $HUMAN_LABELS_DIR ($HUMAN_COUNT files)"
echo "Output:       $OUTPUT_DIR"
echo "================================================================"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/fuse_labels.py \
        --weak_labels_dir /work/data/training/lstv_yolo_full/labels/train \
        --human_labels_dir /work/data/training/human_refined/labels \
        --weak_images_dir /work/data/training/lstv_yolo_full/images/train \
        --human_images_dir /work/data/training/human_refined/images \
        --output_dir /work/data/training/lstv_yolo_refined \
        --iou_threshold 0.3

echo "================================================================"
echo "Label fusion complete!"
echo "End time: $(date)"
echo "================================================================"

# Check output
if [[ -f "$OUTPUT_DIR/dataset.yaml" ]]; then
    echo ""
    echo "✓ Refined dataset created successfully!"
    echo ""
    echo "Location: $OUTPUT_DIR"
    echo ""
    
    FUSED_COUNT=$(find $OUTPUT_DIR/images/train -name "*.jpg" 2>/dev/null | wc -l)
    echo "Total images: $FUSED_COUNT"
    echo ""
    
    if [[ -f "$OUTPUT_DIR/fusion_metrics.json" ]]; then
        echo "Fusion metrics:"
        cat $OUTPUT_DIR/fusion_metrics.json
        echo ""
    fi
    
    echo "================================================================"
    echo "NEXT STEP: Train refined model"
    echo "================================================================"
    echo ""
    echo "  sbatch slurm_scripts/09_train_full_refined.sh"
    echo ""
    echo "This will train on the fused dataset (weak + human labels)"
    echo "Expected improvement: +15-25% over baseline"
    echo ""
else
    echo "ERROR: Refined dataset not created!"
    exit 1
fi

echo "================================================================"
