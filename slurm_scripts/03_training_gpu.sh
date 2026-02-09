#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --constraint=v100
#SBATCH --time=72:00:00
#SBATCH --job-name=train_spine_yolo
#SBATCH -o logs/training_%j.out
#SBATCH -e logs/training_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

echo "================================================================"
echo "Spine Level Identification - YOLOv8 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# Check GPU status
echo "GPU Status:"
nvidia-smi

# Setup environment
# 1. SETUP - Conda environment with Nextflow (for Singularity)
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

# Verify singularity available
which singularity || echo "WARNING: singularity not found in PATH"

# 2. CACHE - Singularity cache directories
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p $XDG_RUNTIME_DIR $NXF_SINGULARITY_CACHEDIR

# 3. SAFETY - Clean environment
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset R_LIBS
unset R_LIBS_USER
unset R_LIBS_SITE

# Project directories
PROJECT_DIR="$(pwd)"
DATA_DIR="${PROJECT_DIR}/data"
MODELS_DIR="${PROJECT_DIR}/models"
RESULTS_DIR="${PROJECT_DIR}/results"

# Create directories
mkdir -p $MODELS_DIR/checkpoints $RESULTS_DIR

# Docker container (auto-converts to Singularity)
DOCKER_USERNAME="go2432"  # Change to your Docker Hub username
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-training:latest"

# Safety: Unset conflicting environment variables
unset LD_LIBRARY_PATH
unset PYTHONPATH

echo "================================================================"
echo "Starting YOLOv8 training..."
echo "Container: $CONTAINER"
echo "================================================================"

# Run training with GPU support (--nv flag)
singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data \
    --bind $MODELS_DIR:/models \
    --bind $RESULTS_DIR:/results \
    --pwd /work \
    "$CONTAINER" \
    python src/training/train_yolo.py \
        --data /data/processed/dataset.yaml \
        --epochs 100 \
        --batch 16 \
        --img 640 \
        --device 0,1 \
        --project /models \
        --name spine_level_v1 \
        --save-period 10 \
        --patience 20 \
        --workers 8

echo "================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "Model saved to: $MODELS_DIR/spine_level_v1"
echo "================================================================"

# Copy best model to checkpoints
if [[ -f "$MODELS_DIR/spine_level_v1/weights/best.pt" ]]; then
    cp "$MODELS_DIR/spine_level_v1/weights/best.pt" \
       "$MODELS_DIR/checkpoints/best_$(date +%Y%m%d).pt"
    echo "Best model backed up to checkpoints/"
fi

echo "DONE."
