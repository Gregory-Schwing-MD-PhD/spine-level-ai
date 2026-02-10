#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=02:00:00
#SBATCH --job-name=lstv_trial
#SBATCH -o logs/lstv_trial_%j.out
#SBATCH -e logs/lstv_trial_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

echo "================================================================"
echo "LSTV Screening Pipeline - TRIAL RUN (5 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

# Check GPU status
echo "GPU Status:"
nvidia-smi

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
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/trial"
SCRIPT_PATH="${PROJECT_DIR}/src/screening/lstv_screen.py"

# CRITICAL: Create writable models cache directory
MODELS_CACHE="${PROJECT_DIR}/spineps_models"
mkdir -p $MODELS_CACHE

# Create directories
mkdir -p $OUTPUT_DIR/logs

# Docker container (auto-converts to Singularity)
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spineps-lstv:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps-lstv.sif"

# Pull/convert container if not exists
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling and converting Docker container to Singularity..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

# Roboflow credentials
ROBOFLOW_KEY="izolWNqCVveKyMrACYzN"
ROBOFLOW_WORKSPACE="spinelevelai"
ROBOFLOW_PROJECT="lstv-candidates"

echo "================================================================"
echo "Starting LSTV screening (TRIAL - 5 studies)..."
echo "Script: $SCRIPT_PATH"
echo "Data: $DATA_DIR"
echo "Series CSV: $SERIES_CSV"
echo "Output: $OUTPUT_DIR"
echo "Models Cache: $MODELS_CACHE"
echo "Roboflow: $ROBOFLOW_WORKSPACE/$ROBOFLOW_PROJECT"
echo "================================================================"

# Run screening with GPU support (--nv flag)
# CRITICAL: Bind models cache to BOTH locations SPINEPS expects
singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/input \
    --bind $OUTPUT_DIR:/data/output \
    --bind $MODELS_CACHE:/app/models \
    --bind $(dirname $SERIES_CSV):/data/raw \
    --bind $MODELS_CACHE:/opt/conda/lib/python3.10/site-packages/spineps/models \
    --env SPINEPS_SEGMENTOR_MODELS=/app/models \
    --env SPINEPS_ENVIRONMENT_DIR=/app/models \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/screening/lstv_screen.py \
        --input_dir /data/input \
        --series_csv /data/raw/train_series_descriptions.csv \
        --output_dir /data/output \
        --limit 5 \
        --roboflow_key "$ROBOFLOW_KEY" \
        --roboflow_workspace "$ROBOFLOW_WORKSPACE" \
        --roboflow_project "$ROBOFLOW_PROJECT" \
        --verbose

exit_code=$?

echo "================================================================"
echo "Trial run complete!"
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "================================================================"

# Show results
if [[ -f "$OUTPUT_DIR/results.csv" ]]; then
    echo "Results summary:"
    head -10 "$OUTPUT_DIR/results.csv"
fi

if [[ -f "$OUTPUT_DIR/progress.json" ]]; then
    echo ""
    echo "Progress checkpoint:"
    cat "$OUTPUT_DIR/progress.json"
fi

echo ""
echo "Output files location:"
echo "  Results CSV:      $OUTPUT_DIR/results.csv"
echo "  Progress JSON:    $OUTPUT_DIR/progress.json"
echo "  Candidate images: $OUTPUT_DIR/candidate_images/"
echo ""

echo "DONE."
