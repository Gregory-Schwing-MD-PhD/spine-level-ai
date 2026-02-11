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

set -euo pipefail
set -x

echo "================================================================"
echo "LSTV Screening Pipeline - TRIAL RUN (5 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

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
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"
SERIES_CSV="${PROJECT_DIR}/data/raw/train_series_descriptions.csv"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/trial"
NIFTI_DIR="${OUTPUT_DIR}/nifti"
SCRIPT_PATH="${PROJECT_DIR}/src/screening/lstv_screen.py"
MODELS_CACHE="${PROJECT_DIR}/spineps_models"

mkdir -p $MODELS_CACHE
mkdir -p $OUTPUT_DIR/logs
mkdir -p $NIFTI_DIR

DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-spineps:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spine-level-ai-spineps.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling SPINEPS container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

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

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/input \
    --bind $OUTPUT_DIR:/data/output \
    --bind $NIFTI_DIR:/data/output/nifti \
    --bind $MODELS_CACHE:/app/models \
    --bind $(dirname $SERIES_CSV):/data/raw \
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

echo "================================================================"
echo "Complete!"
echo "End time: $(date)"
echo "================================================================"
