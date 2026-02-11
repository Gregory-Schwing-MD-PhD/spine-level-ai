#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=download_rsna
#SBATCH -o logs/download_%j.out
#SBATCH -e logs/download_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

echo "================================================================"
echo "RSNA 2024 Dataset Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

which singularity || echo "WARNING: singularity not found"

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
DATA_DIR="${PROJECT_DIR}/data/raw"
mkdir -p "$DATA_DIR"

DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-preprocessing:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spine-level-ai-preprocessing.sif"

KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: Kaggle credentials not found at $KAGGLE_JSON"
    echo ""
    echo "Setup instructions:"
    echo "  1. Go to: https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Save kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "Kaggle credentials found: $KAGGLE_JSON"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling preprocessing container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

mkdir -p ${PROJECT_DIR}/.kaggle_tmp
cp ${HOME}/.kaggle/kaggle.json ${PROJECT_DIR}/.kaggle_tmp/
chmod 600 ${PROJECT_DIR}/.kaggle_tmp/kaggle.json

echo "================================================================"
echo "Starting download..."
echo "This will take several hours for ~150 GB dataset"
echo "================================================================"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/raw \
    --bind ${PROJECT_DIR}/.kaggle_tmp:/root/.kaggle \
    --pwd /work \
    "$IMG_PATH" \
    python /work/download_rsna_dataset.py

exit_code=$?

rm -rf ${PROJECT_DIR}/.kaggle_tmp

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Download failed"
    exit $exit_code
fi

echo "================================================================"
echo "Download complete!"
echo "End time: $(date)"
echo "================================================================"

ls -lh ${DATA_DIR}/
echo ""
echo "Next step: sbatch slurm_scripts/04_lstv_screen_trial.sh"
echo "================================================================"
