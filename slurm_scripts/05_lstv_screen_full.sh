#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=48:00:00
#SBATCH --job-name=lstv_full
#SBATCH -o logs/lstv_full_%j.out
#SBATCH -e logs/lstv_full_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

echo "================================================================"
echo "LSTV Screening Pipeline - FULL RUN (ALL STUDIES)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
echo "================================================================"

nvidia-smi

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

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
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/full"
SCRIPT_PATH="${PROJECT_DIR}/src/screening/lstv_screen.py"
MODELS_CACHE="${PROJECT_DIR}/spineps_models"

mkdir -p $MODELS_CACHE

IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spineps.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: Container not found at $IMG_PATH"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

ROBOFLOW_KEY="izolWNqCVveKyMrACYzN"
ROBOFLOW_WORKSPACE="spinelevelai"
ROBOFLOW_PROJECT="lstv-candidates"

echo "================================================================"
echo "Starting FULL LSTV screening..."
echo "Script: $SCRIPT_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "================================================================"

TOTAL_STUDIES=$(find $DATA_DIR -maxdepth 1 -type d | wc -l)
echo "Total studies to process: $TOTAL_STUDIES"

singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/input \
    --bind $OUTPUT_DIR:/data/output \
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
        --roboflow_key "$ROBOFLOW_KEY" \
        --roboflow_workspace "$ROBOFLOW_WORKSPACE" \
        --roboflow_project "$ROBOFLOW_PROJECT" \
        --verbose

exit_code=$?

echo "================================================================"
echo "Full screening complete!"
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "================================================================"

if [[ -f "$OUTPUT_DIR/results.csv" ]]; then
    total=$(wc -l < "$OUTPUT_DIR/results.csv")
    lstv=$(grep -c "True" "$OUTPUT_DIR/results.csv" || true)
    normal=$(grep -c "False" "$OUTPUT_DIR/results.csv" || true)
    
    echo "Results summary:"
    echo "  Total processed: $total"
    echo "  LSTV candidates: $lstv"
    echo "  Normal: $normal"
fi
