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
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/full"
NIFTI_DIR="${OUTPUT_DIR}/nifti"  # ← ADDED (was missing!)
SCRIPT_PATH="${PROJECT_DIR}/src/screening/lstv_screen.py"
MODELS_CACHE="${PROJECT_DIR}/spineps_models"

mkdir -p $MODELS_CACHE
mkdir -p $OUTPUT_DIR/logs  # ← ADDED
mkdir -p $NIFTI_DIR        # ← ADDED (was missing!)

DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-spineps:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spine-level-ai-spineps.sif"  # ← FIXED (matches trial!)

# Auto-pull if missing (like trial script)
if [[ ! -f "$IMG_PATH" ]]; then
    echo "Pulling SPINEPS container..."
    singularity pull "$IMG_PATH" "$CONTAINER"
fi

echo "Container ready: $IMG_PATH"

ROBOFLOW_KEY="izolWNqCVveKyMrACYzN"
ROBOFLOW_WORKSPACE="spinelevelai"
ROBOFLOW_PROJECT="lstv-candidates"

echo "================================================================"
echo "Starting FULL LSTV screening..."
echo "Script: $SCRIPT_PATH"
echo "Data: $DATA_DIR"
echo "Series CSV: $SERIES_CSV"
echo "Output: $OUTPUT_DIR"
echo "Models Cache: $MODELS_CACHE"
echo "Roboflow: $ROBOFLOW_WORKSPACE/$ROBOFLOW_PROJECT"
echo "================================================================"

TOTAL_STUDIES=$(find $DATA_DIR -maxdepth 1 -type d | wc -l)
echo "Total studies available: $TOTAL_STUDIES"
echo ""
echo "This will process ALL studies in the dataset."
echo "Expected LSTV candidates: ~500 (18% of total)"
echo "Expected duration: 24-48 hours"
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

    echo ""
    echo "Results summary:"
    echo "  Total processed: $total"
    echo "  LSTV candidates: $lstv (~$((lstv * 100 / total))%)"
    echo "  Normal: $normal (~$((normal * 100 / total))%)"
    echo ""
    
    if [[ $lstv -lt 400 ]]; then
        echo "⚠️  WARNING: Expected ~500 LSTV candidates, found $lstv"
        echo "   This may indicate issues with screening criteria"
    elif [[ $lstv -gt 600 ]]; then
        echo "⚠️  WARNING: Found $lstv LSTV candidates (expected ~500)"
        echo "   This may indicate overly permissive screening"
    else
        echo "✓ LSTV candidate count within expected range"
    fi
    echo ""
fi

echo "Output locations:"
echo "  Results CSV:      $OUTPUT_DIR/results.csv"
echo "  NIfTI files:      $OUTPUT_DIR/nifti/"
echo "  Segmentations:    $OUTPUT_DIR/segmentations/"
echo "  Candidate images: $OUTPUT_DIR/candidate_images/"
echo ""
echo "Next step:"
echo "  sbatch slurm_scripts/06_generate_weak_labels_full.sh"
echo ""
echo "================================================================"
