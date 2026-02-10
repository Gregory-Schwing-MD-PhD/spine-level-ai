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
echo "LSTV Screening Pipeline - FULL PRODUCTION RUN"
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
DATA_DIR="${PROJECT_DIR}/data/raw/train_images"  # RSNA dataset location
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/full"  # ← Fixed: outputs in results/
SCRIPT_PATH="${PROJECT_DIR}/src/screening/lstv_screen.py"  # ← Fixed: script location

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

# Roboflow API key (set this as environment variable)
ROBOFLOW_KEY="${ROBOFLOW_API_KEY:-your_roboflow_key_here}"

echo "================================================================"
echo "Starting LSTV screening (FULL DATASET - ~2700 studies)..."
echo "Script: $SCRIPT_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Expected runtime: 24-48 hours"
echo "================================================================"

# Run screening with GPU support (--nv flag) - NO LIMIT
singularity exec --nv \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/input \
    --bind $OUTPUT_DIR:/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/screening/lstv_screen.py \
        --input_dir /data/input \
        --output_dir /data/output \
        --roboflow_key "$ROBOFLOW_KEY" \
        --roboflow_workspace "lstv-screening" \
        --roboflow_project "lstv-candidates" \
        --verbose

exit_code=$?

echo "================================================================"
echo "Full screening complete!"
echo "End time: $(date)"
echo "Exit code: $exit_code"
echo "================================================================"

# Generate summary report
if [[ -f "$OUTPUT_DIR/results.csv" ]]; then
    echo "Generating summary statistics..."
    
    singularity exec \
        --bind $OUTPUT_DIR:/data/output \
        "$IMG_PATH" \
        python -c "
import pandas as pd
df = pd.read_csv('/data/output/results.csv')
print('='*60)
print('FINAL SUMMARY STATISTICS')
print('='*60)
print(f'Total studies processed: {len(df)}')
print(f'LSTV candidates flagged: {df[\"is_lstv_candidate\"].sum()}')
print(f'')
print('Vertebra count distribution:')
print(df['vertebra_count'].value_counts().sort_index())
print(f'')
print(f'Fusion detected: {df[\"fusion_detected\"].sum()}')
print('='*60)
"
fi

echo ""
echo "Output files location:"
echo "  Results CSV:      $OUTPUT_DIR/results.csv"
echo "  Progress JSON:    $OUTPUT_DIR/progress.json"
echo "  Candidate images: $OUTPUT_DIR/candidate_images/"
echo "  NIfTI files:      $OUTPUT_DIR/nifti/"
echo "  Segmentations:    $OUTPUT_DIR/segmentations/"
echo ""

echo "DONE."
