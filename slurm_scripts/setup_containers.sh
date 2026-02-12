#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=setup_containers
#SBATCH -o logs/setup_containers_%j.out
#SBATCH -e logs/setup_containers_%j.err

set -euo pipefail
# Removed 'set -x' to avoid cluttering the logs with valid skip checks
# set -x 

echo "================================================================"
echo "Pulling Missing LSTV Detection Containers"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "================================================================"

# --- 1. Fast Scratch Setup ---
export SINGULARITY_TMPDIR="/tmp/${USER}_setup_containers_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="$SINGULARITY_TMPDIR/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"

mkdir -p "$SINGULARITY_TMPDIR" "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

# Cleanup trap
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT

# Debug info: Check space on the node local disk
echo "Checking local scratch space availability..."
df -h "$SINGULARITY_TMPDIR"

# --- 2. Environment Setup ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

# --- 3. Container Pull Logic ---
CACHE_DIR="${HOME}/singularity_cache"
echo "Cache Directory: $CACHE_DIR"

# --- PREPROCESSING ---
SIF_PRE="${CACHE_DIR}/spine-level-ai-preprocessing.sif"
echo ""
if [[ -f "$SIF_PRE" ]]; then
    echo "✓ [1/3] Preprocessing container exists. Skipping pull."
else
    echo "[1/3] Pulling Preprocessing container..."
    singularity -v pull "$SIF_PRE" docker://go2432/spine-level-ai-preprocessing:latest
fi

# --- SPINEPS ---
SIF_SPINEPS="${CACHE_DIR}/spine-level-ai-spineps.sif"
echo ""
if [[ -f "$SIF_SPINEPS" ]]; then
    echo "✓ [2/3] SPINEPS container exists. Skipping pull."
else
    echo "[2/3] Pulling SPINEPS container..."
    # Note: Removed --force so we don't accidentally overwrite if logic fails
    singularity -v pull "$SIF_SPINEPS" docker://go2432/spine-level-ai-spineps:latest
fi

# --- YOLO ---
SIF_YOLO="${CACHE_DIR}/spine-level-ai-yolo.sif"
echo ""
if [[ -f "$SIF_YOLO" ]]; then
    echo "✓ [3/3] YOLOv11 container exists. Skipping pull."
else
    echo "[3/3] Pulling YOLOv11 container..."
    singularity -v pull "$SIF_YOLO" docker://go2432/spine-level-ai-yolo:latest
fi

echo ""
echo "================================================================"
echo "Container Setup Complete!"
echo "----------------------------------------------------------------"
ls -lh "$CACHE_DIR"/*.sif
echo "----------------------------------------------------------------"
echo "End time: $(date)"
echo "================================================================"
