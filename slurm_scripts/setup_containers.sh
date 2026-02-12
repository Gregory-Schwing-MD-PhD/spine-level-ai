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
set -x

echo "================================================================"
echo "Pulling All LSTV Detection Containers - VERBOSE MODE"
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

# --- 3. Container Pull Logic with MAX VERBOSITY ---
# Using -v (verbose) to show progress and download details
CACHE_DIR="${HOME}/singularity_cache"

echo ""
echo "[1/3] Pulling Preprocessing container..."
singularity -v pull --force \
    "${CACHE_DIR}/spine-level-ai-preprocessing.sif" \
    docker://go2432/spine-level-ai-preprocessing:latest

echo ""
echo "[2/3] Pulling SPINEPS container..."
# Using -v here will show you if it's correctly identifying the 4.45GB layers
singularity -v pull --force \
    "${CACHE_DIR}/spine-level-ai-spineps.sif" \
    docker://go2432/spine-level-ai-spineps:latest

echo ""
echo "[3/3] Pulling YOLOv11 container..."
singularity -v pull --force \
    "${CACHE_DIR}/spine-level-ai-yolo.sif" \
    docker://go2432/spine-level-ai-yolo:latest

echo ""
echo "================================================================"
echo "All containers ready!"
echo "Location: $CACHE_DIR"
echo "End time: $(date)"
echo "================================================================"
