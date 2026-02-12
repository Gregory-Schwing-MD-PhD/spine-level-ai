#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=build_spineps
#SBATCH -o logs/build_spineps_%j.out
#SBATCH -e logs/build_spineps_%j.err

set -euo pipefail
set -x

echo "================================================================"
echo "Building SPINEPS Singularity Container"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "================================================================"

# --- 1. Environment Setup (Replicated Exactly) ---
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:$PATH"
unset JAVA_HOME

which singularity

# --- 2. CRITICAL CHANGE: Use Fast Scratch Space ---
# We use /wsu/tmp because Home is too slow and default /tmp is too small.
# We create a unique subfolder based on Job ID to avoid collisions.
SCRATCH_DIR="/wsu/tmp/${USER}/build_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"

export SINGULARITY_TMPDIR="$SCRATCH_DIR"
export XDG_RUNTIME_DIR="${HOME}/xdr"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"

# Create directories
mkdir -p "$XDG_RUNTIME_DIR" "$NXF_SINGULARITY_CACHEDIR"

# Check available space in scratch before starting
echo "Checking space in scratch ($SCRATCH_DIR):"
df -h "$SCRATCH_DIR"

export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset R_LIBS
unset R_LIBS_USER
unset R_LIBS_SITE

# --- 3. Define Paths ---
DOCKER_USERNAME="go2432"
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-spineps:latest"
IMG_PATH="${NXF_SINGULARITY_CACHEDIR}/spine-level-ai-spineps.sif"

echo "Target Image Path: $IMG_PATH"
echo "Build Temporary Dir: $SINGULARITY_TMPDIR"

# --- 4. Pull the Container ---
# We use --force to overwrite any broken/partial files from previous attempts
echo "================================================================"
echo "Starting Pull..."
echo "================================================================"

singularity pull --force "$IMG_PATH" "$CONTAINER"

# --- 5. Cleanup ---
# CRITICAL: Delete the temp folder immediately to free up the 35GB for others
echo "================================================================"
echo "Cleaning up scratch space..."
rm -rf "$SCRATCH_DIR"
echo "Cleanup complete."
echo "================================================================"

echo "Complete!"
echo "End time: $(date)"
echo "================================================================"
