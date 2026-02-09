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
DATA_DIR="${PROJECT_DIR}/data/raw"

# Docker container (will be auto-converted to Singularity)
DOCKER_USERNAME="go2432"  # Change to your Docker Hub username
CONTAINER="docker://${DOCKER_USERNAME}/spine-level-ai-preprocessing:latest"

# Check for Kaggle credentials
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: Kaggle credentials not found at $KAGGLE_JSON"
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New Token'"
    echo "3. Move kaggle.json to ~/.kaggle/"
    echo "4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "Kaggle credentials found: $KAGGLE_JSON"

# Create download script
cat > ${PROJECT_DIR}/tmp_download.py << 'PYEOF'
import os
import subprocess
from pathlib import Path

def download_rsna():
    """Download RSNA 2024 dataset using Kaggle API"""
    
    data_dir = Path("/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading RSNA 2024 Lumbar Spine dataset...")
    print("This is ~150GB and will take several hours.")
    
    # Download using kaggle CLI
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "rsna-2024-lumbar-spine-degenerative-classification",
        "-p", str(data_dir)
    ]
    
    subprocess.run(cmd, check=True)
    
    print("\nDownload complete!")
    print("Extracting archives...")
    
    # Extract zip files
    import zipfile
    zip_files = list(data_dir.glob("*.zip"))
    
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip after extraction to save space
        zip_file.unlink()
        print(f"  ✓ Extracted and removed {zip_file.name}")
    
    print("\nAll done!")

if __name__ == "__main__":
    download_rsna()
PYEOF

# Run download in Singularity container (auto-converts from Docker)
echo "Starting download with Singularity (Docker container)..."
echo "Container: $CONTAINER"

singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $DATA_DIR:/data/raw \
    --bind $HOME/.kaggle:/root/.kaggle:ro \
    --pwd /work \
    "$CONTAINER" \
    python tmp_download.py

# Cleanup
rm -f ${PROJECT_DIR}/tmp_download.py

echo "================================================================"
echo "Download complete!"
echo "End time: $(date)"
echo "Data location: $DATA_DIR"
echo "================================================================"

# Create dataset summary
echo "Creating dataset summary..."
singularity exec \
    --bind $DATA_DIR:/data/raw \
    "$CONTAINER" \
    bash -c "ls -lh /data/raw && du -sh /data/raw"

echo "DONE."
