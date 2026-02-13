#!/bin/bash
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --time=02:00:00
#SBATCH --job-name=lstv_diagnostic
#SBATCH -o logs/lstv_diagnostic_%j.out
#SBATCH -e logs/lstv_diagnostic_%j.err

set -euo pipefail

# ============================================================================
# CRITICAL FIX: Extract BOTH instance AND semantic SPINEPS outputs
# ============================================================================

PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/lstv_screening/trial_enhanced/nifti"
SEG_DIR="${PROJECT_DIR}/results/lstv_screening/trial_enhanced/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/results/lstv_screening/trial_enhanced"
DIAGNOSTIC_DIR="${OUTPUT_DIR}/diagnostics"

mkdir -p "$DIAGNOSTIC_DIR"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║ SPINEPS OUTPUT DIAGNOSTIC                                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Determine if SPINEPS semantic segmentation (with ribs) is available"
echo ""

# Setup container
IMG_PATH="${HOME}/singularity_cache/spine-level-ai-spineps.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "✗ Container not found: $IMG_PATH"
    exit 1
fi

# Find a study to diagnose
STUDY_ID=$(ls -1 "$NIFTI_DIR" | grep "sub-" | head -1 | sed 's/sub-//' | sed 's/_T2w.nii.gz//')

if [ -z "$STUDY_ID" ]; then
    echo "✗ No studies found in $NIFTI_DIR"
    exit 1
fi

echo "Diagnosing study: $STUDY_ID"
echo ""

# Run diagnostic
singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $NIFTI_DIR:/data/nifti \
    --bind $SEG_DIR:/data/seg \
    --bind $DIAGNOSTIC_DIR:/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/screening/diagnose_spineps_output.py \
        --nifti_dir /data/nifti \
        --seg_dir /data/seg \
        --study_id "$STUDY_ID" \
        --output_dir /data/output

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║ DIAGNOSTIC COMPLETE                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results: $DIAGNOSTIC_DIR/${STUDY_ID}_diagnostic_report.json"
echo ""
echo "Next steps:"
echo "  1. Review diagnostic report to see if semantic labels exist"
echo "  2. If semantic labels exist → ribs can be detected from labels"
echo "  3. If not → need to modify SPINEPS extraction or use intensity fallback"
echo ""
