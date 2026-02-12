#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=weak_labels_full
#SBATCH -o logs/weak_labels_full_%j.out
#SBATCH -e logs/weak_labels_full_%j.err
#SBATCH --mail-user=go2432@wayne.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "================================================================"
echo "WEAK LABEL GENERATION - FULL (500 studies)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/lstv_screening/full/nifti"
SEG_DIR="${PROJECT_DIR}/results/lstv_screening/full/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_full"

IMG_PATH="${HOME}/singularity_cache/spine-level-ai-yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

if [[ ! -d "$NIFTI_DIR" ]]; then
    echo "ERROR: NIfTI directory not found at $NIFTI_DIR"
    echo "Run full screening first: sbatch slurm_scripts/05_lstv_screen_full.sh"
    exit 1
fi

if [[ ! -d "$SEG_DIR" ]]; then
    echo "ERROR: Segmentation directory not found at $SEG_DIR"
    echo "Run full screening first: sbatch slurm_scripts/05_lstv_screen_full.sh"
    exit 1
fi

NIFTI_COUNT=$(find $NIFTI_DIR -name "*.nii.gz" | wc -l)
SEG_COUNT=$(find $SEG_DIR -name "*_seg.nii.gz" | wc -l)

echo "Generating weak labels with SPINE-AWARE slice selection..."
echo "Input NIfTI:   $NIFTI_DIR ($NIFTI_COUNT files)"
echo "Input Segs:    $SEG_DIR ($SEG_COUNT files)"
echo "Output:        $OUTPUT_DIR"
echo "Limit:         500 studies (for training)"
echo "Container:     $IMG_PATH"
echo ""
echo "SPINE-AWARE MODE:"
echo "  - Intelligent midline detection using segmentation"
echo "  - Quantitative metrics will be calculated"
echo "  - No comparison images (too many for full run)"
echo "================================================================"

if [[ $SEG_COUNT -lt 100 ]]; then
    echo "WARNING: Only $SEG_COUNT segmentations found"
    echo "Expected ~500 LSTV candidates from full screening"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Run WITHOUT comparison generation (too many files)
singularity exec \
    --bind $PROJECT_DIR:/work \
    --bind $NIFTI_DIR:/data/nifti \
    --bind $SEG_DIR:/data/seg \
    --bind $OUTPUT_DIR:/data/output \
    --pwd /work \
    "$IMG_PATH" \
    python /work/src/training/generate_weak_labels.py \
        --nifti_dir /data/nifti \
        --seg_dir /data/seg \
        --output_dir /data/output \
        --limit 500

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Label generation failed"
    exit $exit_code
fi

echo ""
echo "================================================================"
echo "Complete! Metrics Summary:"
echo "================================================================"

# Display metrics
if [[ -f "$OUTPUT_DIR/spine_aware_metrics_report.json" ]]; then
    echo ""
    echo "SPINE-AWARE SLICE SELECTION RESULTS:"
    python3 << PYEOF
import json
with open('$OUTPUT_DIR/spine_aware_metrics_report.json') as f:
    stats = json.load(f)

print(f"Total cases:          {stats['total_cases']}")
print(f"Spine-aware success:  {stats['spine_aware_cases']} ({stats['spine_aware_cases']/stats['total_cases']*100:.1f}%)")
print(f"\nOffset corrections applied:")
print(f"  Mean:   {stats['offset_statistics']['mean_mm']:.1f} ± {stats['offset_statistics']['std_mm']:.1f} mm")
print(f"  Median: {stats['offset_statistics']['median_mm']:.1f} mm")
print(f"  Max:    {stats['offset_statistics']['max_mm']:.1f} mm")
print(f"\nSpine density improvement:")
print(f"  Mean:   {stats['improvement_statistics']['mean_ratio']:.2f}x")
print(f"  Median: {stats['improvement_statistics']['median_ratio']:.2f}x")
print(f"\nCorrection distribution:")
for key, value in stats['correction_needed'].items():
    pct = stats['correction_needed_percent'][key]
    print(f"  {key}: {value} ({pct:.1f}%)")

needs_correction = stats['correction_needed']['small_correction_1_5_voxels'] + \
                  stats['correction_needed']['medium_correction_6_15_voxels'] + \
                  stats['correction_needed']['large_correction_16plus_voxels']
pct_corrected = (needs_correction / stats['total_cases'] * 100)

print(f"\n✅ Successfully corrected {needs_correction} cases ({pct_corrected:.1f}%)")
print(f"   These would have had suboptimal slice selection with geometric centering")
PYEOF
    echo ""
fi

echo "================================================================"
echo "Dataset Summary:"
echo "================================================================"

if [[ -f "$OUTPUT_DIR/metadata.json" ]]; then
    echo ""
    cat $OUTPUT_DIR/metadata.json
    echo ""
fi

echo "================================================================"
echo "Output location: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - Labels:  $OUTPUT_DIR/labels/train/"
echo "  - Images:  $OUTPUT_DIR/images/train/"
echo "  - Config:  $OUTPUT_DIR/dataset.yaml"
echo "  - Metrics: $OUTPUT_DIR/spine_aware_metrics_report.json"
echo "  - Summary: $OUTPUT_DIR/quality_validation_summary.png"
echo ""
echo "Next step:"
echo "  sbatch slurm_scripts/07_train_yolo_baseline.sh"
echo "================================================================"

echo ""
echo "End time: $(date)"
echo "================================================================"
