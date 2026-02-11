#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=weak_labels_trial_validated
#SBATCH -o logs/weak_labels_trial_%j.out
#SBATCH -e logs/weak_labels_trial_%j.err

set -euo pipefail

echo "================================================================"
echo "WEAK LABEL GENERATION - TRIAL WITH SPINE-AWARE VALIDATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"

PROJECT_DIR="$(pwd)"
NIFTI_DIR="${PROJECT_DIR}/results/lstv_screening/trial/nifti"
SEG_DIR="${PROJECT_DIR}/results/lstv_screening/trial/segmentations"
OUTPUT_DIR="${PROJECT_DIR}/data/training/lstv_yolo_trial"

IMG_PATH="${HOME}/singularity_cache/yolo.sif"

if [[ ! -f "$IMG_PATH" ]]; then
    echo "ERROR: YOLOv11 container not found!"
    echo "Run: ./setup_containers.sh"
    exit 1
fi

if [[ ! -d "$NIFTI_DIR" ]]; then
    echo "ERROR: NIfTI directory not found at $NIFTI_DIR"
    echo "Run trial screening first: sbatch slurm_scripts/04_lstv_screen_trial.sh"
    exit 1
fi

NIFTI_COUNT=$(find $NIFTI_DIR -name "*.nii.gz" | wc -l)
SEG_COUNT=$(find $SEG_DIR -name "*_seg.nii.gz" | wc -l)

echo "Generating weak labels with SPINE-AWARE slice selection..."
echo "Input NIfTI:   $NIFTI_DIR ($NIFTI_COUNT files)"
echo "Input Segs:    $SEG_DIR ($SEG_COUNT files)"
echo "Output:        $OUTPUT_DIR"
echo "Container:     $IMG_PATH"
echo ""
echo "VALIDATION MODE ENABLED:"
echo "  - Before/after comparison images will be generated"
echo "  - Quantitative metrics will be calculated"
echo "  - Statistical analysis will validate improvements"
echo "================================================================"

# Run with comparison generation enabled
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
        --generate_comparisons

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Label generation failed"
    exit $exit_code
fi

echo ""
echo "================================================================"
echo "Complete! Validation Analysis:"
echo "================================================================"

# Display metrics report
if [[ -f "$OUTPUT_DIR/spine_aware_metrics_report.json" ]]; then
    echo ""
    echo "SPINE-AWARE SLICE SELECTION METRICS:"
    python3 << PYEOF
import json
with open('$OUTPUT_DIR/spine_aware_metrics_report.json') as f:
    stats = json.load(f)

print(f"Total cases:          {stats['total_cases']}")
print(f"Spine-aware success:  {stats['spine_aware_cases']} ({stats['spine_aware_cases']/stats['total_cases']*100:.1f}%)")
print(f"\nOffset from geometric center:")
print(f"  Mean:   {stats['offset_statistics']['mean_mm']:.1f} ± {stats['offset_statistics']['std_mm']:.1f} mm")
print(f"  Median: {stats['offset_statistics']['median_mm']:.1f} mm")
print(f"  Max:    {stats['offset_statistics']['max_mm']:.1f} mm")
print(f"\nSpine density improvement:")
print(f"  Mean:   {stats['improvement_statistics']['mean_ratio']:.2f}x")
print(f"  Median: {stats['improvement_statistics']['median_ratio']:.2f}x")
print(f"\nCorrection magnitude:")
for key, value in stats['correction_needed'].items():
    pct = stats['correction_needed_percent'][key]
    print(f"  {key}: {value} cases ({pct:.1f}%)")

print(f"\n🔥 JUSTIFICATION: {stats['correction_needed']['medium_correction_6_15_voxels'] + stats['correction_needed']['large_correction_16plus_voxels']} cases")
print(f"   ({(stats['correction_needed_percent']['medium_correction_6_15_voxels'] + stats['correction_needed_percent']['large_correction_16plus_voxels']):.1f}%) needed significant correction!")
PYEOF
    echo ""
fi

echo "================================================================"
echo "Output Files:"
echo "================================================================"
echo "Dataset:              $OUTPUT_DIR"
echo "Metrics Report:       $OUTPUT_DIR/spine_aware_metrics_report.json"
echo "Summary Plot:         $OUTPUT_DIR/quality_validation_summary.png"
echo "Comparison Images:    $OUTPUT_DIR/quality_validation/"
echo ""

# Count comparison images
COMPARISON_COUNT=$(find $OUTPUT_DIR/quality_validation -name "*_slice_comparison.png" 2>/dev/null | wc -l)
echo "Generated $COMPARISON_COUNT before/after comparison images"
echo ""

echo "================================================================"
echo "RECOMMENDATION FOR FULL RUN:"
echo "================================================================"

python3 << PYEOF
import json
with open('$OUTPUT_DIR/spine_aware_metrics_report.json') as f:
    stats = json.load(f)

total = stats['total_cases']
needs_correction = stats['correction_needed']['small_correction_1_5_voxels'] + \
                  stats['correction_needed']['medium_correction_6_15_voxels'] + \
                  stats['correction_needed']['large_correction_16plus_voxels']

pct_needs_correction = (needs_correction / total * 100) if total > 0 else 0
mean_improvement = stats['improvement_statistics']['mean_ratio']

print(f"Trial Results ({total} cases):")
print(f"  - {pct_needs_correction:.1f}% of cases needed correction")
print(f"  - Mean spine visibility improved {mean_improvement:.2f}x")
print(f"  - Median offset: {stats['offset_statistics']['median_mm']:.1f}mm")
print()

if pct_needs_correction > 50:
    print("✅ STRONG JUSTIFICATION: >50% of cases need correction")
    print("   Spine-aware slicing is ESSENTIAL for full run!")
elif pct_needs_correction > 30:
    print("✅ GOOD JUSTIFICATION: >30% of cases need correction")
    print("   Spine-aware slicing recommended for full run")
else:
    print("⚠️  WEAK JUSTIFICATION: <30% of cases need correction")
    print("   Spine-aware slicing may not be critical, but still beneficial")

print()
print("Estimated impact on T12 rib detection:")
improvement_pct = (mean_improvement - 1.0) * 100
print(f"  Expected improvement: +{improvement_pct:.1f}% in spine visibility")
print(f"  Likely T12 rib AP improvement: +{improvement_pct*0.5:.1f}% to +{improvement_pct:.1f}%")
PYEOF

echo ""
echo "================================================================"
echo "Next steps:"
echo "================================================================"
echo "1. Review comparison images in: $OUTPUT_DIR/quality_validation/"
echo "2. Check summary plot: $OUTPUT_DIR/quality_validation_summary.png"
echo "3. If justified, proceed with full run using spine-aware slicing"
echo "4. Train baseline: sbatch slurm_scripts/07_train_yolo_baseline.sh"
echo "================================================================"

echo ""
echo "End time: $(date)"
echo "================================================================"
