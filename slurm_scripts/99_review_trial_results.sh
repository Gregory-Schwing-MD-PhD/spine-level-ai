#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=review_trial
#SBATCH -o logs/review_trial_%j.out
#SBATCH -e logs/review_trial_%j.err

set -euo pipefail

echo "================================================================"
echo "TRIAL RESULTS REVIEW - SPINE-AWARE VALIDATION ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"
echo ""

PROJECT_DIR="$(pwd)"

# Check if trial completed
if [[ ! -f "data/training/lstv_yolo_trial/spine_aware_metrics_report.json" ]]; then
    echo "ERROR: Trial not complete yet!"
    echo "Weak labels not generated or still running."
    echo ""
    echo "Check job status:"
    echo "  squeue -u $USER"
    echo ""
    if [[ -f "trial_job_ids.txt" ]]; then
        source trial_job_ids.txt
        echo "Pipeline jobs:"
        echo "  Screening:    $SCREEN_JOB"
        echo "  Weak Labels:  $LABELS_JOB"
        echo "  Training:     $TRAIN_JOB"
        echo ""
        echo "Check specific jobs:"
        echo "  squeue -j $SCREEN_JOB,$LABELS_JOB,$TRAIN_JOB"
    fi
    exit 1
fi

# ================================================================
# SPINE-AWARE VALIDATION METRICS
# ================================================================

echo "================================================================"
echo "SPINE-AWARE SLICE SELECTION VALIDATION RESULTS"
echo "================================================================"
echo ""

python3 << 'PYEOF'
import json
from pathlib import Path
import sys

metrics_file = Path('data/training/lstv_yolo_trial/spine_aware_metrics_report.json')

if not metrics_file.exists():
    print("ERROR: Metrics file not found!")
    sys.exit(1)

with open(metrics_file) as f:
    stats = json.load(f)

total = stats['total_cases']
spine_aware = stats['spine_aware_cases']
fallback = stats['geometric_fallback_cases']

mean_offset_mm = stats['offset_statistics']['mean_mm']
std_offset_mm = stats['offset_statistics']['std_mm']
median_offset_mm = stats['offset_statistics']['median_mm']
max_offset_mm = stats['offset_statistics']['max_mm']

mean_improvement = stats['improvement_statistics']['mean_ratio']
median_improvement = stats['improvement_statistics']['median_ratio']

no_corr = stats['correction_needed']['no_correction']
small_corr = stats['correction_needed']['small_correction_1_5_voxels']
medium_corr = stats['correction_needed']['medium_correction_6_15_voxels']
large_corr = stats['correction_needed']['large_correction_16plus_voxels']

print("TRIAL DATASET:")
print(f"  Total cases:        {total}")
print(f"  Spine-aware used:   {spine_aware} ({spine_aware/total*100:.1f}%)")
print(f"  Geometric fallback: {fallback} ({fallback/total*100:.1f}%)")
print()

print("OFFSET FROM GEOMETRIC CENTER:")
print(f"  Mean:   {mean_offset_mm:.1f} ± {std_offset_mm:.1f} mm")
print(f"  Median: {median_offset_mm:.1f} mm")
print(f"  Max:    {max_offset_mm:.1f} mm")
print()

print("SPINE VISIBILITY IMPROVEMENT:")
print(f"  Mean:   {mean_improvement:.2f}x")
print(f"  Median: {median_improvement:.2f}x")
print()

print("CORRECTION DISTRIBUTION:")
print(f"  No correction (0 voxels):      {no_corr:2d} ({stats['correction_needed_percent']['no_correction']:.1f}%)")
print(f"  Small (1-5 voxels):            {small_corr:2d} ({stats['correction_needed_percent']['small_correction_1_5_voxels']:.1f}%)")
print(f"  Medium (6-15 voxels):          {medium_corr:2d} ({stats['correction_needed_percent']['medium_correction_6_15_voxels']:.1f}%)")
print(f"  Large (>15 voxels):            {large_corr:2d} ({stats['correction_needed_percent']['large_correction_16plus_voxels']:.1f}%)")
print()

needs_correction = small_corr + medium_corr + large_corr
pct_needs_correction = (needs_correction / total * 100)

print("="*70)
print(f"SUMMARY: {needs_correction}/{total} cases ({pct_needs_correction:.1f}%) needed correction")
print("="*70)
print()

# Decision criteria
if mean_offset_mm >= 8:
    justification = "STRONG"
    emoji = "🔥"
    color = "\033[92m"  # Green
elif mean_offset_mm >= 5:
    justification = "GOOD"
    emoji = "✅"
    color = "\033[92m"  # Green
elif mean_offset_mm >= 3:
    justification = "MODERATE"
    emoji = "⚠️"
    color = "\033[93m"  # Yellow
else:
    justification = "WEAK"
    emoji = "❌"
    color = "\033[91m"  # Red

reset = "\033[0m"

print("JUSTIFICATION ANALYSIS:")
print()
print(f"{color}{emoji} {justification} JUSTIFICATION for spine-aware slicing{reset}")
print()

if justification in ["STRONG", "GOOD"]:
    print("Criteria met:")
    if mean_offset_mm >= 5:
        print(f"  ✓ Mean offset {mean_offset_mm:.1f}mm (>5mm threshold)")
    if pct_needs_correction >= 30:
        print(f"  ✓ {pct_needs_correction:.1f}% need correction (>30% threshold)")
    if mean_improvement >= 1.3:
        print(f"  ✓ {mean_improvement:.2f}x improvement (>1.3x threshold)")
    print()
    print(f"{color}RECOMMENDATION: Proceed with spine-aware for full run{reset}")
    
elif justification == "MODERATE":
    print("Mixed results:")
    if mean_offset_mm < 5:
        print(f"  ⚠ Mean offset {mean_offset_mm:.1f}mm (<5mm threshold)")
    if pct_needs_correction < 30:
        print(f"  ⚠ {pct_needs_correction:.1f}% need correction (<30% threshold)")
    print()
    print(f"{color}RECOMMENDATION: Spine-aware provides modest improvement{reset}")
    print("                Proceed if aiming for maximum performance")
    
else:
    print("Criteria NOT met:")
    print(f"  ✗ Mean offset {mean_offset_mm:.1f}mm (<3mm threshold)")
    print(f"  ✗ {pct_needs_correction:.1f}% need correction (<20% threshold)")
    print()
    print(f"{color}RECOMMENDATION: Geometric centering may be sufficient{reset}")
    print("                Consider spine-aware for outlier cases only")

print()

# Estimate impact
improvement_pct = (mean_improvement - 1.0) * 100
estimated_t12_boost_min = improvement_pct * 0.3
estimated_t12_boost_max = improvement_pct * 0.6

print("ESTIMATED IMPACT ON T12 RIB DETECTION:")
print(f"  Spine visibility improved: {improvement_pct:.1f}%")
print(f"  Expected T12 AP boost:     +{estimated_t12_boost_min:.1f}% to +{estimated_t12_boost_max:.1f}%")
print()
print("  Example scenario:")
print(f"    Geometric baseline:  58.0% T12 AP")
print(f"    Spine-aware:         {58 + estimated_t12_boost_max:.1f}% T12 AP")
print(f"    + Human refinement:  ~80-85% T12 AP (clinical target)")

# Save decision for next steps
decision_data = {
    'justification': justification,
    'mean_offset_mm': mean_offset_mm,
    'pct_needs_correction': pct_needs_correction,
    'mean_improvement': mean_improvement,
    'proceed_with_spine_aware': justification in ["STRONG", "GOOD"],
    'estimated_t12_boost_max': estimated_t12_boost_max,
}

import json
with open('trial_decision.json', 'w') as f:
    json.dump(decision_data, f, indent=2)

print()
print("✓ Decision saved to: trial_decision.json")

PYEOF

# ================================================================
# TRAINING RESULTS
# ================================================================

echo ""
echo "================================================================"
echo "BASELINE TRAINING RESULTS (Trial)"
echo "================================================================"
echo ""

if [[ ! -f "runs/lstv/baseline_trial/final_metrics.json" ]]; then
    echo "⚠️  Training not complete yet or failed."
    echo ""
    echo "Check status:"
    if [[ -f "trial_job_ids.txt" ]]; then
        source trial_job_ids.txt
        echo "  squeue -j $TRAIN_JOB"
        echo "  tail -f logs/yolo_baseline_trial_*.out"
    else
        echo "  squeue -u $USER"
    fi
else
    python3 << 'PYEOF'
import json
from pathlib import Path

metrics_file = Path('runs/lstv/baseline_trial/final_metrics.json')

with open(metrics_file) as f:
    metrics = json.load(f)

print("OVERALL METRICS:")
print(f"  mAP@50:    {metrics.get('map50', 0):.4f}")
print(f"  mAP@50-95: {metrics.get('map50_95', 0):.4f}")
print(f"  Precision: {metrics.get('precision', 0):.4f}")
print(f"  Recall:    {metrics.get('recall', 0):.4f}")
print()

if 'per_class_ap' in metrics:
    print("PER-CLASS AP@50:")
    classes = ['t12_vertebra', 't12_rib', 'l5_vertebra', 'l5_transverse_process', 
               'sacrum', 'l4_vertebra', 'l5_s1_disc']
    
    for cls in classes:
        if cls in metrics['per_class_ap']:
            ap = metrics['per_class_ap'][cls].get('ap50', 0)
            print(f"  {cls:25s}: {ap:.4f}")
    
    print()
    
    # Highlight T12 rib
    if 't12_rib' in metrics['per_class_ap']:
        t12_ap = metrics['per_class_ap']['t12_rib'].get('ap50', 0)
        
        if t12_ap >= 0.70:
            color = "\033[92m"  # Green
            status = "EXCELLENT"
            msg = "Ready for clinical use!"
        elif t12_ap >= 0.60:
            color = "\033[92m"  # Green
            status = "GOOD"
            msg = "Suitable baseline, will improve with refinement"
        elif t12_ap >= 0.50:
            color = "\033[93m"  # Yellow
            status = "ACCEPTABLE"
            msg = "Needs human refinement"
        else:
            color = "\033[91m"  # Red
            status = "INSUFFICIENT"
            msg = "Review weak labels and training"
        
        reset = "\033[0m"
        
        print(f"🎯 T12 RIB DETECTION: {color}{t12_ap:.4f}{reset}")
        print(f"   {color}{status}{reset} - {msg}")

PYEOF
fi

# ================================================================
# VISUALIZATION FILES
# ================================================================

echo ""
echo "================================================================"
echo "VALIDATION VISUALIZATIONS AVAILABLE"
echo "================================================================"
echo ""

COMPARISON_DIR="data/training/lstv_yolo_trial/quality_validation"
SUMMARY_PLOT="data/training/lstv_yolo_trial/quality_validation_summary.png"

if [[ -d "$COMPARISON_DIR" ]]; then
    COMP_COUNT=$(find $COMPARISON_DIR -name "*_slice_comparison.png" 2>/dev/null | wc -l)
    echo "Before/After Comparison Images: $COMP_COUNT"
    echo "Location: $COMPARISON_DIR/"
    echo ""
    echo "View examples:"
    echo "  eog $COMPARISON_DIR/*.png &"
    echo "  # or"
    echo "  xdg-open $COMPARISON_DIR/"
    echo ""
fi

if [[ -f "$SUMMARY_PLOT" ]]; then
    echo "Summary Plot: $SUMMARY_PLOT"
    echo "View:"
    echo "  xdg-open $SUMMARY_PLOT"
    echo ""
fi

# ================================================================
# NEXT STEPS
# ================================================================

echo "================================================================"
echo "NEXT STEPS - ACTIONABLE RECOMMENDATIONS"
echo "================================================================"
echo ""

if [[ -f "trial_decision.json" ]]; then
    python3 << 'PYEOF'
import json

with open('trial_decision.json') as f:
    decision = json.load(f)

justification = decision['justification']
proceed = decision['proceed_with_spine_aware']
mean_offset = decision['mean_offset_mm']
pct_corrected = decision['pct_needs_correction']

if proceed:
    print("✅ PROCEED WITH SPINE-AWARE SLICING FOR FULL RUN")
    print()
    print("Action items:")
    print()
    print("1. Review validation images to visually confirm improvements:")
    print("   ls data/training/lstv_yolo_trial/quality_validation/")
    print("   xdg-open data/training/lstv_yolo_trial/quality_validation_summary.png")
    print()
    print("2. Generate full weak labels with spine-aware slicing (500 studies):")
    print("   sbatch slurm_scripts/06_generate_weak_labels_full.sh")
    print()
    print("3. Train baseline on full dataset:")
    print("   sbatch slurm_scripts/07_train_yolo_baseline.sh")
    print()
    print("4. Prepare for med student annotations:")
    print("   - Review docs/ANNOTATION_GUIDELINES_MED_STUDENTS.md")
    print("   - Grant Roboflow access")
    print("   - Target: 200 images, 20 person-hours")
    print()
    print(f"Expected outcomes with full pipeline:")
    boost = decision['estimated_t12_boost_max']
    print(f"  - Baseline T12 rib: ~{65+boost:.1f}% (with spine-aware)")
    print(f"  - After refinement: ~80-85% (clinical target)")
    print(f"  - Overall mAP@50: ~0.85-0.90")
    
elif justification == "MODERATE":
    print("⚠️  MODERATE IMPROVEMENT - DECISION REQUIRED")
    print()
    print(f"Trial showed mean offset of {mean_offset:.1f}mm with {pct_corrected:.1f}% needing correction.")
    print()
    print("Option A: Proceed with spine-aware (recommended for max performance)")
    print("  sbatch slurm_scripts/06_generate_weak_labels_full.sh")
    print()
    print("Option B: Use geometric centering (simpler, slightly lower performance)")
    print("  - Modify weak label scripts to skip spine-aware selection")
    print("  - Expect ~5-10% lower T12 detection")
    print()
    print("Recommendation: Proceed with spine-aware if targeting clinical use (>75% T12)")
    
else:
    print("❌ WEAK JUSTIFICATION FOR SPINE-AWARE")
    print()
    print(f"Trial showed mean offset of {mean_offset:.1f}mm with {pct_corrected:.1f}% needing correction.")
    print()
    print("Dataset appears to have consistent patient positioning.")
    print()
    print("Options:")
    print("1. Proceed with geometric centering (simpler)")
    print("2. Still use spine-aware for robustness (small overhead)")
    print()
    print("Either approach acceptable for this dataset.")

PYEOF
fi

echo ""
echo "================================================================"
echo "FILES GENERATED BY THIS ANALYSIS"
echo "================================================================"
echo ""
echo "  trial_decision.json                    # Decision metrics"
echo "  logs/review_trial_${SLURM_JOB_ID}.out  # This report"
echo ""
echo "================================================================"
echo "Analysis complete!"
echo "End time: $(date)"
echo "================================================================"
