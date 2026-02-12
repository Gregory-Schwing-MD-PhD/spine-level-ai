#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:1:00
#SBATCH --job-name=submit_trial_pipeline
#SBATCH -o logs/submit_trial_pipeline_%j.out
#SBATCH -e logs/submit_trial_pipeline_%j.err

set -euo pipefail

echo "================================================================"
echo "SPINE LEVEL AI - AUTOMATED TRIAL PIPELINE SUBMITTER"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"
echo ""
echo "This job will submit the complete trial workflow:"
echo "  1. LSTV Screening (5 studies)"
echo "  2. Weak Label Generation (spine-aware validation)"
echo "  3. Baseline Training (trial)"
echo "  4. Results Analysis & Recommendation"
echo ""
echo "All jobs will be submitted with proper dependencies."
echo "================================================================"

PROJECT_DIR="$(pwd)"

# Verify files exist
echo ""
echo "Verifying required files..."

required_files=(
    "slurm_scripts/04_lstv_screen_trial.sh"
    "slurm_scripts/06_generate_weak_labels_trial.sh"
    "slurm_scripts/07_train_yolo_baseline_trial.sh"
    "slurm_scripts/99_review_trial_results.sh"
)

missing=0
for file in "${required_files[@]}"; do
    if [[ ! -f "$PROJECT_DIR/$file" ]]; then
        echo "ERROR: Missing $file"
        missing=1
    fi
done

if [[ $missing -eq 1 ]]; then
    echo "ERROR: Missing required files!"
    exit 1
fi

echo "✓ All files present"

# ================================================================
# SUBMIT JOBS WITH DEPENDENCIES
# ================================================================

echo ""
echo "================================================================"
echo "SUBMITTING JOBS"
echo "================================================================"

# JOB 1: LSTV Screening
echo ""
echo "[1/4] Submitting LSTV Screening (trial)..."

SCREEN_JOB=$(sbatch --parsable slurm_scripts/04_lstv_screen_trial.sh)

if [[ -z "$SCREEN_JOB" ]]; then
    echo "ERROR: Failed to submit screening job"
    exit 1
fi

echo "  ✓ Job ID: $SCREEN_JOB"
echo "  Queue: gpu"
echo "  Duration: ~30 minutes"
echo "  Output: results/lstv_screening/trial/"

# JOB 2: Weak Labels with Validation
echo ""
echo "[2/4] Submitting Weak Label Generation (spine-aware validation)..."

LABELS_JOB=$(sbatch --parsable \
    --dependency=afterok:$SCREEN_JOB \
    slurm_scripts/06_generate_weak_labels_trial.sh)

if [[ -z "$LABELS_JOB" ]]; then
    echo "ERROR: Failed to submit weak labels job"
    exit 1
fi

echo "  ✓ Job ID: $LABELS_JOB"
echo "  Queue: primary"
echo "  Dependency: afterok:$SCREEN_JOB"
echo "  Duration: ~10 minutes"
echo "  Output: data/training/lstv_yolo_trial/"
echo "  VALIDATION: quality_validation/ with comparisons + metrics"

# JOB 3: Baseline Training
echo ""
echo "[3/4] Submitting Baseline Training (trial)..."

TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:$LABELS_JOB \
    slurm_scripts/07_train_yolo_baseline_trial.sh)

if [[ -z "$TRAIN_JOB" ]]; then
    echo "ERROR: Failed to submit training job"
    exit 1
fi

echo "  ✓ Job ID: $TRAIN_JOB"
echo "  Queue: gpu"
echo "  Dependency: afterok:$LABELS_JOB"
echo "  Duration: ~3-4 hours"
echo "  Output: runs/lstv/baseline_trial/"

# JOB 4: Results Review & Analysis
echo ""
echo "[4/4] Submitting Results Analysis & Recommendation..."

REVIEW_JOB=$(sbatch --parsable \
    --dependency=afterok:$TRAIN_JOB \
    slurm_scripts/99_review_trial_results.sh)

if [[ -z "$REVIEW_JOB" ]]; then
    echo "ERROR: Failed to submit review job"
    exit 1
fi

echo "  ✓ Job ID: $REVIEW_JOB"
echo "  Queue: primary"
echo "  Dependency: afterok:$TRAIN_JOB"
echo "  Duration: ~1 minute"
echo "  Output: logs/review_trial_*.out + trial_decision.json"

# ================================================================
# SAVE JOB IDS
# ================================================================

cat > trial_job_ids.txt << EOF
# Trial Pipeline Job IDs
# Submitted: $(date)
SCREEN_JOB=$SCREEN_JOB
LABELS_JOB=$LABELS_JOB
TRAIN_JOB=$TRAIN_JOB
REVIEW_JOB=$REVIEW_JOB
EOF

echo ""
echo "================================================================"
echo "PIPELINE SUBMITTED SUCCESSFULLY!"
echo "================================================================"
echo ""
echo "Job Chain:"
echo "  1. Screening:    $SCREEN_JOB (starts immediately)"
echo "  2. Weak Labels:  $LABELS_JOB (waits for #1)"
echo "  3. Training:     $TRAIN_JOB (waits for #2)"
echo "  4. Analysis:     $REVIEW_JOB (waits for #3) ← AUTO ANALYSIS!"
echo ""
echo "Timeline:"
echo "  Now:        Screening starts (~30 min)"
echo "  +30 min:    Weak labels start (~10 min)"
echo "  +40 min:    Training starts (~3-4 hrs)"
echo "  +4-5 hrs:   Analysis starts (~1 min)"
echo "  +4-5 hrs:   COMPLETE WITH RECOMMENDATION!"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo "  squeue -j $SCREEN_JOB,$LABELS_JOB,$TRAIN_JOB,$REVIEW_JOB"
echo ""
echo "View logs:"
echo "  tail -f logs/lstv_trial_*.out"
echo "  tail -f logs/weak_labels_trial_*.out"
echo "  tail -f logs/yolo_baseline_trial_*.out"
echo "  tail -f logs/review_trial_*.out          ← FINAL REPORT!"
echo ""
echo "Job IDs saved to: trial_job_ids.txt"
echo ""
echo "================================================================"
echo "AFTER COMPLETION (~4-5 hours):"
echo "================================================================"
echo ""
echo "The analysis job will automatically:"
echo "  ✓ Display spine-aware validation metrics"
echo "  ✓ Show training performance"
echo "  ✓ Analyze justification (STRONG/GOOD/MODERATE/WEAK)"
echo "  ✓ Provide actionable next steps"
echo "  ✓ Save decision to trial_decision.json"
echo ""
echo "To view the full analysis report:"
echo "  cat logs/review_trial_${REVIEW_JOB}.out"
echo ""
echo "Quick decision check:"
echo "  cat trial_decision.json"
echo ""
echo "View validation images:"
echo "  xdg-open data/training/lstv_yolo_trial/quality_validation_summary.png"
echo "  ls data/training/lstv_yolo_trial/quality_validation/"
echo ""
echo "================================================================"
echo "ONE SUBMISSION = COMPLETE VALIDATED TRIAL + RECOMMENDATION"
echo "================================================================"
echo ""
echo "Just run:"
echo "  sbatch slurm_scripts/00_submit_trial_pipeline.sh"
echo ""
echo "Then in ~4-5 hours, check:"
echo "  cat logs/review_trial_${REVIEW_JOB}.out"
echo ""
echo "That's it! Fully automated trial with decision guidance!"
echo ""
echo "================================================================"
echo "Submitter job complete!"
echo "End time: $(date)"
echo "================================================================"
