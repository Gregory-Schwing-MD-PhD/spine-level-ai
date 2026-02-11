#!/bin/bash
# Run the complete LSTV detection pipeline with inference
# This is a convenience script to run all steps in sequence

set -e

echo "================================================================"
echo "LSTV DETECTION - COMPLETE PIPELINE WITH INFERENCE"
echo "================================================================"
echo ""
echo "This will run:"
echo "  1. Trial screening (5 studies)"
echo "  2. Weak label generation"
echo "  3. YOLOv11 training (trial)"
echo "  4. Model evaluation"
echo "  5. Inference on trial candidates"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "Step 1: Trial Screening"
SCREEN_JOB=$(sbatch --parsable slurm_scripts/04_lstv_screen_trial.sh)
echo "  Job ID: $SCREEN_JOB"

echo ""
echo "Step 2: Generate Weak Labels (depends on Step 1)"
LABELS_JOB=$(sbatch --parsable --dependency=afterok:$SCREEN_JOB slurm_scripts/06_generate_weak_labels_trial.sh)
echo "  Job ID: $LABELS_JOB"

echo ""
echo "Step 3: Train YOLOv11 (depends on Step 2)"
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$LABELS_JOB slurm_scripts/07_train_yolo_trial.sh)
echo "  Job ID: $TRAIN_JOB"

echo ""
echo "Step 4: Evaluate Model (depends on Step 3)"
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm_scripts/09_evaluate_model.sh)
echo "  Job ID: $EVAL_JOB"

echo ""
echo "Step 5: Run Inference on Trial Candidates (depends on Step 3)"
INFER_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm_scripts/11_classify_trial_batch.sh)
echo "  Job ID: $INFER_JOB"

echo ""
echo "================================================================"
echo "Pipeline submitted!"
echo "================================================================"
echo ""
echo "Job IDs:"
echo "  Screening:     $SCREEN_JOB"
echo "  Labels:        $LABELS_JOB"
echo "  Training:      $TRAIN_JOB"
echo "  Evaluation:    $EVAL_JOB"
echo "  Inference:     $INFER_JOB"
echo ""
echo "Monitor:"
echo "  squeue -u $USER"
echo "  tail -f logs/*.out"
echo "================================================================"
