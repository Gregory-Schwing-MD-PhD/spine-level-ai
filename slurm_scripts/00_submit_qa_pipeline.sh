#!/bin/bash
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:01:00
#SBATCH --job-name=submit_qa_pipeline
#SBATCH -o logs/submit_qa_pipeline_%j.out
#SBATCH -e logs/submit_qa_pipeline_%j.err

set -euo pipefail

# ============================================================================
# ANSI COLORS
# ============================================================================
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo "================================================================"
echo "SPINE LEVEL AI - QA PIPELINE SUBMITTER"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "================================================================"
echo ""
echo -e "${BOLD}${PURPLE}This pipeline will:${NC}"
echo -e "  ${CYAN}1. LSTV Screening${NC}        (5 studies, ~30 min, GPU)"
echo -e "  ${CYAN}2. QA Report Generation${NC} (PDF reports with labels, ~5 min)"
echo -e "  ${CYAN}3. Weak Label Generation${NC} (v6.1 tuned, ~10 min)"
echo -e "  ${CYAN}4. Baseline Training${NC}     (YOLOv11, ~3-4 hrs, GPU)"
echo ""
echo "================================================================"

PROJECT_DIR="$(pwd)"

# ============================================================================
# VERIFY FILES
# ============================================================================

echo ""
echo "Verifying required files..."

required_files=(
    "slurm_scripts/04_lstv_screen_trial.sh"
    "slurm_scripts/05_generate_qa_reports.sh"
    "slurm_scripts/06_generate_weak_labels_trial.sh"
    "slurm_scripts/07_train_yolo_baseline_trial.sh"
    "src/screening/visualize_lstv_labels.py"
)

missing=0
for file in "${required_files[@]}"; do
    if [[ ! -f "$PROJECT_DIR/$file" ]]; then
        echo -e "${RED}ERROR: Missing $file${NC}"
        missing=1
    fi
done

if [[ $missing -eq 1 ]]; then
    echo "ERROR: Missing required files!"
    exit 1
fi

echo "✓ All files present"

# ============================================================================
# SUBMIT JOBS
# ============================================================================

echo ""
echo "================================================================"
echo "SUBMITTING JOBS WITH DEPENDENCIES"
echo "================================================================"

# JOB 1: LSTV Screening
echo ""
echo -e "${BOLD}[1/4] Submitting LSTV Screening...${NC}"

SCREEN_JOB=$(sbatch --parsable slurm_scripts/04_lstv_screen_trial_enhanced.sh)

if [[ -z "$SCREEN_JOB" ]]; then
    echo "ERROR: Failed to submit screening job"
    exit 1
fi

echo -e "  ${GREEN}✓ Job ID: $SCREEN_JOB${NC}"
echo "  Queue: gpu"
echo "  Duration: ~30 minutes"
echo "  Output: results/lstv_screening/trial/"

# JOB 2: QA Reports
echo ""
echo -e "${BOLD}[2/4] Submitting QA Report Generation...${NC}"

QA_JOB=$(sbatch --parsable \
    --dependency=afterok:$SCREEN_JOB \
    slurm_scripts/05_generate_qa_reports.sh)

if [[ -z "$QA_JOB" ]]; then
    echo "ERROR: Failed to submit QA job"
    exit 1
fi

echo -e "  ${GREEN}✓ Job ID: $QA_JOB${NC}"
echo "  Queue: primary"
echo "  Dependency: afterok:$SCREEN_JOB"
echo "  Duration: ~5 minutes"
echo -e "  ${PURPLE}Output: results/lstv_screening/trial/qa_reports/${NC}"
echo -e "  ${PURPLE}Features: PDF reports + confidence scoring${NC}"

# JOB 3: Weak Labels
echo ""
echo -e "${BOLD}[3/4] Submitting Weak Label Generation...${NC}"

LABELS_JOB=$(sbatch --parsable \
    --dependency=afterok:$SCREEN_JOB \
    slurm_scripts/06_generate_weak_labels_trial.sh)

if [[ -z "$LABELS_JOB" ]]; then
    echo "ERROR: Failed to submit weak labels job"
    exit 1
fi

echo -e "  ${GREEN}✓ Job ID: $LABELS_JOB${NC}"
echo "  Queue: primary"
echo "  Dependency: afterok:$SCREEN_JOB"
echo "  Duration: ~10 minutes"
echo "  Output: data/training/lstv_yolo_v6_trial/"

# JOB 4: Training
echo ""
echo -e "${BOLD}[4/4] Submitting Baseline Training...${NC}"

TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:$LABELS_JOB \
    slurm_scripts/07_train_yolo_baseline_trial.sh)

if [[ -z "$TRAIN_JOB" ]]; then
    echo "ERROR: Failed to submit training job"
    exit 1
fi

echo -e "  ${GREEN}✓ Job ID: $TRAIN_JOB${NC}"
echo "  Queue: gpu"
echo "  Dependency: afterok:$LABELS_JOB"
echo "  Duration: ~3-4 hours"
echo "  Output: runs/lstv/baseline_trial/"

# ============================================================================
# SAVE JOB IDS
# ============================================================================

cat > qa_pipeline_jobs.txt << EOF
# QA Pipeline Job IDs
# Submitted: $(date)
SCREEN_JOB=$SCREEN_JOB
QA_JOB=$QA_JOB
LABELS_JOB=$LABELS_JOB
TRAIN_JOB=$TRAIN_JOB
EOF

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "================================================================"
echo "PIPELINE SUBMITTED SUCCESSFULLY!"
echo "================================================================"
echo ""
echo -e "${BOLD}${CYAN}Job Chain:${NC}"
echo "  1. Screening:    $SCREEN_JOB (starts immediately)"
echo "  2. QA Reports:   $QA_JOB (waits for #1) ← PDF + CONFIDENCE"
echo "  3. Weak Labels:  $LABELS_JOB (waits for #1)"
echo "  4. Training:     $TRAIN_JOB (waits for #3)"
echo ""
echo -e "${BOLD}${CYAN}Timeline:${NC}"
echo "  Now:        Screening starts (~30 min)"
echo "  +30 min:    QA reports + weak labels start (~5-10 min)"
echo "  +40 min:    Training starts (~3-4 hrs)"
echo "  +4-5 hrs:   COMPLETE!"
echo ""
echo -e "${BOLD}${CYAN}Monitor:${NC}"
echo "  squeue -u \$USER"
echo "  squeue -j $SCREEN_JOB,$QA_JOB,$LABELS_JOB,$TRAIN_JOB"
echo ""
echo -e "${BOLD}${CYAN}View logs:${NC}"
echo "  tail -f logs/lstv_trial_*.out"
echo "  tail -f logs/lstv_qa_reports_*.out  ← QA CONFIDENCE ANALYSIS"
echo "  tail -f logs/weak_labels_v6.1_*.out"
echo "  tail -f logs/yolo_baseline_trial_*.out"
echo ""
echo "Job IDs saved to: qa_pipeline_jobs.txt"
echo ""
echo "================================================================"
echo "CRITICAL: REVIEW QA REPORTS BEFORE ROBOFLOW UPLOAD!"
echo "================================================================"
echo ""
echo -e "${BOLD}${PURPLE}After QA job completes (~35 min):${NC}"
echo ""
echo "1. Check QA summary:"
echo "   cat results/lstv_screening/trial/qa_reports/qa_summary.json"
echo ""
echo "2. View HIGH confidence PDFs:"
echo "   ls results/lstv_screening/trial/qa_reports/*_QA_report.pdf"
echo ""
echo "3. Filter for Roboflow upload:"
echo "   → Upload HIGH confidence only"
echo "   → Manual review MEDIUM confidence"
echo "   → Reject LOW confidence"
echo ""
echo "================================================================"
echo "Submitter complete!"
echo "End time: $(date)"
echo "================================================================"
