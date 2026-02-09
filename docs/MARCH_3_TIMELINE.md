# March 3rd Abstract Deadline - Execution Timeline

## 🚨 CRITICAL DATES

**Today**: February 9, 2026
**Abstract Deadline**: March 3, 2026 (11:59 PM)
**Time Available**: 22 days

---

## ⏰ Week-by-Week Timeline

### **Week 1: Feb 9-15 (Setup & Download)** 
**Days Available**: 7

#### Monday Feb 9 (TODAY)
- [ ] Create GitHub repo: `spine-level-ai`
- [ ] Clone to local machine
- [ ] Build Docker containers locally
- [ ] Push to Docker Hub
- [ ] SSH into Wayne State Grid
- [ ] Clone repo to HPC
- [ ] Setup Kaggle credentials

**Commands for today**:
```bash
# Local machine
cd ~/spine-level-ai
chmod +x docker/build_and_push.sh
./docker/build_and_push.sh

# HPC (Wayne State Grid)
ssh go2432@grid.wayne.edu
cd ~
git clone https://github.com/yourusername/spine-level-ai
cd spine-level-ai
```

#### Tuesday Feb 10
- [ ] Submit data download job (24-hour runtime)
- [ ] While downloading: Read AACA abstract guidelines
- [ ] Draft abstract outline

```bash
sbatch slurm_scripts/01_download_data.sh
qme  # Monitor job
```

#### Wednesday Feb 11 - Friday Feb 14
- [ ] Monitor download progress
- [ ] Setup Roboflow account for labeling
- [ ] Recruit 2-3 labeling helpers (med students?)
- [ ] Create labeling protocol document
- [ ] Test Singularity container conversion locally

**Expected**: Download completes by Feb 12-13

#### Saturday-Sunday Feb 14-15
- [ ] Submit preprocessing job
- [ ] Verify processed images look correct
- [ ] Begin pilot labeling (50 images)

---

### **Week 2: Feb 16-22 (Labeling Sprint)** ⚠️ CRITICAL WEEK
**Days Available**: 7

**Goal**: Label 500-1000 images
**Time per image**: 30-45 seconds
**Total time needed**: 8-15 hours
**Strategy**: Divide among team

#### Monday-Tuesday Feb 16-17: Labeling Setup
- [ ] Upload first batch to Roboflow
- [ ] Train helpers on annotation protocol
- [ ] Each person labels 50 images to standardize
- [ ] Review and correct any inconsistencies

#### Wednesday-Friday Feb 18-20: Labeling Blitz
- [ ] **Target**: 150-200 images/person/day
- [ ] Quality check every 100 images
- [ ] Keep running total

**Daily quota per person**:
- Morning: 75 images (2 hours)
- Afternoon: 75 images (2 hours)

#### Weekend Feb 21-22: Buffer & Training Start
- [ ] Complete any remaining labels
- [ ] Export YOLO format from Roboflow
- [ ] Download to HPC
- [ ] Submit training job (72-hour runtime)

```bash
# Upload labels to HPC
scp -r labels/ go2432@grid.wayne.edu:~/spine-level-ai/data/processed/

# Start training
sbatch slurm_scripts/03_training_gpu.sh
```

---

### **Week 3: Feb 23-Mar 1 (Training & Writing)**
**Days Available**: 7

#### Monday-Wednesday Feb 23-25: Training Monitoring
- [ ] Monitor training progress
- [ ] Check validation metrics every 12 hours
- [ ] If poor performance: adjust hyperparameters, restart
- [ ] Begin abstract writing

**Training should complete**: Feb 25-26 (72 hours from Feb 22)

#### Thursday Feb 26: Results Analysis
- [ ] Training completes
- [ ] Run validation set
- [ ] Calculate final metrics:
  - mAP@0.5
  - Sensitivity/specificity for LSTV
  - Inference time
- [ ] Create sample visualizations

#### Friday Feb 27: Abstract Finalization
- [ ] Insert final numbers into abstract
- [ ] Write complete draft
- [ ] Send to advisor for review

#### Weekend Feb 28-Mar 1: Revisions
- [ ] Incorporate feedback
- [ ] Proofread carefully
- [ ] Have 2-3 people review
- [ ] Prepare submission materials

---

### **Week 4: Mar 2-3 (Submission)**
**Days Available**: 2

#### Monday Mar 2: Final Review
- [ ] One last proofread
- [ ] Verify all co-authors approve
- [ ] Check AACA submission requirements
- [ ] Prepare any supplementary materials

#### Tuesday Mar 3: Submit
- [ ] Submit abstract by 11:59 PM
- [ ] Confirm submission received
- [ ] Save confirmation email

---

## 🎯 Minimum Viable Abstract

**If behind schedule, minimum requirements**:
- 300 labeled images (bare minimum)
- 50 epochs training (vs 100 optimal)
- Basic validation metrics
- Can still submit with preliminary results

**Backup plan** (use ONLY if desperate):
- Label just 200 images
- Train on small subset
- Report as "pilot study" in abstract
- Focus on **feasibility** not performance

---

## 📊 Success Metrics

### Minimum for Abstract Acceptance:
- ✅ mAP@0.5: >75%
- ✅ LSTV detection sensitivity: >85%
- ✅ Working demo: Yes
- ✅ Novel contribution: Automated LSTV warning system

### Ideal Numbers:
- 🎯 mAP@0.5: >85%
- 🎯 LSTV sensitivity: >90%
- 🎯 Inference time: <1 sec
- 🎯 False positive rate: <5%

---

## ⚡ Daily Time Commitment

### Week 1 (Setup): 2-3 hours/day
- Mostly waiting for downloads
- Administrative tasks

### Week 2 (Labeling): 4-6 hours/day ⚠️ 
- Most time-intensive
- Can parallelize with helpers

### Week 3 (Training/Writing): 2-3 hours/day
- Monitoring jobs
- Writing abstract

### Week 4 (Polish): 1-2 hours/day
- Final edits

**Total commitment**: ~60-80 hours over 22 days

---

## 🚨 Risk Mitigation

### Risk 1: Download fails
**Mitigation**: Start today, 24hr buffer
**Backup**: Manual download from Kaggle website

### Risk 2: Labeling takes too long
**Mitigation**: Recruit helpers early (Feb 10)
**Backup**: Label only 300 images, report as pilot

### Risk 3: Training fails/poor performance
**Mitigation**: Start training by Feb 22 (10-day buffer)
**Backup**: Use pre-trained model, fine-tune minimally

### Risk 4: GPU queue wait time
**Mitigation**: Submit multiple jobs with different priorities
**Backup**: Use CPU training (slower but works)

### Risk 5: Abstract rejected
**Mitigation**: Have advisor review early
**Backup**: TechFair acceptance usually easier than platform talks

---

## 📝 Abstract Template (Fill in by Feb 27)

**TITLE**: AI-Assisted Vertebral Level Identification: Preventing Wrong-Level Surgery Through Automated LSTV Detection

**INTRODUCTION**: Wrong-level spine surgery is a catastrophic "never event" with costs exceeding $100,000 per case. Lumbosacral transitional vertebrae (LSTV) are present in 4-8% of the population and frequently cause enumeration errors. We developed an AI system to automatically detect vertebral landmarks and flag LSTV risk.

**METHODS**: A YOLOv8 object detection model was trained on **[N=_____]** whole-spine MRI localizer scans from the RSNA 2024 dataset. The model identifies three key landmarks: T12 last rib, L5 transverse process, and sacral promontory. An automated enumeration algorithm counts vertebrae from T12 and detects LSTV by identifying L5-S1 fusion patterns.

**RESULTS**: On a holdout test set (N=**[_____]**), the model achieved **[XX.X%]** mAP@0.5 for landmark detection. LSTV detection sensitivity was **[XX%]** with **[XX%]** specificity. Average inference time was **[X.X]** seconds per patient. The system correctly identified **[XX/XX]** Castellvi Type II-IV cases in our validation cohort.

**SIGNIFICANCE**: This open-source tool provides automated warnings for vertebral enumeration risk, potentially preventing wrong-level surgery. Integration into surgical planning workflows requires minimal technical infrastructure. Live demonstration will be provided at the TechFair booth.

---

## ✅ Daily Checklist Template

```
Date: ___/___/2026

Morning:
[ ] Check job status (qme)
[ ] Review overnight progress
[ ] Plan today's tasks

Afternoon:
[ ] Execute primary task
[ ] Document progress
[ ] Plan tomorrow

Evening:
[ ] Update timeline
[ ] Identify blockers
[ ] Email updates to team
```

---

## 🎯 Critical Path Items (Cannot Slip)

1. **Feb 10**: Data download started
2. **Feb 16**: Labeling begins
3. **Feb 22**: Training started
4. **Feb 27**: Draft abstract complete
5. **Mar 3**: Submit by 11:59 PM

**If any date slips by >2 days, activate backup plan**

---

## 📧 Team Communication

### Who to recruit:
- 2-3 med students for labeling
- Advisor for abstract review
- Radiology resident (optional) for validation

### Update schedule:
- Daily: Quick Slack/email to advisor
- Weekly: Detailed progress report
- Feb 27: Draft abstract for review

---

## 🏁 Success Definition

**Minimum Success** (still worth submitting):
- Abstract submitted on time
- Pilot results (300 images, >75% accuracy)
- Working proof-of-concept

**Full Success**:
- Abstract submitted on time
- Strong results (500+ images, >85% accuracy)
- TechFair acceptance
- Dr. Tubbs collaboration initiated

**You have 22 days. Let's do this.** 🚀
