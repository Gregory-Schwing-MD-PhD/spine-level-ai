# QUICKSTART - Docker-Based Workflow

## 🚨 MARCH 3RD DEADLINE

**Today**: February 9, 2026
**Abstract Due**: March 3, 2026 (22 days)
**Read**: `docs/MARCH_3_TIMELINE.md` for day-by-day plan

---

## ⚡ Setup in 10 Minutes

### Step 1: Build & Push Docker Containers (Local Machine)

```bash
# On your LOCAL machine with Docker installed
cd spine-level-ai

# Edit docker/build_and_push.sh - change DOCKER_USERNAME to yours
nano docker/build_and_push.sh  # Change go2432 to YOUR username

# Build and push to Docker Hub
chmod +x docker/build_and_push.sh
./docker/build_and_push.sh

# Login to Docker Hub when prompted
# Builds take ~30-45 minutes
```

**What this does**:
- Builds preprocessing container (~2GB)
- Builds training container (~15GB)
- Pushes both to Docker Hub
- Makes them available to HPC via Singularity

---

### Step 2: Setup HPC (Wayne State Grid)

```bash
# SSH to Grid
ssh go2432@wayne.edu@grid.wayne.edu

# Clone repo
cd ~
git clone https://github.com/YOUR_USERNAME/spine-level-ai
cd spine-level-ai

# Create directories
mkdir -p ~/xdr ~/singularity_cache logs

# Setup Kaggle credentials
mkdir -p ~/.kaggle
# Upload your kaggle.json here (see below)
chmod 600 ~/.kaggle/kaggle.json
```

**Get Kaggle credentials**:
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New Token"
3. Downloads `kaggle.json`
4. Upload to Grid: `scp kaggle.json go2432@grid.wayne.edu:~/.kaggle/`

---

### Step 3: Update SLURM Scripts

**Edit all SLURM scripts** - change Docker Hub username:

```bash
cd slurm_scripts

# In each .sh file, change this line:
DOCKER_USERNAME="go2432"  # Change to YOUR Docker Hub username
```

Files to edit:
- `01_download_data.sh`
- `02_preprocessing.sh`  
- `03_training_gpu.sh`

---

### Step 4: Start Data Download

```bash
# Submit download job (runs 24 hours)
sbatch slurm_scripts/01_download_data.sh

# Monitor progress
qme
tail -f logs/download_*.out
```

**What happens**:
- Singularity automatically pulls `docker://YOUR_USERNAME/spine-level-ai-preprocessing:latest`
- Converts to `.sif` file
- Caches in `~/singularity_cache/`
- Downloads RSNA dataset (~150GB)

---

## 📊 Why Docker → Singularity?

### Advantages:
✅ **Portability**: Build once, run anywhere
✅ **Version control**: Tag containers like `v1.0`, `v1.1`
✅ **Sharing**: Anyone can pull from Docker Hub
✅ **No root needed**: Singularity runs as regular user on HPC
✅ **Storage**: Docker Hub hosts containers for free

### Workflow:
```
Local Machine          Docker Hub              HPC
------------          -----------          ---------
Build Dockerfile  →   Push image    →   Singularity pull
(Docker)              (Registry)         (Auto-convert)
```

---

## 🐳 Docker Hub Organization

After pushing, your containers are at:
- `https://hub.docker.com/r/YOUR_USERNAME/spine-level-ai-preprocessing`
- `https://hub.docker.com/r/YOUR_USERNAME/spine-level-ai-training`

Anyone can pull them:
```bash
# Local machine
docker pull YOUR_USERNAME/spine-level-ai-training:latest

# HPC (auto-converts)
singularity pull docker://YOUR_USERNAME/spine-level-ai-training:latest
```

---

## 📅 22-Day Execution Plan

### Week 1 (Feb 9-15): Setup
- **Today**: Build containers, start download
- **Feb 10-13**: Monitor download, setup labeling
- **Feb 14-15**: Preprocessing, pilot labeling

### Week 2 (Feb 16-22): Labeling ⚠️
- **Feb 16-17**: Train labeling team
- **Feb 18-20**: Label 500-1000 images
- **Feb 21-22**: Export labels, start training

### Week 3 (Feb 23-Mar 1): Training & Writing
- **Feb 23-25**: Monitor training (72 hours)
- **Feb 26**: Analyze results
- **Feb 27**: Draft abstract
- **Feb 28-Mar 1**: Revisions

### Week 4 (Mar 2-3): Submission
- **Mar 2**: Final review
- **Mar 3**: Submit by 11:59 PM

---

## 🎯 Critical Milestones

| Date | Milestone | Status |
|------|-----------|--------|
| Feb 10 | Download started | ⏳ |
| Feb 16 | Labeling begins | ⏳ |
| Feb 22 | Training started | ⏳ |
| Feb 27 | Draft complete | ⏳ |
| Mar 3 | Abstract submitted | ⏳ |

**If any date slips >2 days, activate backup plan** (see timeline doc)

---

## 🆘 Common Issues

### "Docker build failed"
```bash
# Check Docker running
docker info

# Clear cache
docker system prune -a

# Rebuild
docker build -f docker/Dockerfile.training -t test .
```

### "Singularity conversion failed"
```bash
# HPC: Check cache
ls ~/singularity_cache/

# Manual pull
singularity pull docker://YOUR_USERNAME/spine-level-ai-training:latest

# Test container
singularity exec training.sif python --version
```

### "Kaggle credentials not found"
```bash
# Verify file exists
ls -la ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json

# Test Kaggle CLI
singularity exec preprocessing.sif kaggle competitions list
```

### "GPU not detected in SLURM"
```bash
# Check job
scontrol show job JOBID

# Verify GPU allocated
echo $CUDA_VISIBLE_DEVICES

# Test in container
singularity exec --nv training.sif nvidia-smi
```

---

## 📝 Abstract Template

Fill in by **Feb 27**:

**TITLE**: AI-Assisted Vertebral Level Identification: Preventing Wrong-Level Surgery Through Automated LSTV Detection

**INTRODUCTION**: Wrong-level spine surgery is a catastrophic "never event" with costs exceeding $100,000 per case. Lumbosacral transitional vertebrae (LSTV) are present in 4-8% of the population and frequently cause enumeration errors. We developed an AI system to automatically detect vertebral landmarks and flag LSTV risk.

**METHODS**: A YOLOv8 object detection model was trained on **N=[____]** whole-spine MRI localizer scans from the RSNA 2024 dataset. The model identifies three key landmarks: T12 last rib, L5 transverse process, and sacral promontory. An automated enumeration algorithm counts vertebrae from T12 and detects LSTV by identifying L5-S1 fusion patterns.

**RESULTS**: On a holdout test set (N=**[____]**), the model achieved **[XX.X%]** mAP@0.5 for landmark detection. LSTV detection sensitivity was **[XX%]** with **[XX%]** specificity. Average inference time was **[X.X]** seconds per patient.

**SIGNIFICANCE**: This open-source tool provides automated warnings for vertebral enumeration risk, potentially preventing wrong-level surgery. Live demonstration will be provided at the TechFair booth.

---

## ✅ Today's Checklist (Feb 9)

- [ ] Build Docker containers locally
- [ ] Push to Docker Hub
- [ ] Clone repo to HPC
- [ ] Setup Kaggle credentials
- [ ] Update SLURM scripts (Docker username)
- [ ] Submit download job
- [ ] Read `docs/MARCH_3_TIMELINE.md`
- [ ] Recruit 2-3 labeling helpers

---

## 🚀 Next Steps

**Tomorrow (Feb 10)**:
- Check download progress
- Setup Roboflow account
- Draft abstract outline
- Contact potential labelers

**This Week**:
- Monitor download (completes Feb 12-13)
- Setup labeling workflow
- Create labeling protocol doc

---

## 📚 Essential Docs

1. **This file** - Quickstart
2. `docs/MARCH_3_TIMELINE.md` - Detailed day-by-day plan
3. `README.md` - Project overview
4. `docker/build_and_push.sh` - Container build script

---

**You have 22 days. Start building now!** 🚀
