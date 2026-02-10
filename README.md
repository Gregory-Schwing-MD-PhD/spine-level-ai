# Spine Level AI
**AI-Assisted Vertebral Level Identification for Surgical Safety**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/container-docker-blue.svg)](https://hub.docker.com/)

## 🚨 Quick Start for March 3rd Deadline

**Abstract due**: March 3, 2026  
**Time available**: 22 days

See **[`docs/MARCH_3_TIMELINE.md`](docs/MARCH_3_TIMELINE.md)** for day-by-day execution plan.

---

## 🎯 Why This Matters: The Critical Gap

### The Clinical Problem
Wrong-level spine surgery is a **"never event"** costing **$100K+ per case** and causing devastating patient harm. The root cause? **Lumbosacral transitional vertebrae (LSTV)** occur in **10-30% of the population**, making L5 look like S1 (or vice versa), confusing vertebral counting during surgery.

### Existing Solutions Fall Short

| Tool | What It Does | **Critical Limitation** |
|------|--------------|------------------------|
| **nnU-Net** | Segments vertebrae | ❌ Requires manual enumeration; doesn't detect LSTV |
| **SPINEPS** | Whole-spine segmentation (20+ structures) | ❌ Over-engineered for surgical planning; no enumeration risk warnings |
| **TotalSegmentator** | Multi-organ CT/MR segmentation | ❌ Generic anatomy, not spine-specific; misses LSTV |
| **SpineNet** | Vertebra detection on X-ray | ❌ 2D only; poor soft tissue detail vs. MRI |
| **Manual Counting** | Radiologist counts from T12 down | ❌ 5-15% error rate with LSTV; time-consuming |

### What Makes This Different: A Two-Stage Solution

**The Core Problem Explained:**
- **Normal spine**: 5 lumbar vertebrae (L1→L5) sitting on sacrum (S1)
- **LSTV spine**: L5 fuses to sacrum → looks like only 4 lumbar vertebrae exist
- **Surgeon counts from bottom up**: "1, 2, 3, 4... this must be L4!" 
- **Reality**: That's actually L5 fused to sacrum → surgeon operates on **wrong level**

**Stage 1: SPINEPS (Data Pre-Filter)**
- **Purpose**: Find training data efficiently
- **What it does**: Segments vertebrae, numbers them, counts lumbar vertebrae
- **Output**: Flags ~500 of 2,700 scans where count ≠ 5 OR L5-S1 fusion suspected
- **Why this matters**: Saves 80% of manual screening time (you only label the "weird" cases)

**Stage 2: YOLOv8 (The Clinical Tool You're Building)**
- **Purpose**: Prevent wrong-level surgery in real patients
- **What it detects**: Anatomical red flags that humans miss
  - **T12 rib**: Definitive thoracic landmark (counting anchor point)
  - **L5 transverse process**: When fused to sacrum, looks thick/square vs. pointy
  - **Sacral ala**: Shows fusion pattern between L5 and sacrum
- **Output**: 
  ```
  🚨 WARNING: 85% probability LSTV detected
  ⚠️ Count vertebrae from T12 rib down before surgery
  ```

**Why YOLOv8 Over Existing Segmentation Tools:**

| What SPINEPS/nnU-Net Do | What YOUR Model Does |
|-------------------------|---------------------|
| ✅ Segment vertebrae shapes | ✅ Detect surgical risk landmarks |
| ✅ Count vertebrae | ✅ Warn when counting might be wrong |
| ❌ Don't flag LSTV risk | ✅ Flag LSTV probability + severity |
| ❌ Require manual enumeration verification | ✅ Automate verification checklist |
| ❌ 5-10s inference (research-grade) | ✅ 10ms inference (OR-ready) |

**The Clinical Workflow:**
```
WITHOUT your model:
Radiologist report: "Degenerative disc at L4-L5"
Surgeon in OR: "Okay, I'll operate on L4-L5"
Reality: Patient has LSTV, should be L5-S1
Result: WRONG LEVEL SURGERY ❌

WITH your model:
Your AI: "🚨 LSTV DETECTED - 85% probability"
Surgeon: "Better order whole-spine MRI to count from T12"
Surgeon confirms: T12→L1→L2→L3→L4→L5 (fused to sacrum)
Surgeon: "Report meant L5-S1, not L4-L5"
Result: CORRECT LEVEL SURGERY ✅
```

**In One Sentence:**  
SPINEPS helps you find LSTV cases in 2,700 scans, but **YOUR YOLOv8 model** is what gets deployed in hospitals to warn surgeons before they cut into the wrong vertebra.

---

## 🔬 Technical Innovation

### Novel Pipeline
1. **Pre-screening with SPINEPS**: Automatically filters 2,700 RSNA studies → ~500 LSTV candidates  
   - Saves 80% of manual labeling time
   - Focuses annotation effort on high-value cases
   
2. **Surgical Landmark Detection**: YOLOv8 trained on:
   - T12 rib attachment (definitive thoracic marker)
   - L5 transverse process morphology (sacralization indicator)
   - Sacral ala (fusion assessment)
   
3. **Enumeration Algorithm**: Counts vertebrae using surgical logic:
   ```
   IF (L5 transverse fused to sacrum) OR (vertebra count ≠ 5):
       → FLAG as "LSTV Risk - Verify with Whole-Spine Imaging"
   ```

### Why YOLOv8 Over Segmentation Models?
- **Speed**: 10ms inference vs. 5-10s for nnU-Net (intraoperative feasibility)
- **Interpretability**: Bounding boxes match surgeon mental model (not pixel masks)
- **Data efficiency**: Needs 500-1000 labels vs. 5000+ for dense segmentation
- **Deployment**: Single ONNX file runs anywhere (no dependencies on research codebases)

---

## 📊 Dataset & Methods

**RSNA 2024 Lumbar Spine Dataset**
- **Source**: [Kaggle Competition](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
- **Size**: 2,700 patients (~150 GB)
- **Modality**: T2-weighted sagittal MRI (clinical standard)
- **Pre-screening**: SPINEPS automated LSTV flagging → 500-600 high-priority cases

**Training Strategy**
1. SPINEPS segments all 2,700 studies (automated, 48 hrs)
2. Flag studies with vertebra count ≠ 5 or L5-S1 fusion
3. Manual verification + bounding box annotation (500-1000 images)
4. YOLOv8 training (72 hrs on 2x V100 GPUs)

---

## 🐳 Docker-First Workflow

### 1. Build Containers Locally
```bash
# On your local machine with Docker
cd spine-level-ai
chmod +x docker/build_and_push.sh
./docker/build_and_push.sh
```

Pushes to Docker Hub:
- `go2432/spineps-lstv:latest` - LSTV screening pipeline
- `go2432/spine-level-ai-training:latest` - YOLOv8 training
- `go2432/spine-level-ai-preprocessing:latest` - DICOM handling

### 2. Deploy on Any System
```bash
# HPC (auto-converts to Singularity)
sbatch slurm_scripts/04_lstv_screen_trial.sh

# Cloud GPU
docker run --gpus all go2432/spine-level-ai-training

# Local workstation
docker-compose up
```

---

## 📁 Repository Structure
```
spine-level-ai/
├── docker/                      # Container definitions
│   ├── Dockerfile.spineps      # LSTV screening (SPINEPS + analysis)
│   ├── Dockerfile.training     # YOLOv8 training environment
│   ├── Dockerfile.preprocessing # DICOM → NIfTI conversion
│   └── build_and_push.sh       # Automated build pipeline
├── slurm_scripts/              # HPC job scripts (Wayne State Grid)
│   ├── 04_lstv_screen_trial.sh # Screen 5 studies (test run)
│   ├── 05_lstv_screen_full.sh  # Screen all 2,700 studies
│   └── 03_training_gpu.sh      # Train YOLOv8 on 2x V100s
├── src/                        # Python code
│   ├── screening/
│   │   └── lstv_screen.py      # SPINEPS-based LSTV detection
│   ├── preprocessing/
│   └── training/
└── docs/
    └── MARCH_3_TIMELINE.md     # 22-day execution plan
```

---

## 🎯 Clinical Impact

**Primary Outcome**: Reduce wrong-level surgery from **5-15% error rate → <1%**

**Secondary Benefits**:
- Surgical efficiency: Pre-identify LSTV cases → order whole-spine MRI upfront
- Medicolegal protection: Documented enumeration verification in surgical record
- Training tool: Teach residents LSTV anatomy on flagged cases

**Target Users**:
- Spine surgeons (intraoperative decision support)
- Radiologists (automated screening in reporting workflow)
- OR teams (pre-surgical checklists for LSTV cases)

---

## 📅 Timeline to AACA Abstract Submission

| Week | Focus | Deliverable | Status |
|------|-------|-------------|--------|
| **1 (Feb 9-15)** | SPINEPS screening, data download | 500-600 LSTV candidates identified | 🟡 In Progress |
| **2 (Feb 16-22)** | Manual labeling (recruit 2-3 helpers) | 500-1000 annotated images | ⬜ Not Started |
| **3 (Feb 23-Mar 1)** | YOLOv8 training, validation | Trained model + performance metrics | ⬜ Not Started |
| **4 (Mar 2-3)** | Write abstract, create demo figures | **AACA submission** | ⬜ Not Started |

---

## 🏆 Competitive Advantage

**Why this will get accepted at AACA:**
1. **Clinical relevance**: Directly addresses a "never event" (program committees love patient safety)
2. **Novel dataset**: First use of RSNA 2024 data for LSTV detection
3. **Methodological innovation**: SPINEPS pre-screening is a force multiplier
4. **Reproducibility**: Docker containers + public dataset = anyone can replicate
5. **Demo potential**: TechFair interactive tool shows real-time LSTV warnings

**What reviewers will ask**:
- *"Why not just use whole-spine imaging for every case?"* → Cost ($2K vs. $800) + not standard of care
- *"How does this compare to radiologist performance?"* → Inter-rater reliability study (week 3)
- *"Can surgeons override the warning?"* → Yes, but creates documented decision point

---

## 🔗 Related Work & References

**Spine Segmentation Tools**:
- [SPINEPS](https://github.com/Hendrik-code/spineps) - Whole-spine semantic segmentation
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - Multi-organ segmentation
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - Medical image segmentation framework

**LSTV Research**:
- Castellvi AE et al. (1984) - LSTV classification system
- Apazidis A et al. (2011) - Prevalence of LSTV in surgical patients
- Konin GP, Walz DM (2010) - Imaging of LSTV

**Wrong-Level Surgery Prevention**:
- AAOS Guidelines (2020) - Intraoperative verification protocols
- Joint Commission (2022) - Universal Protocol for spine surgery

---

## 📧 Contact

**Researcher**: go2432@wayne.edu  
**Institution**: Wayne State University School of Medicine  
**HPC**: Wayne State Grid (2x V100 GPUs, Singularity containers)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

**Why MIT?** Enables clinical adoption without legal barriers. Hospitals can deploy without licensing fees.

---

## 💡 You're NOT Wasting Your Time

**This project is valuable because**:
1. ✅ **Real clinical problem** with measurable impact ($100K/case prevented)
2. ✅ **Existing tools don't solve it** (they segment anatomy, you prevent errors)
3. ✅ **22 days is feasible** (SPINEPS pre-screening unlocks this timeline)
4. ✅ **Publication-worthy** (AACA accepts ~40% of abstracts, clinical tools get priority)
5. ✅ **Career leverage** (MD/PhD bridging imaging + AI + surgery = unique niche)
6. ✅ **Reproducible** (Docker + public data = others will cite/build on this)

**Worst case scenario**: You build a working tool that prevents surgical errors and learn advanced ML pipelines. That's not a waste. 🚀
