# Spine Level AI

**AI-Assisted Vertebral Level Identification for Surgical Safety**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/container-docker-blue.svg)](https://hub.docker.com/)

## 🚨 Quick Start for March 3rd Deadline

**Abstract due**: March 3, 2026
**Time available**: 22 days

See **[`docs/MARCH_3_TIMELINE.md`](docs/MARCH_3_TIMELINE.md)** for day-by-day execution plan.

## Overview

An AI system that automatically identifies vertebral levels on whole-spine MRI scans to prevent wrong-level surgery. Uses YOLOv8 to detect anatomical landmarks (T12 rib, L5 transverse process, sacrum) and flags lumbosacral transitional vertebrae (LSTV).

**Clinical Problem**: Wrong-level spine surgery is a "never event" costing $100K+ per case, often caused by LSTV confusing vertebral counting.

**Solution**: Automated warnings for surgical enumeration risk.

## Docker-First Workflow

### 1. Build Containers Locally
```bash
# On your local machine with Docker
cd spine-level-ai
chmod +x docker/build_and_push.sh
./docker/build_and_push.sh
```

This pushes to Docker Hub:
- `go2432/spine-level-ai-preprocessing:latest`
- `go2432/spine-level-ai-training:latest`

### 2. Use on HPC (Auto-converts to Singularity)
```bash
# Wayne State Grid automatically converts Docker → Singularity
sbatch slurm_scripts/01_download_data.sh
```

Singularity pulls from Docker Hub on-the-fly!

## Repository Structure

```
spine-level-ai/
├── docker/                      # Dockerfiles
│   ├── Dockerfile.training     # PyTorch + YOLOv8
│   ├── Dockerfile.preprocessing # DICOM handling
│   └── build_and_push.sh       # Build & push to Docker Hub
├── slurm_scripts/              # HPC job scripts
│   ├── 01_download_data.sh
│   ├── 02_preprocessing.sh
│   └── 03_training_gpu.sh
├── src/                        # Python code
│   ├── preprocessing/
│   └── training/
└── docs/                       # Documentation
    └── MARCH_3_TIMELINE.md     # Detailed execution plan
```

## Method

1. **Landmark Detection**: YOLOv8 identifies T12 rib, L5 transverse process, sacrum
2. **Enumeration**: Algorithm counts vertebrae from T12 downward
3. **LSTV Detection**: Flags L5-S1 fusion (Castellvi Types)
4. **Warning System**: Alerts surgeon to enumeration risk

## Dataset

**RSNA 2024 Lumbar Spine Degenerative Classification**
- Source: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
- Size: ~150 GB
- Patients: 2000+

## Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 (Feb 9-15) | Setup, download data | Dataset ready |
| 2 (Feb 16-22) | Label images | Training data |
| 3 (Feb 23-Mar 1) | Train model, write | Results + draft |
| 4 (Mar 2-3) | Polish, submit | Abstract submitted |

## Target Venue

**American Association of Clinical Anatomists (AACA) Annual Meeting**
- TechFair: Interactive demo
- Platform talk: Research presentation

## Contact

go2432@wayne.edu
Wayne State University School of Medicine

## License

MIT License - see LICENSE file
