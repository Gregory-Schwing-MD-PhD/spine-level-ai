# LSTV SCREENING v4.0 - COMPLETE SETUP AND USAGE GUIDE

## WHAT THIS SYSTEM DOES

1. **Finds sagittal T2 MRI series** using CSV descriptions (CSV IS TRUTH)
2. **Converts DICOM → NIfTI** without orientation filtering
3. **Runs SPINEPS** to get instance + semantic segmentations
4. **Extracts 4 views per study**:
   - `midline`: TRUE anatomical midline (max spine visibility) → L1-L5-Sacrum segmentation
   - `left`: Left parasagittal (30mm off midline) → T12 left rib detection
   - `mid`: Mid parasagittal (same as midline for now) → L5 transverse processes
   - `right`: Right parasagittal (30mm off midline) → T12 right rib detection
5. **Generates YOLO weak labels** automatically for each view
6. **Creates QA images** showing all 4 views
7. **Calculates multi-view confidence** scores per anatomical structure

## FILES CREATED

- `lstv_screen_production_v4_SIMPLIFIED.py` - Main screening script (v4.0)
- `00_master_pipeline_v4.sh` - SLURM orchestration script
- `dataset.yaml` - YOLO training config (auto-generated)
- `USAGE.md` - This file

## QUICK START

### 1. Run Diagnostic (5 studies)
```bash
sbatch slurm_scripts/00_master_pipeline_v4.sh
```

This will:
- Process 5 studies
- Generate 4 views per study (20 images total)
- Create weak labels automatically
- Generate QA images
- Show you multi-view confidence metrics

### 2. Review Output
```bash
ls results/lstv_pipeline_v4/01_diagnostic/
# images/          - Raw 4-view images
# weak_labels/     - YOLO format labels (.txt files)
# qa_images/       - 4-view comparison images
# results.csv      - Detection metrics
# progress.json    - Processing status
```

### 3. Train YOLO (after full screening)
```bash
# After full pipeline completes
cd results/lstv_pipeline_v4/03_full_screening/

# Dataset is ready - just point YOLO at it
yolo train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## DIRECTORY STRUCTURE

```
results/lstv_pipeline_v4/
├── 01_diagnostic/              # 5 studies
│   ├── images/                 # 4 views × 5 = 20 JPGs
│   ├── weak_labels/            # 20 TXT files (YOLO format)
│   ├── qa_images/              # 20 QA JPGs  
│   ├── nifti/                  # Converted NIfTI files
│   ├── segmentations/          # SPINEPS outputs
│   ├── results.csv             # Per-study metrics
│   ├── progress.json           # Processing status
│   └── dataset.yaml            # YOLO config (auto-generated)
│
├── 02_trial/                   # 50 studies (same structure)
└── 03_full_screening/          # ALL studies (same structure)
```

## WEAK LABEL FORMAT

Each image gets a corresponding `.txt` file with YOLO format labels:

```
# study_12345_midline.txt
0 0.512 0.634 0.123 0.234    # Class 0 (T12 vertebra)
3 0.487 0.712 0.156 0.289    # Class 3 (L5 vertebra)
5 0.501 0.823 0.178 0.312    # Class 5 (Sacrum)
```

Classes:
- 0: T12 vertebra
- 1: T12 left rib
- 2: T12 right rib
- 3: L5 vertebra
- 4: L5 transverse processes
- 5: Sacrum
- 6-9: L4, L1, L2, L3 vertebrae

## MULTI-VIEW CONFIDENCE

The system tracks per-class confidence across all 4 views:

**Example output in `results.csv`:**
```json
{
  "study_id": "12345",
  "multi_view_confidence": [
    {
      "class_id": 0,
      "class_name": "t12_vertebra",
      "aggregate_confidence": 0.87,
      "recommended_view": "midline",
      "entropy_score": 0.23
    },
    {
      "class_id": 1,
      "class_name": "t12_rib_left",
      "aggregate_confidence": 0.64,
      "recommended_view": "left",
      "entropy_score": 0.51
    }
  ]
}
```

- **aggregate_confidence**: Mean confidence across all views that detected this class
- **entropy_score**: Lower = more agreement between views (more reliable)
- **recommended_view**: Which view had highest confidence for this class

## CUSTOMIZATION

### Change Slice Selection
Edit `FourViewSliceSelector` in the main script:

```python
def get_four_slices(self, seg_data, sag_axis):
    optimal_mid = self.find_optimal_midline(seg_data, sag_axis)
    offset_voxels = int(self.parasagittal_offset_mm / self.voxel_spacing_mm)
    
    return {
        'midline': optimal_mid,
        'left': optimal_mid - offset_voxels,  # Adjust offset here
        'mid': optimal_mid,  # Or make this different from midline
        'right': optimal_mid + offset_voxels,
    }
```

### Change MIP Thickness
Edit thickness in `extract_four_view_images`:

```python
if view_name == 'midline':
    thickness = 3  # Thin for vertebrae (change this)
elif view_name in ['left', 'right']:
    thickness = 15  # Thick for ribs (change this)
else:  # mid
    thickness = 10  # Medium for TPs (change this)
```

### Change Confidence Threshold
```bash
# In SLURM script or direct call:
python lstv_screen_production_v4_SIMPLIFIED.py \
  --confidence_threshold 0.5  # Lower = more labels generated
```

## TROUBLESHOOTING

### "No series found"
- Check that `train_series_descriptions.csv` exists
- Verify study_id and series_id columns are integers
- Look for "Sagittal T2" patterns in series_description column

### "SPINEPS failed"
- Check GPU availability: `nvidia-smi`
- Verify container: `singularity exec spine-level-ai-spineps.sif which python`
- Check model cache: `ls spineps_models/`

### "No weak labels generated"
- Check confidence threshold (try lowering it)
- Look at `results.csv` for per-class confidences
- Review QA images to see if segmentation worked

### "Images look wrong"
- This is expected! Different views serve different purposes:
  - **midline**: Should show clear L1-L5-Sacrum alignment
  - **left/right**: May look "off-center" - that's correct for rib detection
  - **mid**: Should show transverse processes laterally

## NEXT STEPS AFTER FULL SCREENING

1. **Review QA Images**
   ```bash
   ls results/lstv_pipeline_v4/03_full_screening/qa_images/ | head -20
   ```

2. **Check Weak Label Quality**
   ```bash
   # Count labels generated
   find results/lstv_pipeline_v4/03_full_screening/weak_labels/ -name "*.txt" | wc -l
   
   # Check class distribution
   cat results/lstv_pipeline_v4/03_full_screening/weak_labels/*.txt | \
     awk '{print $1}' | sort | uniq -c
   ```

3. **Train YOLO**
   ```bash
   cd results/lstv_pipeline_v4/03_full_screening/
   
   yolo train \
     data=dataset.yaml \
     model=yolov8n.pt \
     epochs=100 \
     imgsz=640 \
     batch=16 \
     device=0
   ```

4. **Validate Model**
   ```bash
   yolo val \
     data=dataset.yaml \
     model=runs/detect/train/weights/best.pt
   ```

## EXPECTED TIMELINE

- **Diagnostic** (5 studies): ~5 minutes
- **Trial** (50 studies): ~45 minutes  
- **Full** (~2700 studies): ~6-8 hours

## SUPPORT

If something breaks:

1. Check `logs/lstv_master_*.err` for errors
2. Look at `progress.json` to see what failed
3. Review this USAGE.md for troubleshooting
4. Check QA images to diagnose what went wrong visually

## WHAT'S NEW IN v4.0

- **4 views instead of 3**: Added dedicated midline view
- **Automatic weak labels**: No separate script needed
- **Multi-view confidence**: Tracks which view is best for each class
- **Entropy scoring**: Measures agreement across views
- **Simplified workflow**: One script does everything
