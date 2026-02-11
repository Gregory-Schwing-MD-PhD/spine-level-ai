# LSTV Detection - Annotation Guidelines for Medical Students

**Project:** AI-Assisted Vertebral Level Identification for Surgical Safety  
**Institution:** Wayne State University School of Medicine  
**Version:** 2.0  
**Date:** February 2026

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Your Role & Contribution](#your-role--contribution)
3. [Annotation Platform Setup](#annotation-platform-setup)
4. [Anatomical Landmarks Guide](#anatomical-landmarks-guide)
5. [Annotation Workflow](#annotation-workflow)
6. [Quality Control Checklist](#quality-control-checklist)
7. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
8. [Time Estimates](#time-estimates)

---

## Project Overview

### The Clinical Problem
- **5-15% of spine surgeries** occur at the wrong vertebral level
- Primary cause: **LSTV (Lumbosacral Transitional Vertebrae)** enumeration errors
- This is the **#1 preventable error** in spine surgery

### Our Solution
1. SPINEPS automatically screens 2,700 MRI studies → identifies ~500 LSTV candidates
2. **YOU** refine landmark annotations on 200 key images
3. YOLOv11 learns from refined labels → detects critical anatomical structures
4. Algorithm warns surgeons: **"Count from T12 down, not from sacrum up"**

### Impact
- **Current error rate:** 5-15%
- **Target with AI:** <1%
- **Your contribution:** The difference between "interesting research" and "clinically viable tool"

---

## Your Role & Contribution

### What You'll Do
Refine and correct bounding box annotations for 7 anatomical landmarks on sagittal T2-weighted MRI images.

### Time Commitment
- **Total:** 10-12 hours per person
- **Schedule:** 2 hours/day × 5-6 days
- **Target:** 200 images total (100 per person)
- **Average:** 5-6 minutes per image

### Authorship
You will be **co-authors** on the resulting publication:
- Methods section will explicitly acknowledge your contribution
- Expected submission: March 2026
- Target conferences: AACA, RSNA, or similar

### Skills You'll Gain
- Medical image annotation
- Understanding of AI training pipelines
- Deep knowledge of LSTV anatomy
- Experience with computer vision tools
- Publication credit

---

## Annotation Platform Setup

### Option A: Roboflow (Recommended - Easiest)

**Why Roboflow:**
- Web-based (no installation)
- Pre-populated with images
- Real-time collaboration
- Export directly to YOLO format

**Access:**
1. Go to: [roboflow.com/spinelevelai/lstv-candidates](https://roboflow.com/spinelevelai/lstv-candidates)
2. Create account with your WSU email
3. Request access from project lead
4. You'll see ~500 images with automated annotations

**Workflow:**
1. Filter images by tag: `needs-review`
2. Review/correct each bounding box
3. Add tag: `human-verified-[YOUR-INITIALS]`
4. Move to next image

**Keyboard Shortcuts:**
- `D` = Draw new box
- `E` = Edit box
- `Delete` = Remove box
- `S` = Save and next
- `1-7` = Select class

### Option B: Label Studio (Advanced)

For local installation (if you prefer):
```bash
# One-time setup
docker pull heartexlabs/label-studio:latest
docker run -p 8080:8080 -v ~/lstv-annotation:/label-studio/data heartexlabs/label-studio:latest

# Then open: http://localhost:8080
```

---

## Anatomical Landmarks Guide

### Class 0: T12 Vertebra ⭐ MOST CRITICAL

**Anatomy:**
- Last thoracic vertebra
- Has rib attachments
- Definitive marker for thoracic enumeration

**Where to look:**
- Top 1/3 of image (if visible)
- Look for ribs extending laterally
- Vertebral body is rectangular

**Bounding box:**
- Include entire vertebral body
- Exclude ribs (they're separate)
- Exclude disc spaces above/below

**Common mistakes:**
- ❌ Confusing with T11 or L1
- ❌ Including ribs in T12 box
- ✅ Use ribs as confirmation (if T12, ribs must be present)

**Clinical importance:**
- **CRITICAL:** Without T12, cannot definitively enumerate
- If T12 not visible: Mark image with tag `t12-not-in-fov`

---

### Class 1: T12 Rib ⭐ CRITICAL

**Anatomy:**
- 12th pair of ribs
- Attach to T12 vertebra
- Extend laterally and anteriorly
- Often shorter than other ribs

**Where to look:**
- **LEFT/RIGHT parasagittal views ONLY**
- Lateral extension from T12
- May curve anteriorly

**Bounding box:**
- Focus on **rib head** (attachment point to T12)
- Include proximal 2-3 cm of rib
- Don't try to capture entire rib (often extends beyond FOV)

**Common mistakes:**
- ❌ Confusing with transverse process
- ❌ Trying to box entire rib (just the head!)
- ❌ Looking for ribs on mid-sagittal view (not visible there)
- ✅ Ribs appear on lateral views only

**If not visible:**
- Mark image with tag: `t12-rib-not-in-fov`
- This is common (~30% of cases)
- **Important for analysis:** We need to know when ribs are absent

---

### Class 2: L5 Vertebra

**Anatomy:**
- Last lumbar vertebra
- Articulates with sacrum below
- In LSTV: May be fused to sacrum (sacralization)

**Where to look:**
- Lower 1/3 of image
- Directly above sacrum
- Look for L5-S1 disc space (or absence in fusion)

**Bounding box:**
- Include entire vertebral body
- Exclude transverse processes
- Include superior/inferior endplates

**Special cases:**
- **Sacralization:** L5 partially fused to sacrum
  - Box what you can see of separate L5 body
  - If completely fused: mark as `l5-fused` tag
- **Lumbarization:** 6 lumbar vertebrae present
  - Box the last mobile segment as L5

**Common mistakes:**
- ❌ Including transverse processes (they're separate)
- ❌ In sacralization, missing small remnant of L5
- ✅ Look for disc space to distinguish L5 from sacrum

---

### Class 3: L5 Transverse Process

**Anatomy:**
- Lateral projections from L5
- Bilateral (left and right)
- In LSTV: Often enlarged and may fuse with sacrum

**Where to look:**
- **MID-SAGITTAL view ONLY**
- Lateral to L5 vertebral body
- Appear as wing-like projections

**Bounding box:**
- Include BOTH left and right transverse processes
- Draw one box encompassing both
- Exclude vertebral body (already boxed separately)

**Special cases:**
- **Large transverse processes:** Suggests LSTV
- **Fusion with sacrum:** Include fused portion
- **Asymmetric:** Include both even if different sizes

**Common mistakes:**
- ❌ Boxing only one side
- ❌ Including vertebral body
- ❌ Looking for this on parasagittal views (won't see both sides)

---

### Class 4: Sacrum

**Anatomy:**
- Fused vertebrae (S1-S5)
- Triangular bone
- Forms posterior pelvis

**Where to look:**
- Bottom of spine
- Below L5
- Large, solid bone

**Bounding box:**
- **TOP of S1 ONLY**
- Do NOT box entire sacrum (too large)
- Include approximately S1 body only

**Common mistakes:**
- ❌ Boxing entire sacrum (too much)
- ❌ Including coccyx
- ✅ Just S1, where it articulates with L5

---

### Class 5: L4 Vertebra

**Anatomy:**
- Second-to-last lumbar vertebra
- Above L5
- Used for validation/counting

**Where to look:**
- Middle-to-lower spine
- Two vertebrae above sacrum
- Separated from L5 by L4-L5 disc

**Bounding box:**
- Include entire vertebral body
- Exclude disc spaces
- Exclude transverse processes

**Purpose:**
- Confirms enumeration
- Helps algorithm count vertebrae
- Less critical than T12 and L5

---

### Class 6: L5-S1 Disc ⭐ NEW

**Anatomy:**
- Intervertebral disc between L5 and sacrum
- In normal anatomy: Clearly visible
- In sacralization: Absent or very thin

**Where to look:**
- Between L5 and sacrum
- Dark signal on T2 (disc is darker than bone)
- May be thin or absent

**Bounding box:**
- Entire disc space
- Include anterior and posterior portions
- Width = vertebral body width

**Special cases:**
- **Normal:** Clear disc space, box it
- **Thinned:** Narrow disc, box what's visible
- **Fused:** No disc space visible, add tag `l5-s1-fused`

**Clinical importance:**
- Disc absence = sacralization
- Very thin disc = partial sacralization
- Critical for classification algorithm

---

## Annotation Workflow

### Daily Workflow (2 hours/day)

```
1. Log in to Roboflow
2. Select "Images needing review"
3. For each image:
   a. View automated boxes (already present)
   b. Check each class (0-6)
   c. Correct/add/remove boxes as needed
   d. Add tags if needed
   e. Click "Save & Next"
4. Track progress (aim for 20-25 images/day)
5. End session, log out
```

### Review Priority (Week 1)

**Day 1-2:** Images where SPINEPS found T12 (verify correctness)
**Day 3-4:** Images where SPINEPS missed T12 (add if visible)
**Day 5:** Sacralization cases (refine L5-sacrum interface)

### Review Priority (Week 2)

**Day 6:** Lumbarization cases (verify L6 if present)
**Day 7:** Random quality control (fix any errors)
**Day 8:** Final review (check your own work)

### Image Selection Strategy

Your PI will assign images:
- **Person 1:** Images 1-100
- **Person 2:** Images 101-200
- **Overlap:** 20 images for inter-rater reliability

---

## Quality Control Checklist

### Before clicking "Save & Next":

- [ ] All visible landmarks are annotated
- [ ] No duplicate boxes for same structure
- [ ] Boxes are tight (not too loose)
- [ ] Correct class labels (0-6)
- [ ] T12 rib only on lateral views
- [ ] Transverse process only on mid view
- [ ] Added tags if structures missing
- [ ] Checked for common mistakes (see below)

### Daily Quality Metrics

Track your own performance:
```
Day 1: ___ images completed, ___ corrections made
Day 2: ___ images completed, ___ corrections made
...
```

### Inter-rater Reliability

20 images will be annotated by BOTH of you:
- Compare your annotations
- Discuss discrepancies
- Develop consensus
- Improve consistency

---

## Common Mistakes to Avoid

### ❌ Mistake #1: Confusing T12 with L1
**Solution:** Look for ribs! T12 has ribs, L1 does not.

### ❌ Mistake #2: Boxing ribs on mid-sagittal view
**Solution:** Ribs only visible on parasagittal (lateral) views.

### ❌ Mistake #3: Including transverse processes in vertebral body
**Solution:** Vertebral body and transverse processes are SEPARATE classes.

### ❌ Mistake #4: Missing small L5 in sacralization
**Solution:** Look carefully for small remnant of L5 above sacrum.

### ❌ Mistake #5: Boxing entire sacrum instead of just S1
**Solution:** Only box top of S1 where it articulates with L5.

### ❌ Mistake #6: Forgetting to add tags when structures are absent
**Solution:** Always tag: `t12-not-in-fov`, `l5-s1-fused`, etc.

### ❌ Mistake #7: Boxes too loose or too tight
**Solution:** Box should tightly encompass structure with small margin.

### ❌ Mistake #8: Not checking all 7 classes per image
**Solution:** Methodically check class 0, 1, 2, 3, 4, 5, 6 for each image.

---

## Time Estimates

### Per Image
- **Fast cases:** 3-4 minutes (mostly correct automated labels)
- **Average cases:** 5-6 minutes (some corrections needed)
- **Difficult cases:** 8-10 minutes (major corrections, missing structures)

### Total Project
- **100 images × 6 minutes = 600 minutes = 10 hours**
- **Add 2 hours for training and quality control**
- **Total: ~12 hours per person**

### Breaking It Down
- **Day 1 (Training):** 1 hour orientation + 1 hour practice = 10 images
- **Days 2-6:** 2 hours/day × 5 days = 18-20 images/day = 90 images
- **Day 7 (QC):** 2 hours quality control and overlap review

---

## Getting Started Checklist

Before you begin:
- [ ] Received Roboflow access
- [ ] Reviewed this document thoroughly
- [ ] Completed 10 practice images with PI
- [ ] Understand all 7 anatomical classes
- [ ] Know how to add tags for missing structures
- [ ] Scheduled your 2-hour daily blocks
- [ ] Set up progress tracking spreadsheet

---

## Questions?

**Contact:**
- PI: [Your email]
- Project lead: go2432@wayne.edu

**Resources:**
- SPINEPS documentation: [link]
- LSTV anatomy review: [link]
- Roboflow tutorials: [link]

---

## Acknowledgment

Your work on this project will be acknowledged in the following ways:

1. **Authorship:** Co-author on resulting publication
2. **Methods section text:**
   > "Initial weak labels were generated from SPINEPS segmentations (version X.X). A subset of 200 images was manually refined by two medical students (J.S. and A.K.) to correct landmark localization errors, add missing T12 rib annotations, and verify sacralization/lumbarization classifications. Inter-rater reliability was assessed on 20 overlapping images (Cohen's kappa = X.XX)."

3. **Acknowledgments section:**
   > "We thank medical students [Your names] for their meticulous annotation work."

---

**Thank you for your contribution to surgical safety!**

*This project aims to reduce wrong-level spine surgery from 5-15% to <1% through AI-assisted enumeration.*
