# Implementation Details: Specific Code Locations

## File: generate_weak_labels_enhanced.py

### SECTION 1: Thick Slab MIP Function

**Location:** Lines 183-212
**Function:** `extract_slice(data, sag_axis, slice_idx, thickness=1)`

```python
def extract_slice(data, sag_axis, slice_idx, thickness=1):
    """
    Extract 2D slice or Thick Slab MIP from 3D volume.
    If thickness > 1, performs Maximum Intensity Projection (MIP).
    """
    if thickness <= 1:
        # Single slice (fast path for backward compatibility)
        if sag_axis == 0:
            return data[slice_idx, :, :]
        elif sag_axis == 1:
            return data[:, slice_idx, :]
        else:
            return data[:, :, slice_idx]
    
    # MIP Logic - NEW
    half_thick = thickness // 2
    start = max(0, slice_idx - half_thick)
    end = min(data.shape[sag_axis], slice_idx + half_thick + 1)

    if sag_axis == 0:
        slab = data[start:end, :, :]
    elif sag_axis == 1:
        slab = data[:, start:end, :]
    else:
        slab = data[:, :, start:end]

    return np.max(slab, axis=sag_axis)  # CRITICAL: Maximum Intensity Projection
```

**Key Changes:**
- Added `thickness` parameter (default 1)
- Extracts 3D slab of given thickness
- Returns maximum intensity along sagittal axis
- Backward compatible (thickness=1 works as before)

---

### SECTION 2: Enhanced Bounding Box Extraction

**Location:** Lines 267-284
**Function:** `extract_bounding_box(mask, label_id)`

```python
def extract_bounding_box(mask, label_id):
    """Extract YOLO format bounding box"""
    label_mask = (mask == label_id)

    if not label_mask.any():
        return None

    coords = np.argwhere(label_mask)
    if len(coords) == 0:
        return None

    y_coords = coords[:, 0]
    x_coords = coords[:, 1]

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    height, width = mask.shape

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    # Reject boxes that are too small or invalid - NEW VALIDATION
    if box_width <= 0 or box_height <= 0 or box_width > 1 or box_height > 1:
        return None

    return [x_center, y_center, box_width, box_height]
```

**Key Changes:**
- Added validation: `box_width > 1 or box_height > 1`
- Prevents invalid bounding boxes (e.g., > image size)
- Early return on invalid dimensions

---

### SECTION 3: Robust T12 Rib Detection

**Location:** Lines 286-375
**Function:** `detect_t12_rib_robust(seg_slice, vertebra_label, side='left')`

```python
def detect_t12_rib_robust(seg_slice, vertebra_label, side='left'):
    """ROBUST T12 rib detection using morphological analysis."""
    
    # STEP 1: Get vertebra mask and centroid
    vert_mask = (seg_slice == vertebra_label)
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    # STEP 2: Get vertebra bounding box (critical for adaptive thresholds)
    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min
    
    height, width = seg_slice.shape
    
    # STEP 3: Define anatomically-correct search region
    if side == 'left':
        # Search 0.8x vertebra width to the LEFT of vertebra
        search_x_min = max(0, int(vert_x_min - vert_width * 0.8))
        search_x_max = int(vert_x_min)
    else:
        # Search 0.8x vertebra width to the RIGHT of vertebra
        search_x_min = int(vert_x_max)
        search_x_max = min(width, int(vert_x_max + vert_width * 0.8))
    
    # Search above/level with vertebra (costotransverse anatomy)
    search_y_min = max(0, int(vert_y_min - vert_height * 0.5))
    search_y_max = min(height, int(vert_y_max + vert_height * 0.2))
    
    # STEP 4: Extract rib candidates (non-vertebral segmentations)
    rib_candidates = np.zeros_like(seg_slice)
    for y in range(search_y_min, search_y_max):
        for x in range(search_x_min, search_x_max):
            # Include pixels that are segmented but not vertebra
            if seg_slice[y, x] > 0 and seg_slice[y, x] != vertebra_label:
                rib_candidates[y, x] = seg_slice[y, x]
    
    if not rib_candidates.any():
        return None
    
    # STEP 5: Connected component analysis
    labeled, num_features = scipy_label(rib_candidates > 0)
    
    if num_features == 0:
        return None
    
    # STEP 6: Size validation (anatomically grounded)
    component_sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    
    # Rib should be at least 15% of vertebra size, at most 70% of search region
    min_size = (vert_width * vert_height) * 0.15
    max_size = (search_y_max - search_y_min) * (search_x_max - search_x_min) * 0.7
    
    valid_components = [
        i + 1 for i, size in enumerate(component_sizes)
        if min_size <= size <= max_size
    ]
    
    if not valid_components:
        return None
    
    # STEP 7: Select largest valid component
    largest_comp = max(valid_components, key=lambda c: component_sizes[c - 1])
    rib_mask = (labeled == largest_comp)
    
    if not rib_mask.any():
        return None
    
    return extract_bounding_box(rib_mask.astype(int), 1)
```

**Key Improvements:**
1. Line 327-335: Adaptive search region based on vertebra size (not fixed pixels)
2. Line 353-360: Morphological extraction (not distance-based)
3. Line 363-364: Connected component analysis
4. Line 374-378: Anatomically-grounded size constraints (15-70% of anatomy)
5. Line 380-384: Only accepts properly-sized components

---

### SECTION 4: Robust L5 Transverse Process Detection

**Location:** Lines 378-516
**Function:** `detect_l5_transverse_process_robust(seg_slice, vertebra_label)`

#### Part A: Setup and Candidate Extraction

```python
def detect_l5_transverse_process_robust(seg_slice, vertebra_label):
    """ROBUST L5 TP detection using bilateral analysis."""
    
    # STEP 1: Get vertebra mask
    vert_mask = (seg_slice == vertebra_label)
    if not vert_mask.any():
        return None
    
    coords = np.argwhere(vert_mask)
    cy, cx = coords[:, 0].mean(), coords[:, 1].mean()
    
    # STEP 2: Get vertebra bounding box
    y_vert = coords[:, 0]
    x_vert = coords[:, 1]
    vert_y_min, vert_y_max = y_vert.min(), y_vert.max()
    vert_x_min, vert_x_max = x_vert.min(), x_vert.max()
    vert_width = vert_x_max - vert_x_min
    vert_height = vert_y_max - vert_y_min
    
    height, width = seg_slice.shape
    
    # STEP 3: Define central column exclusion (anatomic)
    # TPs extend from core vertebra outward
    central_width = vert_width * 0.25
    
    # STEP 4: Extract TP candidates (lateral non-vertebral pixels)
    tp_candidates = np.zeros_like(seg_slice)
    
    for y in range(int(vert_y_min - vert_height * 0.1), 
                   int(vert_y_max + vert_height * 0.1)):
        if y < 0 or y >= height:
            continue
        for x in range(width):
            if seg_slice[y, x] > 0 and seg_slice[y, x] != vertebra_label:
                # Check if it's lateral to vertebra
                dist_from_center_x = abs(x - cx)
                if dist_from_center_x > central_width:
                    tp_candidates[y, x] = seg_slice[y, x]
    
    if not tp_candidates.any():
        return None
```

#### Part B: Component Analysis with Position Tracking

```python
    # STEP 5: Connected component analysis
    labeled, num_features = scipy_label(tp_candidates > 0)
    
    if num_features < 2:
        # Need at least 2 for bilateral TPs (or will be single TP case)
        # Handle below
        pass
    
    # STEP 6: Analyze components (position + size)
    component_info = []
    for comp_id in range(1, num_features + 1):
        comp_mask = (labeled == comp_id)
        comp_size = comp_mask.sum()
        comp_coords = np.argwhere(comp_mask)
        
        if len(comp_coords) == 0:
            continue
        
        # Track position and extent
        comp_y_mean = comp_coords[:, 0].mean()
        comp_x_mean = comp_coords[:, 1].mean()
        
        # Component must be reasonably sized
        if comp_size < (vert_width * vert_height * 0.1):
            continue  # Too small
        
        # Must be substantially lateral
        if abs(comp_x_mean - cx) < central_width:
            continue  # Inside vertebra
        
        component_info.append({
            'id': comp_id,
            'size': comp_size,
            'y_mean': comp_y_mean,
            'x_mean': comp_x_mean,
            'y_extent': (comp_coords[:, 0].min(), comp_coords[:, 0].max()),
            'x_extent': (comp_coords[:, 1].min(), comp_coords[:, 1].max()),
        })
    
    if not component_info:
        return None
    
    # Sort by size (largest first)
    component_info.sort(key=lambda c: c['size'], reverse=True)
```

#### Part C: Bilateral Validation

```python
    # STEP 7: Bilateral analysis
    top_components = component_info[:2]
    
    if len(top_components) < 2:
        # Only 1 large component - unilateral case
        if top_components[0]['size'] >= (vert_width * vert_height * 0.2):
            # Still return it (possible unilateral hypoplasia)
            final_mask = np.zeros_like(seg_slice, dtype=bool)
            final_mask[labeled == top_components[0]['id']] = True
            return extract_bounding_box(final_mask.astype(int), 1)
        return None
    
    # STEP 8: Check bilateral symmetry
    left_comp = min(top_components, key=lambda c: c['x_mean'])
    right_comp = max(top_components, key=lambda c: c['x_mean'])
    
    # Calculate size ratio
    size_ratio = max(left_comp['size'], right_comp['size']) / \
                 min(left_comp['size'], right_comp['size'])
    
    # Allow up to 2.5:1 asymmetry (accounting for anatomy variation)
    if size_ratio > 2.5:
        # Suspicious asymmetry - return only the larger one
        return extract_bounding_box((labeled == component_info[0]['id']).astype(int), 1)
    
    # STEP 9: Combine bilateral components
    final_mask = np.zeros_like(seg_slice, dtype=bool)
    final_mask[labeled == left_comp['id']] = True
    final_mask[labeled == right_comp['id']] = True
    
    return extract_bounding_box(final_mask.astype(int), 1)
```

**Key Improvements:**
1. Line 419: Adaptive central exclusion (25% of vertebra, not fixed pixels)
2. Line 422-440: Component tracking with position info
3. Line 444-454: Component filtering by size and position
4. Line 472-482: Left/right identification and symmetry check
5. Line 484-490: Graceful handling of unilateral cases

---

### SECTION 5: Main Label Generation Loop

**Location:** Lines 666-748
**Function:** `create_yolo_labels_multiview(...)`

#### Critical Section: Adaptive MIP Application

```python
    # ... (after getting slice_info and views) ...
    
    label_count = 0

    for view_name, slice_idx in views.items():
        # CRITICAL: ADAPTIVE MIP THICKNESS
        # Ribs need thick slab (15mm) to capture curved anatomy
        # Midline needs thin slab (5mm) to keep spinal canal sharp
        thickness = 15 if view_name in ['left', 'right'] else 5
        
        # CRITICAL: Apply MIP to BOTH MRI and SEGMENTATION
        mri_slice = extract_slice(mri_data, sag_axis, slice_idx, thickness=thickness)
        seg_slice = extract_slice(seg_data, sag_axis, slice_idx, thickness=thickness)

        mri_normalized = normalize_slice(mri_slice)

        image_filename = f"{study_id}_{view_name}.jpg"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), mri_normalized)

        yolo_labels = []

        # Class 0: T12 vertebra
        t12_box = extract_bounding_box(seg_slice, SPINEPS_LABELS['T12'])
        if t12_box:
            yolo_labels.append([0] + t12_box)

        # Class 1: T12 rib (parasagittal only) - ROBUST DETECTION
        if view_name in ['left', 'right']:
            t12_rib_box = detect_t12_rib_robust(seg_slice, SPINEPS_LABELS['T12'], 
                                                  side=view_name)
            if t12_rib_box:
                yolo_labels.append([1] + t12_rib_box)

        # ... (other vertebrae) ...

        # Class 3: L5 transverse process (mid only) - ROBUST DETECTION
        if view_name == 'mid':
            transverse_box = detect_l5_transverse_process_robust(seg_slice, 
                                                                   SPINEPS_LABELS['L5'])
            if transverse_box:
                yolo_labels.append([3] + transverse_box)

        # ... (remaining classes) ...

        if yolo_labels:
            label_filename = f"{study_id}_{view_name}.txt"
            label_path = output_dir / label_filename

            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} "
                           f"{label[3]:.6f} {label[4]:.6f}\n")

            label_count += len(yolo_labels)
```

**Key Changes:**
1. Line 673: Adaptive thickness (15mm for ribs, 5mm for midline)
2. Line 675-677: BOTH MRI and segmentation extracted with MIP
3. Line 689: Robust rib detection on MIP'd segmentation
4. Line 703: Robust TP detection on MIP'd segmentation

---

## Parameter Tuning Guide

### If You Want to Adjust Sensitivity

#### Rib Detection Sensitivity
```python
# File: Line 374-378
min_size = (vert_width * vert_height) * 0.15  # ← Increase to reduce false positives
max_size = (search_mask.sum()) * 0.7           # ← Decrease to be more selective

# Example: More sensitive (catch smaller ribs)
min_size = (vert_width * vert_height) * 0.10  # Down from 0.15
max_size = (search_mask.sum()) * 0.8           # Up from 0.7
```

#### TP Detection Sensitivity
```python
# File: Line 419
central_width = vert_width * 0.25  # ← Decrease to be more selective, increase to be more sensitive

# Example: More selective (only clear TPs)
central_width = vert_width * 0.30  # Up from 0.25
```

#### Symmetry Tolerance
```python
# File: Line 484
if size_ratio > 2.5:  # ← Increase to accept more asymmetry, decrease for strict bilateral

# Example: Stricter bilateral requirement
if size_ratio > 2.0:  # Down from 2.5
```

#### MIP Thickness
```python
# File: Line 673
thickness = 15 if view_name in ['left', 'right'] else 5  # ← Adjust here

# Example: Thinner slabs for sharper detail
thickness = 10 if view_name in ['left', 'right'] else 3
```

---

## Validation Checklist

### Before Running on Full Dataset

- [ ] Extract slice function correctly handles thickness parameter
- [ ] T12 rib detection operates on 15mm MIP segmentation
- [ ] L5 TP detection operates on 5mm MIP segmentation
- [ ] Size constraints are relative to vertebra (not absolute pixels)
- [ ] Connected component analysis is being used
- [ ] Bilateral validation checks symmetry
- [ ] Bounding box validation prevents invalid sizes
- [ ] Segmentation MIP applied (critical!)
- [ ] Output format matches YOLO standards

### Expected Output Structure

```
output_dir/
├── labels/train/
│   ├── studyID_left.txt   (3 classes: vertebrae + rib)
│   ├── studyID_mid.txt    (5 classes: vertebrae + TP + disc)
│   └── studyID_right.txt  (3 classes: vertebrae + rib)
├── images/train/
│   ├── studyID_left.jpg
│   ├── studyID_mid.jpg
│   └── studyID_right.jpg
├── labels/val/
├── images/val/
├── quality_validation/
│   └── studyID_slice_comparison.png
├── dataset.yaml
└── metadata.json
```

---

## Quick Debugging Checklist

```python
# Add to detect_t12_rib_robust() if debugging:
print(f"Rib search: x=[{search_x_min}, {search_x_max}], y=[{search_y_min}, {search_y_max}]")
print(f"Rib candidates found: {rib_candidates.any()}")
print(f"Components: {num_features}")
print(f"Sizes: {component_sizes}")
print(f"Valid components: {valid_components}")

# Add to detect_l5_transverse_process_robust() if debugging:
print(f"Central exclusion width: {central_width}")
print(f"Components: {len(component_info)}")
print(f"Symmetry ratio: {size_ratio}")
```

