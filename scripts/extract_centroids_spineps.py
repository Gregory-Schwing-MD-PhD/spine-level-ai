#!/usr/bin/env python3
"""
Extract Centroids from SPINEPS Segmentations

This script should be run AFTER SPINEPS segmentation to generate
centroid JSON files needed for the uncertainty fusion pipeline.

Run this in your SPINEPS environment/container!

Usage:
    python extract_centroids_spineps.py \
        --spineps_dir /path/to/spineps/results \
        --output_dir /path/to/spineps/results/centroids \
        --mode prod
"""

import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_centroids_from_spineps(seg_path: Path) -> dict:
    """
    Extract 3D centroids from SPINEPS instance segmentation
    
    Args:
        seg_path: Path to *_seg-vert.nii.gz file
        
    Returns:
        Dict with centroid data for each vertebra instance
    """
    # Load segmentation
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata().astype(int)
    affine = seg_nii.affine
    
    logger.debug(f"Loaded segmentation: shape={seg_data.shape}, unique_labels={np.unique(seg_data)}")
    
    centroids = {}
    
    # SPINEPS labels:
    # 1-25: Vertebra instances
    # 101-125: Intervertebral discs (100 + vertebra ID above)
    # 201-225: Endplates (200 + vertebra ID)
    
    # Extract only vertebra instances (1-25)
    for instance_id in range(1, 26):
        mask = (seg_data == instance_id)
        
        if mask.sum() == 0:
            # No voxels for this instance
            continue
        
        # Calculate centroid in voxel space
        centroid_voxel = center_of_mass(mask)
        
        # Transform to world coordinates using affine
        centroid_voxel_homo = np.array([*centroid_voxel, 1.0])
        centroid_world = (affine @ centroid_voxel_homo)[:3]
        
        # Calculate volume
        volume_voxels = int(mask.sum())
        
        # Store centroid data
        vertebra_key = f"vertebra_{instance_id}"
        centroids[vertebra_key] = {
            'instance_id': int(instance_id),
            'centroid_voxel': [float(c) for c in centroid_voxel],  # [i, j, k]
            'centroid_world': [float(c) for c in centroid_world],  # [x, y, z] in mm
            'volume_voxels': volume_voxels
        }
    
    logger.debug(f"Extracted {len(centroids)} vertebra centroids")
    
    return centroids


def save_centroids_json(centroids: dict, output_path: Path):
    """Save centroids as JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(centroids, f, indent=2)
    
    logger.debug(f"Saved centroids to: {output_path}")


def process_all_studies(spineps_dir: Path, output_dir: Path, mode: str = 'prod'):
    """
    Process all SPINEPS segmentations to extract centroids
    
    Args:
        spineps_dir: Directory containing *_seg-vert.nii.gz files
        output_dir: Output directory for centroid JSON files
        mode: 'trial' (10 studies), 'debug' (1 study), or 'prod' (all)
    """
    logger.info("="*60)
    logger.info("SPINEPS Centroid Extraction")
    logger.info("="*60)
    logger.info(f"SPINEPS directory: {spineps_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {mode}")
    
    # Find all segmentation files
    seg_files = sorted(spineps_dir.glob("*_seg-vert.nii.gz"))
    logger.info(f"Found {len(seg_files)} SPINEPS segmentation files")
    
    if len(seg_files) == 0:
        logger.error("No segmentation files found!")
        logger.error(f"Expected files like: {spineps_dir}/*_seg-vert.nii.gz")
        return
    
    # Select files based on mode
    if mode == 'trial':
        seg_files = seg_files[:10]
        logger.info(f"Trial mode: Processing {len(seg_files)} files")
    elif mode == 'debug':
        seg_files = seg_files[:1]
        logger.info(f"Debug mode: Processing {len(seg_files)} file")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each study
    success_count = 0
    error_count = 0
    
    for seg_file in tqdm(seg_files, desc="Extracting centroids"):
        # Extract study ID from filename
        # Format: {study_id}_seg-vert.nii.gz
        study_id = seg_file.stem.replace("_seg-vert", "")
        
        try:
            # Extract centroids
            centroids = extract_centroids_from_spineps(seg_file)
            
            if len(centroids) == 0:
                logger.warning(f"No vertebrae found in {study_id}")
                error_count += 1
                continue
            
            # Save as JSON
            output_path = output_dir / f"{study_id}_centroids.json"
            save_centroids_json(centroids, output_path)
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {study_id}: {e}")
            error_count += 1
    
    # Summary
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Verify centroid JSON files were created")
    logger.info("2. Run lstv-uncertainty-detection with --centroid_dir flag")
    logger.info("3. Heatmaps will be sampled at these SPINEPS centroid locations")


def verify_centroid_file(centroid_path: Path):
    """
    Verify a centroid JSON file has the expected structure
    
    Args:
        centroid_path: Path to centroid JSON file
    """
    logger.info(f"\nVerifying: {centroid_path}")
    
    with open(centroid_path, 'r') as f:
        centroids = json.load(f)
    
    logger.info(f"  Number of vertebrae: {len(centroids)}")
    
    # Show first vertebra as example
    first_key = list(centroids.keys())[0]
    first_centroid = centroids[first_key]
    
    logger.info(f"  Example vertebra: {first_key}")
    logger.info(f"    Instance ID: {first_centroid['instance_id']}")
    logger.info(f"    Centroid (voxel): {first_centroid['centroid_voxel']}")
    logger.info(f"    Centroid (world mm): {first_centroid['centroid_world']}")
    logger.info(f"    Volume: {first_centroid['volume_voxels']} voxels")
    
    # Check for expected keys
    required_keys = ['instance_id', 'centroid_voxel', 'centroid_world', 'volume_voxels']
    for key in required_keys:
        if key not in first_centroid:
            logger.error(f"    ✗ Missing key: {key}")
        else:
            logger.info(f"    ✓ Has key: {key}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract centroids from SPINEPS segmentations for uncertainty fusion'
    )
    
    parser.add_argument('--spineps_dir', type=str, required=True,
                       help='Directory with SPINEPS segmentation files (*_seg-vert.nii.gz)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for centroid JSON files')
    parser.add_argument('--mode', type=str, choices=['trial', 'debug', 'prod'],
                       default='prod',
                       help='Processing mode: trial (10), debug (1), prod (all)')
    parser.add_argument('--verify', type=str, default=None,
                       help='Verify a single centroid JSON file')
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify mode
        verify_centroid_file(Path(args.verify))
    else:
        # Extraction mode
        spineps_dir = Path(args.spineps_dir)
        output_dir = Path(args.output_dir)
        
        if not spineps_dir.exists():
            logger.error(f"SPINEPS directory not found: {spineps_dir}")
            return
        
        process_all_studies(spineps_dir, output_dir, args.mode)


if __name__ == '__main__':
    main()
