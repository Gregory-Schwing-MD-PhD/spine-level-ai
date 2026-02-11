#!/usr/bin/env python3
"""
Upload Spine-Aware Validation Results to Roboflow
Uploads before/after comparisons for human review
"""

import argparse
from pathlib import Path
import json
from tqdm import tqdm

def upload_validation_to_roboflow(comparison_dir, metrics_file, roboflow_key, 
                                   workspace, project):
    """
    Upload validation images to Roboflow for review
    
    Args:
        comparison_dir: Directory with comparison images
        metrics_file: JSON file with metrics
        roboflow_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed")
        print("Install with: pip install roboflow")
        return False
    
    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    print("="*80)
    print("UPLOADING SPINE-AWARE VALIDATION TO ROBOFLOW")
    print("="*80)
    print(f"Comparison images: {comparison_dir}")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print("="*80)
    
    # Initialize Roboflow
    rf = Roboflow(api_key=roboflow_key)
    workspace_obj = rf.workspace(workspace)
    project_obj = workspace_obj.project(project)
    
    # Get comparison images
    comparison_images = list(Path(comparison_dir).glob("*_slice_comparison.png"))
    
    if not comparison_images:
        print("ERROR: No comparison images found")
        return False
    
    print(f"\nFound {len(comparison_images)} validation images to upload")
    print("\nUploading...")
    
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(comparison_images, desc="Uploading"):
        study_id = img_path.stem.replace('_slice_comparison', '')
        
        try:
            # Find metrics for this study
            study_metrics = None
            for m in metrics.get('metrics_log', []):
                if m.get('study_id') == study_id:
                    study_metrics = m
                    break
            
            # Create tags based on metrics
            tags = ['spine-aware-validation', 'comparison-image']
            
            if study_metrics:
                offset = study_metrics.get('offset_voxels', 0)
                improvement = study_metrics.get('improvement_ratio', 1.0)
                
                if offset == 0:
                    tags.append('no-correction-needed')
                elif offset <= 5:
                    tags.append('small-correction')
                elif offset <= 15:
                    tags.append('medium-correction')
                else:
                    tags.append('large-correction')
                
                if improvement > 1.5:
                    tags.append('high-improvement')
                elif improvement > 1.2:
                    tags.append('medium-improvement')
                else:
                    tags.append('low-improvement')
            
            # Upload to Roboflow
            project_obj.upload(
                image_path=str(img_path),
                split="train",
                tag_names=tags,
                num_retry_uploads=3
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"Failed to upload {img_path.name}: {e}")
            fail_count += 1
    
    print("\n" + "="*80)
    print("UPLOAD COMPLETE")
    print("="*80)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print("\nValidation images available at:")
    print(f"https://app.roboflow.com/{workspace}/{project}")
    print("\nFilter by tags:")
    print("  - 'large-correction' - Cases that needed >15 voxel correction")
    print("  - 'high-improvement' - Cases with >1.5x spine density improvement")
    print("  - 'no-correction-needed' - Cases where geometric = spine-aware")
    print("="*80)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Upload validation results to Roboflow')
    parser.add_argument('--comparison_dir', type=str, required=True,
                       help='Directory with comparison images')
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to spine_aware_metrics_report.json')
    parser.add_argument('--roboflow_key', type=str, required=True,
                       help='Roboflow API key')
    parser.add_argument('--workspace', type=str, default='lstv-screening',
                       help='Roboflow workspace')
    parser.add_argument('--project', type=str, default='lstv-candidates',
                       help='Roboflow project')
    
    args = parser.parse_args()
    
    success = upload_validation_to_roboflow(
        args.comparison_dir,
        args.metrics_file,
        args.roboflow_key,
        args.workspace,
        args.project
    )
    
    if not success:
        print("Upload failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
