#!/usr/bin/env python3
"""
RSNA 2024 Dataset Download Script
Uses Kaggle Python API v1.8.4 with progress monitoring
"""

import sys
import os
from pathlib import Path
import zipfile
import time
import threading
from kaggle.api.kaggle_api_extended import KaggleApi

def monitor_download(zip_path, stop_event):
    """Monitor download progress by watching file size"""
    print("\n" + "="*60)
    print("DOWNLOAD PROGRESS")
    print("="*60)
    sys.stdout.flush()
    
    last_size = 0
    start_time = time.time()
    stall_count = 0
    expected_total_gb = 28.2
    
    while not stop_event.is_set():
        if zip_path.exists():
            current_size = zip_path.stat().st_size
            size_gb = current_size / (1024**3)
            elapsed = time.time() - start_time
            
            if elapsed > 0:
                speed_mb = (current_size / (1024**2)) / elapsed
                
                if current_size > last_size:
                    percent = (size_gb / expected_total_gb) * 100
                    remaining_gb = expected_total_gb - size_gb
                    eta_seconds = (remaining_gb * 1024) / speed_mb if speed_mb > 0 else 0
                    
                    print(f"[{percent:5.1f}%] {size_gb:6.2f} / {expected_total_gb:.1f} GB | "
                          f"{speed_mb:6.2f} MB/s | ETA: {eta_seconds/60:5.1f} min", 
                          flush=True)
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count < 6:  # Only show first few stalls
                        print(f"[{size_gb:6.2f} GB] Waiting... ({stall_count})", flush=True)
            
            last_size = current_size
        else:
            print("Waiting for download to start...", flush=True)
        
        time.sleep(10)  # Update every 10 seconds

def download_rsna():
    """Download RSNA 2024 dataset using Kaggle Python API"""
    
    data_dir = Path("/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RSNA 2024 LUMBAR SPINE DATASET DOWNLOAD")
    print("="*60)
    print(f"Target: {data_dir}")
    print(f"Expected size: ~150 GB")
    print("="*60)
    sys.stdout.flush()
    
    # Initialize Kaggle API
    print("\n[1/4] Authenticating with Kaggle...")
    sys.stdout.flush()
    
    try:
        api = KaggleApi()
        api.authenticate()
        print("      ✓ Authenticated successfully")
    except Exception as e:
        print(f"      ✗ Authentication failed: {e}")
        return
    
    sys.stdout.flush()
    
    # Verify competition access
    print("\n[2/4] Verifying competition access...")
    sys.stdout.flush()
    
    try:
        files = api.competition_list_files('rsna-2024-lumbar-spine-degenerative-classification')
        print(f"      ✓ Competition accessible ({len(files)} files)")
    except Exception as e:
        print(f"      ✗ Cannot access competition: {e}")
        print("\n      SOLUTION: Accept rules at:")
        print("      https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification")
        return
    
    sys.stdout.flush()
    
    # Start download with monitoring
    print("\n[3/4] Starting download...")
    print("      This will take several hours for ~150 GB")
    sys.stdout.flush()
    
    zip_file = data_dir / "rsna-2024-lumbar-spine-degenerative-classification.zip"
    
    # Start monitor thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_download,
        args=(zip_file, stop_monitor),
        daemon=True
    )
    monitor_thread.start()
    
    # Download (this blocks until complete)
    try:
        api.competition_download_files(
            competition='rsna-2024-lumbar-spine-degenerative-classification',
            path=str(data_dir),
            quiet=True  # Suppress API's own output
        )
    except Exception as e:
        stop_monitor.set()
        monitor_thread.join(timeout=2)
        print(f"\n      ✗ Download failed: {e}")
        return
    
    # Stop monitor
    stop_monitor.set()
    monitor_thread.join(timeout=2)
    
    print("\n" + "="*60)
    print("      ✓ Download complete!")
    print("="*60)
    sys.stdout.flush()
    
    # Extract
    print("\n[4/4] Extracting archive...")
    sys.stdout.flush()
    
    if not zip_file.exists():
        print("      ✗ ERROR: Zip file not found!")
        return
    
    size_gb = zip_file.stat().st_size / (1024**3)
    print(f"      File: {zip_file.name} ({size_gb:.2f} GB)")
    sys.stdout.flush()
    
    try:
        print("      Opening archive...")
        sys.stdout.flush()
        
        with zipfile.ZipFile(zip_file, 'r') as z:
            members = z.namelist()
            total = len(members)
            print(f"      Total files in archive: {total:,}")
            sys.stdout.flush()
            
            print("\n      Extracting...")
            start_time = time.time()
            
            for i, member in enumerate(members, 1):
                z.extract(member, data_dir)
                
                # Progress every 500 files
                if i % 500 == 0 or i == total:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total - i) / rate if rate > 0 else 0
                    percent = (i * 100) // total
                    
                    print(f"      [{percent:3d}%] {i:,} / {total:,} files | "
                          f"{rate:.1f} files/s | ETA: {remaining/60:.1f} min",
                          flush=True)
        
        print("\n      ✓ Extraction complete!")
        print(f"      Removing zip file to save {size_gb:.2f} GB...")
        sys.stdout.flush()
        
        zip_file.unlink()
        print("      ✓ Cleanup complete!")
        
    except zipfile.BadZipFile as e:
        print(f"      ✗ ERROR: {e}")
        print("      File may be corrupted")
        return
    except Exception as e:
        print(f"      ✗ ERROR during extraction: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("ALL OPERATIONS COMPLETE")
    print("="*60)
    
    extracted_files = [f for f in data_dir.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in extracted_files)
    
    print(f"Location:     {data_dir}")
    print(f"Total files:  {len(extracted_files):,}")
    print(f"Total size:   {total_size / (1024**3):.2f} GB")
    print("="*60)
    sys.stdout.flush()

if __name__ == "__main__":
    download_rsna()
