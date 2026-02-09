#!/usr/bin/env python3
"""
RSNA 2024 Dataset Download Script
Uses Kaggle Python API v1.8.4 with integrity checks
"""

import sys
import hashlib
from pathlib import Path
import zipfile
import time
from kaggle.api.kaggle_api_extended import KaggleApi

def calculate_md5(filepath, chunk_size=8192):
    """Calculate MD5 hash of a file"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def verify_zip_integrity(zip_path):
    """Check if zip file is valid and complete"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # testzip() returns None if all files are OK
            bad_file = z.testzip()
            if bad_file is not None:
                print(f"      Corrupted file in archive: {bad_file}")
                return False
            return True
    except zipfile.BadZipFile:
        print("      File is not a valid zip")
        return False
    except Exception as e:
        print(f"      Error checking zip: {e}")
        return False

def download_rsna():
    """Download RSNA 2024 dataset using Kaggle Python API"""
    
    data_dir = Path("/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RSNA 2024 LUMBAR SPINE DATASET DOWNLOAD")
    print("="*60)
    print(f"Target: {data_dir}")
    print("="*60)
    sys.stdout.flush()
    
    # Initialize Kaggle API
    print("\n[1/5] Authenticating with Kaggle...")
    sys.stdout.flush()
    
    try:
        api = KaggleApi()
        api.authenticate()
        print("      ✓ Authenticated successfully")
    except Exception as e:
        print(f"      ✗ Authentication failed: {e}")
        return
    
    sys.stdout.flush()
    
    # Verify competition access and get expected file info
    print("\n[2/5] Verifying competition access...")
    sys.stdout.flush()
    
    try:
        response = api.competition_list_files('rsna-2024-lumbar-spine-degenerative-classification')
        files = response.files
        
        # Find the main zip file info
        main_file = None
        for f in files:
            if f.name == 'rsna-2024-lumbar-spine-degenerative-classification.zip':
                main_file = f
                break
        
        if main_file:
            expected_size = main_file.total_bytes
            expected_size_gb = expected_size / (1024**3)
            print(f"      ✓ Competition accessible ({len(files)} files)")
            print(f"      ✓ Expected download size: {expected_size_gb:.1f} GB")
        else:
            # Fallback: sum all files
            expected_size = sum(f.total_bytes for f in files)
            expected_size_gb = expected_size / (1024**3)
            print(f"      ✓ Competition accessible ({len(files)} files)")
            print(f"      ✓ Total size: {expected_size_gb:.1f} GB")
            
    except Exception as e:
        print(f"      ✗ Cannot access competition: {e}")
        print("\n      SOLUTION: Accept rules at:")
        print("      https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification")
        return
    
    sys.stdout.flush()
    
    # Check if file already exists
    zip_file = data_dir / "rsna-2024-lumbar-spine-degenerative-classification.zip"
    
    print("\n[3/5] Checking existing files...")
    sys.stdout.flush()
    
    download_needed = True
    
    if zip_file.exists():
        current_size = zip_file.stat().st_size
        current_size_gb = current_size / (1024**3)
        print(f"      Found existing file: {zip_file.name} ({current_size_gb:.2f} GB)")
        
        # Check size match
        if current_size == expected_size:
            print(f"      File size matches expected ({expected_size_gb:.1f} GB)")
            
            # Verify zip integrity
            print("      Verifying zip integrity (this may take a minute)...")
            sys.stdout.flush()
            
            if verify_zip_integrity(zip_file):
                print("      ✓ File is complete and valid!")
                print("      Skipping download.")
                download_needed = False
            else:
                print("      ✗ File is corrupted, will re-download")
                zip_file.unlink()
        else:
            print(f"      ✗ Size mismatch: {current_size_gb:.2f} GB (expected {expected_size_gb:.1f} GB)")
            print("      Removing incomplete file and re-downloading")
            zip_file.unlink()
    else:
        print("      No existing file found")
    
    sys.stdout.flush()
    
    # Download if needed
    if download_needed:
        print("\n[4/5] Starting download...")
        print(f"      Progress will appear in the error log (.err file)")
        sys.stdout.flush()
        
        try:
            api.competition_download_files(
                competition='rsna-2024-lumbar-spine-degenerative-classification',
                path=str(data_dir),
                quiet=False,
                force=True  # Force re-download to overwrite any partial files
            )
        except Exception as e:
            print(f"\n      ✗ Download failed: {e}")
            return
        
        print("\n" + "="*60)
        print("      ✓ Download complete!")
        print("="*60)
        sys.stdout.flush()
    else:
        print("\n[4/5] Download skipped (file already verified)")
        sys.stdout.flush()
    
    # Extract
    print("\n[5/5] Extracting archive...")
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
