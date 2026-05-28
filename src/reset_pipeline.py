#!/usr/bin/env python3
"""
PAFID Reset Pipeline Utility

This script restores the database to its canonical 350-item baseline.
- Overwrites Foodpictures_information_dynamic.csv with the reference version.
- Deletes non-canonical images and metadata from rendered_images/.
- Clears the resized_images/ cache.

Usage:
  python src/reset_pipeline.py
"""

import os
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RENDERED_DIR = ROOT / "rendered_images"
RESIZED_DIR = ROOT / "resized_images"

REF_CSV = DATA_DIR / "Foodpictures_information_reference.csv"
DYN_CSV = DATA_DIR / "Foodpictures_information_dynamic.csv"

def confirm_reset():
    print("WARNING: This will permanently delete all non-canonical generated images and reset your progress.")
    print("The 350 baseline images and metadata will be preserved.")
    response = input("Are you sure you want to proceed? (y/N): ").strip().lower()
    return response == 'y'

def reset_csv():
    print(f"Reseting {DYN_CSV.name} from {REF_CSV.name}...")
    shutil.copy2(REF_CSV, DYN_CSV)

def get_canonical_filenames():
    df = pd.read_csv(REF_CSV)
    return set(df['filename'].tolist())

def prune_rendered_images(safe_list):
    print("Pruning non-canonical files in rendered_images/...")
    count = 0
    # Core files to always keep
    core_files = {"stimuli_master.json", "original master json"}
    
    for item in RENDERED_DIR.iterdir():
        if item.name in core_files or item.name.startswith('.'):
            continue
            
        # Get filename (handle both .png and .json metadata)
        name_stem = item.stem
        png_name = f"{name_stem}.png"
        
        if png_name not in safe_list:
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1
    print(f"Removed {count} extra items from rendered_images/.")

def clear_resized_cache():
    print("Clearing resized_images/ cache...")
    count = 0
    for item in RESIZED_DIR.iterdir():
        if item.name.startswith('.'):
            continue
            
        if item.is_file():
            item.unlink()
            count += 1
        elif item.is_dir() and item.name == "images":
            # Clear all images in the images subdirectory
            for img in item.iterdir():
                if img.is_file():
                    img.unlink()
                    count += 1
    print(f"Cleared {count} files from resized_images/.")

def main():
    if not confirm_reset():
        print("Reset cancelled.")
        return

    try:
        reset_csv()
        safe_list = get_canonical_filenames()
        prune_rendered_images(safe_list)
        clear_resized_cache()
        print("\nReset complete! Your PAFID database is back to its factory state.")
    except Exception as e:
        print(f"Error during reset: {e}")

if __name__ == "__main__":
    main()
