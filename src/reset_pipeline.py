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
import json
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
    print(f"Restoring {DYN_CSV.name} from {REF_CSV.name}...")
    bak = DYN_CSV.with_suffix(".csv.bak")
    if DYN_CSV.exists():
        shutil.copy2(DYN_CSV, bak)
        print(f"  Backup saved: {bak.name}")
    shutil.copy2(REF_CSV, DYN_CSV)

def get_canonical_filenames():
    df = pd.read_csv(REF_CSV)
    return set(df['filename'].tolist())

def prune_rendered_images(safe_list):
    print("Pruning non-canonical files and metadata in rendered_images/...")
    count = 0
    # Core files to always keep
    core_files = {"stimuli_master.json"}
    
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
    
    # Also prune the stimuli_master.json itself
    master_path = RENDERED_DIR / "stimuli_master.json"
    if master_path.exists():
        try:
            bak = master_path.with_suffix(".json.bak")
            shutil.copy2(master_path, bak)
            print(f"  Backup saved: {bak.name}")
            with master_path.open("r") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                original_len = len(data)
                # Keep only entries that are in the safe list
                data = [entry for entry in data if entry.get("image_file") in safe_list]
                
                with master_path.open("w") as f:
                    json.dump(data, f, indent=2)
                
                pruned_json = original_len - len(data)
                if pruned_json > 0:
                    print(f"Pruned {pruned_json} non-canonical entries from stimuli_master.json.")
        except Exception as e:
            print(f"Warning: Could not prune stimuli_master.json: {e}")

    print(f"Removed {count} extra items from rendered_images/ directory.")

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
        print("\nReset complete. The database has been restored to the canonical 350-item baseline.")
    except Exception as e:
        print(f"Error during reset: {e}")

if __name__ == "__main__":
    main()
