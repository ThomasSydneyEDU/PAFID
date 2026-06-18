#!/usr/bin/env python3
"""
PAFID Reset Pipeline Utility

Two modes:

  Full reset (default):
    Restores the database to the canonical 350-item baseline.
    - Overwrites Foodpictures_information_dynamic.csv with the reference version.
    - Deletes non-canonical images and metadata from rendered_images/.
    - Clears the resized_images/ cache.

  Source-aware reset (--stimulus-set <label>):
    Removes only items tagged with a specific stimulus_set label.
    - Leaves the canonical 350 and all other extension sets untouched.
    - Useful for undoing an extension run without touching the baseline.

Usage:
  python src/reset_pipeline.py
  python src/reset_pipeline.py --stimulus-set foodspace_extension_2026
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description="Reset PAFID database to canonical baseline.")
    p.add_argument(
        "--stimulus-set", type=str, default=None,
        help="Remove only items with this stimulus_set label (e.g. 'foodspace_extension_2026'). "
             "Leaves the canonical 350 and any other extension sets untouched. "
             "If omitted, resets the full database to the 350-item canonical baseline."
    )
    return p.parse_args()


def confirm_reset(stimulus_set=None):
    if stimulus_set:
        print(f"WARNING: This will remove all items with stimulus_set='{stimulus_set}' from the database.")
        print("The canonical 350-item baseline and other extension sets will be preserved.")
    else:
        print("WARNING: This will permanently delete all non-canonical generated images and reset your progress.")
        print("The 350 baseline images and metadata will be preserved.")
    response = input("Are you sure you want to proceed? (y/N): ").strip().lower()
    return response == 'y'


# ── Full reset helpers ────────────────────────────────────────────────────────

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
    core_files = {"stimuli_master.json"}

    for item in RENDERED_DIR.iterdir():
        if item.name in core_files or item.name.startswith('.'):
            continue
        name_stem = item.stem
        png_name = f"{name_stem}.png"
        if png_name not in safe_list:
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1

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
            for img in item.iterdir():
                if img.is_file():
                    img.unlink()
                    count += 1
    print(f"Cleared {count} files from resized_images/.")


# ── Source-aware reset helpers ────────────────────────────────────────────────

def remove_by_stimulus_set(stimulus_set: str):
    """Remove all items tagged with the given stimulus_set from master JSON and dynamic CSV."""
    master_path = RENDERED_DIR / "stimuli_master.json"
    removed_files = set()

    # 1. Prune stimuli_master.json
    if master_path.exists():
        bak = master_path.with_suffix(".json.bak")
        shutil.copy2(master_path, bak)
        print(f"  Backup saved: {bak.name}")
        with master_path.open("r") as f:
            data = json.load(f)
        if isinstance(data, list):
            original_len = len(data)
            keep = [e for e in data if e.get("stimulus_set") != stimulus_set]
            removed = [e for e in data if e.get("stimulus_set") == stimulus_set]
            removed_files = {e["image_file"] for e in removed if e.get("image_file")}
            with master_path.open("w") as f:
                json.dump(keep, f, indent=2)
            print(f"  Removed {original_len - len(keep)} entries from stimuli_master.json.")

    # 2. Delete image files for removed entries
    deleted = 0
    for fname in removed_files:
        for p in [RENDERED_DIR / fname, RENDERED_DIR / (Path(fname).stem + ".json")]:
            if p.exists():
                p.unlink()
                deleted += 1
    print(f"  Deleted {deleted} files from rendered_images/.")

    # 3. Prune dynamic CSV
    if DYN_CSV.exists():
        bak = DYN_CSV.with_suffix(".csv.bak")
        shutil.copy2(DYN_CSV, bak)
        print(f"  Backup saved: {bak.name}")
        df = pd.read_csv(DYN_CSV, encoding="utf-8-sig")
        if "stimulus_set" in df.columns:
            original_len = len(df)
            df = df[df["stimulus_set"] != stimulus_set]
            df.to_csv(DYN_CSV, index=False, encoding="utf-8-sig")
            print(f"  Removed {original_len - len(df)} rows from {DYN_CSV.name}.")
        else:
            print(f"  [WARN] 'stimulus_set' column not found in {DYN_CSV.name} — CSV unchanged.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not confirm_reset(args.stimulus_set):
        print("Reset cancelled.")
        return

    try:
        if args.stimulus_set:
            print(f"\nRemoving items with stimulus_set='{args.stimulus_set}'...")
            remove_by_stimulus_set(args.stimulus_set)
            print(f"\nDone. Items with stimulus_set='{args.stimulus_set}' have been removed.")
        else:
            reset_csv()
            safe_list = get_canonical_filenames()
            prune_rendered_images(safe_list)
            clear_resized_cache()
            print("\nReset complete. The database has been restored to the canonical 350-item baseline.")
    except Exception as e:
        print(f"Error during reset: {e}")


if __name__ == "__main__":
    main()
