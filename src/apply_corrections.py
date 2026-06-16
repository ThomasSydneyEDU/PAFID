#!/usr/bin/env python3
"""
Apply manual category corrections to stimuli_master.json and
Foodpictures_information_dynamic.csv.

Reads:  data/category_corrections.csv
Writes: rendered_images/stimuli_master.json (in-place, with .bak backup)
        data/Foodpictures_information_dynamic.csv (in-place, with .bak backup)

category_corrections.csv format (one row per correction):
  food             - exact food name (matched case-insensitively)
  column           - column to correct (e.g. Category_WHO_10, Category_Intuitive_7)
  corrected_value  - the new value to write
  reason           - brief justification
  reviewed_by      - who approved the correction
  date             - date of correction (YYYY-MM-DD)

Usage:
  python src/apply_corrections.py
  python src/apply_corrections.py --dry-run     # preview changes without writing
  python src/apply_corrections.py --corrections data/my_corrections.csv
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECTIONS = ROOT / "data" / "category_corrections.csv"
MASTER_PATH = ROOT / "rendered_images" / "stimuli_master.json"
DYNAMIC_CSV = ROOT / "data" / "Foodpictures_information_dynamic.csv"

# The reference CSV is the frozen canonical baseline. This script must never
# write to it — it is set once manually by the authors after the initial
# correction run and is thereafter read-only.
REFERENCE_CSV = ROOT / "data" / "Foodpictures_information_reference.csv"

# Columns that corrections are permitted to touch.
# Extend this list if new label columns are added to the pipeline.
ALLOWED_COLUMNS = {
    "Category_WHO_10",
    "Category_Intuitive_7",
    "Category_Culinary_9",
    "Category_NOVA_4",
    "natural_vs_transformed",
    "Natural_vs_transformed",
    "Transformation_score",
}


def norm(s: str) -> str:
    """Normalise a food name for case-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def load_corrections(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Corrections file not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    required = {"food", "column", "corrected_value"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Corrections CSV missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df.dropna(subset=["food", "column", "corrected_value"])
    return df


def apply_to_master(corrections: pd.DataFrame, dry_run: bool) -> int:
    if not MASTER_PATH.exists():
        print(f"[WARN] stimuli_master.json not found at {MASTER_PATH} — skipping.")
        return 0

    with MASTER_PATH.open("r", encoding="utf-8") as f:
        master = json.load(f)

    index = {norm(e.get("food", "")): i for i, e in enumerate(master)}
    applied = 0

    for _, row in corrections.iterrows():
        food_key = norm(row["food"])
        col = str(row["column"]).strip()
        val = row["corrected_value"]

        if col not in ALLOWED_COLUMNS:
            print(f"[WARN] Column '{col}' not in allowed list — skipping correction for '{row['food']}'.")
            continue

        if food_key not in index:
            print(f"[WARN] '{row['food']}' not found in stimuli_master.json — skipping.")
            continue

        entry = master[index[food_key]]
        old = entry.get(col, "<missing>")

        # Normalise NOVA to int if needed
        if col == "Category_NOVA_4":
            try:
                val = int(val)
            except (ValueError, TypeError):
                pass

        if dry_run:
            print(f"[DRY] stimuli_master: {row['food']} | {col}: {old!r} → {val!r}")
        else:
            entry[col] = val
            entry.setdefault("manual_corrections", {})[col] = {
                "previous_value": old,
                "corrected_value": val,
                "reason": str(row.get("reason", "")),
                "reviewed_by": str(row.get("reviewed_by", "")),
                "date": str(row.get("date", "")),
            }
            print(f"[OK] stimuli_master: {row['food']} | {col}: {old!r} → {val!r}")
            applied += 1

    if not dry_run and applied:
        bak = MASTER_PATH.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy2(MASTER_PATH, bak)
            print(f"[INFO] Backup: {bak.name}")
        with MASTER_PATH.open("w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        print(f"[INFO] stimuli_master.json updated ({applied} correction(s)).")

    return applied


def apply_to_dynamic_csv(corrections: pd.DataFrame, dry_run: bool) -> int:
    if not DYNAMIC_CSV.exists():
        print(f"[WARN] Dynamic CSV not found at {DYNAMIC_CSV} — skipping.")
        return 0

    df = pd.read_csv(DYNAMIC_CSV, encoding="utf-8-sig")

    # Build a normalised food-name → index mapping.
    # The dynamic CSV uses a 'food' column (lowercase).
    key_col = "food" if "food" in df.columns else "Food"
    index = {norm(str(v)): i for i, v in df[key_col].items()}

    applied = 0

    for _, row in corrections.iterrows():
        food_key = norm(row["food"])
        col = str(row["column"]).strip()
        val = row["corrected_value"]

        # Map Natural_vs_transformed → natural_vs_transformed for the CSV
        csv_col = col
        if col == "Natural_vs_transformed":
            csv_col = "natural_vs_transformed"

        if csv_col not in ALLOWED_COLUMNS and col not in ALLOWED_COLUMNS:
            print(f"[WARN] Column '{col}' not in allowed list — skipping for dynamic CSV.")
            continue

        if food_key not in index:
            print(f"[WARN] '{row['food']}' not found in dynamic CSV — skipping.")
            continue

        idx = index[food_key]

        # Use the CSV column name if it exists; fall back to original col name
        target_col = csv_col if csv_col in df.columns else (col if col in df.columns else None)
        if target_col is None:
            print(f"[WARN] Column '{col}' not present in dynamic CSV — skipping for '{row['food']}'.")
            continue

        old = df.at[idx, target_col]

        if dry_run:
            print(f"[DRY] dynamic CSV:    {row['food']} | {target_col}: {old!r} → {val!r}")
        else:
            df.at[idx, target_col] = val
            print(f"[OK] dynamic CSV:    {row['food']} | {target_col}: {old!r} → {val!r}")
            applied += 1

    if not dry_run and applied:
        bak = DYNAMIC_CSV.with_suffix(".csv.bak")
        if not bak.exists():
            shutil.copy2(DYNAMIC_CSV, bak)
            print(f"[INFO] Backup: {bak.name}")
        df.to_csv(DYNAMIC_CSV, index=False, encoding="utf-8-sig")
        print(f"[INFO] Foodpictures_information_dynamic.csv updated ({applied} correction(s)).")

    return applied


def main():
    parser = argparse.ArgumentParser(description="Apply manual category corrections to the PAFID database.")
    parser.add_argument("--corrections", type=str, default=str(DEFAULT_CORRECTIONS),
                        help=f"Path to corrections CSV (default: {DEFAULT_CORRECTIONS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing anything.")
    args = parser.parse_args()

    corrections_path = Path(args.corrections).resolve()

    # Hard guard: refuse to write to the reference CSV under any circumstances.
    if corrections_path == REFERENCE_CSV.resolve():
        print(
            "[ERROR] apply_corrections.py must never be pointed at "
            "Foodpictures_information_reference.csv.\n"
            "        That file is a frozen canonical baseline. "
            "Edit category_corrections.csv instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    corrections = load_corrections(corrections_path)

    if corrections.empty:
        print("[INFO] No corrections to apply.")
        return 0

    print(f"[INFO] Loaded {len(corrections)} correction(s) from {args.corrections}")
    if args.dry_run:
        print("[INFO] DRY RUN — no files will be modified.\n")

    n_master = apply_to_master(corrections, args.dry_run)
    n_csv = apply_to_dynamic_csv(corrections, args.dry_run)

    if args.dry_run:
        print(f"\n[DRY] Would apply {len(corrections)} correction(s) to stimuli_master.json and dynamic CSV.")
    else:
        print(f"\n[DONE] Applied {n_master} correction(s) to stimuli_master.json, "
              f"{n_csv} to dynamic CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
