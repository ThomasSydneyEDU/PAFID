#!/usr/bin/env python3
"""
Iterative editorial review for the PAFID 350-item baseline.

Reads data/food_category_flags_to_review.csv, which must contain an `action`
column (and optionally `confirmed_value` and `generation_notes` columns).

Supported action values
-----------------------
regenerate      Re-generate the image for this food item with any extra notes
                appended to the generation prompt, then re-run QC, blind
                ratings, and visual-feature extraction for that item only.
correct_labels  Preview the label correction that will be committed by
                apply_corrections.py. No files are written by this script.
accept          This item has been reviewed and no change is needed. Skip.
(empty / NaN)   Item not yet reviewed. Print a summary and skip.

Workflow
--------
1. Authors fill in the `action` column (and `confirmed_value` /
   `generation_notes` as required) in food_category_flags_to_review.csv.
2. Run this script as many times as needed until all items are actioned.
3. Once satisfied, run apply_corrections.py once to commit label corrections.
4. Manually copy the dynamic CSV over the reference CSV to freeze the baseline.

Usage
-----
  python src/run_editorial_review.py
  python src/run_editorial_review.py --dry-run   # preview without API calls
  python src/run_editorial_review.py --food "Fruit leather"  # single item
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FLAGS_CSV = ROOT / "data" / "QC" / "food_category_flags_to_review.csv"
CORRECTIONS_CSV = ROOT / "data" / "QC" / "category_corrections.csv"

# Columns that correct_labels actions are permitted to modify.
LABEL_COLUMNS = {
    "Category_WHO_10",
    "Category_Intuitive_7",
    "Category_Culinary_9",
    "Category_NOVA_4",
    "natural_vs_transformed",
    "Natural_vs_transformed",
    "Transformation_score",
}

# Columns required in the flags CSV (added by authors during review).
REQUIRED_COLS = {"food", "action"}


def load_flags(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Review file not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(
            f"[ERROR] food_category_flags_to_review.csv is missing required columns: {missing}\n"
            "        Add an 'action' column (values: regenerate / correct_labels / accept)",
            file=sys.stderr,
        )
        sys.exit(1)
    return df


def run_generate(food: str, generation_notes: str | None, dry_run: bool) -> None:
    """Re-generate image for one food item, then re-run downstream scripts."""
    cmd_gen = [
        sys.executable, str(ROOT / "src" / "generate_images.py"),
        "--food", food,
        "--overwrite",
    ]
    if generation_notes:
        cmd_gen += ["--extra-prompt", generation_notes]
    if dry_run:
        print(f"  [DRY] Would run: {' '.join(cmd_gen)}")
    else:
        print(f"  Running: {' '.join(cmd_gen)}")
        result = subprocess.run(cmd_gen, cwd=ROOT)
        if result.returncode != 0:
            print(f"  [WARN] generate_images.py exited with code {result.returncode}")
            return

    # Re-run QC for this item only.
    cmd_qc = [
        sys.executable, str(ROOT / "src" / "run_qc.py"),
        "--food", food, "--overwrite",
    ]
    if dry_run:
        print(f"  [DRY] Would run: {' '.join(cmd_qc)}")
    else:
        print(f"  Running: {' '.join(cmd_qc)}")
        subprocess.run(cmd_qc, cwd=ROOT)

    # Re-run blind ratings for this item.
    cmd_rate = [
        sys.executable, str(ROOT / "src" / "rate_images.py"),
        "--food", food, "--overwrite",
    ]
    if dry_run:
        print(f"  [DRY] Would run: {' '.join(cmd_rate)}")
    else:
        print(f"  Running: {' '.join(cmd_rate)}")
        subprocess.run(cmd_rate, cwd=ROOT)

    # Re-run visual feature extraction, merging back into canonical output.
    cmd_feat = [
        sys.executable, str(ROOT / "src" / "extract_visual_features.py"),
        "--food", food, "--merge-canonical",
    ]
    if dry_run:
        print(f"  [DRY] Would run: {' '.join(cmd_feat)}")
    else:
        print(f"  Running: {' '.join(cmd_feat)}")
        subprocess.run(cmd_feat, cwd=ROOT)


def preview_label_correction(row: pd.Series) -> None:
    """Print what label correction would be committed by apply_corrections.py."""
    food = str(row["food"])
    columns_to_review = str(row.get("columns_to_review", "")) if pd.notna(row.get("columns_to_review")) else ""
    confirmed_value = str(row.get("confirmed_value", "")) if pd.notna(row.get("confirmed_value")) else ""
    rationale = str(row.get("rationale", "")) if pd.notna(row.get("rationale")) else ""
    suggested_direction = str(row.get("suggested_direction", "")) if pd.notna(row.get("suggested_direction")) else ""

    print(f"  Food:             {food}")
    print(f"  Column(s):        {columns_to_review}")
    print(f"  Suggested change: {suggested_direction}")
    print(f"  Confirmed value:  {confirmed_value if confirmed_value else '(not yet set)'}")
    print(f"  Rationale:        {rationale}")
    if not confirmed_value:
        print("  [NOTE] Set 'confirmed_value' in the CSV before running apply_corrections.py.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Iterative editorial review for PAFID flagged items."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview actions without making any API calls or file changes."
    )
    parser.add_argument(
        "--food", type=str, default=None,
        help="Process a single food item only (exact match, case-insensitive)."
    )
    args = parser.parse_args(argv)

    df = load_flags(FLAGS_CSV)

    # Filter to a single food if requested.
    if args.food:
        mask = df["food"].str.strip().str.lower() == args.food.strip().lower()
        df = df[mask]
        if df.empty:
            print(f"[ERROR] '{args.food}' not found in {FLAGS_CSV.name}", file=sys.stderr)
            sys.exit(1)

    counts = {"regenerate": 0, "correct_labels": 0, "accept": 0, "pending": 0}

    for _, row in df.iterrows():
        food = str(row["food"]).strip()
        action_raw = row.get("action", "")
        action = str(action_raw).strip().lower() if pd.notna(action_raw) else ""

        print(f"\n{'─' * 60}")
        print(f"Food: {food}  |  Action: {action or '(not set)'}")

        if action == "accept":
            print("  → Accepted. No changes needed.")
            counts["accept"] += 1

        elif action == "regenerate":
            generation_notes = None
            if "generation_notes" in row.index and pd.notna(row.get("generation_notes")):
                generation_notes = str(row["generation_notes"]).strip() or None
            print(f"  → Regenerating image{' with notes: ' + generation_notes if generation_notes else ''}.")
            run_generate(food, generation_notes, args.dry_run)
            counts["regenerate"] += 1

        elif action == "correct_labels":
            print("  → Label correction preview:")
            preview_label_correction(row)
            print("  [INFO] Run apply_corrections.py when all corrections are confirmed.")
            counts["correct_labels"] += 1

        else:
            severity = row.get("severity", "")
            flag_type = row.get("flag_type", "")
            suggested = row.get("suggested_direction", "")
            print(f"  Severity: {severity}  |  Flag type: {flag_type}")
            print(f"  Suggestion: {suggested}")
            print("  → No action set. Add 'regenerate', 'correct_labels', or 'accept' to the CSV.")
            counts["pending"] += 1

    print(f"\n{'═' * 60}")
    print("Summary:")
    print(f"  {counts['regenerate']:3d}  regenerated")
    print(f"  {counts['correct_labels']:3d}  label corrections previewed")
    print(f"  {counts['accept']:3d}  accepted (no change)")
    print(f"  {counts['pending']:3d}  pending (no action set)")

    if counts["correct_labels"] > 0:
        print(
            "\nNext step: verify confirmed_value entries in food_category_flags_to_review.csv,\n"
            "then run:  python src/apply_corrections.py"
        )
    if counts["pending"] > 0:
        print(
            f"\n{counts['pending']} item(s) still need an action. "
            "Edit food_category_flags_to_review.csv and re-run this script."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
