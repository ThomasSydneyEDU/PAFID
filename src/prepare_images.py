"""
prepare_images.py

Utility script to get AI-rendered food images into an experiment-ready state.

Current behaviour
------------------
1. Read `stimuli_master.json` from the `rendered_images/` directory.
2. Use the `"image_file"` field from the JSON records.
3. Verify that every listed image exists in the top-level `rendered_images/` directory.
4. Resize all images in `rendered_images/` that are referenced in the JSON to 384x384 px.
   - This size is a good compromise for online experiments: small enough for
     fast loading but large enough to look clean on typical laptop/desktop
     displays when three images are shown side-by-side.
5. Write:
     * `resized_images/Filtered_Foodpictures_information.csv`
     * `resized_images/Filtered_ImageList.json`
   one level up from `src/`.

Usage
-----
From the repo root (recommended):

    python -m src.prepare_images

Or from inside `src/`:

    python prepare_images.py
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_paths(
    stimuli_dir_arg: Optional[str] = None,
    master_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None,
) -> dict[str, Path]:
  """
  Return key project paths inferred from this file's location, allowing
  optional overrides for the stimuli directory, master file, and output directory.
  """
  this_file = Path(__file__).resolve()
  repo_root = this_file.parents[1]

  if stimuli_dir_arg is not None:
    stim_dir = Path(stimuli_dir_arg).expanduser()
    if not stim_dir.is_absolute():
      stim_dir = (Path.cwd() / stim_dir).resolve()
    else:
      stim_dir = stim_dir.resolve()
  else:
    stim_dir = repo_root / "rendered_images"

  if master_arg is not None:
    json_path = Path(master_arg)
    if not json_path.is_absolute():
      json_path = (repo_root / json_path).resolve()
  else:
    json_path = stim_dir / "stimuli_master.json"

  csv_path = repo_root / "data" / "Foodpictures_information_dynamic.csv"

  if output_dir_arg is not None:
    expt_dir = Path(output_dir_arg).expanduser().resolve()
  else:
    expt_dir = repo_root / "resized_images"

  return {
    "json_path": json_path,
    "csv_path": csv_path,
    "stim_dir": stim_dir,
    "expt_dir": expt_dir,
    "repo_root": repo_root,
  }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")


def verify_images_exist(
  filenames: Sequence[str],
  stim_dir: Path,
) -> list[str]:
  """
  Verify that all filenames exist under `stim_dir`.
  Returns a list of any missing files (relative names).
  """
  missing: list[str] = []
  for name in filenames:
    if not name or str(name) == "nan":
      continue
    path = stim_dir / name
    if not path.is_file():
      missing.append(name)
  return sorted(set(missing))


def resize_image(
  path: Path,
  out_path: Path,
  target_size: int = 384,
) -> None:
  """Resize an image for the experiment without cropping.

  Assumes upstream generation produces square images (e.g. 1024x1024).
  The image is resized directly to (target_size x target_size).
  """
  with Image.open(path) as im:
    im = im.convert("RGB")

    w, h = im.size
    if w != h:
      print(
        f"[WARN] Image is not square ({w}x{h}): {path.name}. "
        "Resizing anyway without cropping."
      )

    if im.size != (target_size, target_size):
      im = im.resize((target_size, target_size), Image.LANCZOS)

    im.save(out_path)


def build_image_list_json(
  filenames: Sequence[str],
  out_path: Path,
) -> None:
  """Write Filtered_ImageList.json — a sorted list of unique image filenames."""
  unique_names = sorted({f for f in filenames if f and str(f) != "nan"})
  out_path.write_text(json.dumps(unique_names, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="Prepare AI-rendered food images for the triplet experiment."
  )
  parser.add_argument(
    "--stimuli-dir",
    type=str,
    default=None,
    help="Stimuli folder containing rendered images and stimuli_master.json. If omitted, auto-detect.",
  )
  parser.add_argument(
    "--master",
    type=str,
    default=None,
    help="Path to stimuli_master.json. If omitted, uses <stimuli-dir>/stimuli_master.json.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for resized images and experiment-ready outputs. "
         "Defaults to resized_images/ inside the PAFID repo. "
         "Use to redirect outputs to an external project.",
  )
  return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
  args = parse_args(sys.argv[1:])

  paths = get_paths(
    stimuli_dir_arg=args.stimuli_dir,
    master_arg=args.master,
    output_dir_arg=args.output_dir,
  )
  json_path: Path = paths["json_path"]
  csv_path: Path = paths["csv_path"]
  stim_dir: Path = paths["stim_dir"]
  expt_dir: Path = paths["expt_dir"]

  expt_img_dir = expt_dir / "images"
  expt_dir.mkdir(parents=True, exist_ok=True)
  expt_img_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] Stimuli folder: {stim_dir}")
  print(f"[INFO] Master JSON:    {json_path}")
  print(f"[INFO] CSV path:       {csv_path}")
  print(f"[INFO] Expt folder:    {expt_dir}")

  if not json_path.is_file():
    sys.exit(f"[ERROR] JSON not found: {json_path}")

  if not stim_dir.is_dir():
    sys.exit(
      f"[ERROR] Stimuli directory not found: {stim_dir}\n"
      "Create it and place your rendered images there."
    )

  # 1. Load JSON
  records = json.loads(json_path.read_text())
  df = pd.DataFrame(records)
  print(f"[INFO] Loaded JSON with {len(df)} records")

  # 2. Use "image_file" column directly
  img_col = "image_file"
  if img_col not in df.columns:
    sys.exit(f"[ERROR] Required image filename field '{img_col}' not found in JSON data.")
  print(f"[INFO] Using image filename field: '{img_col}'")

  filenames = df[img_col].astype(str).tolist()

  # 3. Verify that all images exist
  missing = verify_images_exist(filenames, stim_dir)
  if missing:
    print("[ERROR] The following images listed in the JSON were not found in stimuli/:")
    for name in missing:
      print(f"  - {name}")
    sys.exit(
      f"[ABORT] Found {len(missing)} missing image(s). "
      "Fix the filenames or render the missing items before proceeding."
    )
  else:
    print(f"[INFO] All {len(set(filenames))} referenced images found in stimuli/.")

  # 4. Resize all referenced images to resized_images/images/
  print("[INFO] Resizing images to 384x384 (no crop).")
  for name in sorted(set(filenames)):
    if not name or str(name) == "nan":
      continue
    src = stim_dir / name
    dst = expt_img_dir / name
    resize_image(src, dst, target_size=384)
  print("[INFO] Image resizing complete.")

  # 5. Write outputs — CSVs go alongside the images dir, not buried inside it
  filtered_csv_path = expt_dir.parent / "Filtered_Foodpictures_information.csv"
  filtered_json_path = expt_dir.parent / "Filtered_ImageList.json"

  df.to_csv(filtered_csv_path, index=False)
  build_image_list_json(filenames, filtered_json_path)

  print(f"[INFO] Wrote CSV:  {filtered_csv_path}")
  print(f"[INFO] Wrote JSON: {filtered_json_path}")
  print("[DONE] Stimuli are now in an experiment-ready format.")


if __name__ == "__main__":
  main()
