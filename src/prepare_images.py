"""
prepare_images.py

Utility script to get AI‑rendered food images into an experiment‑ready state.

Current behaviour
------------------
1. Read `stimuli_master.json` from the `rendered_images/` directory.
2. Use the `"image_file"` field from the JSON records.
3. Verify that every listed image exists in the top‑level `rendered_images/` directory.
4. Resize all images in `rendered_images/` that are referenced in the JSON to 384x384 px.
   - This size is a good compromise for online experiments: small enough for
     fast loading but large enough to look clean on typical laptop/desktop
     displays when three images are shown side‑by‑side.
5. Write:
     * `resized_images/Filtered_Foodpictures_information.csv`
     * `resized_images/Filtered_ImageList.json`
   one level up from `src/`.
6. Optionally generate a set of fixed triplets for Phase 1 and write them to
   `fixed_triplets.js` at the project root. These fixed triplets are only
   regenerated when a specific flag is passed.

We can extend this later with further filtering or metadata munging, but this
gets you the basic “expt‑ready” bundle.

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
import random
import textwrap
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence, Any

import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_paths(
    stimuli_dir_arg: Optional[str] = None,
    master_arg: Optional[str] = None,
) -> dict[str, Path]:
  """
  Return key project paths inferred from this file’s location, allowing
  optional overrides for the stimuli directory and master file.
  """
  this_file = Path(__file__).resolve()
  # PAFID root is one level up from src/
  repo_root = this_file.parents[1]

  # Determine stim_dir
  if stimuli_dir_arg is not None:
    stim_dir = Path(stimuli_dir_arg)
    if not stim_dir.is_absolute():
      stim_dir = (repo_root / stim_dir).resolve()
  else:
    # Default to the output directory in the PAFID structure
    stim_dir = repo_root / "rendered_images"

  # Determine master JSON path
  if master_arg is not None:
    json_path = Path(master_arg)
    if not json_path.is_absolute():
      json_path = (repo_root / json_path).resolve()
  else:
    json_path = stim_dir / "stimuli_master.json"

  # Use the dynamic CSV in the data folder
  csv_path = repo_root / "data" / "Foodpictures_information_dynamic.csv"
  # Final processed stimuli go into resized_images
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

  Notes:
  - This does NOT modify the original file; it writes the processed image
    into the experiment images folder.
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
  """
  Write Filtered_ImageList.json.

  For now this is a simple list of image filenames, e.g.:

      ["apple_01.png", "banana_02.png", ...]

  This mirrors the original usage where the experiment code only needed
  filenames; richer metadata stays in the CSV.
  """
  unique_names = sorted({f for f in filenames if f and str(f) != "nan"})
  out_path.write_text(json.dumps(unique_names, indent=2), encoding="utf-8")


def write_fixed_triplets_js(
    filenames: Sequence[str],
    out_dir: Path,
    n_triplets: int = 100,
    seed: Optional[int] = None,
) -> Path:
  """
  Randomly sample a set of fixed triplets from the image filenames and
  write them to `fixed_triplets.js` in the experiment stimuli directory.

  The JS file will define a global `FIXED_TRIPLETS` constant containing
  an array of objects with fields: `id`, `StimA`, `StimB`, `StimC`.

  Parameters
  ----------
  filenames : sequence of str
      List of image filenames (may contain duplicates or NaNs; these
      are cleaned internally).
  out_dir : Path
      Directory where `fixed_triplets.js` will be written (experiment stimuli directory).
  n_triplets : int
      Number of triplets to sample.
  seed : int or None
      Optional random seed for reproducible sampling.
  """
  rng = random.Random(seed)

  # Use unique, non-empty filenames.
  unique_files = sorted({f for f in filenames if f and str(f) != "nan"})
  n_items = len(unique_files)
  if n_items < 3:
    raise ValueError(
      f"Need at least 3 unique images to form triplets, got {n_items}."
    )

  # Sample unique triplets by index, without enumerating all combinations.
  # We cap the number of attempts to avoid pathological infinite loops.
  index_range = list(range(n_items))
  combos: set[tuple[int, int, int]] = set()
  max_attempts = max(n_triplets * 50, n_triplets + 10)
  attempts = 0

  while len(combos) < n_triplets and attempts < max_attempts:
    i, j, k = rng.sample(index_range, 3)
    triple = tuple(sorted((i, j, k)))
    if triple not in combos:
      combos.add(triple)
    attempts += 1

  if len(combos) < n_triplets:
    print(
      f"[WARN] Requested {n_triplets} fixed triplets but only "
      f"constructed {len(combos)} unique combinations."
    )

  # Build records using filenames in the format expected by the experiment.
  # Each record has:
  #   - triplet_id: unique string ID
  #   - images: [StimA, StimB, StimC] (bare filenames; JS code will prefix paths)
  #   - correct_choice: null/None by default (no predefined correct answer)
  combos_sorted = sorted(combos)
  records: list[dict[str, object]] = []
  for idx, (i, j, k) in enumerate(combos_sorted):
    records.append(
      {
        "triplet_id": f"fixed_{idx}",
        "images": [
          unique_files[i],
          unique_files[j],
          unique_files[k],
        ],
        "correct_choice": None,
      }
    )

  js_path = out_dir / "fixed_triplets.js"

  header = textwrap.dedent(
    """\
    // Auto-generated by src/prepare_images.py
    // WARNING: This file is overwritten whenever you run the script with
    //          the --regen-fixed-triplets flag. Do not edit by hand if you
    //          want your changes to persist.

    """
  )

  js_body = json.dumps(records, indent=2)
  js_content = (
    header
    + "export const FIXED_TRIPLETS = "
    + js_body
    + ";\n\n"
    + "export default FIXED_TRIPLETS;\n"
  )

  js_path.write_text(js_content, encoding="utf-8")
  print(f"[INFO] Wrote fixed triplets file: {js_path} "
        f"(n_triplets={len(records)})")
  return js_path


def write_dense_sample_js(
    filenames: Sequence[str],
    out_dir: Path,
    n_items: int = 30,
    seed: Optional[int] = None,
) -> Path:
  """
  Randomly sample a subset of image filenames to be used for dense sampling
  in the experiment, and write them to `dense_sample.js` in the experiment stimuli directory.

  The JS file will define a `DENSE_ITEMS` export containing
  an array of image filename strings.

  Parameters
  ----------
  filenames : sequence of str
      List of image filenames (may contain duplicates or NaNs; these
      are cleaned internally).
  out_dir : Path
      Directory where `dense_sample.js` will be written (experiment stimuli directory).
  n_items : int
      Number of items to sample for dense sampling (default: 30).
  seed : int or None
      Optional random seed for reproducible sampling.
  """
  rng = random.Random(seed)

  # Use unique, non-empty filenames.
  unique_files = sorted({f for f in filenames if f and str(f) != "nan"})
  n_available = len(unique_files)
  if n_available < n_items:
    print(
      f"[WARN] Requested {n_items} dense-sample items but only "
      f"{n_available} unique images are available. Using all available images."
    )
    sample_files = unique_files
  else:
    sample_files = rng.sample(unique_files, n_items)

  js_path = out_dir / "dense_sample.js"

  header = textwrap.dedent(
    """\
    // Auto-generated by src/prepare_images.py
    // WARNING: This file is overwritten whenever you run the script with
    //          the --regen-dense-sample flag. Do not edit by hand if you
    //          want your changes to persist.

    """
  )

  js_body = json.dumps(sample_files, indent=2)
  js_content = (
    header
    + "export const DENSE_ITEMS = "
    + js_body
    + ";\n\n"
    + "export default DENSE_ITEMS;\n"
  )

  js_path.write_text(js_content, encoding="utf-8")
  print(f"[INFO] Wrote dense sample file: {js_path} (n_items={len(sample_files)})")
  return js_path


# ---------------------------------------------------------------------------
# Catch trials (Gemini)
# ---------------------------------------------------------------------------

def _catch_extract_text(resp: Any) -> str:
  """Best-effort extraction across google-genai response shapes."""
  txt = getattr(resp, "text", None)
  if isinstance(txt, str) and txt.strip():
    return txt.strip()

  parsed = getattr(resp, "parsed", None)
  if parsed is not None:
    try:
      return json.dumps(parsed).strip()
    except Exception:
      pass

  cands = getattr(resp, "candidates", None)
  if isinstance(cands, list) and cands:
    content = getattr(cands[0], "content", None)
    parts = getattr(content, "parts", None)
    if isinstance(parts, list):
      chunks: list[str] = []
      for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str) and t:
          chunks.append(t)
      joined = "".join(chunks).strip()
      if joined:
        return joined

  if isinstance(resp, dict):
    for key in ("text", "output_text"):
      v = resp.get(key)
      if isinstance(v, str) and v.strip():
        return v.strip()

  return ""


def _catch_slug_to_label(filename: str) -> str:
  """Convert e.g. 'apple-pear-nashi.png' -> 'apple pear nashi' (prompt-only label)."""
  stem = Path(filename).stem
  stem = stem.replace("_", "-")
  stem = re.sub(r"[-]+", " ", stem).strip()
  return stem


def _catch_write_js(out_path: Path, triplets: list[dict[str, Any]]) -> None:
  header = (
    "// Auto-generated by src/prepare_images.py\n"
    "// WARNING: This file is overwritten when you regenerate catch trials.\n\n"
  )
  body = json.dumps(triplets, indent=2)
  js = (
    header
    + "export const CATCH_TRIALS = "
    + body
    + ";\n\n"
    + "export default CATCH_TRIALS;\n"
  )
  out_path.write_text(js, encoding="utf-8")


def _catch_validate_triplets(triplets: list[dict[str, Any]], available: set[str]) -> tuple[bool, list[str]]:
  """Validate triplets structure and that filenames exist in the available set."""
  errors: list[str] = []
  for i, t in enumerate(triplets):
    imgs = t.get("images")
    odd = t.get("correct_choice")
    if odd is None:
      odd = t.get("odd_index")
    if not isinstance(imgs, list) or len(imgs) != 3 or not all(isinstance(x, str) for x in imgs):
      errors.append(f"triplet[{i}]: 'images' must be list of 3 strings")
      continue
    if len(set(imgs)) != 3:
      errors.append(f"triplet[{i}]: images must be distinct: {imgs}")
    missing = [x for x in imgs if x not in available]
    if missing:
      errors.append(f"triplet[{i}]: unknown filenames (not in list): {missing}")
    if odd not in (0, 1, 2):
      errors.append(f"triplet[{i}]: correct_choice must be 0,1,2 (got {odd})")
  return (len(errors) == 0), errors


def _catch_gemini_generate_triplets(
  model: str,
  filenames: list[str],
  n: int,
  seed: int,
  max_attempts: int = 5,
) -> list[dict[str, Any]]:
  """Ask Gemini to propose N catch trials strictly from the provided filenames."""
  # Lazy import so this script still runs without google-genai unless catch trials are requested.
  try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
  except Exception as e:
    raise RuntimeError(
      "google-genai is required for catch trial generation. Install with: pip install -U google-genai"
    ) from e

  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")

  client = genai.Client(api_key=api_key)
  available_set = set(filenames)
  rng = random.Random(seed)

  system = (
    "You generate easy attention-check 'odd-one-out' triplets for a food image task. "
    "Two items must be very similar (visually and semantically). "
    "The third must be extremely different (different food type/category). "
    "IMPORTANT: You must ONLY use filenames from the provided list. "
    "Return STRICT JSON only (no markdown)."
  )

  schema_hint = {
    "triplets": [
      {
        "triplet_id": "catch_0",
        "images": ["orange.png", "mandarin.png", "beef-steak.png"],
        "correct_choice": 2,
        "pair_reason": "orange and mandarin are both citrus fruits; steak is meat",
        "odd_reason": "steak is a very different category",
      }
    ]
  }

  for attempt in range(1, max_attempts + 1):
    subset_size = min(160, len(filenames))
    subset = rng.sample(filenames, k=subset_size) if len(filenames) > subset_size else list(filenames)
    name_table = [{"file": f, "label": _catch_slug_to_label(f)} for f in subset]

    prompt = {
      "task": "Create catch trials",
      "n": n,
      "rules": [
        "Use only filenames from the provided list.",
        "Each triplet must contain exactly 3 distinct filenames.",
        "Use 'correct_choice' as the index (0/1/2) of the odd-one-out image.",
        "Two items should be nearly the same kind of food (e.g., orange vs mandarin; spinach vs kale; white bread vs wholegrain bread).",
        "The odd item must be VERY different (e.g., meat vs fruit; dessert vs vegetable; seafood vs bread).",
        "Avoid subtle distinctions that could confuse (make these super easy).",
        "Prefer common everyday items when possible.",
      ],
      "available_items": name_table,
      "output_format": schema_hint,
    }

    combined = (
      system
      + "\n\n"
      + "Return JSON ONLY with the exact schema shown in output_format.\n"
      + json.dumps(prompt, ensure_ascii=False)
    )

    resp = client.models.generate_content(
      model=model,
      contents=combined,
      config=types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=2000,
        response_mime_type="application/json",
      ),
    )

    text = _catch_extract_text(resp)
    if not text.strip():
      if attempt == max_attempts:
        raise RuntimeError(
          "Gemini returned empty output (response.text/parsed empty). "
          "This is often caused by structured-output issues or token limits."
        )
      continue

    try:
      obj = json.loads(text)
      if isinstance(obj, list):
        obj = {"triplets": obj}
      triplets = obj.get("triplets")
      if not isinstance(triplets, list):
        raise ValueError("Missing 'triplets' list")

      for t in triplets:
        if isinstance(t, dict):
          if "correct_choice" not in t and "odd_index" in t:
            t["correct_choice"] = t.get("odd_index")

      for idx, t in enumerate(triplets):
        if isinstance(t, dict) and "triplet_id" not in t:
          t["triplet_id"] = f"catch_{idx}"

      ok, errs = _catch_validate_triplets(triplets, available_set)
      if ok and triplets:
        if len(triplets) >= n:
          return triplets[:n]

      # If invalid, retry.
      system += " If your output fails validation, correct it and try again."

    except Exception as e:
      if attempt == max_attempts:
        raise RuntimeError(
          f"Failed to parse/validate Gemini output after {max_attempts} attempts.\n"
          f"Last error: {e}\nLast text:\n{text}"
        )

  raise RuntimeError("Catch-trial generation failed: attempts exhausted")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parse command line arguments for this script."""
  parser = argparse.ArgumentParser(
    description="Prepare AI‑rendered food images for the triplet experiment."
  )
  parser.add_argument(
    "--stimuli-dir",
    type=str,
    default=None,
    help="Stimuli folder containing rendered images and stimuli_master.json (e.g., rendered_images). If omitted, auto-detect.",
  )
  parser.add_argument(
    "--master",
    type=str,
    default=None,
    help="Path to stimuli_master.json. If omitted, uses <stimuli-dir>/stimuli_master.json.",
  )
  parser.add_argument(
    "--require-csv-match",
    action="store_true",
    help="Require that foods in the CSV match foods in the master JSON; abort on mismatch. Default is off.",
  )
  parser.add_argument(
    "--exclude-mismatch",
    action="store_true",
    help="Exclude items where QC label_match is 'mismatch'.",
  )
  parser.add_argument(
    "--exclude-partial",
    action="store_true",
    help="Exclude items where QC label_match is 'partial'.",
  )
  parser.add_argument(
    "--exclude-unclear",
    action="store_true",
    help="Exclude items where QC label_match is 'unclear'.",
  )
  parser.add_argument(
    "--exclude-qc-issue",
    action="append",
    default=[],
    help="Exclude any item that contains this qc issue string (can be provided multiple times).",
  )
  parser.add_argument(
    "--regen-fixed-triplets",
    action="store_true",
    help=(
      "Regenerate fixed_triplets.js with a new random sample of triplets. "
      "WARNING: this will overwrite any existing fixed_triplets.js and "
      "change the fixed trials used in the experiment."
    ),
  )
  parser.add_argument(
    "--fixed-triplets",
    type=int,
    default=100,
    help="Number of fixed triplets to sample if --regen-fixed-triplets is set (default: 100).",
  )
  parser.add_argument(
    "--fixed-seed",
    type=int,
    default=None,
    help="Optional random seed for fixed triplet sampling (for reproducibility).",
  )
  parser.add_argument(
    "--regen-dense-sample",
    action="store_true",
    help=(
      "Regenerate dense_sample.js with a new random set of items used for "
      "dense sampling in the experiment. WARNING: this will overwrite any "
      "existing dense_sample.js and change which items are treated as the "
      "dense-sampled subset."
    ),
  )
  parser.add_argument(
    "--dense-items",
    type=int,
    default=30,
    help="Number of items to sample for dense_sample.js if --regen-dense-sample is set (default: 30).",
  )
  parser.add_argument(
    "--dense-seed",
    type=int,
    default=None,
    help="Optional random seed for dense sample item selection (for reproducibility).",
  )
  parser.add_argument(
    "--regen-catch-trials",
    action="store_true",
    help=(
      "Regenerate catch_trials.js using Gemini. WARNING: this will overwrite any existing catch_trials.js."
    ),
  )
  parser.add_argument(
    "--catch-n",
    type=int,
    default=5,
    help="Number of catch triplets to generate if --regen-catch-trials is set (default: 5).",
  )
  parser.add_argument(
    "--catch-model",
    type=str,
    default="gemini-2.5-pro",
    help="Gemini model to use for catch trial generation (default: gemini-2.5-pro).",
  )
  parser.add_argument(
    "--catch-seed",
    type=int,
    default=20260127,
    help="Random seed used for catch-trial candidate sampling (default: 20260127).",
  )
  return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_qc_issues(value) -> list[str]:
  """Parse qc_issues field to a list of strings (helper for QC filtering)."""
  import json as _json
  import pandas as _pd
  if value is None or (isinstance(value, float) and _pd.isna(value)):
    return []
  if isinstance(value, list):
    return [str(x) for x in value]
  if isinstance(value, str):
    s = value.strip()
    if not s:
      return []
    try:
      parsed = _json.loads(s)
      if isinstance(parsed, list):
        return [str(x) for x in parsed]
    except Exception:
      pass
    return [s]
  return [str(value)]


def main() -> None:
  args = parse_args(sys.argv[1:])

  paths = get_paths(
    stimuli_dir_arg=args.stimuli_dir,
    master_arg=args.master,
  )
  json_path: Path = paths["json_path"]
  csv_path: Path = paths["csv_path"]
  stim_dir: Path = paths["stim_dir"]
  expt_dir: Path = paths["expt_dir"]
  repo_root: Path = paths["repo_root"]

  expt_img_dir = expt_dir / "images"
  # Ensure the experiment output directory exists before creating the images subfolder.
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

  expt_dir.mkdir(parents=True, exist_ok=True)

  # 1. Load JSON
  records = json.loads(json_path.read_text())
  df = pd.DataFrame(records)
  print(f"[INFO] Loaded JSON with {len(df)} records")

  # CSV <-> JSON consistency check (optional)
  if args.require_csv_match:
    if not csv_path.is_file():
      print(f"[WARN] CSV not found at {csv_path}; skipping CSV↔JSON consistency check.")
    else:
      csv_df = pd.read_csv(csv_path)
      if "Food" not in csv_df.columns:
        print("[WARN] 'Food' column not found in CSV; skipping CSV↔JSON consistency check.")
      else:
        csv_foods = set(csv_df["Food"].astype(str).str.strip().str.lower())
        if "food" not in df.columns:
          print("[WARN] 'food' field not found in JSON; skipping CSV↔JSON consistency check.")
        else:
          json_foods = set(df["food"].astype(str).str.strip().str.lower())
          missing_in_json = sorted(csv_foods - json_foods)
          extra_in_json = sorted(json_foods - csv_foods)
          if missing_in_json:
            print("[ERROR] The following foods are listed in the CSV but have no entry in stimuli_master.json (likely not rendered yet):")
            for item in missing_in_json:
              print(f"  - {item}")
            sys.exit("[ABORT] CSV↔JSON mismatch: some CSV foods have no JSON record. Re‑run rendering or fix stimuli_master.json before proceeding.")
          elif extra_in_json:
            print("[WARN] The following foods appear in stimuli_master.json but not in the CSV (extra or outdated entries):")
            for item in extra_in_json:
              print(f"  - {item}")
            # Do NOT drop JSON records anymore, just warn.
          else:
            print("[INFO] CSV and JSON are consistent at the food name level.")
  else:
    print("[INFO] CSV↔JSON consistency check disabled (use --require-csv-match to enable).")

  # QC-based filtering (optional)
  before = len(df)
  filters_applied = False
  mask = pd.Series([True] * len(df))
  # label_match filtering
  if any([args.exclude_mismatch, args.exclude_partial, args.exclude_unclear]):
    if "label_match" in df.columns:
      col = df["label_match"].astype(str).str.strip().str.lower()
      if args.exclude_mismatch:
        mask = mask & (col != "mismatch")
        filters_applied = True
      if args.exclude_partial:
        mask = mask & (col != "partial")
        filters_applied = True
      if args.exclude_unclear:
        mask = mask & (col != "unclear")
        filters_applied = True
  # qc_issues filtering
  if args.exclude_qc_issue and "qc_issues" in df.columns:
    # For each row, parse qc_issues and exclude if any issue matches
    search_terms = [s.lower() for s in args.exclude_qc_issue]
    def qc_issue_row_exclude(val):
      issues = parse_qc_issues(val)
      issues_lc = [str(x).lower() for x in issues]
      for term in search_terms:
        if any(term in issue for issue in issues_lc):
          return False
      return True
    mask = mask & df["qc_issues"].apply(qc_issue_row_exclude)
    filters_applied = True
  if filters_applied:
    kept = int(mask.sum())
    dropped = int(before - kept)
    print(f"[INFO] QC filtering: kept {kept}/{before} (dropped {dropped})")
    df = df.loc[mask].reset_index(drop=True)

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

  # 5. Write resized_images outputs
  filtered_csv_path = expt_dir / "Filtered_Foodpictures_information.csv"
  filtered_json_path = expt_dir / "Filtered_ImageList.json"

  df.to_csv(filtered_csv_path, index=False)
  build_image_list_json(filenames, filtered_json_path)

  print(f"[INFO] Wrote CSV:  {filtered_csv_path}")
  print(f"[INFO] Wrote JSON: {filtered_json_path}")

  # 6. Optionally (re)generate fixed triplets for Phase 1
  fixed_js_path = expt_dir / "fixed_triplets.js"
  if args.regen_fixed_triplets:
    print(
      "[WARN] Regenerating fixed_triplets.js with a NEW random set of fixed "
      "trials. This will change which triplets are treated as fixed anchors "
      "in the experiment."
    )
    write_fixed_triplets_js(
      filenames,
      out_dir=expt_dir,
      n_triplets=args.fixed_triplets,
      seed=args.fixed_seed,
    )
  else:
    if fixed_js_path.is_file():
      print(
        f"[INFO] Existing fixed_triplets.js detected in resized_images/: {fixed_js_path}. "
        "Not modifying it. Run with --regen-fixed-triplets if you intend to "
        "change the fixed trials."
      )
    else:
      print(
        "[INFO] No fixed_triplets.js found in the experiment stimuli folder. "
        "Run this script with --regen-fixed-triplets to create a first set "
        "of fixed trials for the experiment."
      )

  dense_js_path = expt_dir / "dense_sample.js"
  if args.regen_dense_sample:
    print(
      "[WARN] Regenerating dense_sample.js with a NEW random set of items "
      "for dense sampling. This will change which items are part of the "
      "dense-sampled subset in the experiment."
    )
    write_dense_sample_js(
      filenames,
      out_dir=expt_dir,
      n_items=args.dense_items,
      seed=args.dense_seed,
    )
  else:
    if dense_js_path.is_file():
      print(
        f"[INFO] Existing dense_sample.js detected in resized_images/: {dense_js_path}. "
        "Not modifying it. Run with --regen-dense-sample if you intend to "
        "change the dense-sampled item subset."
      )
    else:
      print(
        "[INFO] No dense_sample.js found in the experiment stimuli folder. "
        "Run this script with --regen-dense-sample to create the initial "
        "dense-sampled item subset for the experiment."
      )

  # 6b. Optionally (re)generate catch trials (LLM-generated)
  catch_js_path = expt_dir / "catch_trials.js"
  if args.regen_catch_trials:
    print(
      "[WARN] Regenerating catch_trials.js using Gemini. "
      "This will overwrite any existing catch_trials.js."
    )
    triplets = _catch_gemini_generate_triplets(
      model=args.catch_model,
      filenames=sorted(set(filenames)),
      n=args.catch_n,
      seed=args.catch_seed,
    )
    _catch_write_js(catch_js_path, triplets)

    catch_meta = {
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "model": args.catch_model,
      "n": args.catch_n,
      "seed": args.catch_seed,
      "output": str(catch_js_path),
    }
    catch_meta_path = catch_js_path.with_suffix(".metadata.json")
    catch_meta_path.write_text(json.dumps(catch_meta, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote catch trials: {catch_js_path}")
    print(f"[INFO] Wrote catch metadata: {catch_meta_path}")
  else:
    if catch_js_path.is_file():
      print(
        f"[INFO] Existing catch_trials.js detected in resized_images/: {catch_js_path}. "
        "Not modifying it. Run with --regen-catch-trials if you intend to regenerate catch trials."
      )
    else:
      print(
        "[INFO] No catch_trials.js found in the experiment stimuli folder. "
        "Run this script with --regen-catch-trials to create a first set of catch trials."
      )

  # 7. Write sampling metadata for reproducibility
  # Script provenance (for reproducibility / maintenance tracking)
  script_path = Path(__file__).resolve()
  try:
      script_rel = script_path.relative_to(repo_root)
  except ValueError:
      # Fallback if the script is not within this repo for some reason
      script_rel = script_path.name

  script_info = {
      "absolute_path": str(script_path),
      "relative_to_project_root": str(script_rel),
      "module_name": __name__,
  }

  metadata = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "script": script_info,
    "fixed_triplets": {
      "regen": bool(args.regen_fixed_triplets),
      "exists": fixed_js_path.is_file(),
      "n_triplets": args.fixed_triplets if args.regen_fixed_triplets else None,
      "seed": args.fixed_seed if args.regen_fixed_triplets else None,
    },
    "dense_sample": {
      "regen": bool(args.regen_dense_sample),
      "exists": dense_js_path.is_file(),
      "n_items": args.dense_items if args.regen_dense_sample else None,
      "seed": args.dense_seed if args.regen_dense_sample else None,
    },
    "catch_trials": {
      "regen": bool(args.regen_catch_trials),
      "exists": catch_js_path.is_file(),
      "n": args.catch_n if args.regen_catch_trials else None,
      "seed": args.catch_seed if args.regen_catch_trials else None,
      "model": args.catch_model if args.regen_catch_trials else None,
    },
  }

  metadata_path = expt_dir / "sampling_metadata.json"
  metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(f"[INFO] Wrote sampling metadata: {metadata_path}")

  print("[DONE] Stimuli are now in an experiment‑ready format.")


if __name__ == "__main__":
  main()
