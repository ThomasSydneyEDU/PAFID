#!/usr/bin/env python3
"""
Caption + QC plated stimuli with Gemini.

Reads:  <stimuli_dir>/stimuli_master.json
Loads:  <stimuli_dir>/<image_file> for each entry
Writes: captions + QC fields back into stimuli_master.json (in-place), with a .bak backup
Outputs: QC issues summary to stdout AND writes <stimuli_dir>/qc_issues.json

Additionally:
- Writes data/Foodpictures_information_dynamic.csv (merged from the input list CSV + AI QC/judgements).

Requires:
  pip install -U google-genai
  export GEMINI_API_KEY=...

Example:
  python src/run_qc.py --stimuli-dir rendered_images --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------- Optional CSV export (paths) ----------------

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DEFAULT_INPUT_LIST_CSV = ROOT / "data" / "food_list_initial_seed.csv"
QC_PLUS_AI_CSV = ROOT / "data" / "Foodpictures_information_dynamic.csv"


# ---------------- Gemini client ----------------

def get_gemini_client():
    try:
        from google import genai  # type: ignore
    except Exception:
        print("[ERROR] Missing google-genai. Install with: pip install -U google-genai", file=sys.stderr)
        raise

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"true", "1", "yes"}

    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. "
                "Export it first, e.g., 'export GOOGLE_CLOUD_PROJECT=usyd-llm'"
            )
        print(f"[INFO] Using Gemini via Vertex AI: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it first, e.g. 'export GEMINI_API_KEY=...'.")
    return genai.Client(api_key=api_key)


def backoff_sleep(attempt: int) -> None:
    # exponential backoff with mild jitter
    delay = min(60, (2 ** attempt)) + 0.25 * attempt
    time.sleep(delay)


# ---------------- Prompts ----------------

SYSTEM_INSTRUCTIONS = (
    "You are doing neutral, visual quality control for experimental food stimuli. "
    "Be factual. Do not mention brands. Do not mention the prompt. Do not add opinions."
)


def build_qc_prompt(
    expected_food: str,
    expected_base_food: str,
    expected_category: Optional[str],
) -> str:
    exp_cat = expected_category or "unknown"

    return f"""You will be shown an image of food on a plate.

EXPECTED LABELS (from dataset):
- expected_food: "{expected_food}"
- expected_base_food: "{expected_base_food}"
- expected_category: "{exp_cat}"

Tasks:
1) Write a brief neutral caption (1 sentence, <= 20 words) describing what is visible.
2) Identify the observed food.
3) Compare observed vs expected and rate label match.
4) Flag obvious visual QC issues.

Return ONLY valid JSON with exactly these keys:
- caption: string (1 sentence, <= 20 words)
- observed_food: short noun phrase (e.g., "apple slices", "steamed broccoli florets")
- label_match: one of ["match","partial","mismatch","unclear"]
- label_confidence: number between 0 and 1 (your confidence in label_match)
- portion_size_ok: boolean (true if it looks like a typical single adult serving)
- plate_rim_visible: boolean (true if some plate rim is visible)
- qc_issues: array of short strings from this set:
  ["sauce_present","multiple_items","bowl_present","text_present","odd_perspective",
   "plate_not_matching","food_unrecognizable","portion_too_small","portion_too_large",
   "not_on_plate","background_busy","hands_present"]
- qc_reasons: array of <= 3 short strings explaining label_match and any issues

Guidance:
- If uncertain, set label_match="unclear".
- Use "partial" if it's clearly related but not exact (e.g., wrong cut/prep form).
- Do NOT invent brands or extra items.
"""


# ---------------- Response extraction ----------------

def _extract_text(response: Any) -> str:
    """
    Extract the model's text output from common google-genai response shapes.
    """
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()

    raise RuntimeError("Could not extract text from Gemini response.")


# ---- Helper to strip Markdown code fences and leading/trailing junk before JSON parsing ----
def _strip_json_fences(s: str) -> str:
    """Remove common Markdown code-fence wrappers around JSON."""
    if not isinstance(s, str):
        return ""
    t = s.strip()

    # Remove ```json ... ``` or ``` ... ``` wrappers
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            # Drop opening fence line
            lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        t = "\n".join(lines).strip()

    # Sometimes models prepend/append stray text; keep the outermost JSON object.
    # Grab from first '{' to last '}' if both exist.
    if "{" in t and "}" in t:
        i = t.find("{")
        j = t.rfind("}")
        if i != -1 and j != -1 and j > i:
            t = t[i:j+1].strip()

    return t


def _clamp_0_100(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return max(0.0, min(100.0, v))


def _normalize_qc_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate the JSON fields we expect.
    """
    allowed_match = {"match", "partial", "mismatch", "unclear"}
    allowed_issues = {
        "sauce_present","multiple_items","bowl_present","text_present","odd_perspective",
        "plate_not_matching","food_unrecognizable","portion_too_small","portion_too_large",
        "not_on_plate","background_busy","hands_present"
    }

    out: Dict[str, Any] = {}
    out["caption"] = str(data.get("caption", "")).strip()
    out["observed_food"] = str(data.get("observed_food", "")).strip()

    label_match = str(data.get("label_match", "unclear")).strip().lower()
    out["label_match"] = label_match if label_match in allowed_match else "unclear"

    try:
        conf = float(data.get("label_confidence", 0.0))
    except Exception:
        conf = 0.0
    out["label_confidence"] = max(0.0, min(1.0, conf))

    out["portion_size_ok"] = bool(data.get("portion_size_ok", False))
    out["plate_rim_visible"] = bool(data.get("plate_rim_visible", False))

    issues = data.get("qc_issues", [])
    if not isinstance(issues, list):
        issues = []
    issues_norm = []
    for x in issues:
        s = str(x).strip()
        if s in allowed_issues:
            issues_norm.append(s)
    out["qc_issues"] = issues_norm[:12]

    reasons = data.get("qc_reasons", [])
    if not isinstance(reasons, list):
        reasons = []
    out["qc_reasons"] = [str(x).strip() for x in reasons if str(x).strip()][:3]

    return out


# ---------------- Caption/QC call ----------------

def qc_one_image(
    client: Any,
    image_path: Path,
    prompt: str,
    model: str,
    max_attempts: int = 6,
) -> Dict[str, Any]:
    """
    Send one image + QC prompt to Gemini and return normalized JSON.
    """
    try:
        from google.genai import types  # type: ignore
    except Exception:
        types = None  # type: ignore

    img_bytes = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    if types is not None and hasattr(types, "Part") and hasattr(types.Part, "from_bytes"):
        try:
            img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
        except TypeError:
            img_part = types.Part.from_bytes(bytes=img_bytes, mime_type=mime)
        text_part = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[img_part, text_part])]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            response_mime_type="application/json",
            temperature=0.2,
        )
    else:
        contents = [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}},
                {"text": prompt},
            ],
        }]
        config = {
            "system_instruction": SYSTEM_INSTRUCTIONS,
            "response_mime_type": "application/json",
            "temperature": 0.2,
        }

    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            raw_text = _extract_text(response)
            raw_text_clean = _strip_json_fences(raw_text)
            parsed = json.loads(raw_text_clean)
            if not isinstance(parsed, dict):
                raise ValueError("Model did not return a JSON object.")
            return _normalize_qc_json(parsed)

        except Exception as e:
            last_err = e
            print(f"[WARN] QC attempt {attempt+1} failed for {image_path.name}: {e}")
            if attempt < max_attempts - 1:
                backoff_sleep(attempt)
            else:
                break

    raise RuntimeError(f"Failed QC for {image_path.name} after {max_attempts} attempts: {last_err}")


# ---------------- Master file I/O ----------------

def load_master(master_path: Path) -> List[Dict[str, Any]]:
    if not master_path.exists():
        raise FileNotFoundError(f"stimuli_master.json not found: {master_path}")
    with master_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {master_path}, got {type(data)}")
    return data


def save_master(master_path: Path, data: List[Dict[str, Any]]) -> None:
    bak = master_path.with_suffix(".json.bak")
    if not bak.exists():
        bak.write_text(master_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[INFO] Backup written: {bak}")
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------- Optional CSV export ----------------

def _stringify_cell(x: Any) -> str:
    """Convert values (including lists/dicts) to a CSV-friendly string."""
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)


def export_qc_plus_ai_csv(stimuli_entries: List[Dict[str, Any]], input_list_csv: Path, out_csv: Path) -> None:
    """
    Write a CSV combining Category/Food from the input list with AI QC + subjective judgements
    stored in stimuli_master.json.
    """
    if not input_list_csv.exists():
        print(f"[WARN] Input list CSV not found at: {input_list_csv}")
        print("[WARN] QC+AI CSV export skipped.")
        return

    def norm_key(x: Any) -> str:
        s = ("" if x is None else str(x)).strip().casefold()
        # collapse internal whitespace
        s = " ".join(s.split())
        return s

    # Build lookup by Food name (normalized match on entry['food'])
    by_food: Dict[str, Dict[str, Any]] = {}
    dup_keys: Dict[str, int] = {}
    for e in stimuli_entries:
        key_raw = e.get("food")
        if key_raw:
            k = norm_key(key_raw)
            if k in by_food:
                dup_keys[k] = dup_keys.get(k, 1) + 1
            by_food[k] = e

    if dup_keys:
        # show up to 10 duplicates
        examples = list(dup_keys.items())[:10]
        print(f"[WARN] Duplicate food keys found in stimuli_master.json (normalized). Using last occurrence. Examples: {examples}")

    rows_out: List[Dict[str, Any]] = []

    with input_list_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or ("Food" not in reader.fieldnames):
            print(f"[WARN] Input list CSV at {input_list_csv} does not contain a 'Food' column.")
            print("[WARN] QC+AI CSV export skipped.")
            return

        for r in reader:
            food_name = (r.get("Food") or "").strip()
            food_key = norm_key(food_name)
            cat = (r.get("Category") or "").strip()

            e = by_food.get(food_key, {})

            out: Dict[str, Any] = {
                "Category": cat,
                "Food": food_name,

                # Classification labels from stimuli_master (written by classify_food_gemini)
                

                # From stimuli_master (if present)
                "filename": e.get("image_file", ""),
                "food": food_name,
                "base_food": e.get("base_food", ""),
                # NOTE: legacy food_classification is no longer exported —
                # replaced by Category_Intuitive_7 (see classify_food_gemini).
                "natural_vs_transformed": e.get("Natural_vs_transformed", ""),
                "Category_WHO_10": e.get("Category_WHO_10", ""),
                "Category_Intuitive_7": e.get("Category_Intuitive_7", ""),
                "Category_Culinary_9": e.get("Category_Culinary_9", ""),
                "Category_NOVA_4": e.get("Category_NOVA_4", ""),
                "Transformation_score": e.get("Transformation_score", ""),
                "prompt": e.get("prompt", ""),
                "model": e.get("model", ""),
                "seed": e.get("seed", ""),
                "created": e.get("created", ""),
                "style_version": e.get("style_version", ""),
                "caption": e.get("caption", ""),
                "aware_observed_food": e.get("observed_food", ""),
                "label_match": e.get("label_match", ""),
                "label_confidence": e.get("label_confidence", ""),
                "portion_size_ok": e.get("portion_size_ok", ""),
                "plate_rim_visible": e.get("plate_rim_visible", ""),
                "qc_issues": e.get("qc_issues", ""),
                "qc_reasons": e.get("qc_reasons", ""),

                # Subjective judgements
                "aware_ai_calorie_density": e.get("calorie_density_0_100", ""),
                "aware_ai_healthiness": e.get("healthiness_0_100", ""),
                "aware_ai_sweetness": e.get("sweetness_0_100", ""),
                "aware_ai_saltiness": e.get("saltiness_0_100", ""),
                "aware_ai_sourness": e.get("sourness_0_100", ""),
                "aware_ai_bitterness": e.get("bitterness_0_100", ""),
                "aware_ai_savoriness": e.get("savoriness_0_100", ""),
                "aware_ai_fattiness": e.get("fatty_flavour_0_100", ""),
                "aware_ai_spiciness": e.get("spiciness_0_100", ""),

                "qc_model": e.get("qc_model", ""),
                "qc_at": e.get("qc_at", ""),
            }

            for k, v in list(out.items()):
                out[k] = _stringify_cell(v)

            rows_out.append(out)

    # Warn about unmatched foods from the input list
    unmatched = [r for r in rows_out if not r.get("filename") and (r.get("Food") or "").strip()]
    if unmatched:
        print(f"[WARN] QC+AI CSV: {len(unmatched)} rows from input list did not match any stimuli_master entry (by Food).")
        for ex in unmatched[:10]:
            print(f"  - Unmatched Food: {ex.get('Food')}")
        if len(unmatched) > 10:
            print(f"  ...and {len(unmatched)-10} more")

    import pandas as pd
    new_df = pd.DataFrame(rows_out)

    # 1. Merge human ratings if available
    human_csv = input_list_csv.parent / "human_ratings.csv"
    if human_csv.exists():
        try:
            human_df = pd.read_csv(human_csv)
            # Ensure no overlapping columns except filename
            cols_to_use = [c for c in human_df.columns if c not in new_df.columns or c == 'filename']
            new_df = pd.merge(new_df, human_df[cols_to_use], on='filename', how='left')
            print(f"[INFO] Integrated empirical human ratings from {human_csv.name}")
        except Exception as e:
            print(f"[WARN] Failed to merge human_ratings.csv: {e}")

    # Apply column ordering
    col_order = [
        'filename', 'food', 'base_food', 'Category_WHO_10', 'Category_Intuitive_7', 'Category_Culinary_9',
        'Category_NOVA_4', 'natural_vs_transformed', 'Transformation_score',
        'prompt', 'model', 'seed', 'created', 'style_version',
        'human_calorie_density', 'human_healthiness', 'human_appeal', 'human_familiarity',
        'human_sweetness', 'human_saltiness', 'human_sourness', 'human_bitterness', 
        'human_savoriness', 'human_fattiness', 'human_spiciness',
        'caption', 'aware_observed_food', 'label_match', 'label_confidence', 
        'portion_size_ok', 'plate_rim_visible', 'qc_issues', 'qc_reasons', 'qc_model', 'qc_at',
        'aware_ai_calorie_density', 'aware_ai_healthiness', 'aware_ai_sweetness', 
        'aware_ai_saltiness', 'aware_ai_sourness', 'aware_ai_bitterness', 
        'aware_ai_savoriness', 'aware_ai_fattiness', 'aware_ai_spiciness', 
        'blind_model', 'blind_observed_food', 'blind_guess_similarity', 
        'blind_ai_calorie_density', 'blind_ai_healthiness', 'blind_ai_sweetness', 
        'blind_ai_saltiness', 'blind_ai_sourness', 'blind_ai_bitterness', 
        'blind_ai_savoriness', 'blind_ai_fattiness', 'blind_ai_spiciness', 
        'll_mean_luminance', 'll_rms_contrast', 
        'll_lab_L_mean', 'll_lab_L_std', 'll_lab_a_mean', 'll_lab_a_std', 
        'll_lab_b_mean', 'll_lab_b_std', 'll_hsv_s_mean', 'll_edge_energy'
    ] + [f"ll_hog_pc{j+1:02d}" for j in range(10)]
    
    # 2. Merge with existing output (to preserve blind_ai and ll_metrics)
    if out_csv.exists():
        old_df = pd.read_csv(out_csv, encoding="utf-8-sig")
        if 'filename' in old_df.columns and 'filename' in new_df.columns:
            merge_key = 'filename'
        else:
            merge_key = 'Food'

        # Guard against duplicate keys: a duplicated row in either frame would
        # multiply rows in the left-merge and then self-perpetuate through
        # every subsequent export.
        for name, frame in (("existing CSV", old_df), ("new export", new_df)):
            n_dup = int(frame.duplicated(subset=merge_key).sum())
            if n_dup:
                print(f"[WARN] Dropping {n_dup} duplicate '{merge_key}' row(s) from {name}.")
        old_df = old_df.drop_duplicates(subset=merge_key, keep='first')
        new_df = new_df.drop_duplicates(subset=merge_key, keep='first')

        cols_to_update = [c for c in new_df.columns if c != merge_key]
        old_df = old_df.drop(columns=[c for c in cols_to_update if c in old_df.columns])
        final_df = pd.merge(new_df, old_df, on=merge_key, how='left')
    else:
        final_df = new_df.drop_duplicates(subset='filename' if 'filename' in new_df.columns else 'Food', keep='first')
        
    # Drop legacy/unrecognized columns to keep output clean
    legacy_cols_to_drop = ['food_classification', 'Category', 'Food', 'll_power_radial_30_json', 'll_lbp_hist_json']
    final_df.drop(columns=[c for c in legacy_cols_to_drop if c in final_df.columns], inplace=True)

    final_order = [c for c in col_order if c in final_df.columns]
    final_order += [c for c in final_df.columns if c not in final_order]
    final_df = final_df[final_order]

    final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote QC+AI judgements CSV to: {out_csv}")
    print(f"[INFO] CSV merged from input list: {input_list_csv}")


# ---------------- QC issue reporting ----------------

def entry_image_exists(entry: Dict[str, Any], stimuli_dir: Path) -> bool:
    img = entry.get("image_file")
    if not img:
        return False
    return (stimuli_dir / str(img)).exists()


def is_potential_issue(entry: Dict[str, Any],
                       conf_threshold: float,
                       flag_partial: bool,
                       ignore_issues: Optional[set[str]] = None) -> bool:
    """
    Heuristic: what should we flag for human review?
    """
    match = str(entry.get("label_match", "unclear")).lower()
    conf = float(entry.get("label_confidence", 0.0) or 0.0)
    issues = entry.get("qc_issues", []) or []
    portion_ok = bool(entry.get("portion_size_ok", True))
    rim_ok = bool(entry.get("plate_rim_visible", True))

    ignore = ignore_issues or set()
    issues = [x for x in issues if str(x) not in ignore]

    if match in {"mismatch", "unclear"}:
        return True
    if flag_partial and match == "partial" and conf >= conf_threshold:
        return True
    if conf < conf_threshold:
        return True
    if issues:
        return True
    if not portion_ok or not rim_ok:
        return True
    return False


def collect_qc_issues(data: List[Dict[str, Any]],
                      stimuli_dir: Path,
                      conf_threshold: float,
                      flag_partial: bool,
                      include_missing: bool,
                      ignore_issues: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in data:
        exists = entry_image_exists(e, stimuli_dir)
        if not exists and not include_missing:
            continue

        if is_potential_issue(e, conf_threshold=conf_threshold, flag_partial=flag_partial, ignore_issues=ignore_issues):
            out.append({
                "image_file": e.get("image_file"),
                "food": e.get("food"),
                "base_food": e.get("base_food"),
                "category": e.get("category"),
                "caption": e.get("caption"),
                "observed_food": e.get("observed_food"),
                "label_match": e.get("label_match"),
                "label_confidence": e.get("label_confidence"),
                "portion_size_ok": e.get("portion_size_ok"),
                "plate_rim_visible": e.get("plate_rim_visible"),
                "qc_issues": e.get("qc_issues"),
                "qc_reasons": e.get("qc_reasons"),
                "file_exists": bool(exists),
                "path": str((stimuli_dir / str(e.get("image_file"))).resolve()) if e.get("image_file") else None,
            })
    return out


# ---------------- CLI ----------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption + QC plated stimuli with Gemini and update stimuli_master.json")
    p.add_argument("--stimuli-dir", type=str, required=True, help="Folder containing stimuli_master.json and images")
    p.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini model for captioning/QC")
    p.add_argument("--limit", type=int, default=None, help="Process at most N items")
    p.add_argument("--offset", type=int, default=0, help="Start at item offset")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing caption/QC fields if present")
    p.add_argument("--dry-run", action="store_true", help="Do not call API; just report what would be processed")
    p.add_argument("--conf-threshold", type=float, default=0.70, help="Flag items with confidence below this")
    p.add_argument("--flag-partial", action="store_true", help="Also flag 'partial' matches for review")
    p.add_argument("--include-missing", action="store_true",
                   help="Include entries whose image file is missing in qc_issues.json (default: skip missing)")
    p.add_argument("--ignore-issues", type=str, default="",
                   help="Comma-separated qc_issues tags to ignore for flagging (e.g. 'sauce_present,multiple_items')")
    p.add_argument("--food-list", type=str, default=None,
                   help="Path to external food-list CSV used as input for the merged CSV export. "
                        "Overrides the default food_list_initial_seed.csv.")
    p.add_argument("--dynamic-csv", type=str, default=None,
                   help="Output path for the merged QC+AI CSV (Foodpictures_information_dynamic.csv). "
                        "Overrides the default data/ location. Use to write output to an external project.")
    return p.parse_args(argv)


def has_qc(entry: Dict[str, Any]) -> bool:
    # We treat presence of caption_model + caption as "already processed"
    return bool(str(entry.get("caption", "")).strip()) and bool(str(entry.get("qc_model", "")).strip())


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    stimuli_dir = Path(args.stimuli_dir).expanduser().resolve()
    master_path = stimuli_dir / "stimuli_master.json"

    data = load_master(master_path)

    # Determine indices to process within the window
    start = args.offset
    end = len(data) if args.limit is None else min(len(data), args.offset + args.limit)

    indices: List[int] = []
    for i in range(start, end):
        if args.overwrite or not has_qc(data[i]):
            indices.append(i)

    print(f"[INFO] stimuli_dir: {stimuli_dir}")
    print(f"[INFO] Using model: {args.model}")
    print(f"[INFO] Entries in master: {len(data)}")
    print(f"[INFO] Will process: {len(indices)} (overwrite={args.overwrite})")
    print(f"[INFO] QC thresholds: conf<{args.conf_threshold}, flag_partial={args.flag_partial}")

    ignore_set = {x.strip() for x in str(args.ignore_issues).split(",") if x.strip()}
    if ignore_set:
        print(f"[INFO] Ignoring qc_issues tags for flagging: {sorted(ignore_set)}")

    if args.dry_run:
        for idx in indices[:10]:
            print(f"[DRY] Would process: {data[idx].get('image_file')}")
        if len(indices) > 10:
            print(f"[DRY] ...and {len(indices)-10} more")
        return 0

    client = get_gemini_client()
    done = 0

    for idx in indices:
        entry = data[idx]
        img_name = entry.get("image_file")
        if not img_name:
            print(f"[WARN] Missing image_file for entry index {idx}; skipping")
            continue

        img_path = stimuli_dir / img_name
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}; skipping")
            continue

        expected_food = str(entry.get("food", "")).strip() or str(entry.get("image_file", "")).strip()
        expected_base = str(entry.get("base_food", "")).strip() or expected_food
        expected_cat = entry.get("category", None)

        prompt = build_qc_prompt(expected_food, expected_base, expected_cat)

        try:
            qc = qc_one_image(client, img_path, prompt=prompt, model=args.model)

            # Store into entry
            entry["caption"] = qc["caption"]
            entry["observed_food"] = qc["observed_food"]
            entry["label_match"] = qc["label_match"]
            entry["label_confidence"] = qc["label_confidence"]
            entry["portion_size_ok"] = qc["portion_size_ok"]
            entry["plate_rim_visible"] = qc["plate_rim_visible"]
            entry["qc_issues"] = qc["qc_issues"]
            entry["qc_reasons"] = qc["qc_reasons"]

            entry["qc_model"] = args.model
            entry["qc_at"] = int(time.time())

            done += 1
            if done % 10 == 0:
                save_master(master_path, data)
                print(f"[INFO] Progress saved: {done}/{len(indices)}")

            print(f"[OK] QC {img_name}: {entry['label_match']} ({entry['label_confidence']:.2f}) | {entry['caption']}")

        except Exception as e:
            print(f"[FAIL] Could not QC {img_name}: {e}")

    save_master(master_path, data)
    print(f"[DONE] Processed {done}/{len(indices)} entries. Updated: {master_path}")

    # --- QC Issues report ---
    issues = collect_qc_issues(
        data,
        stimuli_dir=stimuli_dir,
        conf_threshold=args.conf_threshold,
        flag_partial=args.flag_partial,
        include_missing=bool(args.include_missing),
        ignore_issues=ignore_set,
    )
    issues_path = stimuli_dir / "qc_issues.json"
    with issues_path.open("w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)

    print("\n[QC] Potential issues flagged:", len(issues))
    print(f"[QC] Wrote: {issues_path}")

    # Print a short, readable summary (first 30)
    for item in issues[:30]:
        print(
            f"- {item.get('image_file')} | expected={item.get('food')} | "
            f"observed={item.get('observed_food')} | match={item.get('label_match')} "
            f"conf={item.get('label_confidence')} | issues={item.get('qc_issues')}"
        )
    if len(issues) > 30:
        print(f"[QC] ...and {len(issues)-30} more (see qc_issues.json)")

    # Make missing images discoverable even when excluded from qc_issues.json.
    missing = []
    for e in data:
        if e.get("image_file") and not entry_image_exists(e, stimuli_dir):
            missing.append({
                "image_file": e.get("image_file"),
                "food": e.get("food"),
                "base_food": e.get("base_food"),
                "category": e.get("category"),
            })
    missing_path = stimuli_dir / "missing_images.json"
    with missing_path.open("w", encoding="utf-8") as f:
        json.dump(missing, f, indent=2, ensure_ascii=False)
    if missing:
        print(f"[WARN] Missing images referenced by master: {len(missing)}")
        print(f"[WARN] Wrote: {missing_path}")

    # --- Export merged CSV (Category/Food from list + QC/judgements from stimuli_master) ---
    input_list_csv = Path(args.food_list).expanduser().resolve() if args.food_list else DEFAULT_INPUT_LIST_CSV
    dynamic_csv    = Path(args.dynamic_csv).expanduser().resolve() if args.dynamic_csv else QC_PLUS_AI_CSV
    dynamic_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Stimulus list CSV: {input_list_csv}")
    print(f"[INFO] Dynamic CSV output: {dynamic_csv}")
    export_qc_plus_ai_csv(data, input_list_csv, dynamic_csv)

    # Exit code: 0 if all processed, 1 if any issues flagged (useful for automation)
    return 0 if len(issues) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())