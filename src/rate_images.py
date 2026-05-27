#!/usr/bin/env python3
"""
Blind Ratings with Gemini: Get food labels and ratings from images alone,
followed by a similarity score against the actual label.

Step 1: Sends ONLY the image to Gemini to get `observed_food` and ratings.
Step 2: Sends a text-only prompt comparing `observed_food` to the actual food label.

Outputs:
- <stimuli_dir>/blind_ratings.json
- <stimuli_dir>/blind_ratings.csv

Requires:
  pip install -U google-genai
  export GEMINI_API_KEY=...
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------- Gemini client ----------------

def get_gemini_client():
    try:
        from google import genai
    except Exception:
        print("[ERROR] Missing google-genai. Install with: pip install -U google-genai", file=sys.stderr)
        raise
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it first.")
    return genai.Client(api_key=api_key)

def backoff_sleep(attempt: int) -> None:
    delay = min(60, (2 ** attempt)) + 0.25 * attempt
    time.sleep(delay)

# ---------------- Prompts ----------------

SYSTEM_INSTRUCTIONS_BLIND = (
    "You are a neutral observer providing visual assessments of food stimuli. "
    "Be factual. Do not mention brands. Do not add opinions beyond the requested ratings. "
    "For any 0–100 ratings (calories/health/flavour), provide best-effort *subjective judgements* "
    "based only on what is visually inferable from the image and typical culinary expectations. "
    "These are not objective measurements. For 'fatty', judge fatty-tasting richness/oiliness/creaminess (mouthfeel), not fat content. If highly uncertain, use 50."
)

BLIND_PROMPT = """You will be shown an image of food on a plate.

Tasks:
1) Identify the food visible in the image.
2) Provide 0–100 ratings as *subjective judgements* of perceived flavour intensity and health attributes (best-effort inferences from visible cues + typical culinary expectations).

Return ONLY valid JSON with exactly these keys:
- observed_food: short noun phrase (e.g., "apple slices", "steamed broccoli florets")
- calorie_density_0_100: number 0-100 (0=very low calorie, 100=very high calorie density)
- healthiness_0_100: number 0-100 (0=very unhealthy, 100=very healthy)
- sweetness_0_100: number 0-100 (subjective perceived sweetness; 0=not sweet, 100=very sweet)
- saltiness_0_100: number 0-100 (subjective perceived saltiness; 0=not salty, 100=very salty)
- sourness_0_100: number 0-100 (subjective perceived sourness; 0=not sour, 100=very sour)
- bitterness_0_100: number 0-100 (subjective perceived bitterness; 0=not bitter, 100=very bitter)
- savoriness_0_100: number 0-100 (subjective perceived savoury/umami; 0=not savoury, 100=very savoury)
- fatty_flavour_0_100: number 0-100 ("fatty" as flavour/mouthfeel: perceived richness/oiliness/creamy mouthfeel, NOT fat content; 0=not fatty-tasting, 100=very fatty-tasting)
- spiciness_0_100: number 0-100 (subjective perceived chilli heat; 0=not spicy, 100=very spicy)

Guidance:
- For 0–100 ratings: these are subjective judgements of *perceived flavour/mouthfeel intensity*.
  "Fatty" specifically means fatty-tasting richness/oiliness/creaminess (mouthfeel), not nutritional fat content.
  Infer only from visible cues and typical culinary expectations.
  If highly uncertain, use 50.
"""

def build_similarity_prompt(guessed_food: str, actual_food: str) -> str:
    return f"""You are a semantic evaluator.
Compare the guessed food identity with the actual food label.

- Guessed Food: "{guessed_food}"
- Actual Food: "{actual_food}"

Task:
Rate the semantic similarity between these two descriptions on a scale of 0 to 100.
- 100 means they refer to the exact same food or are perfect synonyms.
- 50 means they are somewhat related (e.g., a specific type of a broader category).
- 0 means they are completely unrelated.

Return ONLY valid JSON with exactly this key:
- similarity_score_0_100: number 0-100
"""

# ---------------- Helpers ----------------

def _extract_text(response: Any) -> str:
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

def _strip_json_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        t = "\n".join(lines).strip()
    if "{" in t and "}" in t:
        i = t.find("{")
        j = t.rfind("}")
        t = t[i:j+1].strip()
    return t

def _clamp_0_100(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return max(0.0, min(100.0, v))
    except Exception:
        return None

# ---------------- API Calls ----------------

def get_blind_ratings(client: Any, image_path: Path, model: str, max_attempts: int = 5) -> Dict[str, Any]:
    """Step 1: Blindly rate and identify the food from the image."""
    img_bytes = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    
    contents = [{
        "role": "user",
        "parts": [
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}},
            {"text": BLIND_PROMPT},
        ],
    }]
    config = {
        "system_instruction": SYSTEM_INSTRUCTIONS_BLIND,
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            raw_text = _extract_text(response)
            parsed = json.loads(_strip_json_fences(raw_text))
            
            for k in ["calorie_density_0_100", "healthiness_0_100", "sweetness_0_100", "saltiness_0_100", 
                      "sourness_0_100", "bitterness_0_100", "savoriness_0_100", "fatty_flavour_0_100", "spiciness_0_100"]:
                parsed[k] = _clamp_0_100(parsed.get(k))
            return parsed
        except Exception as e:
            print(f"[WARN] Blind rating attempt {attempt+1} failed for {image_path.name}: {e}")
            if attempt < max_attempts - 1:
                backoff_sleep(attempt)
    raise RuntimeError(f"Failed blind ratings for {image_path.name}")

def get_similarity_score(client: Any, guessed_food: str, actual_food: str, model: str, max_attempts: int = 3) -> Optional[float]:
    """Step 2: Compare the guess to the actual label."""
    prompt = build_similarity_prompt(guessed_food, actual_food)
    
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    config = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            raw_text = _extract_text(response)
            parsed = json.loads(_strip_json_fences(raw_text))
            return _clamp_0_100(parsed.get("similarity_score_0_100"))
        except Exception as e:
            print(f"[WARN] Similarity score attempt {attempt+1} failed: {e}")
            if attempt < max_attempts - 1:
                backoff_sleep(attempt)
    return None

# ---------------- Main Logic ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", type=str, required=True, help="Folder containing images")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing blind ratings in canonical CSV")
    args = parser.parse_args()

    stimuli_dir = Path(args.stimuli_dir).resolve()
    src_dir = Path(__file__).resolve().parent
    canonical_csv = src_dir.parent / "data" / "Foodpictures_information_dynamic.csv"

    if not canonical_csv.exists():
        print(f"[ERROR] Canonical CSV not found at {canonical_csv}")
        return

    # Load Canonical Data
    print(f"[INFO] Loading canonical data from: {canonical_csv}")
    with open(canonical_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Define new columns for blind ratings
    new_fields = [
        "blind_observed_food",
        "blind_guess_similarity",
        "blind_ai_calorie_density",
        "blind_ai_healthiness",
        "blind_ai_sweetness",
        "blind_ai_saltiness",
        "blind_ai_sourness",
        "blind_ai_bitterness",
        "blind_ai_savoriness",
        "blind_ai_fattiness",
        "blind_ai_spiciness",
        "blind_model"
    ]
    for f in new_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    # Key mapping from Gemini JSON to canonical CSV
    key_map = {
        "observed_food": "blind_observed_food",
        "similarity_score_0_100": "blind_guess_similarity",
        "calorie_density_0_100": "blind_ai_calorie_density",
        "healthiness_0_100": "blind_ai_healthiness",
        "sweetness_0_100": "blind_ai_sweetness",
        "saltiness_0_100": "blind_ai_saltiness",
        "sourness_0_100": "blind_ai_sourness",
        "bitterness_0_100": "blind_ai_bitterness",
        "savoriness_0_100": "blind_ai_savoriness",
        "fatty_flavour_0_100": "blind_ai_fattiness",
        "spiciness_0_100": "blind_ai_spiciness",
        "model": "blind_model"
    }

    # Filter rows that need processing
    rows_to_process = []
    for row in rows:
        img_name = row.get("filename", "")
        if not img_name:
            continue
        # Skip if already processed, unless overwrite is set
        if not args.overwrite and str(row.get("blind_observed_food", "")).strip():
            continue
        
        img_path = stimuli_dir / img_name
        if not img_path.exists():
            continue

        rows_to_process.append(row)

    if args.limit:
        rows_to_process = rows_to_process[:args.limit]

    print(f"[INFO] {len(rows_to_process)} images need blind ratings.")
    if len(rows_to_process) == 0:
        print("[DONE] Nothing to do.")
        return

    client = get_gemini_client()
    backup_results = []
    out_json = stimuli_dir / "blind_ratings_backup.json"

    for i, row in enumerate(rows_to_process):
        img_name = row["filename"]
        img_path = stimuli_dir / img_name
        actual_food = row.get("food", Path(img_name).stem.replace("-", " "))
        
        try:
            # 1. Blindly rate and identify
            ratings = get_blind_ratings(client, img_path, args.model)
            guessed_food = ratings.get("observed_food", "")
            
            # 2. Compare guess to actual label
            similarity = get_similarity_score(client, guessed_food, actual_food, args.model)
            
            # Add metadata for backup JSON
            ratings["image_file"] = img_name
            ratings["actual_food"] = actual_food
            ratings["similarity_score_0_100"] = similarity
            ratings["model"] = args.model
            ratings["timestamp"] = int(time.time())
            backup_results.append(ratings)
            
            # Update the row in memory
            for orig_k, new_k in key_map.items():
                val = ratings.get(orig_k)
                row[new_k] = str(val) if val is not None else ""

            print(f"[{i+1}/{len(rows_to_process)}] OK: {img_name} | Actual: '{actual_food}' | Guessed: '{guessed_food}' | Sim: {similarity}")
            
            # Save progressively every 10 images
            if (i + 1) % 10 == 0:
                with open(canonical_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

        except Exception as e:
            print(f"[{i+1}/{len(rows_to_process)}] FAIL: {img_name}: {e}")

    # Final Save
    with open(canonical_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    with open(out_json, "w") as f:
        json.dump(backup_results, f, indent=2)

    print(f"[DONE] Successfully updated {canonical_csv}")
    print(f"[INFO] Backup saved to {out_json}")

if __name__ == "__main__":
    main()
