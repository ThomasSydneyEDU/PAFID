#!/usr/bin/env python3
"""
Blind Ratings with Gemini: Get food labels and ratings from images alone.

Step 1: Sends ONLY the image to Gemini to get `observed_food` and ratings.
Step 2: Sends a text-only prompt comparing `observed_food` to the actual food label.

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
    try: from google import genai
    except: raise RuntimeError("Missing google-genai.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=api_key)

def backoff_sleep(attempt: int) -> None:
    time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)

# ---------------- Prompts ----------------

SYSTEM_INSTRUCTIONS_BLIND = (
    "You are a neutral observer providing visual assessments of food stimuli. "
    "Be factual. Do not mention brands. Do not add opinions beyond the requested ratings. "
    "For any 0–100 ratings (calories/health/flavour), provide best-effort *subjective judgements* "
    "based only on what is visually inferable from the image and typical culinary expectations. "
    "For 'fatty', judge fatty-tasting richness/oiliness/creaminess (mouthfeel). If highly uncertain, use 50."
)

BLIND_PROMPT = """You will be shown an image of food on a plate.
Tasks:
1) Identify the food visible in the image.
2) Provide 0–100 ratings for health/flavour (subjective judgements).

Return ONLY JSON:
- observed_food, calorie_density_0_100, healthiness_0_100, sweetness_0_100, saltiness_0_100, 
- sourness_0_100, bitterness_0_100, savoriness_0_100, fatty_flavour_0_100, spiciness_0_100
"""

def build_similarity_prompt(guessed_food: str, actual_food: str) -> str:
    return f"""Compare guessed identity vs actual label.
- Guess: "{guessed_food}" | Actual: "{actual_food}"
Rate semantic similarity 0-100 (100=same, 50=related, 0=unrelated).
Return ONLY JSON: {"similarity_score_0_100": N}"""

# ---------------- Logic ----------------

def _extract_text(response: Any) -> str:
    txt = getattr(response, "text", None)
    if isinstance(txt, str) and txt.strip(): return txt.strip()
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t.strip(): return t.strip()
    raise RuntimeError("Could not extract text.")

def _strip_json_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines: lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        t = "\n".join(lines).strip()
    if "{" in t and "}" in t:
        i, j = t.find("{"), t.rfind("}")
        t = t[i:j+1].strip()
    return t

def get_blind_ratings(client: Any, image_path: Path, model: str) -> Dict[str, Any]:
    img_bytes = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    contents = [{"role": "user", "parts": [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}},
        {"text": BLIND_PROMPT}
    ]}]
    config = {"system_instruction": SYSTEM_INSTRUCTIONS_BLIND, "response_mime_type": "application/json", "temperature": 0.2}
    for attempt in range(5):
        try:
            res = client.models.generate_content(model=model, contents=contents, config=config)
            return json.loads(_strip_json_fences(_extract_text(res)))
        except: backoff_sleep(attempt)
    raise RuntimeError(f"Blind rating failed: {image_path}")

def get_similarity_score(client: Any, guessed: str, actual: str, model: str) -> float:
    contents = [{"role": "user", "parts": [{"text": build_similarity_prompt(guessed, actual)}]}]
    for attempt in range(3):
        try:
            res = client.models.generate_content(model=model, contents=contents, config={"response_mime_type": "application/json"})
            return float(json.loads(_strip_json_fences(_extract_text(res)))["similarity_score_0_100"])
        except: backoff_sleep(attempt)
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True, help="Path to the dynamic CSV to update")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stimuli_dir = Path(args.stimuli_dir).resolve()
    with open(args.csv, "r") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
        for fld in ["blind_observed_food", "blind_guess_similarity", "blind_ai_calorie_density", "blind_ai_healthiness", 
                    "blind_ai_sweetness", "blind_ai_saltiness", "blind_ai_sourness", "blind_ai_bitterness", 
                    "blind_ai_savoriness", "blind_ai_fattiness", "blind_ai_spiciness", "blind_model"]:
            if fld not in fieldnames: fieldnames.append(fld)

    client = get_gemini_client()
    done = 0
    for row in rows:
        if args.limit and done >= args.limit: break
        if not args.overwrite and row.get("blind_observed_food"): continue
        
        img_path = stimuli_dir / row["filename"]
        if not img_path.exists(): continue
        
        print(f"Blind Rating: {row['Food']}")
        try:
            ratings = get_blind_ratings(client, img_path, args.model)
            sim = get_similarity_score(client, ratings.get("observed_food",""), row["Food"], args.model)
            
            row.update({
                "blind_observed_food": ratings.get("observed_food"),
                "blind_guess_similarity": sim,
                "blind_ai_calorie_density": ratings.get("calorie_density_0_100"),
                "blind_ai_healthiness": ratings.get("healthiness_0_100"),
                "blind_ai_sweetness": ratings.get("sweetness_0_100"),
                "blind_ai_saltiness": ratings.get("saltiness_0_100"),
                "blind_ai_sourness": ratings.get("sourness_0_100"),
                "blind_ai_bitterness": ratings.get("bitterness_0_100"),
                "blind_ai_savoriness": ratings.get("savoriness_0_100"),
                "blind_ai_fattiness": ratings.get("fatty_flavour_0_100"),
                "blind_ai_spiciness": ratings.get("spiciness_0_100"),
                "blind_model": args.model
            })
            done += 1
            if done % 10 == 0:
                with open(args.csv, "w") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writeheader()
                    csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
        except Exception as e: print(f"Error: {e}")

    with open(args.csv, "w") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
    print(f"Updated: {args.csv}")

if __name__ == "__main__":
    main()
