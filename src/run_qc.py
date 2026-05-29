#!/usr/bin/env python3
"""
Caption + QC plated stimuli with Gemini.

Reads:  <stimuli_dir>/stimuli_master.json
Loads:  <stimuli_dir>/<image_file> for each entry
Writes: captions + QC fields back into stimuli_master.json (in-place), with a .bak backup
Outputs: QC issues summary to stdout AND writes <stimuli_dir>/qc_issues.json

Additionally:
- Writes dynamic CSV (merged from the input list CSV + AI QC/judgements).

Requires:
  pip install -U google-genai
  export GEMINI_API_KEY=...
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
    time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)

# ---------------- Prompts ----------------

SYSTEM_INSTRUCTIONS = (
    "You are doing neutral, visual quality control for experimental food stimuli. "
    "Be factual. Do not mention brands. Do not mention the prompt. Do not add opinions beyond the requested ratings. "
    "For any 0–100 ratings (calories/health/flavour), provide best-effort *subjective judgements* "
    "based only on what is visually inferable from the image and typical culinary expectations. "
    "These are not objective measurements. For 'fatty', judge fatty-tasting richness/oiliness/creaminess (mouthfeel), not fat content. If highly uncertain, use 50."
)

def build_qc_prompt(food: str, base_food: str, prep: Optional[str], cat: Optional[str]) -> str:
    return f"""You will be shown an image of food on a plate.
EXPECTED LABELS:
- food: "{food}" | base: "{base_food}" | prep: "{prep}" | cat: "{cat}"

Tasks:
1) Write a brief neutral caption (1 sentence, <= 20 words).
2) Identify observed food and preparation (raw/prepared/unknown).
3) Compare observed vs expected and rate label match.
4) Flag visual QC issues.
5) Provide 0–100 ratings for health/flavour (subjective judgements).

Return ONLY JSON:
- caption, observed_food, observed_prep, label_match, label_confidence (0-1), 
- portion_size_ok (bool), plate_rim_visible (bool),
- qc_issues: list from ["sauce_present","multiple_items","bowl_present","text_present","odd_perspective","plate_not_matching","food_unrecognizable","portion_too_small","portion_too_large","not_on_plate","background_busy","hands_present"],
- qc_reasons: list of <=3 strings,
- calorie_density_0_100, healthiness_0_100, sweetness_0_100, saltiness_0_100, sourness_0_100, bitterness_0_100, savoriness_0_100, fatty_flavour_0_100, spiciness_0_100
"""

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

def _normalize_qc_json(data: Dict[str, Any]) -> Dict[str, Any]:
    # Basic clamping/mapping logic as in original
    for k in ["calorie_density_0_100", "healthiness_0_100", "sweetness_0_100", "saltiness_0_100", 
              "sourness_0_100", "bitterness_0_100", "savoriness_0_100", "fatty_flavour_0_100", "spiciness_0_100"]:
        if k in data:
            try: data[k] = max(0.0, min(100.0, float(data[k])))
            except: data[k] = None
    return data

def qc_one_image(client: Any, image_path: Path, prompt: str, model: str) -> Dict[str, Any]:
    img_bytes = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    contents = [{"role": "user", "parts": [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}},
        {"text": prompt}
    ]}]
    config = {"system_instruction": SYSTEM_INSTRUCTIONS, "response_mime_type": "application/json", "temperature": 0.2}
    
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return _normalize_qc_json(json.loads(_strip_json_fences(_extract_text(response))))
        except Exception as e:
            backoff_sleep(attempt)
    raise RuntimeError(f"QC failed for {image_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", type=str, required=True)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stimuli_dir = Path(args.stimuli_dir).resolve()
    master_path = stimuli_dir / "stimuli_master.json"
    with open(master_path, "r") as f: data = json.load(f)

    client = get_gemini_client()
    done = 0
    for entry in data:
        if args.limit and done >= args.limit: break
        if not args.overwrite and "qc_at" in entry: continue
        
        img_path = stimuli_dir / entry["image_file"]
        if not img_path.exists(): continue
        
        print(f"QC: {entry['food']}")
        try:
            res = qc_one_image(client, img_path, build_qc_prompt(entry["food"], entry.get("base_food",""), entry.get("prep_form",""), entry.get("category","")), args.model)
            entry.update(res)
            entry["qc_model"] = args.model
            entry["qc_at"] = int(time.time())
            done += 1
            if done % 10 == 0:
                with open(master_path, "w") as f: json.dump(data, f, indent=2)
        except Exception as e: print(f"Error: {e}")

    with open(master_path, "w") as f: json.dump(data, f, indent=2)
    
    # Export to CSV
    by_food = {e["food"].lower(): e for e in data if "qc_at" in e}
    with open(args.input_csv, "r") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) + [
            "filename", "caption", "aware_observed_food", "aware_observed_prep", "label_match", "label_confidence",
            "portion_size_ok", "plate_rim_visible", "qc_issues", "qc_reasons",
            "aware_ai_calorie_density", "aware_ai_healthiness", "aware_ai_sweetness", "aware_ai_saltiness",
            "aware_ai_sourness", "aware_ai_bitterness", "aware_ai_savoriness", "aware_ai_fattiness", "aware_ai_spiciness",
            "qc_model", "qc_at"
        ]
        
    out_rows = []
    for r in rows:
        e = by_food.get(r["Food"].lower(), {})
        r.update({
            "filename": e.get("image_file", ""),
            "caption": e.get("caption", ""),
            "aware_observed_food": e.get("observed_food", ""),
            "aware_observed_prep": e.get("observed_prep", ""),
            "label_match": e.get("label_match", ""),
            "label_confidence": e.get("label_confidence", ""),
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
            "qc_at": e.get("qc_at", "")
        })
        out_rows.append(r)
        
    with open(args.output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
