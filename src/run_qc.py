#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, csv, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional

def get_gemini_client():
    try: from google import genai
    except: raise RuntimeError("Missing google-genai.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=api_key)

def backoff_sleep(attempt: int) -> None:
    time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)

SYSTEM_INSTRUCTIONS = "You are doing neutral, visual quality control for experimental food stimuli. Be factual."

def build_qc_prompt(food: str, base_food: str, prep: Optional[str], cat: Optional[str]) -> str:
    return f"""You will be shown an image of food on a plate.
EXPECTED LABELS: food: "{food}" | base: "{base_food}" | prep: "{prep}" | cat: "{cat}"
Return ONLY JSON with keys:
caption, observed_food, observed_prep, label_match, label_confidence, 
portion_size_ok, plate_rim_visible, qc_issues, qc_reasons,
calorie_density_0_100, healthiness_0_100, sweetness_0_100, saltiness_0_100, sourness_0_100, bitterness_0_100, savoriness_0_100, fatty_flavour_0_100, spiciness_0_100
"""

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
        except: backoff_sleep(attempt)
    raise RuntimeError(f"QC failed for {image_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", type=str, required=True)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
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
        
        img_file = entry.get("image_file", "").strip()
        if not img_file: continue
        img_path = stimuli_dir / img_file
        if not img_path.is_file(): continue
        
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
    
    by_food = {e["food"].lower(): e for e in data if "qc_at" in e}
    with open(args.input_csv, "r") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) + [
            "filename", "caption", "aware_observed_food", "aware_observed_prep", "label_match", "label_confidence",
            "aware_ai_calorie_density", "aware_ai_healthiness", "aware_ai_sweetness", "aware_ai_saltiness",
            "aware_ai_sourness", "aware_ai_bitterness", "aware_ai_savoriness", "aware_ai_fattiness", "aware_ai_spiciness",
            "qc_model", "qc_at"
        ]
        
    for r in rows:
        e = by_food.get(r["Food"].lower(), {})
        r.update({
            "filename": e.get("image_file", ""), "caption": e.get("caption", ""),
            "aware_observed_food": e.get("observed_food", ""), "aware_observed_prep": e.get("observed_prep", ""),
            "label_match": e.get("label_match", ""), "label_confidence": e.get("label_confidence", ""),
            "aware_ai_calorie_density": e.get("calorie_density_0_100", ""), "aware_ai_healthiness": e.get("healthiness_0_100", ""),
            "aware_ai_sweetness": e.get("sweetness_0_100", ""), "aware_ai_saltiness": e.get("saltiness_0_100", ""),
            "aware_ai_sourness": e.get("sourness_0_100", ""), "aware_ai_bitterness": e.get("bitterness_0_100", ""),
            "aware_ai_savoriness": e.get("savoriness_0_100", ""), "aware_ai_fattiness": e.get("fatty_flavour_0_100", ""),
            "aware_ai_spiciness": e.get("spiciness_0_100", ""), "qc_model": e.get("qc_model", ""), "qc_at": e.get("qc_at", "")
        })
        
    with open(args.output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
