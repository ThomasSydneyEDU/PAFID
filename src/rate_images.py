#!/usr/bin/env python3
import argparse, base64, csv, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from google import genai

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

def backoff_sleep(attempt: int) -> None:
    time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)

def build_similarity_prompt(guessed, actual):
    return f"Compare guess '{guessed}' vs actual '{actual}'.\nRate similarity 0-100. Return JSON: " + '{"similarity_score_0_100": N}'

def get_blind_ratings(client, path, model):
    img = Path(path).read_bytes()
    mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
    contents = [{"role": "user", "parts": [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(img).decode("utf-8")}},
        {"text": "Identify the food visible in the image and provide 0-100 ratings for its healthiness, calorie density, and flavor profile (sweet, salt, sour, bitter, savory, fatty, spicy). Return the results as a JSON object."}
    ]}]
    for attempt in range(5):
        try:
            res = client.models.generate_content(model=model, contents=contents, config={"response_mime_type": "application/json"})
            return json.loads(res.text.strip())
        except: backoff_sleep(attempt)
    raise RuntimeError("Blind rating failed.")

def get_similarity(client, g, a, model):
    for attempt in range(3):
        try:
            res = client.models.generate_content(model=model, contents=build_similarity_prompt(g, a), config={"response_mime_type": "application/json"})
            return float(json.loads(res.text.strip())["similarity_score_0_100"])
        except: backoff_sleep(attempt)
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli-dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.csv) as f: rows = list(csv.DictReader(f))
    client = get_gemini_client()
    
    fieldnames = list(rows[0].keys())
    new_fields = ["blind_observed_food", "blind_guess_similarity", "blind_ai_calorie_density", "blind_ai_healthiness", 
                  "blind_ai_sweetness", "blind_ai_saltiness", "blind_ai_sourness", "blind_ai_bitterness", 
                  "blind_ai_savoriness", "blind_ai_fattiness", "blind_ai_spiciness", "blind_model"]
    for f in new_fields:
        if f not in fieldnames: fieldnames.append(f)
    
    done = 0
    for r in rows:
        if args.limit and done >= args.limit: break
        if r.get("blind_observed_food"): continue
        
        img = Path(args.stimuli_dir) / r["filename"]
        if not img.is_file(): continue
        
        print(f"Blind Rating: {r['Food']}")
        try:
            res = get_blind_ratings(client, img, args.model)
            sim = get_similarity(client, res.get('observed_food', ''), r['Food'], args.model)
            
            r.update({
                "blind_observed_food": res.get("observed_food"),
                "blind_guess_similarity": sim,
                "blind_ai_calorie_density": res.get("calorie_density_0_100"),
                "blind_ai_healthiness": res.get("healthiness_0_100"),
                "blind_ai_sweetness": res.get("sweetness_0_100"),
                "blind_ai_saltiness": res.get("saltiness_0_100"),
                "blind_ai_sourness": res.get("sourness_0_100"),
                "blind_ai_bitterness": res.get("bitterness_0_100"),
                "blind_ai_savoriness": res.get("savoriness_0_100"),
                "blind_ai_fattiness": res.get("fatty_flavour_0_100"),
                "blind_ai_spiciness": res.get("spiciness_0_100"),
                "blind_model": args.model
            })
            done += 1
            if done % 10 == 0:
                with open(args.csv, 'w') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows)
        except Exception as e: print(f"Error: {e}")

    with open(args.csv, 'w') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

if __name__ == '__main__': main()
