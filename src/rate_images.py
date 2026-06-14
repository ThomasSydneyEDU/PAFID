#!/usr/bin/env python3
"""
Blind AI ratings for plated food stimuli.

Sends each image to Gemini with NO food label ("blind"): the model must
identify the food and provide 0-100 subjective judgements. A second call
scores the similarity between the model's guess and the true food name.

Reads & writes (in-place, resumable): data/Foodpictures_information_dynamic.csv
Rows that already have `blind_observed_food` are skipped, so re-running the
pipeline only rates NEW stimuli.

Columns written:
    blind_observed_food, blind_guess_similarity,
    blind_ai_calorie_density, blind_ai_healthiness, blind_ai_sweetness,
    blind_ai_saltiness, blind_ai_sourness, blind_ai_bitterness,
    blind_ai_savoriness, blind_ai_fattiness, blind_ai_spiciness, blind_model

Requires:
  pip install -U google-genai
  export GEMINI_API_KEY=...

Example:
  python src/rate_images.py --stimuli-dir rendered_images/
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DEFAULT_CSV = ROOT / "data" / "Foodpictures_information_dynamic.csv"

# Default matches the model used for the canonical 350-item database so that
# ratings for new stimuli are comparable to the existing ones.
DEFAULT_MODEL = "gemini-2.5-pro"

BLIND_FIELDS = [
    "blind_observed_food", "blind_guess_similarity",
    "blind_ai_calorie_density", "blind_ai_healthiness",
    "blind_ai_sweetness", "blind_ai_saltiness", "blind_ai_sourness",
    "blind_ai_bitterness", "blind_ai_savoriness", "blind_ai_fattiness",
    "blind_ai_spiciness", "blind_model",
]

# ---------------- Prompts (documented in README; keep in sync) ----------------

SYSTEM_INSTRUCTIONS = (
    "You are a neutral observer providing visual assessments of food stimuli. "
    "Be factual. Do not mention brands. Do not add opinions beyond the requested ratings. "
    "For any 0–100 ratings (calories/health/flavour), provide best-effort *subjective judgements* "
    "based only on what is visually inferable from the image and typical culinary expectations. "
    "These are not objective measurements. For 'fatty', judge fatty-tasting "
    "richness/oiliness/creaminess (mouthfeel), not fat content. If highly uncertain, use 50."
)

BLIND_PROMPT = """You will be shown an image of food on a plate.

Tasks:
1) Identify the food visible in the image.
2) Provide 0-100 ratings as *subjective judgements* of perceived flavour intensity and health attributes (best-effort inferences from visible cues + typical culinary expectations).

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
- Do NOT invent brands or extra items.
- If highly uncertain about a rating, use 50.
"""


def build_similarity_prompt(guessed: str, actual: str) -> str:
    return (
        f"Compare guess '{guessed}' vs actual '{actual}'.\n"
        "Rate similarity 0-100. Return JSON: " + '{"similarity_score_0_100": N}'
    )


# ---------------- Gemini helpers ----------------

def get_gemini_client():
    """Vertex AI (preferred) or AI Studio API key — same logic as run_qc.py."""
    try:
        from google import genai  # noqa: deferred import so --help etc. work without the package
    except ImportError:
        print("[ERROR] Missing google-genai. Install with: pip install -U google-genai", file=sys.stderr)
        raise

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"true", "1", "yes"}
    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. Export it first, e.g. 'export GOOGLE_CLOUD_PROJECT=usyd-llm'"
            )
        print(f"[INFO] Using Gemini via Vertex AI: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini credentials: set GOOGLE_GENAI_USE_VERTEXAI=True (+ GOOGLE_CLOUD_PROJECT) "
            "for Vertex AI, or export GEMINI_API_KEY for AI Studio."
        )
    return genai.Client(api_key=api_key)


def backoff_sleep(attempt: int) -> None:
    time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)


def _strip_json_fences(s: str) -> str:
    """Remove Markdown code fences / stray text around a JSON object."""
    t = (s or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if "{" in t and "}" in t:
        i, j = t.find("{"), t.rfind("}")
        if j > i:
            t = t[i:j + 1].strip()
    return t


def _clamp_0_100(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return max(0.0, min(100.0, v))


def get_blind_ratings(client, path: Path, model: str, max_attempts: int = 5) -> Dict[str, Any]:
    img = Path(path).read_bytes()
    mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
    contents = [{"role": "user", "parts": [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(img).decode("utf-8")}},
        {"text": BLIND_PROMPT},
    ]}]
    config = {
        "system_instruction": SYSTEM_INSTRUCTIONS,
        "response_mime_type": "application/json",
        "temperature": 0.2,
    }
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            res = client.models.generate_content(model=model, contents=contents, config=config)
            parsed = json.loads(_strip_json_fences(res.text))
            if not isinstance(parsed, dict):
                raise ValueError("Model did not return a JSON object.")
            return parsed
        except Exception as e:
            last_err = e
            print(f"[WARN] Blind rating attempt {attempt+1} failed for {Path(path).name}: {e}")
            if attempt < max_attempts - 1:
                backoff_sleep(attempt)
    raise RuntimeError(f"Blind rating failed for {Path(path).name}: {last_err}")


def get_similarity(client, guessed: str, actual: str, model: str, max_attempts: int = 3) -> float:
    for attempt in range(max_attempts):
        try:
            res = client.models.generate_content(
                model=model,
                contents=build_similarity_prompt(guessed, actual),
                config={"response_mime_type": "application/json"},
            )
            return float(json.loads(_strip_json_fences(res.text))["similarity_score_0_100"])
        except Exception as e:
            print(f"[WARN] Similarity attempt {attempt+1} failed ('{guessed}' vs '{actual}'): {e}")
            if attempt < max_attempts - 1:
                backoff_sleep(attempt)
    return 0.0


# ---------------- CSV I/O ----------------

def write_rows(csv_path: Path, fieldnames, rows) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Blind AI ratings (image only, no food label) with Gemini")
    parser.add_argument("--stimuli-dir", type=str, required=True,
                        help="Folder containing the stimulus images (e.g. rendered_images/)")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help=f"CSV to read/update in-place (default: {DEFAULT_CSV})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL}; matches the canonical database)")
    parser.add_argument("--limit", type=int, default=None, help="Rate at most N new items")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-rate rows that already have blind ratings")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # utf-8-sig: run_qc.py writes this CSV with a BOM
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[ERROR] No rows in {csv_path}", file=sys.stderr)
        return 1

    fieldnames = list(rows[0].keys())
    for fld in BLIND_FIELDS:
        if fld not in fieldnames:
            fieldnames.append(fld)

    todo = [r for r in rows if args.overwrite or not (r.get("blind_observed_food") or "").strip()]
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Rows: {len(rows)} total, {len(todo)} to rate (overwrite={args.overwrite})")

    if not todo:
        print("[DONE] Nothing to do — all rows already have blind ratings.")
        return 0

    client = get_gemini_client()
    done = 0

    for r in todo:
        if args.limit and done >= args.limit:
            break

        img = Path(args.stimuli_dir) / str(r.get("filename", ""))
        if not r.get("filename") or not img.is_file():
            print(f"[WARN] Image not found, skipping: {img}")
            continue

        food_name = (r.get("food") or r.get("Food") or "").strip()
        print(f"Blind rating: {food_name or r['filename']}")
        try:
            res = get_blind_ratings(client, img, args.model)
            guessed = str(res.get("observed_food", "")).strip()
            sim = get_similarity(client, guessed, food_name, args.model)

            r.update({
                "blind_observed_food": guessed,
                "blind_guess_similarity": sim,
                "blind_ai_calorie_density": _clamp_0_100(res.get("calorie_density_0_100")),
                "blind_ai_healthiness": _clamp_0_100(res.get("healthiness_0_100")),
                "blind_ai_sweetness": _clamp_0_100(res.get("sweetness_0_100")),
                "blind_ai_saltiness": _clamp_0_100(res.get("saltiness_0_100")),
                "blind_ai_sourness": _clamp_0_100(res.get("sourness_0_100")),
                "blind_ai_bitterness": _clamp_0_100(res.get("bitterness_0_100")),
                "blind_ai_savoriness": _clamp_0_100(res.get("savoriness_0_100")),
                "blind_ai_fattiness": _clamp_0_100(res.get("fatty_flavour_0_100")),
                "blind_ai_spiciness": _clamp_0_100(res.get("spiciness_0_100")),
                "blind_model": args.model,
            })
            done += 1
            if done % 10 == 0:
                write_rows(csv_path, fieldnames, rows)
                print(f"[INFO] Progress saved: {done}/{len(todo)}")
        except Exception as e:
            print(f"[FAIL] {r.get('filename')}: {e}")

    write_rows(csv_path, fieldnames, rows)
    print(f"[DONE] Rated {done} item(s). Updated: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
