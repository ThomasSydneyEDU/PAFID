#!/usr/bin/env python3
"""
PAFID Stimulus Image Generation (Stage 1, Part 2)

Loads a food list CSV, builds highly standardized visual prompts using structural
and culinary heuristics, and generates high-resolution images using Gemini
(gemini-3-pro-image-preview) or OpenAI DALL-E.

Updates and appends generation details directly into stimuli_master.json, preserving
any classification metadata generated in Stage 1, Part 1 (classify_food.py).

Requires:
  pip install -U google-genai openai pandas requests
  export GEMINI_API_KEY=... or export OPENAI_API_KEY=...
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import requests

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
CSV_PATH = ROOT / "data" / "food_list_initial_seed.csv"
OUT_DIR = ROOT / "rendered_images"

DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "standard"
DEFAULT_BACKEND = "gemini"

# Default models
DEFAULT_MODEL = "gpt-image-1"  # OpenAI / DALL-E default
DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"

VALID_SIZES = {"256x256", "512x512", "1024x1024"}

STYLE_VERSION = "v3"


@dataclass
class RenderSpec:
    food: str
    who10_category: str = ""
    intuitive7_category: str = ""
    culinary9_category: str = ""
    nat_vs_trans: str = ""
    transformation_score: int = -1
    nova_category: int = -1
    additional_prompt: str = ""
    size: str = "1024x1024"
    quality: str = "standard"
    model: str = DEFAULT_GEMINI_MODEL
    out_dir_override: Optional[Path] = None
    stimulus_set: Optional[str] = None
    seed: Optional[int] = None

    @property
    def out_dir(self) -> Path:
        return self.out_dir_override if self.out_dir_override is not None else OUT_DIR

    @property
    def stem(self) -> str:
        return slugify(self.food)

    @property
    def image_path(self) -> Path:
        return self.out_dir / f"{self.stem}.png"


def is_truthy(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "y", "on")


def normalize_size(s: str) -> str:
    s = str(s).strip().lower()
    if s in VALID_SIZES:
        return s
    if s == "1024":
        return "1024x1024"
    if s == "512":
        return "512x512"
    if s == "256":
        return "256x256"
    print(f"[INFO] Unsupported size '{s}'. Falling back to 1024x1024. Supported: {sorted(VALID_SIZES)}")
    return "1024x1024"


def slugify(name: str) -> str:
    s = str(name).strip().replace(" ", "-")
    s = re.sub(r"[^\w\-]", "", s)
    return s.lower()


# --------- Client builders ---------

def get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print("[ERROR] Missing openai package. Install with: pip install openai", file=sys.stderr)
        raise
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it first, e.g., 'export OPENAI_API_KEY=sk-...'"
        )
    return OpenAI()


def get_gemini_client():
    try:
        from google import genai  # type: ignore
    except Exception as e:
        print("[ERROR] Missing google-genai package. Install with: pip install -U google-genai", file=sys.stderr)
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
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it first, e.g., 'export GEMINI_API_KEY=...'"
        )
    print("[INFO] Using Gemini Developer API with GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# --------- Prompt Heuristics ---------

PROMPT_TEMPLATE = (
    "A food photograph of {food}. "
    "{plate_clause} "
    "{vessel_clause}"
    "The food is presented cleanly in a single standard adult serving size, "
    "centered in the frame. Soft, natural studio lighting from the side, "
    "casting soft shadows. The shot is captured at a standard three-quarter "
    "downward camera view (40-45 degrees), focusing solely on the plated food. "
    "The background is a completely plain, solid, flat matte light grey surface, "
    "with no table settings, no utensils, no napkins, no textures, and no other items "
    "visible. Professional, sharp, high-resolution culinary photography."
)

PLATE_STANDARD = "The food is served on a plain, solid white, round plate."

NEGATIVE_PROMPT = (
    "logos, text, branding, words, signs, cutlery, fork, knife, spoon, hands, "
    "people, busy background, wooden table, tablecloth, napkin, decorations, "
    "multiple servings, buffet, messy presentation, extreme close-up, flat lay, "
    "overhead shot, black plate, patterned plate, square plate, bowl"
)


def needs_bowl(food_name: str) -> bool:
    """
    Return True if the food is a soup, stew, cereal, porridge, or liquid/semi-liquid dish
    that is traditionally served in a bowl.
    """
    fn = food_name.lower()
    keywords = [
        "soup", "ramen", "laksa", "stew", "curry", "dahl", "porridge", "oatmeal",
        "muesli", "cereal", "yogurt", "yoghurts", "custard", "jelly", "pudding",
        "sorbet", "gelato", "ice-cream", "ice cream", "chili con carne", "goulash",
        "acai", "smoothie-bowl", "smoothie bowl"
    ]
    return any(kw in fn for keywords_list in [keywords] for kw in keywords_list)


def bowl_clause_for(spec: RenderSpec) -> str:
    if needs_bowl(spec.food):
        return "The food is served inside a plain, solid white, round bowl that sits centered on the white plate. "
    return ""


def build_prompt(spec: RenderSpec) -> str:
    food = spec.food
    
    # 1. Base vessel/plate assignments
    plate_clause = PLATE_STANDARD
    vessel_clause = bowl_clause_for(spec)

    # 2. Granular food heuristic (loose mound presentation)
    granular_keywords = [
        "rice", "couscous", "quinoa", "sprouts", "lentils", "peas", "berries",
        "nuts", "seeds", "beans", "popcorn", "crisps", "chips", "sweets", "candy"
    ]
    granular_clause = ""
    if any(kw in food.lower() for kw in granular_keywords):
        granular_clause = (
            "The items are presented in a natural, loose, neat mound in the center. "
            "Individual pieces, textures, and details are clearly distinct, not mashed together. "
        )

    # 3. Structural ambiguity heuristic (whole + cut piece)
    ambiguous_keywords = ["taro", "water chestnut", "truffle", "fig", "apricot", "plum"]
    disambig_clause = ""
    if any(kw in food.lower() for kw in ambiguous_keywords):
        disambig_clause = (
            "The presentation includes one whole, uncut item next to one sliced or halved piece "
            "to show both the exterior texture and the internal structure and flesh clearly. "
        )

    # 4. Handle manual prompt overrides/additions
    additional = spec.additional_prompt.strip()
    if additional:
        if not additional.endswith("."):
            additional += "."
        # If the manual prompt starts with 'overwrite:', replace the base template entirely
        if additional.lower().startswith("overwrite:"):
            return additional[len("overwrite:"):].strip()
        additional = " " + additional
    
    base_prompt = PROMPT_TEMPLATE.format(
        food=food,
        plate_clause=plate_clause,
        vessel_clause=vessel_clause
    )
    
    final_prompt = base_prompt + additional + " " + granular_clause + disambig_clause
    return final_prompt.strip()


# --------- Master File Synchronization ---------

def write_png_b64(b64_str: str, output_path: Path):
    img_data = base64.b64decode(b64_str)
    with open(output_path, "wb") as f:
        f.write(img_data)


def write_meta(spec: RenderSpec, prompt: str, backend: str, style_version: str):
    """
    Append or update metadata for this stimulus in stimuli_master.json, preserving
    any pre-existing metadata such as classifications or evaluations.
    """
    master_path = spec.out_dir / "stimuli_master.json"
    master: List[Dict[str, Any]] = []

    if master_path.exists():
        try:
            with master_path.open("r", encoding="utf-8") as f:
                master = json.load(f)
            if not isinstance(master, list):
                master = []
        except Exception as e:
            print(f"[WARN] Could not read stimuli_master.json, starting fresh: {e}")
            master = []

    # Match by food name (normalized)
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    key = norm(spec.food)
    entry: Optional[Dict[str, Any]] = None
    entry_idx = -1

    for idx, e in enumerate(master):
        if norm(e.get("food", e.get("Food", ""))) == key:
            entry = e
            entry_idx = idx
            break

    if entry is None:
        entry = {
            "food": spec.food,
            "image_file": spec.image_path.name,
            "base_food": spec.food
        }
    else:
        # Ensure base keys are set
        entry["image_file"] = spec.image_path.name
        if "food" not in entry:
            entry["food"] = spec.food

    # Inject rendering credentials/parameters
    entry["prompt"] = prompt
    entry["model"] = spec.model
    entry["size"] = spec.size
    entry["quality"] = spec.quality
    if spec.seed is not None:
        entry["seed"] = spec.seed
    entry["created"] = int(time.time())
    entry["source"] = f"ai-{backend}"
    entry["style_version"] = style_version
    if spec.stimulus_set is not None:
        entry["stimulus_set"] = spec.stimulus_set

    # Write back any specified classification parameters if they were provided in the spec
    if spec.who10_category:
        entry["Category_WHO_10"] = spec.who10_category
    if spec.intuitive7_category:
        entry["Category_Intuitive_7"] = spec.intuitive7_category
    if spec.culinary9_category:
        entry["Category_Culinary_9"] = spec.culinary9_category
    if spec.nat_vs_trans:
        entry["Natural_vs_transformed"] = spec.nat_vs_trans
    if spec.transformation_score != -1:
        entry["Transformation_score"] = spec.transformation_score
    if spec.nova_category != -1:
        entry["Category_NOVA_4"] = spec.nova_category

    if entry_idx != -1:
        master[entry_idx] = entry
    else:
        master.append(entry)

    # Save to JSON
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)


# --------- Image Generation Engines ---------

def generate_image_b64_openai(client, prompt: str, size: str, quality: str, model: str, seed: Optional[int]) -> str:
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
        **({"seed": seed} if seed is not None else {}),
    )
    item = resp.data[0]
    b64 = getattr(item, "b64_json", None) or getattr(item, "image_base64", None)
    if b64:
        return b64

    url = getattr(item, "url", None)
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        import base64 as _b64
        return _b64.b64encode(r.content).decode("utf-8")

    raise RuntimeError("Images API returned neither base64 nor URL.")


def generate_image_b64_gemini(client, prompt: str, size: str, quality: str, model: str, seed: Optional[int]) -> str:
    try:
        from google.genai import types  # type: ignore
    except Exception:
        types = None  # type: ignore

    if types is not None and hasattr(types, "Part") and hasattr(types.Part, "from_text") and hasattr(types, "Content"):
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    else:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

    print(f"      [DEBUG] Calling Gemini API (model: {model})...")
    
    # Structure of Imagen requests in Google GenAI client SDK:
    # client.models.generate_images(...)
    aspect_ratio = "1:1"  # Default aspect ratio
    person_generation = "DONT_ALLOW"

    try:
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio=aspect_ratio,
                person_generation=person_generation,
                **({"seed": seed} if seed is not None else {})
            )
        )
        # Extract base64
        for image_obj in response.generated_images:
            b64_bytes = image_obj.image.image_bytes
            return base64.b64encode(b64_bytes).decode("utf-8")
        raise RuntimeError("No images returned by Gemini API")
    except AttributeError:
        # Fallback if using developer SDK wrappers
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={"temperature": 0.4}
        )
        txt = response.text or ""
        raise RuntimeError(f"Gemini model returned text instead of image: {txt[:200]}")


def render_one(spec: RenderSpec, client, dry_run: bool = False, overwrite: bool = False, backend: str = "gemini") -> bool:
    img_path = spec.image_path
    food = spec.food

    # Skip if file already exists
    if not overwrite and img_path.exists():
        print(f"      [SKIP] Image already exists: {img_path.name}")
        return True

    prompt = build_prompt(spec)
    print(f"      [PROMPT] \"{prompt}\"")

    if dry_run:
        print(f"      [DRY-RUN] Would render '{food}' -> {img_path.name}")
        # Even in dry-run, we write empty/mock values into stimuli_master.json for layout tracking
        write_meta(spec, prompt, backend, STYLE_VERSION)
        return True

    print(f"      [RENDER] Generating image for '{food}' via {backend}...")
    try:
        t0 = time.time()
        if backend == "openai":
            b64 = generate_image_b64_openai(
                client, prompt, size=spec.size, quality=spec.quality, model=spec.model, seed=spec.seed
            )
        elif backend == "gemini":
            b64 = generate_image_b64_gemini(
                client, prompt, size=spec.size, quality=spec.quality, model=spec.model, seed=spec.seed
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Save image
        write_png_b64(b64, img_path)
        print(f"      [OK] Wrote image to: {img_path.name} (took {time.time()-t0:.1f}s)")

        # Sync metadata
        write_meta(spec, prompt, backend, STYLE_VERSION)
        return True
    except Exception as e:
        print(f"      [ERROR] Rendering failed for '{food}': {e}", file=sys.stderr)
        return False


def load_items(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Food list CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Food" not in df.columns:
        raise ValueError(f"Food list CSV must contain a 'Food' column. Columns: {list(df.columns)}")
    return df


def main():
    p = argparse.ArgumentParser(description="Stage 1, Part 2: AI Plated Food Image Generation")
    p.add_argument("--food-list", type=str, default=None, help="Path to food list CSV")
    p.add_argument("--output-dir", type=str, default=None, help="Output folder for PNGs and master JSON")
    p.add_argument("--food", type=str, default=None, help="Render a single Food item (exact match)")
    p.add_argument("--limit", type=int, default=None, help="Render at most N items")
    p.add_argument("--offset", type=int, default=0, help="Start at row offset")
    p.add_argument("--size", type=str, default=DEFAULT_SIZE, help="Image size, e.g., 512x512 or 1024x1024")
    p.add_argument("--quality", type=str, default=DEFAULT_QUALITY, choices=["standard", "high"], help="Image quality tier")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Image model, e.g., gpt-image-1")
    p.add_argument(
        "--backend", type=str, default=DEFAULT_BACKEND, choices=["openai", "gemini"],
        help="Model provider backend: 'openai' for DALL-E, 'gemini' for Vertex AI / Imagen"
    )
    p.add_argument("--seed", type=int, default=None, help="Optional seed for determinism")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    p.add_argument("--dry-run", action="store_true", help="Do not call image API; just print prompts and sync metadata")
    p.add_argument("--stimulus-set", type=str, default="pafid_v1",
                   help="Provenance label stored in stimuli_master.json and the dynamic CSV. "
                        "Defaults to 'pafid_v1' (the canonical set); extension runs pass their own label "
                        "(e.g. foodspace_extension_2026).")
    p.add_argument("--extra-prompt", type=str, default=None, help="Additional prompt instructions to append (from editorial review)")
    args = p.parse_args()

    args.size = normalize_size(args.size)
    backend = args.backend
    requested_model = args.model

    if backend == "gemini":
        placeholder_models = (
            DEFAULT_MODEL,
            "gpt-image-1",
            "gemini-3-pro-image",
            "imagen-4.0-fast-generate-001",
        )
        if requested_model in placeholder_models:
            args.model = DEFAULT_GEMINI_MODEL

    csv_path = Path(args.food_list).expanduser().resolve() if args.food_list else CSV_PATH
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_items(csv_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    # Filtering
    if args.food:
        df = df[df["Food"].str.lower() == args.food.lower()]
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.head(args.limit)

    # Initialize client unless dry-run
    if args.dry_run:
        client = None
    else:
        if backend == "openai":
            client = get_openai_client()
        elif backend == "gemini":
            client = get_gemini_client()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # Load master entries for synchronization
    master_path = out_dir / "stimuli_master.json"
    master_by_food = {}
    if master_path.exists():
        try:
            with master_path.open("r", encoding="utf-8") as f:
                m_data = json.load(f)
            if isinstance(m_data, list):
                master_by_food = {re.sub(r"[^a-z0-9]", "", str(e.get("food", "")).lower()): e for e in m_data}
        except Exception as e:
            print(f"[WARN] Could not load stimuli_master.json for category sync: {e}")

    to_render = []
    skipped_count = 0
    for _, row in df.iterrows():
        food = row["Food"]
        has_extra = bool(str(row.get("Additional Prompt", "")).strip()) or bool(args.extra_prompt)
        spec = RenderSpec(food=food, out_dir_override=out_dir)
        if not args.overwrite and spec.image_path.exists() and not has_extra:
            skipped_count += 1
        else:
            to_render.append(row)

    if skipped_count > 0:
        print(f"[INFO] {skipped_count} images skipped as they already exist.")

    active_count = len(to_render)
    print(f"[INFO] Items to render: {active_count}")

    success = skipped_count
    for i, row in enumerate(to_render, 1):
        food_name = row["Food"]
        print(f"[INFO] ({i}/{active_count}) Processing: {food_name}")

        # Synchronize classification categories if they exist in the master JSON
        key = re.sub(r"[^a-z0-9]", "", food_name.lower())
        e = master_by_food.get(key, {})

        additional_text = str(row.get("Additional Prompt", "") or "").strip()
        if args.extra_prompt:
            additional_text = f"{additional_text} {args.extra_prompt.strip()}".strip()

        spec = RenderSpec(
            food=food_name,
            who10_category=e.get("Category_WHO_10", ""),
            intuitive7_category=e.get("Category_Intuitive_7", ""),
            culinary9_category=e.get("Category_Culinary_9", ""),
            nat_vs_trans=e.get("Natural_vs_transformed", ""),
            transformation_score=e.get("Transformation_score", -1),
            nova_category=e.get("Category_NOVA_4", -1),
            additional_prompt=additional_text,
            size=args.size,
            quality=args.quality,
            model=args.model,
            out_dir_override=out_dir if args.output_dir else None,
            stimulus_set=args.stimulus_set,
            seed=args.seed,
        )

        ok = render_one(spec, client=client, dry_run=args.dry_run, overwrite=args.overwrite, backend=backend)
        success += int(ok)

    print(f"[DONE] {success}/{len(df)} succeeded")
    return 0 if success == len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
