#!/usr/bin/env python3
"""
LLM Image Render Pipeline for Food Stimuli

- Reads: data/food_list_initial_seed.csv (default) or --csv
- Generates: one photorealistic, brand-free image per Food using OpenAI or Gemini image generation APIs
- Saves: PNG images and per-item JSON metadata under rendered_images/ (default) or --out-dir

Usage examples:
  python src/generate_stimuli.py --dry-run --limit 5
  python src/generate_stimuli.py --category Fruit --size 1024 --quality high --seed 42 --n 1
  python src/generate_stimuli.py --csv path/to/my_foods.csv --out-dir path/to/images

Env vars required:
  GEMINI_API_KEY   # your project API key
  OPENAI_API_KEY   # only if using OpenAI backend
"""

from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

# ---- Auto-label helpers (used if certain columns are missing) ----
PROCESSED_KWS = [
    "bread", "cake", "cookie", "pie", "pasta", "burger", "fried", "roast",
    "baked", "cooked", "soup", "salad", "sauce", "chips", "bar", "sandwich",
    "ice cream", "chocolate", "jam", "juice", "soda", "beer", "milkshake",
    "smoothie", "muffin", "donut", "pudding", "casserole", "stew", "curry",
    "taco", "pizza", "burrito", "wrap", "noodle", "dumpling", "rice", "lasagna",
    "bowl", "mixed",
]

SWEET_CATS = {"Dessert", "Fruit"}
SWEET_KWS = [
    "sweet", "chocolate", "cake", "cookie", "pie", "ice cream", "candy",
    "jam", "honey", "sugar", "pudding", "dessert", "syrup", "fruit",
    "donut", "brownie", "cupcake", "muffin", "milkshake", "smoothie",
    "acai", "baklava",
]


def _natural_vs_transformed(food: str) -> str:
    f = str(food).lower()
    return "Transformed" if any(k in f for k in PROCESSED_KWS) else "Natural"

def _sweet_vs_savory(food: str, category: str) -> str:
    f = str(food).lower()
    return "Sweet" if (category in SWEET_CATS or any(k in f for k in SWEET_KWS)) else "Savory"


# --------- Configuration Defaults ---------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = ROOT / "data" / "food_list_initial_seed.csv"
DEFAULT_OUT_DIR = ROOT / "rendered_images"

DEFAULT_MODEL = "gpt-image-1"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"

TRUTHY = {"1", "1.0", "true", "t", "yes", "y", "on"}

def is_truthy(val: Any) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, float):
        return val != 0.0
    s = str(val).strip().lower()
    if not s:
        return False
    return s in TRUTHY

DEFAULT_BACKEND = "gemini"
DEFAULT_SIZE =  "1024x1024"
DEFAULT_QUALITY = "standard"
DEFAULT_BG = "plain matte light grey background"
DEFAULT_LIGHTING = "even, soft studio lighting"
DEFAULT_VIEW = "consistent 45-degree three-quarter view, camera slightly above and in front of a simple plain white plate, angled down toward the centre, centered in frame"

VALID_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

def normalize_size(s: str) -> str:
    s = str(s).lower()
    if s == "512x512":
        return "1024x1024"
    if s not in VALID_SIZES:
        return "1024x1024"
    return s

_slugify_re = re.compile(r"[^a-z0-9]+")

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = _slugify_re.sub("-", s)
    s = s.strip("-")
    return s[:80]

@dataclass
class RenderSpec:
    food: str
    category: str
    nat_vs_trans: str
    sweet_vs_savory: str
    out_dir: Path
    base_food: Optional[str] = None
    prep_form: Optional[str] = None
    additional_prompt: str = ""
    size: str = DEFAULT_SIZE
    quality: str = DEFAULT_QUALITY
    model: str = DEFAULT_MODEL
    seed: Optional[int] = None
    plate_image: Optional[Path] = None

    @property
    def stem(self) -> str:
        return slugify(self.food)

    @property
    def image_path(self) -> Path:
        return self.out_dir / f"{self.stem}.png"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / f"{self.stem}.json"


BOWL_KWS = [
    "soup", "stew", "ramen", "pho", "laksa", "miso soup", "tom yum", "tom yum soup",
    "noodles", "udon", "vermicelli", "cereal", "oatmeal", "porridge", "risotto",
    "curry", "dahl", "custard", "pudding", "gelato", "sorbet", "ice cream", "bowl",
]

def needs_bowl(food: str, category: str) -> bool:
    f = str(food).lower()
    c = str(category).lower()
    if c in {"dessert"} and any(k in f for k in ["ice cream", "gelato", "sorbet", "custard", "pudding"]):
        return True
    if any(k in f for k in BOWL_KWS):
        return True
    return False

def bowl_clause_for(spec: RenderSpec) -> str:
    if not needs_bowl(spec.food, spec.category):
        return ""
    if spec.plate_image is not None:
        return (
            "If this food is traditionally served in a bowl, place it in a simple plain white bowl that MATCHES the reference plate style (same white tone and minimal design). "
            "The bowl should sit centered ON TOP of the same reference plate (do not replace the plate). "
            "No patterns, no branding, no colored rims; keep it minimal and consistent."
        )
    return (
        "If this food is traditionally served in a bowl, place it in a simple plain white bowl that matches the plate (same plain white tone and minimal design). "
        "The bowl should be centered on the plate. No patterns, no branding, no colored rims."
    )

PROMPT_TEMPLATE = (
    "A photorealistic studio photograph of {food}, single subject, no packaging, "
    "no brand labels or text. {plate_clause} {vessel_clause} "
    "Render the image on a square 1024x1024 pixel canvas (1:1 aspect ratio). "
    "Do not change the aspect ratio or return a rectangular image. "
    "Center the plate within the square frame with equal margins on all sides; "
    "ensure the entire plate (and bowl, if present) is fully visible with no cropping. "
    "Maintain a FIXED three-quarter camera geometry corresponding to approximately a 40–45° downward tilt from vertical (classic food photography angle). "
    "The round plate must appear as an ellipse with a consistent major-to-minor axis ratio across all images. "
    "Do NOT vary camera tilt, camera height, focal length, or perspective to better show the food. "
    "The camera is positioned above and slightly in front of the plate, angled down toward the centre "
    "(not top-down, not side-on). "
    "Show a typical single-serving portion size of {food} as commonly served to one adult "
    "(not a tiny sample and not an oversized platter). "
    "Keep the portion neatly arranged and fully visible on the plate with some rim still visible "
    "(do not cover the entire plate). "
    "The {food} is shown in a typical ready-to-eat, edible form appropriate to the requested preparation. "
    "The food must be physically grounded on the plate with natural contact shadows and occlusion at the base; "
    "avoid white halos, cutout edges, pasted-on appearance, or floating pieces. "
    "Ensure consistent lighting direction and shadow softness between the food and the plate. "
    "Use realistic surface micro-texture (e.g., pores, fibers, grain separation) and avoid waxy or plastic-looking surfaces. "
    "Ensure physically plausible scale and proportions (e.g., slices must match the size of the whole item). "
    "High detail, natural colors, minimal shadows, sharp focus, stock-photo style."
)

NEGATIVE_PROMPT = (
    "No logos, no brand names, no watermarks, no text, no human hands, "
    "no patterned plates, no cutting boards, no trays, no complex props, "
    "no busy backgrounds, no steam to indicate temperature, "
    "no multiple items, no multiple servings, no family-style platters, "
    "no collages, no extreme side views, no fisheye or distorted perspective, "
    "no camera angle changes, no perspective drift, no adaptive framing"
)


def build_prompt(spec: RenderSpec) -> str:
    if spec.plate_image is not None:
        plate_clause = (
            "Using the provided reference image of an empty plain white plate, place the food on that EXACT SAME plate. "
            "The plate must be IDENTICAL to the reference: do not change the plate shape, rim width, rim height, curvature, color temperature, texture, gloss, lighting reflections, shadows, background, camera angle, or framing. "
            "Preserve the apparent ellipse of the plate exactly as in the reference image; this ellipse DEFINES the viewing angle. "
            "Do not adjust camera tilt, height, or perspective to accommodate the food. "
            "Do not stylize, redraw, resize, rotate, crop, warp, or substitute the plate. "
            "Only add the food item onto the plate while preserving all plate pixels and appearance. "
            "Maintain the same camera angle and framing; keep some rim visible."
        )
    else:
        plate_clause = f"Placed on a simple plain white round plate on a {DEFAULT_BG}, {DEFAULT_LIGHTING}."

    vessel_clause = bowl_clause_for(spec)
    prep = (spec.prep_form or "").strip().lower() if spec.prep_form is not None else ""
    if prep == "raw":
        prep_clause = (
            "Render the food RAW and UNCOOKED (no steam/roast/grill marks, no sauteing). "
            "No sauces or heavy seasoning; if cutting is typical, show simple raw cuts (e.g., slices/florets)."
        )
    elif prep == "prepared":
        prep_clause = (
            "Render the food in a typical PLAIN PREPARED form (cooked if commonly cooked): "
            "lightly steamed, roasted, boiled, or sauteed as appropriate. "
            "No sauces; minimal visible seasoning (avoid garnishes)."
        )
    else:
        prep_clause = ""

    granular_kws = ["rice", "couscous", "quinoa", "bulgur", "millet"]
    granular_clause = ""
    if any(k in spec.food.lower() for k in granular_kws):
        granular_clause = (
            " Depict granular foods as a natural loose mound with visible individual grains and micro-shadowing; "
            "avoid smooth, perfectly domed, or overly compact piles."
        )

    base = PROMPT_TEMPLATE.format(
        food=spec.food,
        plate_clause=plate_clause,
        vessel_clause=vessel_clause,
    ) + granular_clause

    ambiguous_kws = ["taro", "water chestnut"]
    if any(k in spec.food.lower() for k in ambiguous_kws):
        base += " Show one whole item alongside one cut piece to clarify internal structure."

    no_bowl_clause = " no bowls." if not vessel_clause else ""
    if prep_clause:
        return f"{base} {prep_clause} {NEGATIVE_PROMPT}{no_bowl_clause}"
    return f"{base} {NEGATIVE_PROMPT}{no_bowl_clause}"


def get_openai_client():
    from openai import OpenAI
    return OpenAI()

def get_gemini_client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

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
        import requests, base64 as _b64
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return _b64.b64encode(r.content).decode("utf-8")
    raise RuntimeError("Images API returned neither base64 nor URL.")

def generate_image_b64_gemini(client, prompt: str, size: str, quality: str, model: str, seed: Optional[int], plate_image: Optional[Path] = None) -> str:
    from google.genai import types
    parts = []
    if plate_image is not None:
        img_bytes = plate_image.read_bytes()
        mime = "image/png" if plate_image.suffix.lower() == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    parts.append(types.Part.from_text(text=prompt))
    contents = [types.Content(role="user", parts=parts)]
    response = client.models.generate_content(model=model, contents=contents)
    for cand in getattr(response, "candidates", []) or []:
        for part in getattr(cand.content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                if isinstance(data, bytes):
                    return base64.b64encode(data).decode("utf-8")
                if isinstance(data, str) and data:
                    return data
    raise RuntimeError("Gemini image generation did not return any inline image data.")

def write_png_b64(b64: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(b64)
    out_path.write_bytes(raw)

def write_meta(spec: RenderSpec, prompt: str, backend: str, style_version: str = "v1_uniform_grey_45deg") -> None:
    master_path = spec.out_dir / "stimuli_master.json"
    entry: Dict[str, Any] = {
        "image_file": spec.image_path.name,
        "food": spec.food,
        "base_food": spec.base_food if spec.base_food is not None else spec.food,
        "prep_form": spec.prep_form,
        "category": spec.category,
        "Natural_vs_transformed": spec.nat_vs_trans,
        "Sweet_vs_savory": spec.sweet_vs_savory,
        "prompt": prompt,
        "model": spec.model,
        "size": spec.size,
        "quality": spec.quality,
        "seed": spec.seed,
        "created": int(time.time()),
        "source": f"ai-{backend}",
        "style_version": style_version,
        "plate_reference": str(spec.plate_image) if spec.plate_image is not None else None,
    }
    if master_path.exists():
        try:
            with master_path.open("r") as f:
                data = json.load(f)
            if not isinstance(data, list): data = []
        except Exception: data = []
    else: data = []
    data.append(entry)
    with master_path.open("w") as f: json.dump(data, f, indent=2)
    with spec.meta_path.open("w") as f: json.dump(entry, f, indent=2)

def render_one(spec: RenderSpec, client=None, dry_run: bool=False, overwrite: bool=False, backend: str = DEFAULT_BACKEND) -> bool:
    if spec.image_path.exists() and not overwrite and not spec.additional_prompt:
        return True
    prompt = build_prompt(spec)
    if spec.additional_prompt:
        prompt = f"{prompt} {spec.additional_prompt}"
    if dry_run:
        write_meta(spec, prompt, backend=backend)
        return True
    if client is None:
        if backend == "openai": client = get_openai_client()
        elif backend == "gemini": client = get_gemini_client()
    for attempt in range(6):
        try:
            if backend == "openai":
                b64 = generate_image_b64_openai(client, prompt, spec.size, spec.quality, spec.model, spec.seed)
            elif backend == "gemini":
                b64 = generate_image_b64_gemini(client, prompt, spec.size, spec.quality, spec.model, spec.seed, plate_image=spec.plate_image)
            write_png_b64(b64, spec.image_path)
            write_meta(spec, prompt, backend=backend)
            return True
        except Exception as e:
            time.sleep(min(60, 2 ** attempt))
    return False

def load_items(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists(): raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"Category", "Food", "Natural_vs_transformed", "Sweet_vs_savory"}
    present = set(df.columns)
    changed = False
    if "Natural_vs_transformed" not in present:
        df["Natural_vs_transformed"] = df["Food"].apply(_natural_vs_transformed)
        changed = True
    if "Sweet_vs_savory" not in present:
        df["Sweet_vs_savory"] = df.apply(lambda x: _sweet_vs_savory(x["Food"], x["Category"]), axis=1)
        changed = True
    if "Additional Prompt" not in df.columns:
        df["Additional Prompt"] = ""; changed = True
    if "Additional Prompt" in df.columns and df["Additional Prompt"].isna().any():
        df["Additional Prompt"] = df["Additional Prompt"].fillna(""); changed = True
    if changed: df.to_csv(csv_path, index=False)
    return df

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate representative images for food stimuli.")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV_PATH), help="Path to input CSV")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Path to output directory")
    p.add_argument("--category", type=str, default=None)
    p.add_argument("--food", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--size", type=str, default=DEFAULT_SIZE)
    p.add_argument("--quality", type=str, default=DEFAULT_QUALITY, choices=["standard", "high"])
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--backend", type=str, default=DEFAULT_BACKEND, choices=["openai", "gemini"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--one-per-category", action="store_true")
    p.add_argument("--plate-image", type=str, default=None)
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.size = normalize_size(args.size)
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plate_image_path = Path(args.plate_image).expanduser().resolve() if args.plate_image else None
    
    if args.backend == "gemini":
        placeholder_models = (DEFAULT_MODEL, "gpt-image-1", "gemini-3-pro-image", "imagen-4.0-fast-generate-001")
        if args.model in placeholder_models:
            args.model = DEFAULT_GEMINI_MODEL

    print(f"[INFO] Backend: {args.backend} | Model: {args.model}")
    print(f"[INFO] CSV: {csv_path} | Output: {out_dir}")

    try:
        df = load_items(csv_path)
    except Exception as e:
        print(f"[ERROR] {e}"); return 2

    full_df = df.copy()
    if args.category: df = df[df["Category"].str.lower() == args.category.lower()]
    if args.food: df = df[df["Food"].str.lower() == args.food.lower()]
    if args.one_per_category: df = df.sort_values(["Category", "Food"]).drop_duplicates(subset=["Category"], keep="first")
    if args.offset: df = df.iloc[args.offset:]
    if args.limit is not None: df = df.head(args.limit)

    client = None if args.dry_run else (get_openai_client() if args.backend == "openai" else get_gemini_client())
    
    total = len(df)
    success = 0
    for i, row in enumerate(df.itertuples(), 1):
        print(f"[INFO] ({i}/{total}) Processing: {row.Food}")
        spec = RenderSpec(
            food=row.Food,
            category=row.Category,
            nat_vs_trans=row.Natural_vs_transformed,
            sweet_vs_savory=row.Sweet_vs_savory,
            out_dir=out_dir,
            base_food=getattr(row, "Base_Food", row.Food),
            prep_form=getattr(row, "Prep_Form", None),
            additional_prompt=getattr(row, "_6", ""), # Additional Prompt is typically col 6
            size=args.size, quality=args.quality, model=args.model, seed=args.seed, plate_image=plate_image_path,
        )
        ok = render_one(spec, client=client, dry_run=args.dry_run, overwrite=args.overwrite, backend=args.backend)
        success += int(ok)

    print(f"[DONE] {success}/{total} succeeded")
    return 0 if success == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
