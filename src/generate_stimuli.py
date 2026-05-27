#!/usr/bin/env python3
"""
LLM Image Render Pipeline for Food Stimuli

- Reads: data/food_list_initial_seed.csv
  (columns expected: Category, Food, Natural_vs_transformed, Sweet_vs_savory, Additional Prompt)
- Generates: one photorealistic, brand-free image per Food using OpenAI or Gemini image generation APIs
- Saves: PNG images and per-item JSON metadata under rendered_images/{slug}.png|.json

Usage examples:
  python src/generate_stimuli.py --dry-run --limit 5
  python src/generate_stimuli.py --category Fruit --size 1024 --quality high --seed 42 --n 1
  python src/generate_stimuli.py --resume

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


# --------- Configuration ---------
# PAFID Root is one level up from src/
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "food_list_initial_seed.csv"
OUT_DIR = ROOT / "rendered_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults
DEFAULT_MODEL = "gpt-image-1"
# Default Gemini image model. Must match a model name returned by the Gemini API.
DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"

# Accept common truthy encodings from CSV cells (strings and numeric flags)
TRUTHY = {"1", "1.0", "true", "t", "yes", "y", "on"}


def is_truthy(val: Any) -> bool:
    """Return True for common truthy encodings in CSVs.

    Handles:
      - bool/int: True/1
      - float: 1.0 (and any non-zero float)
      - str: '1', '1.0', 'true', 'yes', etc.
      - pandas/NumPy NaN -> False
    """
    if val is None:
        return False

    # Treat NaN as False (pandas/NumPy)
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass

    # Fast paths for numeric/bool
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
DEFAULT_BACKEND = "gemini"  # "openai" or "gemini"
DEFAULT_SIZE =  "1024x1024"  # 512x512, 1024x1024, etc.
DEFAULT_QUALITY = "standard"  # "standard" or "high" (see pricing)
DEFAULT_BG = "plain matte light grey background"
DEFAULT_LIGHTING = "even, soft studio lighting"
DEFAULT_VIEW = "consistent 45-degree three-quarter view, camera slightly above and in front of a simple plain white plate, angled down toward the centre, centered in frame"

# ---- Supported sizes and normalization ----
VALID_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

def normalize_size(s: str) -> str:
    s = str(s).lower()
    if s == "512x512":
        print("[INFO] Mapping size 512x512 -> 1024x1024 (not supported by API).")
        return "1024x1024"
    if s not in VALID_SIZES:
        print(f"[INFO] Unsupported size '{s}'. Falling back to 1024x1024. Supported: {sorted(VALID_SIZES)}")
        return "1024x1024"
    return s

# --------- Utility helpers ---------
_slugify_re = re.compile(r"[^a-z0-9]+")

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = _slugify_re.sub("-", s)
    s = s.strip("-")
    return s[:80]  # keep filenames manageable

@dataclass
class RenderSpec:
    # Required (non-default) fields must come first for dataclasses
    food: str
    category: str
    nat_vs_trans: str
    sweet_vs_savory: str

    # Optional / defaulted fields
    base_food: Optional[str] = None
    prep_form: Optional[str] = None  # "raw" or "prepared"
    additional_prompt: str = ""
    size: str = DEFAULT_SIZE
    quality: str = DEFAULT_QUALITY
    model: str = DEFAULT_MODEL
    seed: Optional[int] = None
    plate_image: Optional[Path] = None
    out_dir_override: Optional[Path] = None

    @property
    def out_dir(self) -> Path:
        return self.out_dir_override if self.out_dir_override is not None else OUT_DIR

    @property
    def stem(self) -> str:
        return slugify(self.food)

    @property
    def image_path(self) -> Path:
        return self.out_dir / f"{self.stem}.png"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / f"{self.stem}.json"


#
# ---- Serving vessel heuristics ----
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
    """Return an instruction string for when the item is traditionally served in a bowl."""
    if not needs_bowl(spec.food, spec.category):
        return ""

    # If a plate reference image is used, the bowl must visually match that plate.
    if spec.plate_image is not None:
        return (
            "If this food is traditionally served in a bowl, place it in a simple plain white bowl that MATCHES the reference plate style (same white tone and minimal design). "
            "The bowl should sit centered ON TOP of the same reference plate (do not replace the plate). "
            "No patterns, no branding, no colored rims; keep it minimal and consistent."
        )

    # Otherwise keep a consistent minimalist set.
    return (
        "If this food is traditionally served in a bowl, place it in a simple plain white bowl that matches the plate (same plain white tone and minimal design). "
        "The bowl should be centered on the plate. No patterns, no branding, no colored rims."
    )

# --------- Prompt builder ---------
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

    # Explicit preparation control (preferred over relying solely on CSV free-text)
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

    # Granular food realism (e.g., rice, couscous)
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

    # Disambiguation for visually ambiguous foods
    ambiguous_kws = ["taro", "water chestnut"]
    if any(k in spec.food.lower() for k in ambiguous_kws):
        base += " Show one whole item alongside one cut piece to clarify internal structure."

    # Conditional negatives: forbid bowls only when no bowl is requested
    no_bowl_clause = " no bowls." if not vessel_clause else ""

    if prep_clause:
        return f"{base} {prep_clause} {NEGATIVE_PROMPT}{no_bowl_clause}"
    return f"{base} {NEGATIVE_PROMPT}{no_bowl_clause}"
    # Optional flavor: reinforce sweet/savory with subtle styling hints (kept neutral)
    # We avoid explicit taste adjectives to remain purely visual and brand-free.


# --------- OpenAI Client (Images API) ---------
# We keep imports local so the script can still run --dry-run without the package.

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


# --------- Gemini Client (Images API) ---------
def get_gemini_client():
    """
    Return a Gemini client for image generation.

    Requires:
      - google-genai package (pip install -U google-genai)
      - GEMINI_API_KEY environment variable (or other configured auth)
    """
    try:
        from google import genai  # type: ignore
    except Exception as e:
        print("[ERROR] Missing google-genai package. Install with: pip install -U google-genai", file=sys.stderr)
        raise

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it first, e.g., 'export GEMINI_API_KEY=...'"
        )
    return genai.Client(api_key=api_key)


def generate_image_b64_openai(client, prompt: str, size: str, quality: str, model: str, seed: Optional[int]) -> str:
    """
    Call the Images API and return a base64 PNG string.
    Some client versions return data[0].b64_json; others may return a URL.
    We handle both.
    """
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,               # e.g., "1024x1024"
        quality=quality,         # "standard" or "high"
        n=1,
        **({"seed": seed} if seed is not None else {}),
        # NOTE: no response_format here
    )
    item = resp.data[0]

    # Preferred: base64 in the response
    b64 = getattr(item, "b64_json", None) or getattr(item, "image_base64", None)
    if b64:
        return b64

    # Fallback: URL (download it)
    url = getattr(item, "url", None)
    if url:
        import requests, base64 as _b64
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return _b64.b64encode(r.content).decode("utf-8")

    raise RuntimeError("Images API returned neither base64 nor URL.")


# Gemini image generation helper
def generate_image_b64_gemini(client, prompt: str, size: str, quality: str, model: str, seed: Optional[int], plate_image: Optional[Path] = None) -> str:
    """
    Call the Gemini image generation API and return a base64 PNG string.
    """
    try:
        from google.genai import types  # type: ignore
    except Exception:
        types = None  # type: ignore

    parts = []

    if plate_image is not None:
        img_bytes = plate_image.read_bytes()
        suffix = plate_image.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        # Prefer strongly-typed parts if available; otherwise use dict fallback.
        if types is not None and hasattr(types, "Part") and hasattr(types.Part, "from_bytes"):
            try:
                # Newer google-genai versions often require keyword-only args
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            except TypeError:
                # Some versions use different keyword names
                parts.append(types.Part.from_bytes(bytes=img_bytes, mime_type=mime))
        else:
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}})

    if types is not None and hasattr(types, "Part") and hasattr(types.Part, "from_text") and hasattr(types, "Content"):
        parts.append(types.Part.from_text(text=prompt))
        contents = [types.Content(role="user", parts=parts)]
    else:
        # Dict-based fallback
        dict_parts = []
        for p in parts:
            if isinstance(p, dict):
                dict_parts.append(p)
        dict_parts.append({"text": prompt})
        contents = [{"role": "user", "parts": dict_parts}]

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    # ---- Extract first inline image part from common response shapes ----
    # Shape A: response.parts
    for part in getattr(response, "parts", []) or []:
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if isinstance(data, bytes):
                return base64.b64encode(data).decode("utf-8")
            if isinstance(data, str) and data:
                return data

    # Shape B: response.candidates[0].content.parts
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                if isinstance(data, bytes):
                    return base64.b64encode(data).decode("utf-8")
                if isinstance(data, str) and data:
                    return data

    raise RuntimeError("Gemini image generation did not return any inline image data.")


# --------- I/O & rendering loop ---------

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_png_b64(b64: str, out_path: Path) -> None:
    ensure_parent(out_path)
    raw = base64.b64decode(b64)
    out_path.write_bytes(raw)


def write_meta(
    spec: RenderSpec,
    prompt: str,
    backend: str = DEFAULT_BACKEND,
    style_version: str = "v1_uniform_grey_45deg"
) -> None:
    """
    Append metadata for this stimulus to a single master JSON file.

    The master file is a list of entries, each containing the image filename
    and associated stimulus metadata.
    """
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

    # Load existing list if present
    if master_path.exists():
        try:
            with master_path.open("r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(entry)

    with master_path.open("w") as f:
        json.dump(data, f, indent=2)


def backoff_sleep(attempt: int) -> None:
    # exponential backoff with jitter
    delay = min(60, 2 ** attempt) + (0.1 * attempt)
    time.sleep(delay)


def render_one(spec: RenderSpec, client=None, dry_run: bool=False, overwrite: bool=False, backend: str = DEFAULT_BACKEND) -> bool:
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    if spec.image_path.exists() and not overwrite and not spec.additional_prompt:
        # Existing image with no overwrite requested and no extra prompt: keep as is.
        print(f"[SKIP] Already exists: {spec.image_path}")
        return True

    # Build the base prompt from the template
    prompt = build_prompt(spec)

    # Optionally append any additional prompt text from the CSV
    if spec.additional_prompt:
        extra = spec.additional_prompt.strip()
        if extra:
            prompt = f"{prompt} {extra}"

    if dry_run:
        print(f"[DRY] Would render: {spec.food} -> {spec.image_path}")
        print(f"      Prompt: {prompt}")
        write_meta(spec, prompt, backend=backend)  # still write metadata for review
        return True

    if client is None:
        if backend == "openai":
            client = get_openai_client()
        elif backend == "gemini":
            client = get_gemini_client()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # Try with retries for rate limits/transient errors
    for attempt in range(6):
        try:
            if backend == "openai":
                b64 = generate_image_b64_openai(client, prompt, spec.size, spec.quality, spec.model, spec.seed)
            elif backend == "gemini":
                b64 = generate_image_b64_gemini(client, prompt, spec.size, spec.quality, spec.model, spec.seed, plate_image=spec.plate_image)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            write_png_b64(b64, spec.image_path)
            write_meta(spec, prompt, backend=backend)
            print(f"[OK] Rendered: {spec.food} -> {spec.image_path}")
            return True
        except Exception as e:
            print(f"[WARN] Attempt {attempt+1} failed for {spec.food}: {e}")
            if attempt < 5:
                backoff_sleep(attempt)
            else:
                print(f"[FAIL] Giving up on {spec.food}")
                return False



# --------- Master metadata utilities ---------

def load_master_df(master_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the consolidated metadata from stimuli_master.json as a pandas DataFrame.

    If the file does not exist or is malformed, returns an empty DataFrame with
    the expected columns.
    """
    if master_path is None:
        master_path = OUT_DIR / "stimuli_master.json"

    if not master_path.exists():
        print(f"[WARN] Master metadata file not found: {master_path}")
        cols = [
            "image_file",
            "food",
            "category",
            "Natural_vs_transformed",
            "Sweet_vs_savory",
            "prompt",
            "model",
            "size",
            "quality",
            "seed",
            "created",
        ]
        return pd.DataFrame(columns=cols)

    try:
        with master_path.open("r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {master_path}: {e}")
        cols = [
            "image_file",
            "food",
            "category",
            "Natural_vs_transformed",
            "Sweet_vs_savory",
            "prompt",
            "model",
            "size",
            "quality",
            "seed",
            "created",
        ]
        return pd.DataFrame(columns=cols)

    if not isinstance(data, list):
        print(f"[WARN] Expected a list in {master_path}, got {type(data)}; returning empty DataFrame.")
        cols = [
            "image_file",
            "food",
            "category",
            "Natural_vs_transformed",
            "Sweet_vs_savory",
            "prompt",
            "model",
            "size",
            "quality",
            "seed",
            "created",
        ]
        return pd.DataFrame(columns=cols)

    return pd.DataFrame.from_records(data)


def list_missing_images(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Return a DataFrame of rows whose image files are missing on disk.

    If *df* is None, the master DataFrame will be loaded automatically
    from stimuli_master.json.

    The returned DataFrame has the same columns as df, plus a boolean
    column `image_exists` (always False).
    """
    if df is None:
        df = load_master_df()

    if df.empty:
        print("[INFO] Master metadata is empty; nothing to check.")
        return df

    image_paths = [OUT_DIR / fname for fname in df["image_file"]]
    exists_flags = [p.exists() for p in image_paths]

    missing_mask = [not flag for flag in exists_flags]
    missing_df = df[missing_mask].copy()
    missing_df["image_exists"] = False

    if missing_df.empty:
        print("[OK] All images referenced in stimuli_master.json exist on disk.")
    else:
        print(f"[WARN] {len(missing_df)} images referenced in stimuli_master.json are missing on disk.")
        for _, row in missing_df.iterrows():
            print(f"  - {row['food']} -> {row['image_file']}")

    return missing_df

# --------- Main CLI ---------

def load_items(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"Category", "Food", "Natural_vs_transformed", "Sweet_vs_savory"}
    present = set(df.columns)

    # Ensure Category/Food exist first
    if not {"Category", "Food"}.issubset(present):
        missing_cf = {"Category", "Food"} - present
        raise ValueError(f"CSV missing required columns: {missing_cf}")

    changed = False

    if "Natural_vs_transformed" not in present:
        df["Natural_vs_transformed"] = df["Food"].apply(_natural_vs_transformed)
        changed = True
        print("[INFO] Added missing column: Natural_vs_transformed")

    if "Sweet_vs_savory" not in present:
        df["Sweet_vs_savory"] = df.apply(lambda x: _sweet_vs_savory(x["Food"], x["Category"]), axis=1)
        changed = True
        print("[INFO] Added missing column: Sweet_vs_savory")

    if "Additional Prompt" not in df.columns:
        df["Additional Prompt"] = ""
        changed = True
        print("[INFO] Added missing column: Additional Prompt")

    # Normalise any NaN values in Additional Prompt to empty strings so they
    # are treated as "no extra instructions" rather than truthy values.
    if "Additional Prompt" in df.columns:
        if df["Additional Prompt"].isna().any():
            df["Additional Prompt"] = df["Additional Prompt"].fillna("")
            changed = True
            print("[INFO] Normalised NaN values in Additional Prompt column to empty strings.")

    # Clear calories column for later processing elsewhere
    if "Calories_per_100g (kcal)" in df.columns:
        if df["Calories_per_100g (kcal)"].notna().any():
            df["Calories_per_100g (kcal)"] = pd.NA
            changed = True
            print("[INFO] Cleared Calories_per_100g (kcal) column in CSV.")

    if changed:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Updated CSV written with missing columns: {csv_path}")

    # Final check
    missing_final = required - set(df.columns)
    if missing_final:
        raise ValueError(f"CSV still missing columns after augmentation: {missing_final}")

    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate representative images for food stimuli.")
    p.add_argument("--category", type=str, default=None, help="Filter to a single Category (e.g., Fruit)")
    p.add_argument("--food", type=str, default=None, help="Render a single Food item (exact match)")
    p.add_argument("--limit", type=int, default=None, help="Render at most N items")
    p.add_argument("--offset", type=int, default=0, help="Start at row offset")
    p.add_argument("--size", type=str, default=DEFAULT_SIZE, help="Image size, e.g., 512x512 or 1024x1024")
    p.add_argument("--quality", type=str, default=DEFAULT_QUALITY, choices=["standard", "high"], help="Image quality tier")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Image model, e.g., gpt-image-1")
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["openai", "gemini"],
        help="Image backend to use: 'openai' (gpt-image-1) or 'gemini' (Gemini image models, default gemini-2.5-flash-image).",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional seed for determinism (if supported)")
    p.add_argument("--n", type=int, default=1, help="Images per item (currently saves the first; extend as needed)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files (default: skip if PNG exists)")
    p.add_argument("--dry-run", action="store_true", help="Do not call API; just print prompts and write metadata")
    p.add_argument("--one-per-category", action="store_true", help="Render only one item per Category (first alphabetical)")
    p.add_argument(
        "--plate-image",
        type=str,
        default=None,
        help="Optional path to a reference plate image. If provided (Gemini backend), the model will place the food onto this exact plate.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.size = normalize_size(args.size)
    plate_image_path: Optional[Path] = Path(args.plate_image).expanduser().resolve() if args.plate_image else None
    if plate_image_path is not None and not plate_image_path.exists():
        print(f"[ERROR] Plate image not found: {plate_image_path}")
        return 2
    
    backend = args.backend
    requested_model = args.model

    # If using Gemini and the user did not explicitly request a valid Gemini model,
    # fall back to our default v3 image model.
    if backend == "gemini":
        # Treat OpenAI-style names and some legacy/experimental Gemini or Imagen
        # names as placeholders; auto-downgrade to a stable default image model.
        placeholder_models = (
            DEFAULT_MODEL,
            "gpt-image-1",
            "gemini-3-pro-image",
            "gemini-3-pro-image-preview",
            "imagen-4.0-fast-generate-001",
        )
        if requested_model in placeholder_models:
            args.model = DEFAULT_GEMINI_MODEL
            if requested_model != args.model:
                print(f"[INFO] Gemini model auto-selected: requested '{requested_model}' -> using '{args.model}'")

    # Echo the backend and model that will actually be used.
    print(f"[INFO] Using backend '{backend}' with model '{args.model}'")

    try:
        df = load_items(CSV_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    # Keep an unfiltered copy of all items for final integrity checks
    full_df = df.copy()

    # Filtering
    if args.category:
        df = df[df["Category"].str.lower() == args.category.lower()]
    if args.food:
        df = df[df["Food"].str.lower() == args.food.lower()]

    if args.one_per_category:
        # Choose one representative per category (alphabetical by Food for determinism)
        df = (df.sort_values(["Category", "Food"]) 
                .drop_duplicates(subset=["Category"], keep="first")
             )

    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.head(args.limit)

    # Prepare client unless dry-run
    if args.dry_run:
        client = None
    else:
        if backend == "openai":
            client = get_openai_client()
        elif backend == "gemini":
            client = get_gemini_client()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    total = len(df)
    if args.one_per_category:
        cats = ", ".join(sorted(df["Category"].unique()))
        print(f"[INFO] One-per-category mode. Categories: {cats}")
    print(f"[INFO] Items to render: {total}")

    success = 0
    for _, row in df.iterrows():
        # Base spec
        spec = RenderSpec(
            food=row["Food"],
            base_food=(str(row.get("Base Food", "") or "").strip() or None),
            prep_form=(str(row.get("Prep Form", "") or "").strip() or None),
            category=row["Category"],
            nat_vs_trans=row["Natural_vs_transformed"],
            sweet_vs_savory=row["Sweet_vs_savory"],
            additional_prompt=str(row.get("Additional Prompt", "") or ""),
            size=args.size,
            quality=args.quality,
            model=args.model,
            seed=args.seed,
            plate_image=plate_image_path,
        )

        # By default, skip if image exists. 
        # Only re-render if --overwrite is set OR if there is an additional prompt 
        # (which implies the user wants a new version with the new instructions).
        has_extra = bool(spec.additional_prompt.strip())
        if not args.overwrite and spec.image_path.exists():
            if not has_extra:
                print(f"[SKIP] Already exists: {spec.image_path}")
                success += 1
                continue
            else:
                print(f"[INFO] Re-rendering {spec.food} because Additional Prompt is present.")

        ok = render_one(spec, client=client, dry_run=args.dry_run, overwrite=args.overwrite, backend=backend)
        success += int(ok)

    print(f"[DONE] {success}/{total} succeeded")

    # Final integrity check: only run when processing the full dataset
    # (no category/food filters, no offsets/limits, no one-per-category).
    if (
        not args.category
        and not args.food
        and args.offset == 0
        and args.limit is None
        and not args.one_per_category
    ):
        print("[INFO] Running final integrity check on stimuli directory...")
        try:
            # Expected image stems from the full CSV
            expected_stems = {slugify(food) for food in full_df["Food"]}
            png_paths = sorted(p for p in OUT_DIR.glob("*.png"))
            existing_stems = {p.stem for p in png_paths}

            missing = sorted(expected_stems - existing_stems)
            extras = sorted(existing_stems - expected_stems)

            if missing:
                print(f"[WARN] {len(missing)} foods in CSV have no image:")
                for stem in missing:
                    print(f"  - {stem}.png")
            else:
                print("[OK] All foods in CSV have corresponding images.")

            if extras:
                print(f"[INFO] Removing {len(extras)} extra image(s) not in CSV:")
                extra_map = {p.stem: p for p in png_paths}
                for stem in extras:
                    path = extra_map.get(stem)
                    if path and path.exists():
                        print(f"  [DELETE] {path}")
                        path.unlink()
            else:
                print("[OK] No extra images found in stimuli directory.")
        except Exception as e:
            print(f"[WARN] Integrity check failed: {e}")

    return 0 if success == total else 1


if __name__ == "__main__":
    raise SystemExit(main())