#!/usr/bin/env python3
"""
LLM Image Render Pipeline for Food Stimuli

- Reads: data/food_list_initial_seed.csv
  (columns expected: Food)
- Uses Gemini text API to assign Category_WHO_10, Category_Intuitive_7,
  Category_Culinary_9, Natural_vs_transformed, and Transformation_score per food item
- Generates: one photorealistic, brand-free image per Food using OpenAI or Gemini image generation APIs
- Saves: PNG images and per-item JSON metadata under rendered_images/{slug}.png|.json

Usage examples:
  python src/generate_stimuli.py --dry-run --limit 5
  python src/generate_stimuli.py --category Fruit --size 1024 --quality high --seed 42 --n 1

Resumable by default: existing PNGs are skipped unless --overwrite is passed.
If the run is interrupted, simply re-run the same command to continue from where it left off.

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
# Gemini text model used for food classification.
DEFAULT_GEMINI_TEXT_MODEL = "gemini-2.5-flash"

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

    # Labels assigned by Gemini classification (populated before image generation)
    who10_category: str = ""
    intuitive7_category: str = ""
    culinary9_category: str = ""
    nat_vs_trans: str = ""
    transformation_score: int = -1
    nova_category: int = -1

    # Optional / defaulted fields
    additional_prompt: str = ""
    size: str = DEFAULT_SIZE
    quality: str = DEFAULT_QUALITY
    model: str = DEFAULT_MODEL
    seed: Optional[int] = None
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

def needs_bowl(food: str) -> bool:
    f = str(food).lower()
    return any(k in f for k in BOWL_KWS)

def bowl_clause_for(spec: RenderSpec) -> str:
    """Return an instruction string for when the item is traditionally served in a bowl."""
    if not needs_bowl(spec.food):
        return ""
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
    plate_clause = f"Placed on a simple plain white round plate on a {DEFAULT_BG}, {DEFAULT_LIGHTING}."
    vessel_clause = bowl_clause_for(spec)

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
    Return a Gemini client.

    If GOOGLE_GENAI_USE_VERTEXAI=True, uses Vertex AI with Application Default
    Credentials (no API key required). Otherwise falls back to the Gemini
    Developer API using GEMINI_API_KEY.
    """
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


WHO10_CATEGORIES = [
    "Dairy and eggs",
    "Fruits",
    "Vegetables",
    "Confectionery and sweets",
    "Bakery wares and cereals",
    "Meat",
    "Fish",
    "Beverages",
    "Ready-to-eat savories",
    "Prepared foods",
]

INTUITIVE7_CATEGORIES = [
    "Vegetable",
    "Plant protein",
    "Animal protein",
    "Fruit",
    "Dessert",
    "Grain",
    "Dish",
]

CULINARY9_CATEGORIES = [
    "Produce - Sweet",
    "Produce - Savory",
    "Carbohydrates & Staples",
    "Animal Protein",
    "Plant Protein",
    "Dairy",
    "Composite Meals",
    "Desserts & Sweets",
    "Snacks & Savory Junk",
]

# Bump this whenever CLASSIFY_PROMPT_TEMPLATE changes. Entries in
# stimuli_master.json stamped with the current version are skipped on
# --classify-only reruns, making the classification step resumable.
CLASSIFY_PROMPT_VERSION = "v2-2026-06-12-intuitive7-folk-categories"

# --- NOVA classification (Monteiro et al., 2016) ---
# Ported from manuscript/nova_classification_prompt.md in the FoodTriplet-Analysis
# repo (the document used for the original manual batch classification of the
# canonical 350 items). Same definitions and ambiguity rules, applied per food.
# Stamped separately from the main 4-scheme prompt so the two can evolve
# independently without re-triggering each other.
NOVA_PROMPT_VERSION = "v1-2026-06-monteiro-2016"

NOVA_CLASSIFY_PROMPT_TEMPLATE = (
    'Classify the food item "{food}" into a NOVA group (1-4) based on the extent and purpose '
    "of industrial food processing (Monteiro et al., 2016).\n\n"
    "NOVA groups:\n"
    "1 — Unprocessed or minimally processed foods: foods in their natural state, or altered by "
    "processes that do not add substances (drying, roasting, freezing, boiling, pasteurisation, "
    "fermentation). Examples: fresh/frozen fruit and vegetables, plain meat and fish, eggs, plain "
    "milk, plain nuts and seeds, dried legumes, plain rice, oats, plain flour, plain yogurt.\n"
    "2 — Processed culinary ingredients: substances extracted from whole foods and used in cooking, "
    "not typically eaten alone. Examples: vegetable oils, butter, lard, sugar, honey, salt, vinegar, "
    "plain starch. (Unlikely to apply to most plated foods.)\n"
    "3 — Processed foods: foods made by adding salt, sugar, oil, or other Group 2 substances to "
    "Group 1 foods; usually two or three ingredients; the alteration is recognisable. Examples: "
    "canned tomatoes, salted nuts, smoked fish, preserved meats, simple cheeses, plain bread, wine.\n"
    "4 — Ultra-processed foods: industrial formulations with many ingredients, typically including "
    "additives not used in home cooking (emulsifiers, stabilisers, flavourings, colours, sweeteners, "
    "preservatives); no whole-food equivalent. Examples: soft drinks, packaged chips/crisps, instant "
    "noodles, reconstituted meat products, commercial breakfast cereals with additives, packaged "
    "cakes and biscuits, flavoured yogurts, fast food items, candy/confectionery.\n\n"
    "Rules for ambiguous cases:\n"
    "- For foods marked '(prepared)', classify the typical home-cooked preparation (usually still "
    "Group 1 or 3). Do not assume industrial processing simply because a food is cooked.\n"
    "- Dried fruits are typically Group 1 (drying is minimal processing) unless they contain added "
    "sugar or sulphites, in which case Group 3.\n"
    "- Plain legumes cooked or canned are Group 1-3 depending on preparation; classify the typical "
    "ready-to-eat form.\n"
    "- Dishes (e.g. biryani, lasagna, pad thai, sushi) are typically assembled from Group 1-3 "
    "ingredients in a home or restaurant kitchen; classify as Group 3 unless clearly an industrial "
    "product (e.g. instant ramen, fast food).\n"
    "- Classify the most typical, widely available form of the food, not the most processed "
    "possible version.\n\n"
    "Reply in exactly this format — two lines, no explanation:\n"
    "NOVA: <integer 1-4>\n"
    "NOTE: <one short phrase only if borderline/ambiguous, otherwise leave empty>"
)


def classify_nova_gemini(client, food: str, retries: int = 8, model: str = "gemini-2.5-flash"):
    """Classify a single food into a NOVA group (1-4). Returns (group:int|None, note:str)."""
    prompt = NOVA_CLASSIFY_PROMPT_TEMPLATE.format(food=food)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0},
            )
            text = response.text.strip()
            group, note = None, ""
            for line in text.splitlines():
                upper = line.upper()
                if upper.startswith("NOVA:"):
                    digits = re.sub(r"\D", "", line.split(":", 1)[1])
                    if digits:
                        group = max(1, min(4, int(digits[0])))
                elif upper.startswith("NOTE:"):
                    note = line.split(":", 1)[1].strip()
            if group is not None:
                return group, note
            raise ValueError(f"Could not parse NOVA group from: {text[:120]!r}")
        except Exception as e:
            print(f"[WARN] NOVA classification error: {e}. Retrying...")
            import time
            time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)
    return None, ""

CLASSIFY_PROMPT_TEMPLATE = (
    'Classify the food item "{food}" using all four schemes below.\n\n'
    "WHO 10 categories — pick exactly one:\n"
    "- Dairy and eggs: milk, cheese, yogurt, butter, cream, eggs\n"
    "- Fruits: fresh, dried, or minimally processed fruit\n"
    "- Vegetables: fresh, dried, or minimally processed vegetables\n"
    "- Confectionery and sweets: chocolate, candy, cake, cookies, pastries, ice cream, desserts\n"
    "- Bakery wares and cereals: bread, crackers, pasta, rice, oats, cereals, grains\n"
    "- Meat: beef, pork, lamb, poultry, game, and processed meats\n"
    "- Fish: fish and all seafood\n"
    "- Beverages: any drink — juice, alcohol, coffee, tea, soda, water\n"
    "- Ready-to-eat savories: nuts, seeds, crisps, pretzels, popcorn, trail mix — snack foods eaten without further preparation\n"
    "- Prepared foods: multi-ingredient dishes and meals where no single ingredient dominates (e.g. pizza, curry, stir-fry, sushi, salad with multiple components)\n\n"
    "AI Intuitive 7 categories — pick exactly one. These are folk categories: choose the bucket "
    "an average person would put the food in, based on what the food fundamentally IS.\n"
    "- Fruit: Foods recognised and eaten as fruit, whether sweet or sour, fresh or dried — e.g. apples, berries, bananas, citrus, grapes, melon, dried fruit.\n"
    "- Vegetable: Produce used as a vegetable in meals, including starchy roots and tubers, and culinary vegetables that are botanically fruits (e.g. tomato, avocado, cucumber, capsicum/pepper, eggplant).\n"
    "- Grain: Foods made primarily of cereal grains or flour that are not primarily sweet — e.g. rice, pasta, noodles, bread, oats, breakfast cereals, tortillas, plain or savoury baked goods.\n"
    "- Animal protein: Foods derived from a single animal source — meat, poultry, fish, seafood, eggs, and dairy (cheese, yogurt). Includes processed or preserved single-source meats (e.g. sausage, ham, bacon, smoked fish).\n"
    "- Plant protein: Plant-derived foods primarily treated as protein sources — e.g. tofu, tempeh, beans, lentils, chickpeas, edamame, legumes, nuts where they function as a protein-rich food.\n"
    "- Dessert: Foods that are primarily sweet and eaten as a treat — e.g. cake, cookies, sweet pastries, chocolate, ice cream, candy, sweet puddings. The food must be primarily sweet: plain or savoury baked goods are Grain, not Dessert.\n"
    "- Dish: Composite meals combining multiple ingredients into a named recipe — e.g. pizza, curry, stir-fry, burger, sandwich, lasagna, sushi, burrito, salad bowl, rice bowl. Only multi-ingredient prepared meals belong here.\n"
    "Decision rules for Intuitive 7:\n"
    "  * Use the food name, including any parenthetical preparation information.\n"
    "  * Judge by folk/culinary intuition, not botanical or nutritional classification.\n"
    "  * The category follows the dominant base food. Cooking, drying, frying, or otherwise processing a single base food does not change its bucket: grilled salmon is Animal protein; roast potato is Vegetable; toast is Grain; dried apple is Fruit.\n"
    "  * Dessert requires the food to be primarily sweet AND treat-like. Being baked or a pastry is not sufficient.\n"
    "  * Dish is only for multi-ingredient composite meals. Neither simple preparation nor being a snack makes something a Dish.\n"
    "  * Savoury snack foods are classified by their dominant base ingredient (e.g. grain-based crackers are Grain; potato-based crisps are Vegetable; nut mixes are Plant protein).\n"
    "  * Dairy foods belong to Animal protein (the only animal-derived bucket).\n\n"
    "Culinary 9 categories — pick exactly one:\n"
    "- Produce - Sweet: Sweet fruits typically eaten as fruit/snacks/dessert. Examples: apple, banana, berries, orange, grapes, melon, mango.\n"
    "- Produce - Savory: Vegetables and savory produce used in meals/salads. Include botanically-fruit-but-culinarily-savory items. Examples: broccoli, carrot, lettuce, spinach, tomato, avocado, cucumber, capsicum/pepper, eggplant.\n"
    "- Carbohydrates & Staples: Starchy or grain-based meal bases, especially when plain or single-dominant. Examples: rice, plain pasta, plain noodles, bread, oats, tortillas, potatoes, sweet potato, corn, porridge.\n"
    "- Animal Protein: Meat, poultry, fish, seafood, eggs when the food is a single-dominant animal-protein item. Examples: steak, chicken breast, salmon fillet, prawns/shrimp, boiled egg.\n"
    "- Plant Protein: Plant foods treated primarily as protein sources. Examples: tofu, tempeh, lentils, beans, chickpeas, edamame, nuts if functioning as protein-rich food.\n"
    "- Dairy: Dairy products when not clearly desserts or composite meals. Examples: milk, cheese, plain yoghurt/yogurt, cottage cheese.\n"
    "- Composite Meals: Multi-ingredient prepared foods or named dishes/meals. Examples: pizza, curry, stir-fry, burger, sandwich, sushi, burrito, lasagna, salad bowl, rice bowl, pasta dish with sauce/multiple ingredients.\n"
    "- Desserts & Sweets: Sweet treats and desserts, regardless of whether they are baked, dairy-based, frozen, or grain-based. Examples: cake, cookies, pastries, donuts, chocolate, ice cream, candy/sweets, pudding.\n"
    "- Snacks & Savory Junk: Savory snack foods, often processed, salty, crunchy, or eaten outside meals. Examples: potato chips/crisps, popcorn, pretzels, crackers, corn chips, savory snack mix.\n"
    "Decision rules for Culinary 9:\n"
    "  * Use culinary/common-sense categories, not botanical categories.\n"
    "  * Simple cooking alone does not make something Composite Meals.\n"
    "  * Multi-ingredient named meals should be Composite Meals unless they are clearly desserts/sweets.\n"
    "  * Sweet treats should always be Desserts & Sweets.\n"
    "  * Savory snack foods should be Snacks & Savory Junk, even if grain- or potato-based.\n"
    "  * Dairy-based desserts, such as ice cream, should be Desserts & Sweets, not Dairy.\n\n"
    "Natural vs Transformed — pick exactly one:\n"
    "- Natural: the food is still visually and conceptually identifiable as a single biological food source, "
    "even if it has undergone minor preparation such as washing, peeling, cutting, drying, freezing, or simple cooking\n"
    "- Transformed: the food has been substantially altered from its original biological source through combination, "
    "reforming, refining, fermentation, baking, frying, industrial processing, or preparation into a dish or product\n\n"
    "Transformation score — assign a single integer from 0 to 100 using this scale:\n"
    "  0–10:  Whole, raw, unmodified (e.g. apple, banana, carrot, tomato)\n"
    " 10–25:  Minimally prepared but source-identifiable (e.g. sliced fruit, peeled orange, roasted nuts, dried fruit)\n"
    " 25–40:  Simply cooked single-source food (e.g. boiled egg, steamed vegetables, grilled fish, plain rice)\n"
    " 40–55:  Mechanically altered single-source food (e.g. mashed potato, minced meat, fruit puree, smoothie)\n"
    " 55–70:  Biochemically or structurally transformed (e.g. cheese, yoghurt, tofu, bread, pasta)\n"
    " 70–85:  Composite prepared dish (e.g. soup, curry, sandwich, sushi, dumplings)\n"
    " 85–100: Highly transformed or manufactured food (e.g. pizza, cake, sausage, chocolate bar, cereal, candy)\n\n"
    "Reply in exactly this format — five lines, no explanation:\n"
    "WHO10: <category>\n"
    "INTUITIVE7: <category>\n"
    "CULINARY9: <category>\n"
    "NAT_TRANS: <Natural or Transformed>\n"
    "SCORE: <integer 0-100>"
)


NAT_TRANS_VALUES = {"Natural", "Transformed"}


def classify_food_gemini(client, food: str, retries: int = 8, model: str = "gemini-2.5-flash") -> tuple:
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(food=food)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0},  # deterministic-as-possible classification
            )
            text = response.text.strip()
            who10, intuitive7, culinary9, nat_trans, score_str = "unknown", "unknown", "unknown", "unknown", ""
            for line in text.splitlines():
                upper = line.upper()
                if upper.startswith("WHO10:"):
                    who10 = line.split(":", 1)[1].strip()
                elif upper.startswith("INTUITIVE7:"):
                    intuitive7 = line.split(":", 1)[1].strip()
                elif upper.startswith("CULINARY9:"):
                    culinary9 = line.split(":", 1)[1].strip()
                elif upper.startswith("NAT_TRANS:"):
                    nat_trans = line.split(":", 1)[1].strip()
                elif upper.startswith("SCORE:"):
                    score_str = line.split(":", 1)[1].strip()

            try:
                score = int(re.sub(r'\D', '', score_str))
                score = max(0, min(100, score))  # clamp to 0–100
            except (ValueError, TypeError):
                score = -1
            
            return who10, intuitive7, culinary9, nat_trans, score
        except Exception as e:
            print(f"[WARN] Classification API error: {e}. Retrying...")
            import time
            time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)
    return "unknown", "unknown", "unknown", "unknown", -1

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
def generate_image_b64_gemini(client, prompt: str, size: str, quality: str, model: str, seed: Optional[int]) -> str:
    """
    Call the Gemini image generation API and return a base64 PNG string.
    """
    try:
        from google.genai import types  # type: ignore
    except Exception:
        types = None  # type: ignore

    if types is not None and hasattr(types, "Part") and hasattr(types.Part, "from_text") and hasattr(types, "Content"):
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    else:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

    print(f"      [DEBUG] Calling Gemini API (model: {model})...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
    )
    print("      [DEBUG] API response received.")

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
        "base_food": spec.food,
        "Category_WHO_10": spec.who10_category,
        "Category_Intuitive_7": spec.intuitive7_category,
        "Category_Culinary_9": spec.culinary9_category,
        "Natural_vs_transformed": spec.nat_vs_trans,
        "Transformation_score": spec.transformation_score,
        "Category_NOVA_4": spec.nova_category if spec.nova_category != -1 else None,
        "prompt": prompt,
        "model": spec.model,
        "size": spec.size,
        "quality": spec.quality,
        "seed": spec.seed,
        "created": int(time.time()),
        "source": f"ai-{backend}",
        "style_version": style_version,
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

    # Also write a per-item JSON file for redundant tracking
    with spec.meta_path.open("w") as f:
        json.dump(entry, f, indent=2)


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
                b64 = generate_image_b64_gemini(client, prompt, spec.size, spec.quality, spec.model, spec.seed)
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
            "Category_WHO_10",
            "Category_Intuitive_7",
            "Category_Culinary_9",
            "Natural_vs_transformed",
            "Transformation_score",
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
            "Category_WHO_10",
            "Category_Intuitive_7",
            "Category_Culinary_9",
            "Natural_vs_transformed",
            "Transformation_score",
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
            "Category_WHO_10",
            "Category_Intuitive_7",
            "Category_Culinary_9",
            "Natural_vs_transformed",
            "Transformation_score",
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

# --------- Classify-only utilities ---------

def run_classify_only(df: pd.DataFrame, client, out_dir: Path = OUT_DIR, text_model: str = DEFAULT_GEMINI_TEXT_MODEL) -> int:
    """
    Classify all foods in df using Gemini text API and inject the four
    classification labels directly into stimuli_master.json. Does not
    touch any images.

    Matching between df rows and master entries is done by normalised food name.
    Returns 0 on full success, 1 if any items failed classification.
    """
    master_path = out_dir / "stimuli_master.json"

    # Load existing master (may already have QC fields, ratings, etc. — preserve all)
    if master_path.exists():
        try:
            with master_path.open("r") as f:
                master = json.load(f)
            if not isinstance(master, list):
                master = []
        except Exception as e:
            print(f"[ERROR] Could not read stimuli_master.json: {e}")
            return 1
    else:
        master = []

    # Build a lookup from normalised food name -> index in master list
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    master_index: Dict[str, int] = {}
    for idx, entry in enumerate(master):
        key = norm(entry.get("food", entry.get("Food", "")))
        master_index[key] = idx

    # Back up existing master ONCE before any incremental writes
    if master_path.exists():
        bak_path = master_path.with_suffix(".json.bak")
        import shutil
        shutil.copy2(master_path, bak_path)
        print(f"[INFO] Backed up existing master to {bak_path.name}")

    def _save():
        with master_path.open("w") as f:
            json.dump(master, f, indent=2)

    LABEL_FIELDS = ["Category_WHO_10", "Category_Intuitive_7", "Category_Culinary_9",
                    "Natural_vs_transformed", "Transformation_score"]

    def _already_done(entry: dict) -> bool:
        """Entry was classified with the CURRENT prompt version and has valid labels."""
        if entry.get("classify_prompt_version") != CLASSIFY_PROMPT_VERSION:
            return False
        return all(str(entry.get(k, "")).strip() not in ("", "unknown", "None") for k in LABEL_FIELDS)

    def _nova_done(entry: dict) -> bool:
        """Entry has a valid NOVA group under the current NOVA prompt version."""
        if entry.get("nova_prompt_version") != NOVA_PROMPT_VERSION:
            return False
        try:
            return int(entry.get("Category_NOVA_4")) in (1, 2, 3, 4)
        except (TypeError, ValueError):
            return False

    import time as _time
    total = len(df)
    failures = 0
    done = 0
    nova_done_n = 0
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        food = row["Food"]
        key = norm(food)
        entry = master[master_index[key]] if key in master_index else None

        main_done = entry is not None and _already_done(entry)
        nova_ok = entry is not None and _nova_done(entry)

        # Resumable: skip foods fully classified under the current prompt versions
        if main_done and nova_ok:
            skipped += 1
            continue

        # --- Main 4-scheme classification ---
        if not main_done:
            print(f"[INFO] ({i}/{total}) Classifying: {food}")
            who10, intuitive7, culinary9, nat_trans, score = classify_food_gemini(client, food, model=text_model)

            # Treat exhausted-retry sentinel ('unknown') as a failure: do NOT write
            # 'unknown' labels into the master — leave the entry untouched so a
            # rerun retries it.
            if (not who10) or who10 == "unknown" or intuitive7 == "unknown":
                print(f"[FAIL] Could not classify: {food} (will retry on next run)")
                failures += 1
                continue

            print(f"      [CLASSIFY] WHO10='{who10}' | Intuitive7='{intuitive7}' | Culinary9='{culinary9}' | {nat_trans} | Score={score}")

            if entry is None:
                # Food not yet in master (e.g. newly added) — create minimal entry
                print(f"      [INFO] No existing master entry for '{food}' — creating new entry.")
                entry = {"food": food, "image_file": f"{slugify(food)}.png"}
                master.append(entry)
                master_index[key] = len(master) - 1

            entry["Category_WHO_10"] = who10
            entry["Category_Intuitive_7"] = intuitive7
            entry["Category_Culinary_9"] = culinary9
            entry["Natural_vs_transformed"] = nat_trans
            entry["Transformation_score"] = score
            entry["classify_prompt_version"] = CLASSIFY_PROMPT_VERSION
            if "Category_Simple_6" in entry:
                del entry["Category_Simple_6"]
            done += 1
            _save()  # checkpoint after every API result
            _time.sleep(1.5)

        # --- NOVA classification (Monteiro et al., 2016) ---
        if not nova_ok:
            print(f"[INFO] ({i}/{total}) NOVA-classifying: {food}")
            nova_group, nova_note = classify_nova_gemini(client, food, model=text_model)
            if nova_group is None:
                print(f"[FAIL] Could not NOVA-classify: {food} (will retry on next run)")
                failures += 1
                continue
            print(f"      [NOVA] Group {nova_group}" + (f" ({nova_note})" if nova_note else ""))
            entry["Category_NOVA_4"] = int(nova_group)
            if nova_note:
                entry["nova_notes"] = nova_note
            entry["nova_prompt_version"] = NOVA_PROMPT_VERSION
            entry["nova_source"] = text_model
            nova_done_n += 1
            _save()  # checkpoint after every API result
            _time.sleep(1.5)

    _save()
    if skipped:
        print(f"[INFO] Skipped {skipped} food(s) already fully classified "
              f"(prompt '{CLASSIFY_PROMPT_VERSION}', NOVA '{NOVA_PROMPT_VERSION}').")
    print(f"[DONE] Classified {done} food(s) + {nova_done_n} NOVA label(s) this run, "
          f"{failures} failure(s). Master updated: {master_path}")
    if failures:
        print("[HINT] Re-run the same command to retry failed items — completed ones will be skipped.")
    return 0 if failures == 0 else 1


# --------- Main CLI ---------

def load_items(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    if "Food" not in df.columns:
        raise ValueError("CSV missing required column: Food")

    changed = False

    # Additional Prompt is optional — add as empty in-memory column if absent,
    # without writing it back to the seed CSV.
    if "Additional Prompt" not in df.columns:
        df["Additional Prompt"] = ""

    # Clear calories column for later processing elsewhere
    if "Calories_per_100g (kcal)" in df.columns:
        if df["Calories_per_100g (kcal)"].notna().any():
            df["Calories_per_100g (kcal)"] = pd.NA
            changed = True
            print("[INFO] Cleared Calories_per_100g (kcal) column in CSV.")

    if changed:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Updated CSV written: {csv_path}")

    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate representative images for food stimuli.")
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
    p.add_argument("--classify-only", action="store_true", help="Run Gemini text classification only — update per-item JSON metadata without generating or touching images")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.size = normalize_size(args.size)
    
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

    # --- Classify-only mode: classify foods and update metadata, skip image generation ---
    if args.classify_only:
        # Always force a text model for classification to avoid 429 Image model quotas
        text_model = DEFAULT_GEMINI_TEXT_MODEL
        print(f"[INFO] Classify-only mode — using text model '{text_model}'")
        gemini_client = get_gemini_client()
        classify_df = df.copy()
        if args.food:
            classify_df = classify_df[classify_df["Food"].str.lower() == args.food.lower()]
        if args.offset:
            classify_df = classify_df.iloc[args.offset:]
        if args.limit is not None:
            classify_df = classify_df.head(args.limit)
        return run_classify_only(classify_df, gemini_client, text_model=text_model)

    # Keep an unfiltered copy of all items for final integrity checks
    full_df = df.copy()

    # Filtering
    if args.food:
        df = df[df["Food"].str.lower() == args.food.lower()]

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

    # Pre-calculate skips to keep logging clean
    to_render = []
    skipped_count = 0
    for _, row in df.iterrows():
        spec = RenderSpec(
            food=row["Food"],
            additional_prompt=str(row.get("Additional Prompt", "") or ""),
        )
        has_extra = bool(spec.additional_prompt.strip())
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

        # Classify food across all schemes via Gemini text API
        if not args.dry_run and backend == "gemini":
            who10, intuitive7, culinary9, nat_trans, score = classify_food_gemini(client, food_name)
            print(f"      [CLASSIFY] WHO10='{who10}' | Intuitive7='{intuitive7}' | Culinary9='{culinary9}' | {nat_trans} | Score={score}")
            nova_group, nova_note = classify_nova_gemini(client, food_name)
            nova_group = nova_group if nova_group is not None else -1
            print(f"      [NOVA] Group {nova_group}" + (f" ({nova_note})" if nova_note else ""))
        else:
            who10, intuitive7, culinary9, nat_trans, score = "", "", "", "", -1
            nova_group = -1

        # Base spec
        spec = RenderSpec(
            food=row["Food"],
            who10_category=who10,
            intuitive7_category=intuitive7,
            culinary9_category=culinary9,
            nat_vs_trans=nat_trans,
            transformation_score=score,
            nova_category=nova_group,
            additional_prompt=str(row.get("Additional Prompt", "") or ""),
            size=args.size,
            quality=args.quality,
            model=args.model,
            seed=args.seed,
        )

        if not args.overwrite and spec.image_path.exists() and bool(spec.additional_prompt.strip()):
            print(f"      [INFO] Re-rendering {spec.food} because Additional Prompt is present.")

        ok = render_one(spec, client=client, dry_run=args.dry_run, overwrite=args.overwrite, backend=backend)
        success += int(ok)

    print(f"[DONE] {success}/{total} succeeded")

    # Final integrity check: only run when processing the full dataset
    # (no category/food filters, no offsets/limits, no one-per-category).
    if (
        not args.food
        and args.offset == 0
        and args.limit is None
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