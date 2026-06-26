#!/usr/bin/env python3
"""
PAFID — Non-Food Object Foil Generator

Generates photorealistic images of non-food objects rendered on the same
plain white plate / grey background as the PAFID food stimuli.  These images
are intended for use as visual control stimuli (foils) in food-related
experiments.  They are NOT part of the core PAFID data release.

Usage examples
--------------
  # Render the built-in foil list (default)
  python generate_nonfood_stimuli.py --dry-run
  python generate_nonfood_stimuli.py --resume

  # Render a single object
  python generate_nonfood_stimuli.py --object "river pebbles"

  # Use a custom CSV (columns: Category, Food)
  python generate_nonfood_stimuli.py --csv my_objects.csv

Env vars required
-----------------
  GEMINI_API_KEY   — for Gemini backend (default)
  OPENAI_API_KEY   — for OpenAI backend

Notes
-----
- Prompts mirror generate_stimuli.py: plain white plate, grey background,
  40–45° camera, studio lighting, 1024×1024 canvas.
- Output is written to data/rendered_images_nonfood/ (separate from food images).
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]          # repo root
OUT_DIR = ROOT / "data" / "rendered_images_nonfood"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional custom CSV (columns: Category, Food).  Falls back to built-in list.
DEFAULT_CSV_PATH = ROOT / "data" / "nonfood_object_list.csv"

# ── Built-in foil list ─────────────────────────────────────────────────────────
BUILTIN_OBJECTS: list[dict[str, str]] = [
    {"Category": "NonFoodObject", "Food": "river pebbles"},
    {"Category": "NonFoodObject", "Food": "seashells"},
    {"Category": "NonFoodObject", "Food": "soap cubes"},
    {"Category": "NonFoodObject", "Food": "candle wax curls"},
    {"Category": "NonFoodObject", "Food": "sponge cubes"},
    {"Category": "NonFoodObject", "Food": "craft clay dough blobs"},
    {"Category": "NonFoodObject", "Food": "bark chips"},
    {"Category": "NonFoodObject", "Food": "twigs like fries"},
    {"Category": "NonFoodObject", "Food": "moss clump greens"},
    {"Category": "NonFoodObject", "Food": "potpourri petals"},
    {"Category": "NonFoodObject", "Food": "marbles"},
    {"Category": "NonFoodObject", "Food": "screws and washers"},
]

# ── Model / backend defaults ──────────────────────────────────────────────────
DEFAULT_MODEL        = "gpt-image-1"
DEFAULT_GEMINI_MODEL = "gemini-3-pro-image-preview"
DEFAULT_BACKEND      = "gemini"
DEFAULT_SIZE         = "1024x1024"
DEFAULT_QUALITY      = "standard"
DEFAULT_BG           = "plain matte light grey background"
DEFAULT_LIGHTING     = "even, soft studio lighting"

VALID_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}


def normalize_size(s: str) -> str:
    s = str(s).lower()
    if s == "512x512":
        print("[INFO] Mapping size 512x512 -> 1024x1024 (not supported by API).")
        return "1024x1024"
    if s not in VALID_SIZES:
        print(f"[INFO] Unsupported size '{s}'. Falling back to 1024x1024.")
        return "1024x1024"
    return s


# ── Helpers ───────────────────────────────────────────────────────────────────
_slugify_re = re.compile(r"[^a-z0-9]+")

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = _slugify_re.sub("-", s)
    return s.strip("-")[:80]


@dataclass
class RenderSpec:
    # Required fields
    food: str
    category: str

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


# ── Prompt builder ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "A photorealistic studio photograph of {food} arranged on a plate. "
    "IMPORTANT: The subject is a NON-FOOD INANIMATE OBJECT. It is NOT edible and must NOT look like food. "
    "No packaging, no brand labels or text. "
    "Placed on a simple plain white round plate on a {bg}, {lighting}. "
    "Render the image on a square 1024x1024 pixel canvas (1:1 aspect ratio). "
    "Do not change the aspect ratio or return a rectangular image. "
    "Center the plate within the square frame with equal margins on all sides; "
    "ensure the entire plate is fully visible with no cropping. "
    "Maintain a FIXED three-quarter camera geometry corresponding to approximately a 40–45° downward "
    "tilt from vertical (classic food photography angle). "
    "The round plate must appear as an ellipse with a consistent major-to-minor axis ratio across all images. "
    "Do NOT vary camera tilt, camera height, focal length, or perspective to better show the object. "
    "The camera is positioned above and slightly in front of the plate, angled down toward the centre "
    "(not top-down, not side-on). "
    "Show a typical plate-sized arrangement of the object, neatly arranged and fully visible with "
    "some rim still visible (do not cover the entire plate). "
    "The object must be physically grounded on the plate with natural contact shadows and occlusion "
    "at the base; avoid white halos, cutout edges, pasted-on appearance, or floating pieces. "
    "Ensure consistent lighting direction and shadow softness between the object and the plate. "
    "Use realistic surface micro-texture and avoid waxy or plastic-looking surfaces unless the "
    "object itself is plastic or wax. "
    "High detail, natural colors, minimal shadows, sharp focus, stock-photo style."
)

NEGATIVE_PROMPT = (
    "No logos, no brand names, no watermarks, no text, no human hands, "
    "no patterned plates, no cutting boards, no trays, no complex props, "
    "no busy backgrounds, no steam, no multiple servings, no family-style platters, "
    "no collages, no extreme side views, no fisheye or distorted perspective, "
    "no camera angle changes, no perspective drift, no adaptive framing, no bowls."
)


def build_prompt(spec: RenderSpec) -> str:
    base = PROMPT_TEMPLATE.format(
        food=spec.food,
        bg=DEFAULT_BG,
        lighting=DEFAULT_LIGHTING,
    )
    return f"{base} {NEGATIVE_PROMPT}"


# ── API clients ───────────────────────────────────────────────────────────────
def get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        print("[ERROR] Missing openai package. Install with: pip install openai", file=sys.stderr)
        raise
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def get_gemini_client():
    try:
        from google import genai  # type: ignore
    except Exception:
        print("[ERROR] Missing google-genai package. Install with: pip install -U google-genai", file=sys.stderr)
        raise
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def generate_image_b64_openai(
    client, prompt: str, size: str, quality: str, model: str, seed: Optional[int]
) -> str:
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
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("utf-8")
    raise RuntimeError("Images API returned neither base64 nor URL.")


def generate_image_b64_gemini(
    client, prompt: str, size: str, quality: str, model: str, seed: Optional[int]
) -> str:
    try:
        from google.genai import types  # type: ignore
    except Exception:
        types = None  # type: ignore

    if types is not None and hasattr(types, "Part") and hasattr(types.Part, "from_text") and hasattr(types, "Content"):
        parts = [types.Part.from_text(text=prompt)]
        contents = [types.Content(role="user", parts=parts)]
    else:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

    response = client.models.generate_content(model=model, contents=contents)

    for part in getattr(response, "parts", []) or []:
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if isinstance(data, bytes):
                return base64.b64encode(data).decode("utf-8")
            if isinstance(data, str) and data:
                return data

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


# ── I/O helpers ───────────────────────────────────────────────────────────────
def write_png_b64(b64: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(b64))


def write_meta(spec: RenderSpec, prompt: str, backend: str = DEFAULT_BACKEND) -> None:
    master_path = spec.out_dir / "stimuli_master.json"
    entry: Dict[str, Any] = {
        "image_file": spec.image_path.name,
        "food": spec.food,
        "category": spec.category,
        "prompt": prompt,
        "model": spec.model,
        "size": spec.size,
        "quality": spec.quality,
        "seed": spec.seed,
        "created": int(time.time()),
        "source": f"ai-{backend}",
        "style_version": "v1_uniform_grey_45deg",
    }
    data: list = []
    if master_path.exists():
        try:
            with master_path.open("r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(entry)
    with master_path.open("w") as f:
        json.dump(data, f, indent=2)


def backoff_sleep(attempt: int) -> None:
    delay = min(60, 2 ** attempt) + (0.1 * attempt)
    time.sleep(delay)


def render_one(
    spec: RenderSpec,
    client=None,
    dry_run: bool = False,
    overwrite: bool = False,
    backend: str = DEFAULT_BACKEND,
) -> bool:
    spec.out_dir.mkdir(parents=True, exist_ok=True)

    if spec.image_path.exists() and not overwrite and not spec.additional_prompt:
        print(f"[SKIP] Already exists: {spec.image_path}")
        return True

    prompt = build_prompt(spec)
    if spec.additional_prompt.strip():
        prompt = f"{prompt} {spec.additional_prompt.strip()}"

    if dry_run:
        print(f"[DRY] Would render: {spec.food} -> {spec.image_path}")
        print(f"      Prompt: {prompt[:120]}...")
        write_meta(spec, prompt, backend=backend)
        return True

    if client is None:
        client = get_gemini_client() if backend == "gemini" else get_openai_client()

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
            print(f"[WARN] Attempt {attempt + 1} failed for {spec.food}: {e}")
            if attempt < 5:
                backoff_sleep(attempt)
            else:
                print(f"[FAIL] Giving up on {spec.food}")
                return False
    return False


# ── Metadata utilities ────────────────────────────────────────────────────────
def load_master_df(master_path: Optional[Path] = None) -> pd.DataFrame:
    if master_path is None:
        master_path = OUT_DIR / "stimuli_master.json"
    cols = ["image_file", "food", "category", "prompt", "model", "size", "quality", "seed", "created"]
    if not master_path.exists():
        return pd.DataFrame(columns=cols)
    try:
        with master_path.open("r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return pd.DataFrame(columns=cols)
        return pd.DataFrame.from_records(data)
    except Exception as e:
        print(f"[ERROR] Failed to read {master_path}: {e}")
        return pd.DataFrame(columns=cols)


def list_missing_images(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        df = load_master_df()
    if df.empty:
        print("[INFO] Master metadata is empty; nothing to check.")
        return df
    image_paths = [OUT_DIR / fname for fname in df["image_file"]]
    missing_mask = [not p.exists() for p in image_paths]
    missing_df = df[missing_mask].copy()
    missing_df["image_exists"] = False
    if missing_df.empty:
        print("[OK] All images referenced in stimuli_master.json exist on disk.")
    else:
        print(f"[WARN] {len(missing_df)} images are missing on disk.")
        for _, row in missing_df.iterrows():
            print(f"  - {row['food']} -> {row['image_file']}")
    return missing_df


# ── Item loader ───────────────────────────────────────────────────────────────
def load_items(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load items from a CSV file (columns: Category, Food) or fall back to the
    built-in BUILTIN_OBJECTS list if no CSV is provided or found.
    """
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        if not {"Category", "Food"}.issubset(set(df.columns)):
            raise ValueError("CSV must contain 'Category' and 'Food' columns.")
        if "Additional Prompt" not in df.columns:
            df["Additional Prompt"] = ""
        df["Additional Prompt"] = df["Additional Prompt"].fillna("")
        print(f"[INFO] Loaded {len(df)} items from {csv_path}")
        return df

    print("[INFO] No CSV provided or found — using built-in object list.")
    df = pd.DataFrame.from_records(BUILTIN_OBJECTS)
    df["Additional Prompt"] = ""
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate non-food object foil images for PAFID experiments."
    )
    p.add_argument("--object", type=str, default=None, help="Render a single object (exact name match)")
    p.add_argument("--category", type=str, default=None, help="Filter to a single category")
    p.add_argument("--csv", type=str, default=None, help="Path to custom CSV (Category, Food columns)")
    p.add_argument("--limit", type=int, default=None, help="Render at most N items")
    p.add_argument("--offset", type=int, default=0, help="Start at row offset")
    p.add_argument("--size", type=str, default=DEFAULT_SIZE, help="Image size, e.g., 1024x1024")
    p.add_argument("--quality", type=str, default=DEFAULT_QUALITY, choices=["standard", "high"])
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Image model")
    p.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["openai", "gemini"],
        help="Image backend to use: 'openai' (gpt-image-1) or 'gemini' (Gemini image models, default gemini-3-pro-image-preview).",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional seed for determinism")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--dry-run", action="store_true", help="Print prompts without calling the API")
    p.add_argument("--resume", action="store_true", help="Skip items that already have an image")
    p.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    p.add_argument("--integrity-check", action="store_true", help="Report missing images vs. master metadata")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.size = normalize_size(args.size)
    backend = args.backend

    # Auto-select Gemini model if user passed an OpenAI-style name
    if backend == "gemini" and args.model in (DEFAULT_MODEL, "gpt-image-1"):
        args.model = DEFAULT_GEMINI_MODEL
        print(f"[INFO] Gemini backend selected — using model '{args.model}'")

    print(f"[INFO] Backend: '{backend}', model: '{args.model}'")

    out_dir_override: Optional[Path] = None
    if args.out_dir:
        out_dir_override = Path(args.out_dir).expanduser().resolve()
        out_dir_override.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv).expanduser().resolve() if args.csv else DEFAULT_CSV_PATH
    try:
        df = load_items(csv_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    full_df = df.copy()

    # Filters
    if args.category:
        df = df[df["Category"].str.lower() == args.category.lower()]
    if args.object:
        df = df[df["Food"].str.lower() == args.object.lower()]
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.head(args.limit)

    client = None
    if not args.dry_run:
        client = get_gemini_client() if backend == "gemini" else get_openai_client()

    print(f"[INFO] Items to render: {len(df)}")
    success = 0
    for _, row in df.iterrows():
        spec = RenderSpec(
            food=str(row["Food"]),
            category=str(row["Category"]),
            additional_prompt=str(row.get("Additional Prompt", "") or ""),
            size=args.size,
            quality=args.quality,
            model=args.model,
            seed=args.seed,
            out_dir_override=out_dir_override,
        )
        if args.resume and spec.image_path.exists():
            print(f"[RESUME] Skipping existing {spec.image_path}")
            success += 1
            continue
        ok = render_one(spec, client=client, dry_run=args.dry_run, overwrite=args.overwrite, backend=backend)
        success += int(ok)

    print(f"[DONE] {success}/{len(df)} succeeded")

    if args.integrity_check:
        effective_out = out_dir_override or OUT_DIR
        expected = {slugify(f) for f in full_df["Food"]}
        existing = {p.stem for p in sorted(effective_out.glob("*.png"))}
        missing = sorted(expected - existing)
        extras = sorted(existing - expected)
        if missing:
            print(f"[WARN] {len(missing)} expected image(s) missing:")
            for s in missing:
                print(f"  - {s}.png")
        else:
            print("[OK] No missing images.")
        if extras:
            print(f"[INFO] {len(extras)} extra image(s) not in object list:")
            for s in extras:
                print(f"  - {s}.png")

    return 0 if success == len(df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
