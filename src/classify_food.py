#!/usr/bin/env python3
"""
PAFID Food List AI Classification (Stage 1, Part 1)

Categorises each food item from a CSV list using Gemini text API,
populating taxonomic culinary/folk classifications and continuous/discrete processing constructs.
Saves/syncs the metadata in stimuli_master.json.

Output schemes:
  - Culinary & Folk Prompt (WHO 10, Intuitive 7, Culinary 9)
  - Processing Focus Prompt (NOVA 4, Natural vs Transformed, Transformation Score)

Requires:
  pip install -U google-genai pandas
  export GEMINI_API_KEY=... (or GOOGLE_GENAI_USE_VERTEXAI=true)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
CSV_PATH = ROOT / "data" / "food_list_initial_seed.csv"
OUT_DIR = ROOT / "rendered_images"

DEFAULT_GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Version stamps for resumable pipelines
CLASSIFY_PROMPT_VERSION = "v3-2026-06-culinary-folk"
NOVA_PROMPT_VERSION = "v2-2026-06-processing-focus"


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


def slugify(name: str) -> str:
    s = str(name).strip().replace(" ", "-")
    s = re.sub(r"[^\w\-]", "", s)
    return s.lower()


# --- Prompt templates (aligned by construct) ---

CLASSIFY_PROMPT_TEMPLATE = (
    'Classify the food item "{food}" using all three schemes below.\n\n'
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
    "  * The category follows the dominant base food. Cooking, drying, frying, or otherwise processing a single base food does not change its bucket: grilled salmon is Animal protein; roast profile is Vegetable; toast is Grain; dried apple is Fruit.\n"
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
    "Reply in exactly this format — three lines, no explanation:\n"
    "WHO10: <category>\n"
    "INTUITIVE7: <category>\n"
    "CULINARY9: <category>"
)

NOVA_CLASSIFY_PROMPT_TEMPLATE = (
    'Classify the food item "{food}" using the processing schemes below.\n\n'
    "1) NOVA group (1-4) based on the extent and purpose "
    "of industrial food processing (Monteiro et al., 2016).\n"
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
    "Rules for ambiguous NOVA cases:\n"
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
    "2) Natural vs Transformed — pick exactly one:\n"
    "- Natural: the food is still visually and conceptually identifiable as a single biological food source, "
    "even if it has undergone minor preparation such as washing, peeling, cutting, drying, freezing, or simple cooking\n"
    "- Transformed: the food has been substantially altered from its original biological source through combination, "
    "reforming, refining, fermentation, baking, frying, industrial processing, or preparation into a dish or product\n\n"
    "3) Transformation score — assign a single integer from 0 to 100 using this scale:\n"
    "  0–10:  Whole, raw, unmodified (e.g. apple, banana, carrot, tomato)\n"
    " 10–25:  Minimally prepared but source-identifiable (e.g. sliced fruit, peeled orange, roasted nuts, dried fruit)\n"
    " 25–40:  Simply cooked single-source food (e.g. boiled egg, steamed vegetables, grilled fish, plain rice)\n"
    " 40–55:  Mechanically altered single-source food (e.g. mashed potato, minced meat, fruit puree, smoothie)\n"
    " 55–70:  Biochemically or structurally transformed (e.g. cheese, yoghurt, tofu, bread, pasta)\n"
    " 70–85:  Composite prepared dish (e.g. soup, curry, sandwich, sushi, dumplings)\n"
    " 85–100: Highly transformed or manufactured food (e.g. pizza, cake, sausage, chocolate bar, cereal, candy)\n\n"
    "Reply in exactly this format — four lines, no explanation:\n"
    "NOVA: <integer 1-4>\n"
    "NAT_TRANS: <Natural or Transformed>\n"
    "SCORE: <integer 0-100>\n"
    "NOTE: <one short phrase only if borderline/ambiguous, otherwise leave empty>"
)


def _clamp_0_100(score_str: str) -> int:
    try:
        score = int(re.sub(r'\D', '', score_str))
        return max(0, min(100, score))
    except (ValueError, TypeError):
        return -1


def classify_food_gemini(client, food: str, retries: int = 8, model: str = DEFAULT_GEMINI_TEXT_MODEL) -> tuple:
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(food=food)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0},
            )
            text = response.text.strip()
            who10, intuitive7, culinary9 = "unknown", "unknown", "unknown"
            for line in text.splitlines():
                upper = line.upper()
                if upper.startswith("WHO10:"):
                    who10 = line.split(":", 1)[1].strip()
                elif upper.startswith("INTUITIVE7:"):
                    intuitive7 = line.split(":", 1)[1].strip()
                elif upper.startswith("CULINARY9:"):
                    culinary9 = line.split(":", 1)[1].strip()
            return who10, intuitive7, culinary9
        except Exception as e:
            print(f"[WARN] Classification API error: {e}. Retrying...")
            time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)
    return "unknown", "unknown", "unknown"


def classify_nova_gemini(client, food: str, retries: int = 8, model: str = DEFAULT_GEMINI_TEXT_MODEL) -> tuple:
    prompt = NOVA_CLASSIFY_PROMPT_TEMPLATE.format(food=food)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0},
            )
            text = response.text.strip()
            group, nat_trans, score_str, note = None, "unknown", "", ""
            for line in text.splitlines():
                upper = line.upper()
                if upper.startswith("NOVA:"):
                    digits = re.sub(r"\D", "", line.split(":", 1)[1])
                    if digits:
                        group = max(1, min(4, int(digits[0])))
                elif upper.startswith("NAT_TRANS:"):
                    nat_trans = line.split(":", 1)[1].strip()
                elif upper.startswith("SCORE:"):
                    score_str = line.split(":", 1)[1].strip()
                elif upper.startswith("NOTE:"):
                    note = line.split(":", 1)[1].strip()

            score = _clamp_0_100(score_str)
            if group is not None:
                return group, nat_trans, score, note
            raise ValueError(f"Could not parse NOVA group from: {text[:120]!r}")
        except Exception as e:
            print(f"[WARN] NOVA classification error: {e}. Retrying...")
            time.sleep(min(60, (2 ** attempt)) + 0.25 * attempt)
    return None, "unknown", -1, ""


def run_classify(df: pd.DataFrame, client, out_dir: Path, text_model: str) -> int:
    master_path = out_dir / "stimuli_master.json"

    if master_path.exists():
        try:
            with master_path.open("r", encoding="utf-8") as f:
                master = json.load(f)
            if not isinstance(master, list):
                master = []
        except Exception as e:
            print(f"[ERROR] Could not read stimuli_master.json: {e}")
            return 1
    else:
        master = []

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    master_index: Dict[str, int] = {}
    for idx, entry in enumerate(master):
        key = norm(entry.get("food", entry.get("Food", "")))
        master_index[key] = idx

    if master_path.exists():
        bak_path = master_path.with_suffix(".json.bak")
        import shutil
        shutil.copy2(master_path, bak_path)
        print(f"[INFO] Backed up existing master to {bak_path.name}")

    def _save():
        with master_path.open("w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)

    LABEL_FIELDS = ["Category_WHO_10", "Category_Intuitive_7", "Category_Culinary_9"]

    def _already_done(entry: dict) -> bool:
        if entry.get("classify_prompt_version") != CLASSIFY_PROMPT_VERSION:
            return False
        return all(str(entry.get(k, "")).strip() not in ("", "unknown", "None") for k in LABEL_FIELDS)

    def _nova_done(entry: dict) -> bool:
        if entry.get("nova_prompt_version") != NOVA_PROMPT_VERSION:
            return False
        try:
            int_val = int(entry.get("Category_NOVA_4"))
            if int_val not in (1, 2, 3, 4):
                return False
        except (TypeError, ValueError):
            return False
        return all(str(entry.get(k, "")).strip() not in ("", "unknown", "None") for k in ["Natural_vs_transformed", "Transformation_score"])

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

        if main_done and nova_ok:
            skipped += 1
            continue

        if not main_done:
            print(f"[INFO] ({i}/{total}) Classifying culinary & folk schemes: {food}")
            who10, intuitive7, culinary9 = classify_food_gemini(client, food, model=text_model)

            if (not who10) or who10 == "unknown" or intuitive7 == "unknown":
                print(f"[FAIL] Could not classify culinary categories: {food} (will retry on next run)")
                failures += 1
                continue

            print(f"      [CLASSIFY] WHO10='{who10}' | Intuitive7='{intuitive7}' | Culinary9='{culinary9}'")

            if entry is None:
                print(f"      [INFO] No existing master entry for '{food}' — creating new entry.")
                entry = {"food": food, "image_file": f"{slugify(food)}.png"}
                master.append(entry)
                master_index[key] = len(master) - 1

            entry["Category_WHO_10"] = who10
            entry["Category_Intuitive_7"] = intuitive7
            entry["Category_Culinary_9"] = culinary9
            entry["classify_prompt_version"] = CLASSIFY_PROMPT_VERSION
            if "Category_Simple_6" in entry:
                del entry["Category_Simple_6"]
            done += 1
            _save()
            time.sleep(1.5)

        if not nova_ok:
            print(f"[INFO] ({i}/{total}) Processing-classifying: {food}")
            nova_group, nat_trans, score, nova_note = classify_nova_gemini(client, food, model=text_model)
            if nova_group is None:
                print(f"[FAIL] Could not processing-classify: {food} (will retry on next run)")
                failures += 1
                continue
            print(f"      [NOVA] Group {nova_group} | {nat_trans} | Score={score}" + (f" ({nova_note})" if nova_note else ""))
            
            if entry is None:
                print(f"      [INFO] No existing master entry for '{food}' — creating new entry.")
                entry = {"food": food, "image_file": f"{slugify(food)}.png"}
                master.append(entry)
                master_index[key] = len(master) - 1

            entry["Category_NOVA_4"] = int(nova_group)
            entry["Natural_vs_transformed"] = nat_trans
            entry["Transformation_score"] = score
            if nova_note:
                entry["nova_notes"] = nova_note
            entry["nova_prompt_version"] = NOVA_PROMPT_VERSION
            entry["nova_source"] = text_model
            nova_done_n += 1
            _save()
            time.sleep(1.5)

    _save()
    if skipped:
        print(f"[INFO] Skipped {skipped} food(s) already fully classified (prompt '{CLASSIFY_PROMPT_VERSION}', NOVA '{NOVA_PROMPT_VERSION}').")
    print(f"[DONE] Classified {done} food(s) + {nova_done_n} NOVA label(s) this run, {failures} failure(s). Master updated: {master_path}")
    return 0 if failures == 0 else 1


def load_items(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Food list CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Food" not in df.columns:
        raise ValueError(f"Food list CSV must contain a 'Food' column. Columns: {list(df.columns)}")
    return df


def main():
    p = argparse.ArgumentParser(description="Stage 1, Part 1: AI Food List Text Classification")
    p.add_argument("--food-list", type=str, default=None, help="Path to alternative food list CSV")
    p.add_argument("--output-dir", type=str, default=None, help="Output folder for master JSON")
    p.add_argument("--food", type=str, default=None, help="Classify a single Food item (exact match)")
    p.add_argument("--limit", type=int, default=None, help="Classify at most N items")
    p.add_argument("--offset", type=int, default=0, help="Start at row offset")
    args = p.parse_args()

    csv_path = Path(args.food_list).expanduser().resolve() if args.food_list else CSV_PATH
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_items(csv_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    if args.food:
        df = df[df["Food"].str.lower() == args.food.lower()]
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.head(args.limit)

    client = get_gemini_client()
    return run_classify(df, client, out_dir=out_dir, text_model=DEFAULT_GEMINI_TEXT_MODEL)


if __name__ == "__main__":
    sys.exit(main())
