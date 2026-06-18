# PROJECT_SPEC — PAFID

*Last updated: 2026-06-18*

---

## What this project is

**PAFID (Public AI-Generated Food Image Database)** is a pipeline for generating, validating, and rating a standardised set of photorealistic food stimulus images using Generative AI. It produces a publicly released database of food images with rich metadata — AI-assigned category labels, AI and human perceptual ratings, visual feature statistics, and QC provenance — suitable for use in behavioural, EEG, and fMRI food perception research.

The canonical database contains **350 food items**. The pipeline is designed to allow extension projects to add new foods to this set without modifying the canonical baseline.

Repository: https://github.com/ThomasSydneyEDU/PAFID  
Companion paper: Stella et al. (in preparation — no citation yet).

---

## Who it is for

- **Researchers** in food perception, cognitive neuroscience, and psychology who need visually consistent, richly labelled food stimuli
- **Extension projects** (e.g. FoodSpace-Extension) that need to generate images and ratings for new food items using the same pipeline and visual style as the canonical 350
- **The broader community** — PAFID is public and designed to be reproducible by any researcher with Google Cloud / Gemini API access

---

## What problem it solves

Creating a standardised food stimulus set manually is slow and inconsistent. Researchers need:
- Visually consistent images (same camera angle, lighting, plate, background) across hundreds of foods
- Multiple category label schemes (WHO, culinary, intuitive, processing level) that are expensive to assign by hand
- Perceptual ratings (sweetness, healthiness, appeal, etc.) that normally require running participants
- A pipeline that is reproducible, extensible, and auditable

PAFID automates all of this. The image generation prompt enforces a fixed photographic style; Gemini assigns five label schemes per food via text classification; AI-based aware and blind rating proxies for human perceptual data; and low-level visual statistics are extracted automatically. Human ratings can be merged in when available.

---

## What PAFID does (current capabilities)

The pipeline has seven sequential stages, all resumable and incremental by default:

| Stage | Script | What it does |
|---|---|---|
| 1. Generate | `generate_stimuli.py` | Renders images via Gemini; assigns 5 label schemes per food |
| 2. QC / aware ratings | `run_qc.py` | Checks image quality; rates food with label visible |
| 3. Blind ratings | `rate_images.py` | Rates food from image alone (no label shown) |
| 4. Visual features | `extract_visual_features.py` | Extracts low-level stats (luminance, contrast, HOG PCs, etc.) |
| 5. Prepare images | `prepare_images.py` | Resizes images for experiment deployment |
| 6. Editorial review | `run_editorial_review.py` | Iterative correction of labels and image regeneration |
| 7. Apply corrections | `apply_corrections.py` | Commits verified corrections to master + dynamic CSV |

All metadata consolidates into `data/Foodpictures_information_dynamic.csv` (the working dataset) and `rendered_images/stimuli_master.json` (the per-item ground truth).

---

## What is explicitly out of scope

- **Non-food stimuli.** `generate_nonfood_stimuli.py` exists for research foils but non-food images are not part of the PAFID release and will not be.
- **Human data collection.** PAFID merges human ratings when provided, but running the surveys is outside scope. The `Food survey/` directory is excluded from version control.
- **Re-running SPoSE or triplet analyses.** PAFID is a stimulus database. Downstream similarity modelling belongs to projects like FoodTriplet-Analysis and FoodSpace-Extension.
- **Multi-image variants per food.** The pipeline generates one canonical image per food. Variant generation (different preparations, cultural versions) would require a new pipeline stage.

---

## Technical constraints

| Constraint | Detail |
|---|---|
| **Image generation** | Google Gemini via Vertex AI (recommended) or AI Studio API key. Image model: Gemini image generation. Classification/rating model: `gemini-2.5-flash` (temperature 0) and `gemini-2.5-pro` |
| **Authentication** | Vertex AI: `gcloud` application-default credentials + `GOOGLE_GENAI_USE_VERTEXAI=True`. AI Studio: `GEMINI_API_KEY` |
| **Python runtime** | Standard venv (`python -m venv venv`). Cross-platform (Mac, Windows, Linux) |
| **Image format** | PNG, 1024×1024, consistent photographic style enforced by prompt. Resized output for experiments in `resized_images/` |
| **Metadata format** | `Foodpictures_information_dynamic.csv` schema must remain backward compatible with FoodTriplet-Analysis and FoodSpace-Extension |

---

## What must not be broken

**The canonical 350-item baseline is immutable.**

- `data/food_list_initial_seed.csv` — the seed list of 350 foods. The pipeline must never write to this file.
- `data/Foodpictures_information_reference.csv` — the static post-correction baseline. No pipeline script writes to it after the one-time author freeze step.
- `rendered_images/` canonical entries — existing images for the 350 original foods must never be overwritten by an extension run.
- **Filename ↔ food name mapping** — slugified filenames are the join key between images and all downstream metadata. Renaming a food or its image breaks all analyses referencing it.
- **Dynamic CSV schema** — column names and types must remain stable. New columns may be added; existing columns must not be renamed or removed without a migration.
- **Prompt versioning** — classification prompts carry version stamps (`v2-2026-06-12-intuitive7-folk-categories`, `v1-2026-06-monteiro-2016`). Updating a prompt must bump its version; the pipeline uses version stamps to determine what needs re-running.

---

## Planned changes — extension support

The following changes are planned to support external projects (e.g. FoodSpace-Extension) adding foods to the database without modifying PAFID's repo:

1. **`food_list_initial_seed.csv` hardened as read-only** — the pipeline will never append to it. This makes the seed a permanent, auditable record of the canonical 350.

2. **`--food-list <path>` flag** — `generate_stimuli.py` and `run_pipeline.sh` will accept an external CSV of new food names at runtime. The calling project provides its own list; PAFID does not look for a hardcoded extension file.

3. **`--output-dir <path>` flag** — outputs (generated images, per-item metadata, updated dynamic CSV) will be redirectable to the calling project's directory. PAFID's own `rendered_images/` and `resized_images/` will only ever contain the canonical 350. This keeps PAFID's repo clean when extension projects run the pipeline.

4. **`--source <label>` flag** — a `source` column will be added to `Foodpictures_information_dynamic.csv`. Original 350 foods get `pafid_v1`; foods added via `--food-list` get a caller-supplied label (e.g. `foodspace_extension_2026`). This enables downstream projects to filter by provenance and is useful for triplet generation and RSA analyses.

5. **`reset_pipeline.py` source-aware reset** — the reset script will support resetting only rows with a given source label, leaving other sources untouched. A full reset (restoring to canonical 350 only) remains available.

### Extension usage (from a calling project)

```bash
cd ../PAFID
bash run_pipeline.sh \
  --food-list ../FoodSpace-Extension/outputs/curated_nominees.csv \
  --output-dir ../FoodSpace-Extension/stimuli \
  --source foodspace_extension_2026
```

PAFID repo is unmodified. Outputs land in the calling project.

---

## Outstanding issues (from PAFID_TODO.md)

### Companion paper (Stella et al.)
- Document current classification prompt (v2, 2026-06-12) — Intuitive 7 folk-category rules
- Document NOVA scheme and its provenance (manual batch → automated, Monteiro et al. 2016)
- Document the udon manual correction (name → noodles, image → composite dish)
- State models used per step and their versions
- Note HOG-PC caveat (PCA basis is run-specific; not comparable across runs)

### README fixes
- Update model name: classification uses `gemini-2.5-flash`, not `gemini-2.0-flash`
- Fix duplicate section numbering (two "6." headings)
- Reconcile `Foodpictures_information_reference.csv` vs `_old.csv` filename discrepancy

### Data hygiene
- Decide fate of backup files (`*.pre_blind_join.bak`, `*.pre_ll_join.bak`, etc.)
- Check "Borrito bowl" spelling in seed list (coordinate with stimuli_master if corrected)
- Commit all uncommitted code + data changes

### Validation
- Diff WHO_10 / Culinary_9 / NvT / Transformation_score against pre-v2 values (temperature-0 rerun may have flipped borderline items)
- Caption-based name-vs-image mismatch scan (the udon pattern) across all 350 items
- Revisit "sweet breakfast cereal" → Grain classification (borderline under v2 sweetness rule)
- End-to-end test of new-stimulus path once extension support is implemented

### QC backlog
- 38 items flagged in `data/QC/food_category_flags_to_review.csv` — image/label mismatches and category disagreements. **Decision: these will not be manually corrected.** They are a documented, transparent part of the database. The README explains their nature and how to handle them in downstream analyses. This is preferable to silent corrections that obscure the limitations of AI-based labelling.

---

## What counts as done

PAFID v1 (canonical release) is done when:

1. `Foodpictures_information_reference.csv` is frozen to the current 350-item baseline (38 flagged items are retained as documented known issues, not corrected)
2. README reflects current model names, correct section numbering, accurate file references, and a clear statement that the 38 flagged items are an intentional part of the documented database
3. Companion paper methods section documents: current classification prompts with version stamps, NOVA scheme provenance, models used per stage, HOG-PC caveat
4. Extension support (planned changes above) is implemented and end-to-end tested with one food item

---

## Key files

| File | Role | Mutable? |
|---|---|---|
| `data/food_list_initial_seed.csv` | Canonical 350-food seed list | No — read-only after v1 freeze |
| `data/Foodpictures_information_reference.csv` | Static post-correction baseline | No — frozen after corrections applied |
| `data/Foodpictures_information_dynamic.csv` | Working metadata (pipeline output) | Yes — pipeline writes here |
| `rendered_images/stimuli_master.json` | Per-item ground truth (images + all metadata) | Yes — pipeline writes here |
| `rendered_images/` | Generated high-res images | Yes — new images added; canonical never overwritten |
| `resized_images/` | Experiment-ready images | Yes — regenerated by `prepare_images.py` |
| `data/QC/food_category_flags_to_review.csv` | Open audit of 38 flagged items | Yes — authors action this |
| `data/QC/category_corrections.csv` | Confirmed human overrides | Yes — authors write here |
