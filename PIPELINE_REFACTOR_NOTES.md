# PAFID Pipeline Refactor: Processing Alignment & Perceptual Blinding

*Date: 2026-06-27*
*Active Branch: `pipeline-refactor`*

---

## 1. Overview of the Changes

To improve the conceptual consistency, scientific rigor, and layout clarity of the PAFID image generation and ratings pipeline, we refactored the pipeline into **6 flat, linear, sequential stages** (Stage 0 through Stage 5):

0.  **Stage 0: Food Selection (Manual)** (Semantic names and list compilation; *no code*)
    *   *Manual Review:* Authors review the compiled food names list to ensure nomenclature precision, distinctiveness, and formatting consistency (e.g. raw, prepared, or cut notes) before initiating the automated pipeline.
1.  **Stage 1: Text Classification (`classify_food.py`)** (Lightweight Text classification)
    *   *Manual Review:* Authors review the resulting classification schemas and processing constructs in `stimuli_master.json` to verify logical taxonomical boundaries before generating images.
2.  **Stage 2: Image Generation & Quality Control (Combined)** (`generate_images.py` + `run_qc.py`)
    *   *Combined Action:* Images are generated using prompt-building heuristics. Immediately after rendering, the automated visual Quality Control script runs to audit plate margins, captions, and flag visual discrepancies in `qc_issues.json`.
3.  **Stage 3: Editorial Review / Apply Manual Suggestions (`run_editorial_review.py`)**
    *   *Iterative Feedback Loop:* This is the human-in-the-loop Suggestions Phase. Authors review the visual flags from Stage 2 and write prompt modifications directly into `food_category_flags_to_review.csv` (the review CSV). Running `run_editorial_review.py` automatically applies these manual suggestions, re-renders the images, and re-runs QC in a closed loop to verify the fixes.
4.  **Stage 4: AI Ratings (`rate_images.py`)** (Strict Perceptual Blinding Ratings: Blind & Aware)
5.  **Stage 5: Visual Feature Extraction (`extract_visual_features.py`)** (Computer Vision Low-Level Statistics)
6.  **Stage 6: Prepare Images (`prepare_images.py`)** (Experimental Resizing & Downsampling)

---

## 2. Rationale

1.  **Methodological Rigor (Eliminating Contamination):** Keeping blind and aware conditions in separate prompt executions is scientifically essential. In a single combined prompt, self-attention on the text label would contaminate ("leak") into the blind evaluations. By running separate calls, the blind condition remains strictly blind.
2.  **Interactive Suggestions Loop:** Combining Image Generation and QC into a single automated step allows us to establish a clean closed-loop feedback phase (Stage 3). Authors provide manual suggestions, and the pipeline re-runs generation + QC in tandem until the problems are resolved.
3.  **Conceptual Cleanliness:** Decouples physical image validation/audit (QC) from cognitive/subjective evaluations (Ratings), and isolates metadata classification from actual image generation.
4.  **Synchronization & Robustness:** `rate_images.py` now writes ratings *both* to the working dataset (`Foodpictures_information_dynamic.csv`) and back into `stimuli_master.json`. This ensures:
    *   The per-item master JSON remains the single source of ground truth.
    *   Re-running Quality Control (`run_qc.py`) will automatically re-read and preserve the ratings, preventing any data-loss during multi-step pipeline executions.

---

## 3. Staging & Execution Guide

### Stage 0: Food Selection (Manual Compilation)
*   **Action:** Manual selection and naming of the food stimuli list, saved to a CSV list (e.g., `data/food_list_initial_seed.csv`).
*   **Manual Review Note:** Before running Stage 1, authors carefully verify all semantic names for distinctiveness, familiarity, and typical preparation notes to prevent parsing ambiguity.

### Stage 1: Text Classification (`src/classify_food.py`)
*   **Action:** Uses a split structure to get (a) culinary taxonomy categories and (b) NOVA processing scores safely from Gemini.
*   **Version Control:** Bumped versions to `v3-2026-06-culinary-folk` and `v2-2026-06-processing-focus` to invalidate outdated metadata.
*   **Manual Review Note:** Before running Stage 2, authors check `stimuli_master.json` to ensure taxonomical classifications are accurate and logical.

### Stage 2: Image Generation & Quality Control (Combined)
*   **Action:** 
    1. `generate_images.py` renders high-res `1024x1024` PNGs based on standard templates and vessel/granular heuristics.
    2. `run_qc.py` immediately audits the rendered PNGs for naturalism, plate rims, hands, text, and generates neutral captions. It outputs flags to `qc_issues.json` and `food_category_flags_to_review.csv`.

### Stage 3: Editorial Review / Apply Manual Suggestions (`src/run_editorial_review.py`)
*   **Action:** Authors inspect flagged visual issues and write custom prompt fixes in the `generation_notes` column of `food_category_flags_to_review.csv`. Running `run_editorial_review.py` reads these manual suggestions and automatically:
    *   Re-renders the image with the custom prompt additions.
    *   Re-runs Quality Control on the newly generated image to verify that the problems are resolved.

### Stage 4: AI Ratings (`src/rate_images.py`)
*   **Action:** Blind Ratings (Image only $\rightarrow$ Guess + Blind ratings) & Aware Ratings (Image + Food Name $\rightarrow$ Aware ratings).
*   **Checkpoints:** Saves both to CSV and `stimuli_master.json` in-place.

### Stage 5: Visual Feature Extraction (`src/extract_visual_features.py`)
*   **Action:** Extracts luminance, RMS contrast, CIELAB channels, edge energy, and run-specific HOG PCA vectors.

### Stage 6: Prepare Images (`src/prepare_images.py`)
*   **Action:** Lanczos resamples standard `1024x1024` PNGs down to experiment-ready `384x384` pixel sizes.

---

## 4. How to Validate

To validate that both scripts execute flawlessly and output correct schemas under this new design, run a test cycle on a limited subset of foods.

```bash
# Activate your environment
source .venv/bin/activate

# 1. Run Classification (Stage 1)
python3 src/classify_food.py --limit 5 --output-dir test_outputs

# 2. Run Image Generation & QC (Stage 2)
python3 src/generate_images.py --limit 5 --output-dir test_outputs --dry-run
python3 src/run_qc.py --stimuli-dir test_outputs --limit 5 --overwrite --dynamic-csv test_outputs/Foodpictures_information_dynamic.csv

# 3. Run AI Ratings (Stage 4)
python3 src/rate_images.py --stimuli-dir test_outputs --csv test_outputs/Foodpictures_information_dynamic.csv --limit 5
```

Check `test_outputs/stimuli_master.json` and `test_outputs/Foodpictures_information_dynamic.csv` to verify metadata and rating columns.
