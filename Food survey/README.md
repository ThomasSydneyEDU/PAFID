# Food Survey Pipeline

Processes raw Qualtrics survey exports, integrates them with the master AI/metadata canonical CSV, and produces a per-participant analysis dataset alongside per-image summaries.

## Prerequisites

- Python 3 with `pandas`, `numpy`, and `openpyxl`.

## Folder Layout

```
Food survey/
├── human_data/                       # drop your raw Qualtrics CSVs here
│   ├── SONA_surveydata.csv
│   └── MTURK_surveydata.csv
├── food_survey_reference.csv         # ImageName <-> ImageID + isEasy flag
├── survey_analysis.py
├── update_canonical_human_means.py
├── image_rating_analysis.py
├── compile.py
└── image_presentation_count.py       # optional helper
```

The master canonical CSV lives one level up at `../data/Foodpictures_information_canonical.csv`.

## Running the Pipeline

From inside `Food survey/`:

```bash
python3 survey_analysis.py                # 1. extract per-participant ratings
python3 update_canonical_human_means.py   # 2. write per-image means into canonical
python3 image_rating_analysis.py          # 3. per-image summary CSV
python3 compile.py                        # 4. joined Excel for AI-vs-human comparison
```

## Step Details

### 1. `survey_analysis.py`

Reads every `*_surveydata.csv` it finds in `human_data/` (auto-discovery), extracts per-food responses, anonymises participant IDs, attaches image filenames via `food_survey_reference.csv`, and computes a per-participant `QualityCheck` percentage based on Familiarity > 2 on "easy" items. The source label (`SONA`, `MTURK`, etc.) is derived from each filename's prefix before the underscore.

To process a custom set of files instead of auto-discovery, pass them as positional args:

```bash
python3 survey_analysis.py path/to/somefile.csv path/to/anotherfile.csv
```

**Output:** `extracted_survey_data.csv` — one row per participant × food item, with columns `DataSource`, `ParticipantNumber`, `Time`, `QualityCheck`, `FoodItem`, `ImageID`, `ImageName`, `IsEasy`, `CalorieDensity`, `Healthiness`, `Appeal`, `Familiarity`, `FoodName`, and the seven taste ratings (`Sweet`, `Salty`, `Sour`, `Bitter`, `Umami`, `Fatty`, `Spicy`).

### 2. `update_canonical_human_means.py`

Aggregates `extracted_survey_data.csv` into per-image means and writes them into the master `../data/Foodpictures_information_canonical.csv`.

Behaviour:

- Drops participants with `QualityCheck < 80%` (threshold is the `QUALITY_THRESHOLD` constant in the script).
- Extracts the leading digit of Familiarity (`"3 - Sometimes"` → `3`) before averaging.
- Resets every `human_*` column to NaN, then writes the new means into rows whose `filename` matches an `ImageName` in the survey. Images not represented in the current survey keep NaN — an explicit "no current data" signal.
- Backs up the previous canonical to `../data/Foodpictures_information_canonical.bak.csv` (single rotating backup, overwritten each run), then writes via atomic temp-file rename.

Survey-to-canonical column mapping:

| Survey | → | Canonical |
|---|---|---|
| CalorieDensity | → | human_calorie_density |
| Healthiness | → | human_healthiness |
| Appeal | → | human_appeal |
| Familiarity | → | human_familiarity |
| Sweet | → | human_sweetness |
| Salty | → | human_saltiness |
| Sour | → | human_sourness |
| Bitter | → | human_bitterness |
| Umami | → | human_savoriness |
| Fatty | → | human_fattiness |
| Spicy | → | human_spiciness |

`human_familiarity` is added to canonical the first time the script runs if not already present.

### 3. `image_rating_analysis.py`

Reads `extracted_survey_data.csv` and produces a per-image summary with means, SDs, the proportion of Familiarity ≥ 4 and ≤ 2, a `High_Appeal` flag (mean ≥ 70), and rank columns for top-N most/least familiar images.

**Output:** `image_rating_full_summary.csv`.

### 4. `compile.py`

Joins `extracted_survey_data.csv` with `../data/Foodpictures_information_canonical.csv` on `ImageName ↔ filename`, renames the human taste columns to `*Human` and the AI taste columns to `*AI`, organises everything into semantic blocks (core IDs, food labels, paired sensory, human-only ratings, AI summaries, survey meta, image meta, QC, vision embeddings), and exports a colour-coded Excel file.

`compile.py` strips the `human_*` columns from canonical on load and rebuilds the human side from raw survey data, so the Excel preserves per-participant ratings rather than the per-image means written by step 2.

**Output:** `ai_vs_human_COMPILED.xlsx` — one row per participant × image.

### 5. `image_presentation_count.py` (optional)

Prints per-image response counts. No file output.

## Outputs Summary

| File | Granularity | Purpose |
|---|---|---|
| `extracted_survey_data.csv` | participant × image | Behavioural-analysis source |
| `image_rating_full_summary.csv` | image | Means, SDs, familiarity proportions, ranks |
| `ai_vs_human_COMPILED.xlsx` | participant × image | Joined survey + AI ratings + image metadata |
| `../data/Foodpictures_information_canonical.csv` | image | Master AI + human means file |
| `../data/Foodpictures_information_canonical.bak.csv` | image | Single rotating backup |

## Console Summary

After step 1, the script prints a per-participant breakdown:

```text
--- Summary ---
Total Unique Participants: 225
Ratings per Participant & Quality Check (Familiarity > 2 on 'Easy' items):
  - SONA: R_9CCjNuNK3rFLe3D: 60 items rated | Time: 0:31:59 | Quality Check: 81.0% (17/21 easy items passed)
  - MTURK: R_5GvjYVcK4mP6jXI: 60 items rated | Time: 0:34:00 | Quality Check: 80.0% (16/20 easy items passed)
...
```

## Troubleshooting

- **"No input CSVs given and no *_surveydata.csv files found":** Place your raw Qualtrics exports in `Food survey/human_data/` with filenames ending in `_surveydata.csv`.
- **"Stimuli reference file not found":** Ensure `food_survey_reference.csv` is in `Food survey/`.
- **"Canonical CSV not found":** The master must exist at `../data/Foodpictures_information_canonical.csv`.
- **Most or all participants dropped by quality filter:** Inspect the `QualityCheck` column in `extracted_survey_data.csv`. Threshold is 80% — adjust `QUALITY_THRESHOLD` in `update_canonical_human_means.py` if your design warrants a different bar.
- **Missing Image Names:** If `ImageName` is empty in the extracted CSV, check that `food_survey_reference.csv` has the correct `ImageID` mappings.
