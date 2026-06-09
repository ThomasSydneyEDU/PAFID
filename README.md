# PAFID: Public AI-Generated Food Image Database Pipeline

PAFID is a modular, extensible pipeline for generating, validating, and rating photorealistic food stimuli using Generative AI. This repository provides the tools to extend the existing 350-item canonical database with new cultural or nutritional variants.

## Directory Structure

```text
PAFID/
├── src/                       # Core pipeline scripts
│   ├── generate_stimuli.py    # Generates images and assigns AI labels from a food list
│   ├── run_qc.py              # Automated Quality Control & aware ratings
│   ├── prepare_images.py      # Resizes and packages images for experiments
│   ├── rate_images.py         # Conducts blind AI ratings
│   └── extract_visual_features.py  # Computes low-level visual statistics
├── data/                      # Input lists and metadata
│   ├── food_list_initial_seed.csv          # Seed list (Food column only)
│   ├── Foodpictures_information_dynamic.csv
│   └── Foodpictures_information_reference.csv
├── assets/                    # Reference assets
│   └── plates/                # Reference plate images for visual consistency
├── rendered_images/           # Generated high-res images and metadata
├── resized_images/            # Experiment-ready images and JS trial scripts
└── requirements.txt           # Python dependencies
```

## Setup

### 1. Download the Code
You can download the code using Git or by downloading a ZIP file.

**Option A: Using Git**
```bash
git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo/PAFID
```
*(Note: Replace `YourUsername/YourRepo` with the actual URL to this repository once published.)*

**Option B: Downloading as a ZIP**
1. Click the green "**Code**" button at the top of the GitHub repository page.
2. Select "**Download ZIP**".
3. Extract the ZIP file to your computer.
4. Open your terminal (Mac/Linux) or Command Prompt/PowerShell (Windows).
5. Navigate to the extracted `PAFID` folder using the `cd` command (e.g., `cd Downloads/FoodStimGeneration-main/PAFID`).

### 2. Set Up a Virtual Environment (Recommended)
To avoid conflicts with other Python projects on your computer, create a virtual environment.

**Create the virtual environment:**
```bash
python -m venv venv
```
*(If `python` doesn't work, try `python3 -m venv venv`)*

**Activate the virtual environment:**
*   **Mac/Linux:**
    ```bash
    source venv/bin/activate
    ```
*   **Windows (Command Prompt):**
    ```cmd
    venv\Scripts\activate.bat
    ```
*   **Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```

### 3. Install Dependencies
Once your virtual environment is active (you should see `(venv)` in your command line prompt), install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Set Up Google Gemini Access

The pipeline uses **Google Gemini** for image generation and automated food labelling. Two authentication methods are supported — see [GOOGLE_API_SETUP.md](GOOGLE_API_SETUP.md) for full instructions.

#### Option A — Vertex AI (Recommended)

Uses your Google Cloud project credentials. No API key required.

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project your-project-id
```

Then set these environment variables (add to `~/.zshrc` or `~/.bash_profile` to make permanent):

*   **Mac/Linux:**
    ```bash
    export GOOGLE_CLOUD_PROJECT="your-project-id"
    export GOOGLE_CLOUD_LOCATION="global"
    export GOOGLE_GENAI_USE_VERTEXAI="True"
    ```
*   **Windows (PowerShell):**
    ```powershell
    [System.Environment]::SetEnvironmentVariable("GOOGLE_CLOUD_PROJECT", "your-project-id", "User")
    [System.Environment]::SetEnvironmentVariable("GOOGLE_CLOUD_LOCATION", "global", "User")
    [System.Environment]::SetEnvironmentVariable("GOOGLE_GENAI_USE_VERTEXAI", "True", "User")
    ```

#### Option B — AI Studio API Key

For quick personal use. Go to [Google AI Studio](https://aistudio.google.com/app/apikey), sign in, and create an API key. Then:

*   **Mac/Linux:**
    ```bash
    export GEMINI_API_KEY="your-gemini-api-key"
    ```
*   **Windows (Command Prompt):**
    ```cmd
    set GEMINI_API_KEY=your-gemini-api-key
    ```
*   **Windows (PowerShell):**
    ```powershell
    $env:GEMINI_API_KEY="your-gemini-api-key"
    ```

*(Optional)* If you plan to use OpenAI's models for image generation instead, set your `OPENAI_API_KEY` in the same manner.

## Data Management

This repository includes two versions of the master metadata:
*   `data/Foodpictures_information_dynamic.csv`: The **working version** that the pipeline updates with new ratings and visual features.
*   `data/Foodpictures_information_reference.csv`: A **static copy** of the original study results (351 items) for alignment and reference.

## AI-Assigned Labels

Rather than requiring labels to be specified manually in the seed list, the pipeline uses Gemini to assign four labels per food item automatically:

| Column | Description |
|---|---|
| `Category_WHO_10` | WHO/FAO food category (10 classes: Fruits, Vegetables, Meat, Fish, Dairy and eggs, Bakery wares and cereals, Confectionery and sweets, Beverages, Ready-to-eat savories, Prepared foods) |
| `Category_Simple_6` | Simplified food category (6 classes: Fruit, Vegetable, Protein, Grain, Dessert, Dish) |
| `Natural_vs_transformed` | Whether the food is **Natural** (identifiable as a single biological source, even if minimally prepared) or **Transformed** (substantially altered through processing, combination, or cooking) |
| `Transformation_score` | Continuous score 0–100 reflecting degree of processing (0–10 = raw/whole; 85–100 = highly manufactured) |

Classification uses `gemini-2.0-flash` via a single text call per food item and does not depend on the generated image.

## Usage Pipeline

### 1. Generate Stimuli
Add new food items to `data/food_list_initial_seed.csv` (one food name per row, `Food` column only). Then run:
```bash
python src/generate_stimuli.py --limit 5
```
*   For each food, Gemini assigns all four labels (see above) before generating the image.
*   **Safe Defaults**: By default, the script will **skip** existing PNG files in `rendered_images/`. Use `--overwrite` only if you explicitly wish to replace an image.
*   **Outputs**: Images and per-item metadata saved to `rendered_images/stimuli_master.json`.

### 2. Classify Without Generating Images
To assign or update labels on an existing set of images without regenerating them:
```bash
python src/generate_stimuli.py --classify-only
```
*   Reads food names from `food_list_initial_seed.csv`.
*   Calls Gemini text API to assign all four labels per food.
*   Updates `rendered_images/stimuli_master.json` in-place, preserving all existing fields (QC ratings, prompts, visual features, etc.).
*   Creates a `.bak` backup of the master before writing.
*   **Images are never touched.**

Useful flags:
```bash
python src/generate_stimuli.py --classify-only --food "Apple"   # single item
python src/generate_stimuli.py --classify-only --limit 10       # first N items
```

### 3. Quality Control & Aware Ratings
Verify the generated images and get "aware" AI ratings (where the AI knows the food label):
```bash
python src/run_qc.py --stimuli-dir rendered_images/
```
*   **Merges results** into `data/Foodpictures_information_dynamic.csv`.

### 4. Blind AI Ratings
Acquire "blind" AI ratings (where the AI only sees the image and must guess what the food is):
```bash
python src/rate_images.py --stimuli-dir rendered_images/
```
*   **Merges results** into `data/Foodpictures_information_dynamic.csv`.

### 5. Extract Visual Features
Compute low-level visual statistics (luminance, contrast, edge energy, etc.):
```bash
python src/extract_visual_features.py --stimuli-dir rendered_images/ --merge-canonical
```
*   **Merges results** into `data/Foodpictures_information_dynamic.csv`.

### 6. Prepare for Experiments
Resize images and generate trial metadata:
```bash
python src/prepare_images.py --stimuli-dir rendered_images/
```

### 6. Reset Pipeline
If you want to undo your changes and return the database to the 350-item baseline:
```bash
python src/reset_pipeline.py
```
*   **Safety**: This script will ask for confirmation before deleting any non-canonical images or metadata you have generated.

## Extending the Database

To add new foods, simply append rows to `data/food_list_initial_seed.csv` with a single `Food` column. All labels (category, natural/transformed, transformation score) are assigned automatically by Gemini when you run the generation script.

```csv
Food
Acai Bowl
Peking Duck
Baklava
```

The seed list includes 3 demo items at the end as an example. You can remove them or use them as a template.

## Citation
If you use this database or pipeline in your research, please cite:
(Insert Citation Here)
