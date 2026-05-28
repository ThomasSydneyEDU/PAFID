# PAFID: Public AI-Generated Food Image Database Pipeline

PAFID is a modular, extensible pipeline for generating, validating, and rating photorealistic food stimuli using Generative AI. This repository provides the tools to extend the existing 350-item canonical database with new cultural or nutritional variants.

## Directory Structure

```text
PAFID/
├── src/                       # Core pipeline scripts
│   ├── generate_stimuli.py    # Generates images from a CSV list
│   ├── run_qc.py              # Automated Quality Control & aware ratings
│   ├── prepare_images.py      # Resizes and packages images for experiments
│   └── rate_images.py         # Conducts blind AI ratings
├── data/                      # Input lists and metadata
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

### 4. Set Your API Keys
The pipeline requires an API key to generate and evaluate images. By default, it uses the **Google Gemini API**. 

*   **Detailed Guide**: For step-by-step instructions on setting up your account and getting a key, see [GOOGLE_API_SETUP.md](GOOGLE_API_SETUP.md).
*   **Quick Start**: Go to [Google AI Studio](https://aistudio.google.com/app/apikey), sign in, and click "Create API Key".

Once you have your key, set it as an environment variable in your terminal:

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

*(Optional)* If you plan to use OpenAI's models instead, set your `OPENAI_API_KEY` in the same manner.

## Data Management

This repository includes two versions of the master metadata:
*   `data/Foodpictures_information_dynamic.csv`: The **working version** that the pipeline updates with new ratings and visual features.
*   `data/Foodpictures_information_reference.csv`: A **static copy** of the original study results (351 items) for alignment and reference.

## Usage Pipeline

### 1. Generate Stimuli
Add new food items to `data/food_list_initial_seed.csv`. Then run:
```bash
python src/generate_stimuli.py --limit 5
```
*   **Safe Defaults**: By default, the script will **skip** existing PNG files in the `rendered_images/` directory. Use `--overwrite` only if you explicitly wish to replace an image.
*   **Outputs**: Images and metadata saved to `rendered_images/`.

### 2. Quality Control & Aware Ratings
Verify the generated images and get initial AI ratings:
```bash
python src/run_qc.py --stimuli-dir rendered_images/
```
*   **Merges results** into `data/Foodpictures_information_dynamic.csv`.

### 3. Prepare for Experiments
Resize images and generate trial metadata:
```bash
python src/prepare_images.py --stimuli-dir rendered_images/
```

### 4. Extract Visual Features
Compute low-level visual statistics:
```bash
python src/extract_visual_features.py --stimuli-dir rendered_images/ --merge-canonical
```
*   **Merges results** into `data/Foodpictures_information_dynamic.csv`.

### 5. Reset Pipeline
If you want to undo your changes and return the database to the 350-item baseline:
```bash
python src/reset_pipeline.py
```
*   **Safety**: This script will ask for confirmation before deleting any non-canonical images or metadata you have generated.

## Extending the Database

**Demo Items:** The end of `data/food_list_initial_seed.csv` includes 3 example items ("Acai Bowl", "Peking Duck", "Baklava"). If you run the pipeline, it will generate images for these items as a demonstration. You can delete them from the CSV if you prefer to start fresh, or use them as a template for adding your own items.

To add new foods:
1.  Append new rows to `data/food_list_initial_seed.csv`.
2.  Specify the `Food`, `Category`, `Natural_vs_transformed`, and `Sweet_vs_savory` labels.
3.  Run the generation script.

## Citation
If you use this database or pipeline in your research, please cite:
(Insert Citation Here)
