import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_CSV = SCRIPT_DIR / "extracted_survey_data.csv"
OUTPUT_CSV = SCRIPT_DIR.parent / "data" / "human_ratings.csv"

QUALITY_THRESHOLD = 80.0  # drop participants whose QualityCheck % is below this

HUMAN_COLUMN_MAP = {
    "CalorieDensity": "human_calorie_density",
    "Healthiness":    "human_healthiness",
    "Appeal":         "human_appeal",
    "Sweet":          "human_sweetness",
    "Salty":          "human_saltiness",
    "Sour":           "human_sourness",
    "Bitter":         "human_bitterness",
    "Umami":          "human_savoriness",
    "Fatty":          "human_fattiness",
    "Spicy":          "human_spiciness",
}

def extract_leading_digit(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    match = re.match(r"^\d+", s)
    return float(match.group(0)) if match else np.nan

def main():
    if not SURVEY_CSV.exists():
        sys.exit(f"[ERROR] Survey extraction not found: {SURVEY_CSV}\n"
                 f"        Ensure you have your raw data extracted.")

    print(f"[INFO] Loading survey: {SURVEY_CSV}")
    survey = pd.read_csv(SURVEY_CSV)
    n_participants_raw = survey["ParticipantNumber"].nunique()
    print(f"        {len(survey):,} rows, {n_participants_raw} participants")

    survey["QualityCheck"] = pd.to_numeric(survey["QualityCheck"], errors="coerce")
    survey = survey[survey["QualityCheck"] >= QUALITY_THRESHOLD].copy()
    n_kept = survey["ParticipantNumber"].nunique()
    print(f"[INFO] Quality filter (>= {QUALITY_THRESHOLD:g}%): "
          f"kept {n_kept}/{n_participants_raw} participants "
          f"({n_participants_raw - n_kept} dropped)")

    if "Familiarity" in survey.columns:
        survey["Familiarity"] = survey["Familiarity"].apply(extract_leading_digit)
    
    for survey_col in HUMAN_COLUMN_MAP:
        if survey_col in survey.columns:
            survey[survey_col] = pd.to_numeric(survey[survey_col], errors="coerce")

    grouped = survey.groupby("ImageName")[[c for c in HUMAN_COLUMN_MAP.keys() if c in survey.columns]].mean()
    grouped.rename(columns=HUMAN_COLUMN_MAP, inplace=True)
    grouped = grouped.round(2)
    grouped.index.name = "filename"
    
    OUTPUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    grouped.to_csv(OUTPUT_CSV)
    print(f"[OK] Wrote aggregated human ratings to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
