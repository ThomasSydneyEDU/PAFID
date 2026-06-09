import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_CSV = SCRIPT_DIR / "extracted_survey_data.csv"
OUTPUT_MEANS_CSV = SCRIPT_DIR.parent / "data" / "human_ratings.csv"
OUTPUT_INDIVIDUAL_CSV = SCRIPT_DIR.parent / "data" / "human_ratings_individual.csv"

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
    "Familiarity":    "human_familiarity"
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

    # ---- Quality filter ----
    survey["QualityCheck"] = pd.to_numeric(survey["QualityCheck"], errors="coerce")
    survey = survey[survey["QualityCheck"] >= QUALITY_THRESHOLD].copy()
    
    # ---- Anonymize Subject IDs ----
    # Combine DataSource + ParticipantNumber to ensure uniqueness across platforms, then map to anonymous ID
    if "DataSource" in survey.columns and "ParticipantNumber" in survey.columns:
        survey["_subject_str"] = survey["DataSource"].astype(str) + "_" + survey["ParticipantNumber"].astype(str)
    else:
        survey["_subject_str"] = survey["ParticipantNumber"].astype(str)
        
    unique_subjects = survey["_subject_str"].unique()
    subject_map = {orig: i+1 for i, orig in enumerate(unique_subjects)}
    survey["anon_subject_id"] = survey["_subject_str"].map(subject_map)
    
    n_kept = len(unique_subjects)
    print(f"[INFO] Quality filter (>= {QUALITY_THRESHOLD:g}%): "
          f"kept {n_kept}/{n_participants_raw} participants "
          f"({n_participants_raw - n_kept} dropped)")

    # ---- Clean Rating Data ----
    if "Familiarity" in survey.columns:
        survey["Familiarity"] = survey["Familiarity"].apply(extract_leading_digit)
    
    for survey_col in HUMAN_COLUMN_MAP:
        if survey_col in survey.columns:
            survey[survey_col] = pd.to_numeric(survey[survey_col], errors="coerce")

    # ---- Rename and Export Individual Data ----
    survey.rename(columns={"ImageName": "filename"}, inplace=True)
    survey.rename(columns=HUMAN_COLUMN_MAP, inplace=True)
    
    # Select only the de-identified tracking and rating columns
    cols_to_keep = ["anon_subject_id", "filename"] + [v for k, v in HUMAN_COLUMN_MAP.items() if v in survey.columns]
    individual_df = survey[cols_to_keep].copy()
    
    # Drop rows where all actual ratings are NaN (just in case)
    rating_cols = [v for k, v in HUMAN_COLUMN_MAP.items() if v in individual_df.columns]
    individual_df.dropna(subset=rating_cols, how="all", inplace=True)
    
    OUTPUT_INDIVIDUAL_CSV.parent.mkdir(exist_ok=True, parents=True)
    individual_df.to_csv(OUTPUT_INDIVIDUAL_CSV, index=False)
    print(f"[OK] Wrote anonymized individual ratings to: {OUTPUT_INDIVIDUAL_CSV}")

    # ---- Compute and Export Means ----
    grouped = individual_df.groupby("filename")[rating_cols].mean()
    grouped = grouped.round(2)
    
    grouped.to_csv(OUTPUT_MEANS_CSV)
    print(f"[OK] Wrote aggregated human ratings to: {OUTPUT_MEANS_CSV}")

if __name__ == "__main__":
    main()
