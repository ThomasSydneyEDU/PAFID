import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_CSV = SCRIPT_DIR / "extracted_survey_data.csv"
STATE_CSV = SCRIPT_DIR / "state_ratings.csv"
OUTPUT_MEANS_CSV = SCRIPT_DIR.parent / "data" / "human_ratings.csv"
OUTPUT_INDIVIDUAL_CSV = SCRIPT_DIR.parent / "data" / "human_ratings_individual.csv"
OUTPUT_STATE_INDIVIDUAL_CSV = SCRIPT_DIR.parent / "data" / "human_state_individual.csv"
OUTPUT_STATE_SUMMARY_CSV = SCRIPT_DIR.parent / "data" / "human_state_summary.csv"

QUALITY_THRESHOLD = 80.0  # drop participants whose QualityCheck % is below this

# Per-participant physiological-state metrics to summarise (from state_ratings.csv).
STATE_METRICS = ["Hunger", "Thirst", "MinutesSinceEaten"]

# Optional sanity cap for "minutes since last ate" (e.g. a stray 30000 = 500 h).
# Values strictly greater than this are treated as missing before summarising.
# Set to None to keep every value as-entered (no filtering).
MINUTES_SINCE_EATEN_CAP = 1440  # 24 h; values above are treated as data-entry errors

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

    # ---- Physiological-state ratings (same participants, same anon IDs) ----
    process_state_ratings(subject_map)

def process_state_ratings(subject_map):
    """Summarise per-participant state ratings (hunger/thirst/time-since-eaten).

    Applies the SAME low-effort exclusion as the food ratings: a participant is
    only included if they survived the QualityCheck filter above. `subject_map`
    holds exactly those survivors (keyed on "<DataSource>_<ParticipantNumber>"),
    so reusing it guarantees identical participant exclusion and reuses the same
    anonymised subject IDs as human_ratings_individual.csv.
    """
    if not STATE_CSV.exists():
        print(f"[WARN] State ratings not found ({STATE_CSV.name}); skipping state summary.")
        return

    print(f"[INFO] Loading state ratings: {STATE_CSV}")
    state = pd.read_csv(STATE_CSV)

    # Rebuild the same subject key used for the food ratings.
    if "DataSource" in state.columns and "ParticipantNumber" in state.columns:
        state["_subject_str"] = (
            state["DataSource"].astype(str) + "_" + state["ParticipantNumber"].astype(str)
        )
    else:
        state["_subject_str"] = state["ParticipantNumber"].astype(str)

    n_state_raw = state["_subject_str"].nunique()

    # Keep only QC-passing participants, and attach their anonymous IDs.
    state = state[state["_subject_str"].isin(subject_map)].copy()
    state["anon_subject_id"] = state["_subject_str"].map(subject_map)
    n_state_kept = state["_subject_str"].nunique()
    print(f"[INFO] State ratings quality filter: kept {n_state_kept}/{n_state_raw} "
          f"participants ({n_state_raw - n_state_kept} dropped to match food ratings)")

    # Coerce metrics to numeric.
    for col in STATE_METRICS + ["TimeSinceEaten_Hours", "TimeSinceEaten_Minutes"]:
        if col in state.columns:
            state[col] = pd.to_numeric(state[col], errors="coerce")

    # Optional sanity cap on time-since-eaten.
    if MINUTES_SINCE_EATEN_CAP is not None and "MinutesSinceEaten" in state.columns:
        n_capped = (state["MinutesSinceEaten"] > MINUTES_SINCE_EATEN_CAP).sum()
        if n_capped:
            print(f"[INFO] Excluding {n_capped} MinutesSinceEaten value(s) > "
                  f"{MINUTES_SINCE_EATEN_CAP} from the summary.")
        state.loc[state["MinutesSinceEaten"] > MINUTES_SINCE_EATEN_CAP, "MinutesSinceEaten"] = np.nan

    # ---- De-identified individual state file ----
    indiv_cols = ["anon_subject_id"] + [
        c for c in ["Hunger", "Thirst", "TimeSinceEaten_Hours",
                    "TimeSinceEaten_Minutes", "MinutesSinceEaten"]
        if c in state.columns
    ]
    state_individual = state[indiv_cols].sort_values("anon_subject_id")
    OUTPUT_STATE_INDIVIDUAL_CSV.parent.mkdir(exist_ok=True, parents=True)
    state_individual.to_csv(OUTPUT_STATE_INDIVIDUAL_CSV, index=False)
    print(f"[OK] Wrote anonymized individual state ratings to: {OUTPUT_STATE_INDIVIDUAL_CSV}")

    # ---- Means / SDs summary (Overall + per DataSource) ----
    metrics = [m for m in STATE_METRICS if m in state.columns]

    def summarise(df, group_label):
        rows = []
        for m in metrics:
            vals = df[m].dropna()
            rows.append({
                "group": group_label,
                "metric": m,
                "n": int(vals.count()),
                "mean": round(vals.mean(), 2) if len(vals) else np.nan,
                "sd": round(vals.std(ddof=1), 2) if len(vals) > 1 else np.nan,
                "min": vals.min() if len(vals) else np.nan,
                "median": vals.median() if len(vals) else np.nan,
                "max": vals.max() if len(vals) else np.nan,
            })
        return rows

    summary_rows = summarise(state, "Overall")
    if "DataSource" in state.columns:
        for src, sub in state.groupby("DataSource"):
            summary_rows.extend(summarise(sub, src))

    summary_df = pd.DataFrame(summary_rows,
                              columns=["group", "metric", "n", "mean", "sd",
                                       "min", "median", "max"])
    summary_df.to_csv(OUTPUT_STATE_SUMMARY_CSV, index=False)
    print(f"[OK] Wrote state ratings summary (means/SDs) to: {OUTPUT_STATE_SUMMARY_CSV}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
