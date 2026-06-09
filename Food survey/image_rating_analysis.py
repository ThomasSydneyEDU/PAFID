import pandas as pd

# -----------------------------
# Load data
# -----------------------------
file_path = "extracted_survey_data.csv"
df = pd.read_csv(file_path)

# Columns to analyze
rating_columns = ["CalorieDensity", "Healthiness", "Appeal", "Familiarity"]

# -----------------------------
# Convert Familiarity to numeric
# -----------------------------
df["Familiarity_numeric"] = df["Familiarity"].str[0].astype(int)

# Ensure other rating columns are numeric
for col in ["CalorieDensity", "Healthiness", "Appeal"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# Group by image and calculate means and SDs
# -----------------------------
grouped = df.groupby(["ImageID", "ImageName"])[["CalorieDensity", "Healthiness", "Appeal", "Familiarity_numeric"]]

means = grouped.mean().reset_index()
stds = grouped.std().reset_index()

# Rename columns
means.columns = ["ImageID", "ImageName"] + [col + "_Mean" for col in ["CalorieDensity", "Healthiness", "Appeal", "Familiarity"]]
stds.columns = ["ImageID", "ImageName"] + [col + "_SD" for col in ["CalorieDensity", "Healthiness", "Appeal", "Familiarity"]]

# Merge summary
summary = pd.merge(means, stds, on=["ImageID", "ImageName"])

# -----------------------------
# Round means & SDs to 2 decimals
# -----------------------------
for col in summary.columns:
    if "_Mean" in col or "_SD" in col:
        summary[col] = summary[col].round(2)

# -----------------------------
# High Appeal flag (threshold 70 / 100)
# -----------------------------
summary["High_Appeal"] = summary["Appeal_Mean"] >= 70

# -----------------------------
# Calculate Familiarity proportions per image
# -----------------------------
df["Familiarity_4_or_5"] = df["Familiarity_numeric"] >= 4
df["Familiarity_2_or_lower"] = df["Familiarity_numeric"] <= 2

fam_high_prop = df.groupby(["ImageID", "ImageName"])["Familiarity_4_or_5"].mean().reset_index()
fam_low_prop = df.groupby(["ImageID", "ImageName"])["Familiarity_2_or_lower"].mean().reset_index()

fam_high_prop.rename(columns={"Familiarity_4_or_5": "Proportion_Familiarity_4_or_5"}, inplace=True)
fam_low_prop.rename(columns={"Familiarity_2_or_lower": "Proportion_Familiarity_2_or_lower"}, inplace=True)

# Round proportions to 2 decimals
fam_high_prop["Proportion_Familiarity_4_or_5"] = fam_high_prop["Proportion_Familiarity_4_or_5"].round(2)
fam_low_prop["Proportion_Familiarity_2_or_lower"] = fam_low_prop["Proportion_Familiarity_2_or_lower"].round(2)

# Merge proportions into summary
summary = pd.merge(summary, fam_high_prop, on=["ImageID", "ImageName"])
summary = pd.merge(summary, fam_low_prop, on=["ImageID", "ImageName"])

# -----------------------------
# Top N most and least familiar (tie-aware ranks)
# -----------------------------
top_N = 20

# Most familiar ranks (ties share the same rank)
summary["Top_Most_Familiar_Rank"] = summary["Proportion_Familiarity_4_or_5"] \
    .rank(method="min", ascending=False)
summary["Top_Most_Familiar_Rank"] = summary["Top_Most_Familiar_Rank"].where(summary["Top_Most_Familiar_Rank"] <= top_N, pd.NA)

# Least familiar ranks (ties share the same rank)
summary["Top_Least_Familiar_Rank"] = summary["Proportion_Familiarity_2_or_lower"] \
    .rank(method="min", ascending=False)
summary["Top_Least_Familiar_Rank"] = summary["Top_Least_Familiar_Rank"].where(summary["Top_Least_Familiar_Rank"] <= top_N, pd.NA)

# -----------------------------
# Save combined summary CSV
# -----------------------------
summary.to_csv("image_rating_full_summary.csv", index=False)
print(f"Saved image_rating_full_summary.csv with {len(summary)} images (means, SDs, proportions rounded, unique ranks)")

# -----------------------------
# Optional: display top 5 most/least familiar
# -----------------------------
top_most = summary.sort_values("Proportion_Familiarity_4_or_5", ascending=False).head(5)
top_least = summary.sort_values("Proportion_Familiarity_2_or_lower", ascending=False).head(5)

print("\nTop 5 most familiar images:")
print(top_most[["ImageID", "ImageName", "Proportion_Familiarity_4_or_5", "Top_Most_Familiar_Rank"]])

print("\nTop 5 least familiar images:")
print(top_least[["ImageID", "ImageName", "Proportion_Familiarity_2_or_lower", "Top_Least_Familiar_Rank"]])
