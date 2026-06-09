"""
============================================================
Image Frequency Counter (Per Participant Response)
============================================================

This script counts how many times each image appears in the
survey dataset (i.e., how many participant responses exist per image).

Each row in the dataset represents one participant's response
to one image.

INPUT:
- extracted_survey_data.csv
  Must contain the column 'ImageName'

PROCESS:
1. Loads the dataset
2. Counts occurrences of each ImageName
3. Sorts results alphabetically
4. Prints counts and summary statistics to the terminal

OUTPUT:
- Printed results only (no files are created or modified)

============================================================
"""

import pandas as pd

# Load dataset (adjust path if needed)
df = pd.read_csv("extracted_survey_data.csv")

# Count occurrences per image
counts = df["ImageName"].value_counts()

# Sort alphabetically by image name
counts = counts.sort_index()

# Print results
print("\n=== Image Appearance Counts (Alphabetical Order) ===\n")

for image, count in counts.items():
    print(f"{image}: {count}")

# Optional summary
print("\n=== Summary ===")
print(f"Total images: {df['ImageName'].nunique()}")
print(f"Total responses: {len(df)}")
print(f"Average responses per image: {len(df) / df['ImageName'].nunique():.2f}")