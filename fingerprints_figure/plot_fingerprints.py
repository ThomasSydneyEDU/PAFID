#!/usr/bin/env python3
"""
Generate fingerprint radar plots for each Category_Culinary_9 category.

Reads:  data/Foodpictures_information_dynamic.csv
Writes: fingerprints_figure/category_fingerprints.pdf
        fingerprints_figure/category_fingerprints.png

Each radar plot shows 10 attributes averaged across all foods in the category:
  - 7 sensory ratings: sweetness, saltiness, savoriness, sourness,
                       bitterness, spiciness, fattiness
  - 2 appraisal dimensions: healthiness, calorie density
  - 1 process variable: transformation score (highlighted in amber)

All axes are on a 0–100 scale. Transformation score is positioned at 12 o'clock
to visually distinguish it from the human-rated attributes.

Usage:
    python fingerprints_figure/plot_fingerprints.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "Foodpictures_information_dynamic.csv"
OUT_DIR = ROOT / "fingerprints_figure"

AXES = [
    "Transformation_score",   # top (12 o'clock) — process variable
    "human_sweetness",
    "human_saltiness",
    "human_savoriness",
    "human_sourness",
    "human_bitterness",
    "human_spiciness",
    "human_fattiness",
    "human_healthiness",
    "human_calorie_density",
]

LABELS = [
    "Transformation",
    "Sweetness",
    "Saltiness",
    "Savoriness",
    "Sourness",
    "Bitterness",
    "Spiciness",
    "Fattiness",
    "Healthiness",
    "Calorie density",
]

CATEGORY_COL = "Category_Culinary_9"

FILL_COLOR    = "#7F77DD"
FILL_ALPHA    = 0.18
LINE_COLOR    = "#7F77DD"
POINT_COLOR   = "#7F77DD"
LABEL_COLOR   = "#5f5e5a"
XFM_COLOR     = "#BA7517"   # amber — highlights transformation score axis
FLAVOUR_COLOR = "#2E86AB"   # blue — flavour/sensory ratings
APPRAISAL_COLOR = "#C0392B" # red — food attributes (healthiness, calorie density)
GRID_COLOR    = "#cccccc"
BG_COLOR      = "white"

FLAVOUR_LABELS    = {"Sweetness", "Saltiness", "Savoriness", "Sourness",
                     "Bitterness", "Spiciness", "Fattiness"}
APPRAISAL_LABELS  = {"Healthiness", "Calorie density"}


def build_radar(ax: plt.Axes, values: list[float], title: str, n: int) -> None:
    N = len(values)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    values_closed = values + [values[0]]

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, values_closed,
            color=LINE_COLOR, linewidth=1.4, zorder=3)
    ax.fill(angles_closed, values_closed,
            color=FILL_COLOR, alpha=FILL_ALPHA, zorder=2)
    ax.scatter(angles, values,
               color=POINT_COLOR, s=14, zorder=4, linewidths=0)

    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--")
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_facecolor(BG_COLOR)

    ax.set_xticks(angles)
    tick_labels = ax.set_xticklabels(LABELS, fontsize=7.5)
    for label in tick_labels:
        text = label.get_text()
        if text == "Transformation":
            label.set_color(XFM_COLOR)
            label.set_fontweight("semibold")
        elif text in FLAVOUR_LABELS:
            label.set_color(FLAVOUR_COLOR)
            label.set_fontweight("semibold")
        elif text in APPRAISAL_LABELS:
            label.set_color(APPRAISAL_COLOR)
            label.set_fontweight("semibold")
        else:
            label.set_color(LABEL_COLOR)

    ax.set_title(f"{title} (n = {n})", fontsize=9.5, fontweight="medium",
                 color="#2c2c2a", pad=14, loc="center")


def main() -> None:
    dyn = pd.read_csv(DATA_CSV)

    categories = (
        dyn.groupby(CATEGORY_COL)[AXES]
        .mean()
        .round(1)
        .reset_index()
        .sort_values(CATEGORY_COL)
    )
    counts = dyn.groupby(CATEGORY_COL).size().rename("n")

    n_cats = len(categories)
    ncols = 3
    nrows = int(np.ceil(n_cats / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(13, 13),
        subplot_kw=dict(polar=True),
    )
    fig.patch.set_facecolor(BG_COLOR)

    for i, row in categories.iterrows():
        cat_name = row[CATEGORY_COL]
        values = row[AXES].tolist()
        n = counts.loc[cat_name]
        ax = axes.flat[i]
        display_name = cat_name.replace(" - ", " – ")
        build_radar(ax, values, display_name, n)

    for j in range(n_cats, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle(
        "Culinary food category fingerprints",
        fontsize=13, fontweight="medium", color="#2c2c2a", y=0.98,
    )

    # Legend for axis label colour coding
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=FLAVOUR_COLOR,    label="Flavour ratings"),
        mpatches.Patch(color=APPRAISAL_COLOR,  label="Food attributes"),
        mpatches.Patch(color=XFM_COLOR,        label="Transformation score"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.subplots_adjust(
        hspace=0.55, wspace=0.45,
        left=0.04, right=0.96,
        top=0.86, bottom=0.03,
    )

    out_pdf = OUT_DIR / "category_fingerprints.pdf"
    out_png = OUT_DIR / "category_fingerprints.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=150)
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=150)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
