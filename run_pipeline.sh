#!/usr/bin/env bash
# PAFID Pipeline
#
# Usage:
#   bash run_pipeline.sh              # Full run: generate + classify + QC + blind + features + prepare
#   bash run_pipeline.sh --safe-rerun # Skip image generation; re-classify and overwrite QC/ratings/features
#
# Flags:
#   --safe-rerun   Skip image generation. Re-runs classification (resumable via
#                  version stamps), then overwrites QC, blind ratings, and visual
#                  features. Useful for backfilling new metadata columns or
#                  re-scoring after prompt changes. Images are never touched.
#   --skip-prepare Skip the prepare_images.py step (resizing + trial metadata).
#                  Implied by --safe-rerun.

set -euo pipefail

SAFE_RERUN=false
SKIP_PREPARE=false

for arg in "$@"; do
    case $arg in
        --safe-rerun)   SAFE_RERUN=true ;;
        --skip-prepare) SKIP_PREPARE=true ;;
        *)
            echo "[ERROR] Unknown flag: $arg"
            echo "Usage: bash run_pipeline.sh [--safe-rerun] [--skip-prepare]"
            exit 1
            ;;
    esac
done

# Vertex AI credentials — adjust project if needed
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=usyd-llm

echo "============================================================"
if $SAFE_RERUN; then
    echo " PAFID Pipeline — SAFE RERUN (no image generation)"
else
    echo " PAFID Pipeline — FULL RUN"
fi
echo "============================================================"

if $SAFE_RERUN; then
    echo ""
    echo "[1/4] Classifying items and updating metadata (skipping rendering)..."
    python src/generate_stimuli.py --classify-only

    echo ""
    echo "[2/4] Running Quality Control & Aware AI Ratings (overwrite)..."
    python src/run_qc.py --stimuli-dir rendered_images/ --overwrite

    echo ""
    echo "[3/4] Running Blind AI Ratings (overwrite)..."
    python src/rate_images.py --stimuli-dir rendered_images/ --overwrite

    echo ""
    echo "[4/4] Extracting Visual Features and merging to canonical CSV..."
    python src/extract_visual_features.py --stimuli-dir rendered_images/ --merge-canonical
else
    echo ""
    echo "[1/5] Generating stimuli and classifying..."
    python src/generate_stimuli.py

    echo ""
    echo "[2/5] Running Quality Control & Aware AI Ratings..."
    python src/run_qc.py --stimuli-dir rendered_images/

    echo ""
    echo "[3/5] Running Blind AI Ratings..."
    python src/rate_images.py --stimuli-dir rendered_images/

    echo ""
    echo "[4/5] Extracting Visual Features and merging to canonical CSV..."
    python src/extract_visual_features.py --stimuli-dir rendered_images/ --merge-canonical

    if ! $SKIP_PREPARE; then
        echo ""
        echo "[5/5] Preparing images for experiments..."
        python src/prepare_images.py --stimuli-dir rendered_images/
    else
        echo ""
        echo "[5/5] Skipping prepare_images (--skip-prepare)."
    fi
fi

echo ""
echo "============================================================"
echo "Pipeline complete! Check data/Foodpictures_information_dynamic.csv"
echo "============================================================"
