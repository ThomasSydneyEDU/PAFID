#!/usr/bin/env bash
# PAFID Pipeline
#
# Usage:
#   bash run_pipeline.sh              # Full run: generate + classify + QC + blind + features + prepare
#   bash run_pipeline.sh --safe-rerun # Skip image generation; re-classify and overwrite QC/ratings/features
#
# Extension flags (redirect outputs to an external project without touching PAFID):
#   --food-list=<path>      External CSV of food names (Food column). Seed list is never modified.
#   --output-dir=<path>     Directory for images, stimuli_master.json, and all derived outputs.
#   --stimulus-set=<label>  Provenance label stored in stimuli_master.json (e.g. foodspace_extension_2026).
#
# Other flags:
#   --safe-rerun   Skip image generation. Re-runs classification (resumable via
#                  version stamps), then overwrites QC, blind ratings, and visual
#                  features. Useful for backfilling new metadata columns or
#                  re-scoring after prompt changes. Images are never touched.
#   --skip-prepare Skip the prepare_images.py step (resizing + trial metadata).
#                  Implied by --safe-rerun.

set -euo pipefail

SAFE_RERUN=false
SKIP_PREPARE=false
FOOD_LIST=""
OUTPUT_DIR=""
STIMULUS_SET=""

for arg in "$@"; do
    case $arg in
        --safe-rerun)        SAFE_RERUN=true ;;
        --skip-prepare)      SKIP_PREPARE=true ;;
        --food-list=*)       FOOD_LIST="${arg#*=}" ;;
        --output-dir=*)      OUTPUT_DIR="${arg#*=}" ;;
        --stimulus-set=*)    STIMULUS_SET="${arg#*=}" ;;
        *)
            echo "[ERROR] Unknown flag: $arg"
            echo "Usage: bash run_pipeline.sh [--safe-rerun] [--skip-prepare]"
            echo "       [--food-list=<path>] [--output-dir=<path>] [--stimulus-set=<label>]"
            exit 1
            ;;
    esac
done

# Build optional extension flags to pass through to each script
EXT_GENERATE=""
STIMULI_DIR="rendered_images/"
EXT_QC=""
EXT_RATE=""
EXT_FEATURES=""
EXT_PREPARE=""

if [ -n "$FOOD_LIST" ]; then
    EXT_GENERATE="$EXT_GENERATE --food-list=$FOOD_LIST"
    EXT_QC="$EXT_QC --food-list=$FOOD_LIST"
fi
if [ -n "$OUTPUT_DIR" ]; then
    EXT_GENERATE="$EXT_GENERATE --output-dir=$OUTPUT_DIR"
    STIMULI_DIR="$OUTPUT_DIR"
    EXT_QC="$EXT_QC --dynamic-csv=$OUTPUT_DIR/Foodpictures_information_dynamic.csv"
    EXT_RATE="--csv=$OUTPUT_DIR/Foodpictures_information_dynamic.csv"
    EXT_FEATURES="--canonical-csv=$OUTPUT_DIR/Foodpictures_information_dynamic.csv"
    EXT_PREPARE="--output-dir=$OUTPUT_DIR/resized"
fi
if [ -n "$STIMULUS_SET" ]; then
    EXT_GENERATE="$EXT_GENERATE --stimulus-set=$STIMULUS_SET"
fi

# Vertex AI credentials — adjust project if needed
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=usyd-llm

echo "============================================================"
if $SAFE_RERUN; then
    echo " PAFID Pipeline — SAFE RERUN (no image generation)"
elif [ -n "$OUTPUT_DIR" ]; then
    echo " PAFID Pipeline — EXTENSION RUN (outputs → $OUTPUT_DIR)"
else
    echo " PAFID Pipeline — FULL RUN"
fi
echo "============================================================"

if $SAFE_RERUN; then
    echo ""
    echo "[1/4] Classifying items and updating metadata (skipping rendering)..."
    python src/generate_stimuli.py --classify-only $EXT_GENERATE

    echo ""
    echo "[2/4] Running Quality Control & Aware AI Ratings (overwrite)..."
    python src/run_qc.py --stimuli-dir "$STIMULI_DIR" --overwrite $EXT_QC

    echo ""
    echo "[3/4] Running Blind AI Ratings (overwrite)..."
    python src/rate_images.py --stimuli-dir "$STIMULI_DIR" --overwrite $EXT_RATE

    echo ""
    echo "[4/4] Extracting Visual Features and merging to canonical CSV..."
    python src/extract_visual_features.py --stimuli-dir "$STIMULI_DIR" --merge-canonical $EXT_FEATURES
else
    echo ""
    echo "[1/5] Generating stimuli and classifying..."
    python src/generate_stimuli.py $EXT_GENERATE

    echo ""
    echo "[2/5] Running Quality Control & Aware AI Ratings..."
    python src/run_qc.py --stimuli-dir "$STIMULI_DIR" $EXT_QC

    echo ""
    echo "[3/5] Running Blind AI Ratings..."
    python src/rate_images.py --stimuli-dir "$STIMULI_DIR" $EXT_RATE

    echo ""
    echo "[4/5] Extracting Visual Features and merging to canonical CSV..."
    python src/extract_visual_features.py --stimuli-dir "$STIMULI_DIR" --merge-canonical $EXT_FEATURES

    if ! $SKIP_PREPARE; then
        echo ""
        echo "[5/5] Preparing images for experiments..."
        python src/prepare_images.py --stimuli-dir "$STIMULI_DIR" $EXT_PREPARE
    else
        echo ""
        echo "[5/5] Skipping prepare_images (--skip-prepare)."
    fi
fi

echo ""
echo "============================================================"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Pipeline complete! Check $OUTPUT_DIR/Foodpictures_information_dynamic.csv"
else
    echo "Pipeline complete! Check data/Foodpictures_information_dynamic.csv"
fi
echo "============================================================"
