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

# Resolve the directory this script lives in so src/ paths work regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    python3 "$SCRIPT_DIR/src/"classify_food.py $EXT_GENERATE

    echo ""
    echo "[2/4] Running Quality Control (overwrite)..."
    python3 "$SCRIPT_DIR/src/"run_qc.py --stimuli-dir "$STIMULI_DIR" --overwrite $EXT_QC || \
        echo "[WARN] QC flagged some items — see qc_issues.json. Continuing pipeline."

    echo ""
    echo "[3/4] Running Perceptual AI Ratings (Blind & Aware; overwrite)..."
    python3 "$SCRIPT_DIR/src/"rate_images.py --stimuli-dir "$STIMULI_DIR" --overwrite $EXT_RATE

    echo ""
    echo "[4/4] Extracting Visual Features and merging to canonical CSV..."
    python3 "$SCRIPT_DIR/src/"extract_visual_features.py --stimuli-dir "$STIMULI_DIR" --merge-canonical $EXT_FEATURES
else
    echo ""
    echo "[1/5] Classifying food categories and processing attributes..."
    python3 "$SCRIPT_DIR/src/"classify_food.py $EXT_GENERATE

    echo ""
    echo "[2/5] Generating food stimulus images & running automated Quality Control..."
    python3 "$SCRIPT_DIR/src/"generate_images.py $EXT_GENERATE
    python3 "$SCRIPT_DIR/src/"run_qc.py --stimuli-dir "$STIMULI_DIR" $EXT_QC || \
        echo "[WARN] QC flagged some items — see qc_issues.json. Continuing pipeline."

    echo ""
    echo "[3/5] Running Perceptual AI Ratings (Blind & Aware)..."
    python3 "$SCRIPT_DIR/src/"rate_images.py --stimuli-dir "$STIMULI_DIR" $EXT_RATE

    echo ""
    echo "[4/5] Extracting Visual Features and merging to canonical CSV..."
    python3 "$SCRIPT_DIR/src/"extract_visual_features.py --stimuli-dir "$STIMULI_DIR" --merge-canonical $EXT_FEATURES

    if ! $SKIP_PREPARE; then
        echo ""
        echo "[5/5] Preparing images for experiments..."
        python3 "$SCRIPT_DIR/src/"prepare_images.py --stimuli-dir "$STIMULI_DIR" $EXT_PREPARE
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
