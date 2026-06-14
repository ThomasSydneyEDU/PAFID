#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================================"
echo "Starting PAFID Pipeline - SAFE RERUN (No Image Rendering)"
echo "============================================================"

# Ensure Vertex AI credentials are set
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=usyd-llm

echo ""
echo "[1/4] Classifying items and updating metadata (skipping rendering)..."
python src/generate_stimuli.py --classify-only

echo ""
echo "[2/4] Running Quality Control & Aware AI Ratings (forcing overwrite)..."
python src/run_qc.py --stimuli-dir rendered_images/ --overwrite

echo ""
echo "[3/4] Running Blind AI Ratings (forcing overwrite)..."
python src/rate_images.py --stimuli-dir rendered_images/ --overwrite

echo ""
echo "[4/4] Extracting Visual Features and merging to canonical CSV..."
python src/extract_visual_features.py --stimuli-dir rendered_images/ --merge-canonical

echo ""
echo "============================================================"
echo "Pipeline complete! Check data/Foodpictures_information_dynamic.csv"
echo "============================================================"
