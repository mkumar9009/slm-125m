#!/bin/bash
# Quick start script for 125M SLM data pipeline
# Run this AFTER setting up .env.local with credentials

set -e

echo "125M SLM Data Pipeline - Quick Start"
echo "===================================="

# Load credentials
if [ ! -f .env.local ]; then
    echo "❌ .env.local not found!"
    echo "1. Copy .env.local.template to .env.local"
    echo "2. Fill in your Modal and HuggingFace credentials"
    echo "3. Run this script again"
    exit 1
fi

source .env.local
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET

# Verify credentials
echo "✓ Credentials loaded from .env.local"

# Create volume if it doesn't exist
echo "Creating Modal volume (slm-125m)..."
modal volume create slm-125m 2>/dev/null || echo "  (Volume already exists)"

echo ""
echo "Modal setup complete! Choose a phase to run:"
echo ""
echo "Phase 0 (Smoke Test):"
echo "  modal run modal_app.py"
echo ""
echo "Phase 0 (Measure Yields):"
echo "  modal run modal_app.py::measure"
echo ""
echo "Phase 1 (Clean):"
echo "  modal run modal_app.py::clean --fineweb-shards 5"
echo ""
echo "Phase 2 (Dedup + Decontaminate):"
echo "  modal run modal_app.py::dedup"
echo ""
echo "Phase 3 (Train Tokenizer):"
echo "  modal run modal_app.py::tokenizer"
echo ""
echo "Phase 4 (Tokenize + Pack):"
echo "  modal run modal_app.py::tokenize"
echo ""
echo "Verify Results:"
echo "  modal volume ls slm-125m /tokens"
echo "  modal volume ls slm-125m /tokenizer"
echo "  modal volume get slm-125m /tokens/index.json ./index.json"
echo ""
echo "Check Spend:"
echo "  modal billing report --json | python3 -c 'import sys,json; print(sum(float(r[\"cost\"]) for r in json.load(sys.stdin)))'"
echo ""
echo "Expected final result:"
echo "  - train: ~2.19B tokens"
echo "  - val: ~22.1M tokens (1% split)"
echo "  - cost: <$1 USD"
echo "  - time: ~40 minutes"
