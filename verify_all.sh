#!/bin/bash
# verify_all.sh — lightweight end-to-end artifact verification
# Requires: uv (https://docs.astral.sh/uv/) and a built C++ kernel (cpp/build/)
# Run from the repo root: bash verify_all.sh
set -e

echo "============================================="
echo "StructPiece Artifact Verification Suite"
echo "============================================="
echo "All steps parse pre-computed JSON logs."
echo "No GPU or heavy compute is required."
echo

echo "1/6. Running PyTest EGC Invariant Suite..."
PYTHONPATH=. uv run pytest tests/ -q

echo
echo "2/6. Verifying Core Ablation Results (Table 4)..."
PYTHONPATH=. uv run python reproduce/run_core_reeval.py --verify-only

echo
echo "3/6. Verifying NanoLM Results (Table 1)..."
PYTHONPATH=. uv run python reproduce/run_nanolm_experiments.py --verify-only

echo
echo "4/6. Verifying Morfessor / BPE-Knockout Baselines (Appendix)..."
PYTHONPATH=. uv run python reproduce/run_morfessor_baseline.py --verify-only

echo
echo "5/6. Verifying Extended Scale Results (Tables 15 & 38)..."
PYTHONPATH=. uv run python reproduce/run_extended_scale.py --verify-only

echo
echo "6/6. Generating LaTeX Tables from Logs..."
PYTHONPATH=. uv run python reproduce/generate_tables.py

echo
echo "============================================="
echo "✅  All Verifications Passed."
echo "============================================="
