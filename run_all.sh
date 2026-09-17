#!/usr/bin/env bash
# Reproduces the headline results in this repo. Uses the committed processed
# data (data/processed/pairs_clean.csv, data/eval/golden_set.csv) -- does NOT
# require downloading the raw Kaggle dataset. Should take well under 15 minutes
# (typically ~1-2 minutes) on a normal laptop CPU.
set -e

echo "==> Training intent classifier (TF-IDF + LogisticRegression)..."
PYTHONPATH=src python3 src/train_classifier.py

echo "==> Running evaluation harness against the golden set..."
python3 eval/run_eval.py

echo "==> Sanity-checking the end-to-end agent on a few examples..."
PYTHONPATH=src python3 src/agent.py

echo ""
echo "Done. See report/REPORT.md for the full write-up of these numbers."
echo "(Optional, requires ANTHROPIC_API_KEY: python3 eval/llm_judge.py --n 30)"
