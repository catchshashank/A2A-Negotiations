#!/bin/bash
# ============================================================
# run_baseline.sh
# Runs a single quick experiment to verify your setup works.
# Uses only the "high" and "low" budget scenarios, 1 trial each.
# Edit model names and paths as needed.
# ============================================================

python main.py \
  --products-file dataset/products.json \
  --buyer-model   gpt-4o-mini \
  --seller-model  gpt-4o-mini \
  --summary-model gpt-4o-mini \
  --max-turns 30 \
  --num-experiments 1 \
  --budget-scenarios high low \
  --output-dir results/baseline
