#!/bin/bash
# ============================================================
# run_full_replication.sh
# Replicates the paper's model-pair matrix across all 5 budget
# scenarios. This is expensive — run selectively or in parallel.
# ============================================================

BUYER_MODELS=("gpt-4o-mini" "gpt-4.1")
SELLER_MODELS=("gpt-4o-mini" "gpt-4.1")
SUMMARY_MODEL="gpt-4o-mini"
N=3        # experiments per scenario (paper uses 5)
TURNS=30

for BUYER in "${BUYER_MODELS[@]}"; do
  for SELLER in "${SELLER_MODELS[@]}"; do
    echo ""
    echo "=============================="
    echo "Buyer: $BUYER | Seller: $SELLER"
    echo "=============================="
    python main.py \
      --products-file dataset/products.json \
      --buyer-model   "$BUYER" \
      --seller-model  "$SELLER" \
      --summary-model "$SUMMARY_MODEL" \
      --max-turns     $TURNS \
      --num-experiments $N \
      --output-dir    results/full_replication
  done
done

echo ""
echo "Full replication complete."
