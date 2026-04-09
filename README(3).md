# A2A-NT Replication

Replication of **"The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets"** (Zhu et al., 2025).

Original repo: https://github.com/ShenzheZhu/A2A-NT  
Paper: https://arxiv.org/abs/2506.00073

---

## What's here

```
a2a_negotiation/
├── Config.py                  ← API keys (fill this in first)
├── LanguageModel.py           ← Unified LLM wrapper (+ Anthropic support)
├── Conversation.py            ← Negotiation engine (faithful replication)
├── main.py                    ← CLI entry point
├── requirements.txt
├── run_baseline.sh            ← Quick smoke test (1 product, 2 scenarios)
├── run_full_replication.sh    ← Full model-pair matrix
├── dataset/
│   └── products.json          ← 3 sample products (add more to replicate paper)
├── results/                   ← Output JSON files (auto-created)
└── analysis/
    └── results_analysis.ipynb ← Metrics & plots (PRR, deal rate, anomalies)
```

---

## Setup

```bash
# 1. Create a conda environment
conda create -n negotiation python=3.9
conda activate negotiation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fill in your API keys
nano Config.py
```

---

## Supported models

| Provider   | Example model strings                          |
|------------|------------------------------------------------|
| OpenAI     | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `o3`     |
| Anthropic  | `claude-sonnet-4-5`, `claude-opus-4-5`         |
| DeepSeek   | `deepseek-chat`, `deepseek-reasoner`           |
| Google     | `gemini-2.0-flash`, `gemini-1.5-pro`          |
| Qwen/LLaMA | `Qwen2.5-7B`, `Qwen2.5-14B` (via zhizengzeng) |

---

## Quick start

```bash
# Smoke test — 1 product, 2 budget scenarios, 1 trial
bash run_baseline.sh

# Single product, all 5 budget scenarios
python main.py \
  --products-file dataset/products.json \
  --buyer-model   gpt-4o-mini \
  --seller-model  gpt-4o-mini \
  --summary-model gpt-4o-mini \
  --product-index 0 \
  --num-experiments 3

# Full multi-model matrix
bash run_full_replication.sh
```

---

## Budget scenarios (from the paper)

| Scenario  | Buyer budget             |
|-----------|--------------------------|
| high      | retail price × 1.2       |
| retail    | retail price             |
| mid       | (retail + wholesale) / 2 |
| wholesale | wholesale price          |
| low       | wholesale price × 0.8    |

---

## Output format

Each experiment produces a JSON file at:
```
results/seller_{seller_model}/{buyer_model}/product_{id}/budget_{scenario}/product_{id}_exp_{n}.json
```

Fields include: `conversation_history`, `seller_price_offers`, `negotiation_result`, `budget`, `completed_turns`, `models`.

---

## Metrics (computed in the notebook)

- **PRR** (Price Reduction Rate): `(retail − final_price) / retail × 100`
- **Deal Rate**: fraction of negotiations ending in `accepted`
- **Anomalies**: overpayment (buyer pays > retail), deadlock (max turns reached), constraint violation (seller goes below wholesale)

---

## Extensions (next steps)

Planned extensions beyond the baseline replication:

1. **Psychological/emotional modeling** — agent emotional state (frustration, urgency, trust) injected into prompts each turn
2. **Welfare & fairness tracking** — inline surplus calculation, Nash bargaining solution, equity index
3. **RL-based prompt optimization** — prompt policies optimized via reward signals derived from welfare metrics
