# ============================================================
# main.py
# ============================================================
# Entry point for the A2A negotiation replication.
#
# Faithfully replicates the original A2A-NT main.py.
# Key restoration: all 5 budget scenarios are active by default
# (the original had 3 commented out for faster iteration).
# ============================================================

import os
import json
import argparse

from Conversation import Conversation


# ------------------------------------------------------------------
# Budget scenarios
# ------------------------------------------------------------------

def calculate_budget_scenarios(retail_price_str: str, wholesale_price_str: str) -> dict:
    """
    Return the five budget scenarios defined in the paper:
      - high      : retail × 1.2
      - retail    : retail price
      - mid       : (retail + wholesale) / 2
      - wholesale : wholesale price
      - low       : wholesale × 0.8
    """
    retail    = float(retail_price_str.replace("$", "").replace(",", ""))
    wholesale = float(wholesale_price_str.replace("$", "").replace(",", ""))

    return {
        "high":      retail * 1.2,
        "retail":    retail,
        "mid":       (retail + wholesale) / 2,
        "wholesale": wholesale,
        "low":       wholesale * 0.8,
    }


# ------------------------------------------------------------------
# Single-product experiment runner
# ------------------------------------------------------------------

def run_experiment(
    product_index: int,
    products_file: str,
    buyer_model: str,
    seller_model: str,
    summary_model: str,
    max_turns: int,
    num_experiments: int = 3,
    output_dir: str = "results",
    append: bool = False,
    budget_scenarios: list = None,   # None = all five
) -> None:
    """Run `num_experiments` negotiations for a single product."""

    print(f"\nLoading product {product_index} from {products_file}...")

    with open(products_file) as f:
        products = json.load(f)

    if not isinstance(products, list):
        raise ValueError(f"{products_file} must contain a JSON list of products.")
    if not (0 <= product_index < len(products)):
        raise IndexError(f"product_index {product_index} is out of range (0–{len(products)-1}).")

    product    = products[product_index]
    product_id = product.get("id", product_index + 1)

    all_budgets = calculate_budget_scenarios(
        product["Retail Price"], product["Wholesale Price"]
    )

    # Filter to requested scenarios (default: all)
    if budget_scenarios:
        budgets = {k: v for k, v in all_budgets.items() if k in budget_scenarios}
    else:
        budgets = all_budgets

    for budget_name, budget_value in budgets.items():
        full_output_dir = os.path.join(
            output_dir,
            f"seller_{seller_model}",
            buyer_model,
            f"product_{product_id}",
            f"budget_{budget_name}",
        )
        os.makedirs(full_output_dir, exist_ok=True)

        # Count already-completed experiments for this scenario
        existing_files = [
            f for f in os.listdir(full_output_dir)
            if f.startswith(f"product_{product_id}_exp_") and f.endswith(".json")
        ]
        existing_count = len(existing_files)

        if existing_count >= num_experiments and not append:
            print(
                f"  Skipping {budget_name} — already has {existing_count} "
                f"experiment(s) (target: {num_experiments})"
            )
            continue

        remaining = num_experiments - existing_count
        start_num = (
            max(int(f.split("_exp_")[1].split(".")[0]) for f in existing_files) + 1
            if append and existing_files
            else existing_count
        )

        print(f"\n{'─'*20}")
        print(f"Scenario : {budget_name}  (${budget_value:.2f})")
        print(f"Product  : {product['Product Name']}")
        print(f"Running  : {remaining} experiment(s)  [existing: {existing_count}]")
        print(f"{'─'*20}")

        for i in range(remaining):
            experiment_num = start_num + i
            print(f"\n  Experiment {existing_count + i + 1}/{num_experiments}  (#{experiment_num})")

            conv = Conversation(
                product_data=product,
                buyer_model=buyer_model,
                seller_model=seller_model,
                summary_model=summary_model,
                max_turns=max_turns,
                experiment_num=experiment_num,
                budget=budget_value,
            )
            conv.budget_scenario = budget_name
            conv.run_negotiation()
            conv.save_conversation(full_output_dir)

            print(f"  Turns: {conv.completed_turns} | Result: {conv.negotiation_result} | "
                  f"Final price: ${conv.current_price_offer:.2f}")


# ------------------------------------------------------------------
# All-products runner
# ------------------------------------------------------------------

def run_all_products(
    products_file: str,
    buyer_model: str,
    seller_model: str,
    summary_model: str,
    max_turns: int,
    num_experiments: int = 5,
    output_dir: str = "results",
    append: bool = False,
    budget_scenarios: list = None,
) -> None:
    """Run experiments for every product in the dataset."""

    with open(products_file) as f:
        products = json.load(f)

    print(f"Found {len(products)} products in {products_file}.")
    print(f"Buyer model  : {buyer_model}")
    print(f"Seller model : {seller_model}")
    print(f"Summary model: {summary_model}")
    print(f"Max turns    : {max_turns}")
    print(f"Experiments  : {num_experiments} per scenario")

    for i, product in enumerate(products):
        print(f"\n{'='*50}")
        print(f"Product {i+1}/{len(products)}: {product['Product Name']}")
        print(f"{'='*50}")

        run_experiment(
            product_index=i,
            products_file=products_file,
            buyer_model=buyer_model,
            seller_model=seller_model,
            summary_model=summary_model,
            max_turns=max_turns,
            num_experiments=num_experiments,
            output_dir=output_dir,
            append=append,
            budget_scenarios=budget_scenarios,
        )

    print("\nAll products completed.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A2A-NT: Agent-to-Agent Negotiation Simulator (replication)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--products-file",    default="dataset/products.json",
                        help="Path to the products JSON file")
    parser.add_argument("--buyer-model",      default="gpt-3.5-turbo",
                        help="LLM for the buyer agent")
    parser.add_argument("--seller-model",     default="gpt-3.5-turbo",
                        help="LLM for the seller agent")
    parser.add_argument("--summary-model",    default="gpt-3.5-turbo",
                        help="LLM used to extract prices and evaluate negotiation state")
    parser.add_argument("--max-turns",        type=int, default=30,
                        help="Hard cap on negotiation turns (safety limit)")
    parser.add_argument("--num-experiments",  type=int, default=3,
                        help="Number of trials per product × budget scenario")
    parser.add_argument("--output-dir",       default="results",
                        help="Root directory for output JSON files")
    parser.add_argument("--append",           action="store_true",
                        help="Add runs to existing results instead of skipping completed scenarios")
    parser.add_argument("--product-index",    type=int, default=None,
                        help="Run a single product by index (0-based); omit to run all")
    parser.add_argument("--budget-scenarios", nargs="+",
                        choices=["high", "retail", "mid", "wholesale", "low"],
                        default=None,
                        help="Which budget scenarios to run (default: all five)")

    args = parser.parse_args()

    if args.product_index is not None:
        run_experiment(
            product_index=args.product_index,
            products_file=args.products_file,
            buyer_model=args.buyer_model,
            seller_model=args.seller_model,
            summary_model=args.summary_model,
            max_turns=args.max_turns,
            num_experiments=args.num_experiments,
            output_dir=args.output_dir,
            append=args.append,
            budget_scenarios=args.budget_scenarios,
        )
    else:
        run_all_products(
            products_file=args.products_file,
            buyer_model=args.buyer_model,
            seller_model=args.seller_model,
            summary_model=args.summary_model,
            max_turns=args.max_turns,
            num_experiments=args.num_experiments,
            output_dir=args.output_dir,
            append=args.append,
            budget_scenarios=args.budget_scenarios,
        )


if __name__ == "__main__":
    main()
