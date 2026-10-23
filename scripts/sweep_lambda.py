#!/usr/bin/env python3
"""
Cost-Sensitivity Parameter (λ) Sweep & Pareto Frontier Generator (Phase 4).
Sweeps λ in [0.0, 1.0] to evaluate the quality vs. cost trade-off curve.
Computes 95% bootstrap confidence intervals for each λ operating point.
Generates assets/quality_vs_cost.png Pareto curve plot.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from router import config, costs
from router.embed import get_client
from router.index import ExemplarIndex
from scripts.baselines import load_eval_data, bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
PLOT_PATH = ASSETS_DIR / "quality_vs_cost.png"


def run_lambda_sweep(n_points: int = 11):
    items = load_eval_data()
    print(f"Loaded {len(items)} evaluation queries for λ parameter sweep.")

    embedder = get_client()
    texts = [item["text"] for item in items]
    print(f"Embedding {len(texts)} queries...")
    embeddings = embedder.embed(texts)

    index = ExemplarIndex()
    index.load(embed_fn=embedder.embed)
    print(f"Loaded index with {len(index.meta)} exemplars.")

    avg_tokens = {
        "simple": (40, 60),
        "reasoning": (100, 250),
    }

    def calc_item_cost(route: str) -> float:
        model = config.MODEL_FOR_ROUTE[route]
        tin, tout = avg_tokens[route]
        return costs.estimate(model, tin, tout)

    lambdas = np.linspace(0.0, 1.0, n_points)
    sweep_results = []

    print("\n" + "=" * 76)
    print(f"{'λ (Cost Weight)':<15} | {'Routing Accuracy (95% CI)':<26} | {'Avg Proxy Cost/Req':<20}")
    print("=" * 76)

    for lam in lambdas:
        correct_list = []
        cost_list = []
        for item, emb in zip(items, embeddings):
            expected = item["expected_route"]
            route, _ = index.route(emb, cost_weight=lam)
            correct_list.append(1.0 if route == expected else 0.0)
            cost_list.append(calc_item_cost(route))

        acc_mean, acc_low, acc_high = bootstrap_ci(correct_list)
        cost_mean, cost_low, cost_high = bootstrap_ci(cost_list)

        acc_str = f"{acc_mean*100:5.1f}% [{acc_low*100:4.1f}%, {acc_high*100:4.1f}%]"
        cost_str = f"${cost_mean:7.5f}"

        print(f"{lam:<15.2f} | {acc_str:<26} | {cost_str:<20}")

        sweep_results.append({
            "lambda": float(lam),
            "accuracy_mean": acc_mean,
            "accuracy_ci": (acc_low, acc_high),
            "cost_mean": cost_mean,
            "cost_ci": (cost_low, cost_high),
        })

    print("-" * 76)

    # Plot Pareto curve
    plot_pareto_curve(sweep_results)
    return sweep_results


def plot_pareto_curve(sweep_results: list[dict]):
    costs_vals = [r["cost_mean"] * 1000 for r in sweep_results]  # in $ / 1k requests
    acc_vals = [r["accuracy_mean"] * 100 for r in sweep_results]
    lams = [r["lambda"] for r in sweep_results]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(costs_vals, acc_vals, marker="o", color="#2563eb", linewidth=2, label="Darwin Router (λ sweep)")

    for c, a, l in zip(costs_vals, acc_vals, lams):
        if l in (0.0, 0.2, 0.5, 0.8, 1.0):
            plt.annotate(
                f"λ={l:.1f}",
                (c, a),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    plt.title("Empirical Quality vs. Proxy Cost Trade-Off Curve (Pareto Frontier)", fontsize=12, fontweight="bold")
    plt.xlabel("Average Proxy Cost per 1k Requests ($)", fontsize=10)
    plt.ylabel("Routing Accuracy (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Saved Pareto curve visualization to {PLOT_PATH}")


if __name__ == "__main__":
    run_lambda_sweep()
