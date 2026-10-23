#!/usr/bin/env python3
"""
Baselines Benchmark Runner (Phase 4).
Evaluates routing policies across the 200-query checkable evaluation benchmark (data/eval_200.jsonl):
1. always-small (100% cheap lane)
2. always-big (100% reasoning lane)
3. random-50 (50/50 uniform random)
4. static-router (cold-start bootstrap exemplars)
5. darwin-verifier-probed (exemplars + active boundary probes)

Computes 95% bootstrap confidence intervals (1,000 resamples) for accuracy and proxy cost.
"""

import json
import random
from pathlib import Path
import numpy as np

from router import config, costs
from router.embed import get_client
from router.index import ExemplarIndex

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_200.jsonl"
EXEMPLARS_PATH = DATA_DIR / "exemplars.jsonl"


def load_eval_data() -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_PATH}. Run curation scripts first.")
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap_ci(values: list[float], n_bootstraps: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and 95% percentile bootstrap confidence intervals."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    means = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(means, 100 * alpha))
    high = float(np.percentile(means, 100 * (1.0 - alpha)))
    return float(arr.mean()), low, high


def run_benchmark():
    items = load_eval_data()
    print(f"Loaded {len(items)} evaluation queries from {EVAL_PATH}.")

    embedder = get_client()
    texts = [item["text"] for item in items]
    print(f"Embedding {len(texts)} test queries with {embedder.backend}...")
    embeddings = embedder.embed(texts)

    # Load active index (probed)
    probed_index = ExemplarIndex()
    probed_index.load(embed_fn=embedder.embed)

    # Build bootstrap-only static index (exclude probe exemplars)
    bootstrap_index = ExemplarIndex()
    if config.EXEMPLARS_PATH.exists():
        with open(config.EXEMPLARS_PATH) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("source") == "bootstrap":
                        bootstrap_index.add(r["text"], r["label"], r["source"], r["embedding"], persist=False)

    # Estimate average token lengths for proxy pricing
    # Average ~40 tokens input, ~60 tokens output for simple; ~100 in, ~250 out for reasoning
    avg_tokens = {
        "simple": (40, 60),
        "reasoning": (100, 250),
    }

    def calc_item_cost(route: str) -> float:
        model = config.MODEL_FOR_ROUTE[route]
        tin, tout = avg_tokens[route]
        return costs.estimate(model, tin, tout)

    # Policy evaluations
    policies = ["always-small", "always-big", "random-50", "static-router", "darwin-probed"]
    results = {p: {"correct": [], "cost": []} for p in policies}

    rng = random.Random(42)

    for item, emb in zip(items, embeddings):
        expected = item["expected_route"]

        # 1. always-small
        route_small = "simple"
        results["always-small"]["correct"].append(1.0 if route_small == expected else 0.0)
        results["always-small"]["cost"].append(calc_item_cost(route_small))

        # 2. always-big
        route_big = "reasoning"
        results["always-big"]["correct"].append(1.0 if route_big == expected else 0.0)
        results["always-big"]["cost"].append(calc_item_cost(route_big))

        # 3. random-50
        route_rand = "simple" if rng.random() < 0.5 else "reasoning"
        results["random-50"]["correct"].append(1.0 if route_rand == expected else 0.0)
        results["random-50"]["cost"].append(calc_item_cost(route_rand))

        # 4. static-router (cold start bootstrap only)
        route_static, _ = bootstrap_index.route(emb)
        results["static-router"]["correct"].append(1.0 if route_static == expected else 0.0)
        results["static-router"]["cost"].append(calc_item_cost(route_static))

        # 5. darwin-probed (bootstrap + active boundary probes)
        route_probed, _ = probed_index.route(emb)
        results["darwin-probed"]["correct"].append(1.0 if route_probed == expected else 0.0)
        results["darwin-probed"]["cost"].append(calc_item_cost(route_probed))

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (200-Query Checkable Evaluation Set with 95% Bootstrap CIs)")
    print("=" * 80)
    print(f"{'Routing Policy':<22} | {'Routing Accuracy (95% CI)':<26} | {'Avg Proxy Cost/Req (95% CI)':<28}")
    print("-" * 80)

    summary_data = {}
    for p in policies:
        acc_mean, acc_low, acc_high = bootstrap_ci(results[p]["correct"])
        cost_mean, cost_low, cost_high = bootstrap_ci(results[p]["cost"])

        acc_str = f"{acc_mean*100:5.1f}% [{acc_low*100:4.1f}%, {acc_high*100:4.1f}%]"
        cost_str = f"${cost_mean:7.5f} [${cost_low:6.5f}, ${cost_high:6.5f}]"

        print(f"{p:<22} | {acc_str:<26} | {cost_str:<28}")

        summary_data[p] = {
            "accuracy_mean": acc_mean,
            "accuracy_ci": (acc_low, acc_high),
            "cost_mean": cost_mean,
            "cost_ci": (cost_low, cost_high),
        }

    print("-" * 80)
    return summary_data


if __name__ == "__main__":
    run_benchmark()
