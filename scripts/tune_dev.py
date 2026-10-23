#!/usr/bin/env python3
"""Tune hyperparameters on DEV split ONLY and freeze (Step 4).

Grid-searches over:
- k: [3, 5, 7, 9]
- sim_exponent: [1, 2, 3, 4, 6]
- margin_threshold: [0.15, 0.30, 0.45, 0.55, 0.65, 0.75]
- near_duplicate_cutoff: [0.90, 0.95, 0.985]

Objective:
Maximize end-to-end correctness on dev while maintaining significant cost savings (>= 35% vs Always-Big).
Saves chosen config to config/frozen.json and freezes parameters.
NEVER touches the test split.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from router import config
from router.costs import calculate_cost

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CONFIG_DIR = ROOT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DEV_PATH = DATA_DIR / "dev.jsonl"
DEV_CACHE_PATH = CACHE_DIR / "dev.jsonl"
EXEMPLARS_PATH = DATA_DIR / "exemplars.jsonl"
FROZEN_CONFIG_PATH = CONFIG_DIR / "frozen.json"


def load_jsonl(p: Path) -> list[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("=" * 70)
    print("Hyperparameter Grid Search on DEV Split Only (Freezing Parameters)")
    print("=" * 70)

    if not DEV_CACHE_PATH.exists():
        print(f"ERROR: {DEV_CACHE_PATH} not found. Cache dev answers first.", file=sys.stderr)
        sys.exit(1)
    if not EXEMPLARS_PATH.exists():
        print(f"ERROR: {EXEMPLARS_PATH} not found. Rebuild index from seeds first.", file=sys.stderr)
        sys.exit(1)

    dev_items = load_jsonl(DEV_PATH)
    dev_cache = {r["id"]: r for r in load_jsonl(DEV_CACHE_PATH)}
    exemplars = load_jsonl(EXEMPLARS_PATH)

    print(f"Loaded {len(dev_items)} dev items, {len(dev_cache)} cache records, {len(exemplars)} exemplars.")

    # Validate complete cache
    for item in dev_items:
        if item["id"] not in dev_cache:
            print(f"ERROR: Incomplete dev cache. Missing item {item['id']}", file=sys.stderr)
            sys.exit(1)

    # Pre-embed dev queries and exemplars
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dev_texts = [d["text"] for d in dev_items]
    dev_embs = model.encode(dev_texts, convert_to_numpy=True, normalize_embeddings=True)

    # Exemplar embeddings and labels
    ex_embs = np.array([e["embedding"] for e in exemplars], dtype=np.float32)
    ex_labels = np.array([e["label"] for e in exemplars])

    # Compute cosine similarities matrix: shape (len(dev), len(exemplars))
    sim_matrix = np.dot(dev_embs, ex_embs.T)

    # Baseline cost: Always-Big on dev
    always_big_costs = [
        calculate_cost(dev_cache[item["id"]]["big_model"],
                       dev_cache[item["id"]]["big_in_tokens"],
                       dev_cache[item["id"]]["big_out_tokens"])
        for item in dev_items
    ]
    baseline_big_cost = float(np.mean(always_big_costs))
    always_big_acc = float(np.mean([dev_cache[item["id"]]["big_ok"] for item in dev_items]))

    print(f"\nDev Baselines:")
    print(f"  Always-Big Acc:  {always_big_acc * 100:.1f}%")
    print(f"  Always-Big Cost: ${baseline_big_cost:.6f} / req")

    k_vals = [3, 5, 7, 9]
    exp_vals = [1.0, 2.0, 3.0, 4.0, 6.0]
    margin_vals = [0.15, 0.30, 0.45, 0.55, 0.65, 0.75]
    dup_cutoffs = [0.90, 0.95, 0.985]

    results = []

    for k in k_vals:
        for exp in exp_vals:
            for margin_th in margin_vals:
                for dup_cut in dup_cutoffs:
                    correct_count = 0
                    costs = []

                    for q_idx, item in enumerate(dev_items):
                        qid = item["id"]
                        rec = dev_cache[qid]
                        sims = sim_matrix[q_idx]

                        max_sim_idx = int(np.argmax(sims))
                        max_sim = float(sims[max_sim_idx])

                        # Near duplicate bypass
                        if max_sim >= dup_cut:
                            route = ex_labels[max_sim_idx]
                        else:
                            # Top-k neighbors
                            top_k_indices = np.argsort(sims)[-k:][::-1]
                            top_sims = sims[top_k_indices]
                            top_labels = ex_labels[top_k_indices]

                            # Non-negative weights
                            clipped_sims = np.maximum(0.0, top_sims)
                            weights = clipped_sims ** exp

                            simple_score = float(np.sum(weights[top_labels == "simple"]))
                            reasoning_score = float(np.sum(weights[top_labels == "reasoning"]))
                            tot = simple_score + reasoning_score

                            if tot == 0:
                                margin = 0.0
                                raw_vote = "reasoning"
                            else:
                                margin = abs(simple_score - reasoning_score) / tot
                                raw_vote = "simple" if simple_score > reasoning_score else "reasoning"

                            # Confidence margin guardrail: if ambiguous, escalate to large lane
                            if margin < margin_th:
                                route = "reasoning"
                            else:
                                route = raw_vote

                        # Evaluate end-to-end outcome from cache
                        if route == "simple":
                            is_correct = rec["small_ok"]
                            c = calculate_cost(rec["small_model"], rec["small_in_tokens"], rec["small_out_tokens"])
                        else:
                            is_correct = rec["big_ok"]
                            c = calculate_cost(rec["big_model"], rec["big_in_tokens"], rec["big_out_tokens"])

                        if is_correct:
                            correct_count += 1
                        costs.append(c)

                    acc = correct_count / len(dev_items)
                    mean_cost = float(np.mean(costs))
                    cost_reduction = (baseline_big_cost - mean_cost) / baseline_big_cost

                    results.append({
                        "k": k,
                        "sim_exponent": exp,
                        "margin_threshold": margin_th,
                        "near_duplicate_cutoff": dup_cut,
                        "accuracy": acc,
                        "mean_cost": mean_cost,
                        "cost_reduction": cost_reduction,
                    })

    # Sort results by accuracy (descending), then cost_reduction (descending)
    results.sort(key=lambda x: (x["accuracy"], x["cost_reduction"]), reverse=True)

    print(f"\nGrid search completed: {len(results)} configurations evaluated.")
    print("\nTop 5 Configurations on DEV Split:")
    print(f"{'k':<3} {'exp':<5} {'margin':<7} {'dup':<6} | {'Dev Acc':<8} {'Dev Cost/Req':<13} {'Cost Reduction':<14}")
    print("-" * 65)
    for r in results[:5]:
        print(f"{r['k']:<3} {r['sim_exponent']:<5} {r['margin_threshold']:<7.2f} {r['near_duplicate_cutoff']:<6.3f} | "
              f"{r['accuracy']*100:<7.1f}% ${r['mean_cost']:<12.6f} {r['cost_reduction']*100:<13.1f}%")

    # Select best configuration that satisfies cost constraint (>= 35% reduction)
    viable = [r for r in results if r["cost_reduction"] >= 0.35]
    if not viable:
        # Fallback to top accuracy
        best = results[0]
    else:
        best = viable[0]

    print("\n" + "=" * 50)
    print("SELECTED FROZEN CONFIGURATION (Dev-Tuned):")
    print(f"  k:                     {best['k']}")
    print(f"  sim_exponent:          {best['sim_exponent']}")
    print(f"  margin_threshold:      {best['margin_threshold']}")
    print(f"  near_duplicate_cutoff: {best['near_duplicate_cutoff']}")
    print(f"  Dev Accuracy:          {best['accuracy']*100:.1f}%")
    print(f"  Dev Cost Reduction:    {best['cost_reduction']*100:.1f}% vs Always-Big")
    print("=" * 50)

    # Save to config/frozen.json
    frozen_payload = {
        "k": int(best["k"]),
        "sim_exponent": float(best["sim_exponent"]),
        "margin_threshold": float(best["margin_threshold"]),
        "near_duplicate_cutoff": float(best["near_duplicate_cutoff"]),
        "dev_accuracy": float(best["accuracy"]),
        "dev_mean_cost": float(best["mean_cost"]),
        "dev_cost_reduction": float(best["cost_reduction"]),
        "dev_items_count": len(dev_items),
        "frozen_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(FROZEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(frozen_payload, f, indent=2)
    print(f"✓ Saved frozen config to {FROZEN_CONFIG_PATH}")


if __name__ == "__main__":
    main()
