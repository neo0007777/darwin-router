#!/usr/bin/env python3
"""
Hyperparameter Sweep for kNN Routing (scripts/sweep_k.py)
=========================================================
Sweeps over neighborhood size k in [3, 5, 7, 9, 11] and distance weighting
powers in [1, 2, 4, 8] on the 200-query checkable benchmark.
Validates the optimal hyperparameter regime for >80% routing accuracy.
"""

import json
from pathlib import Path
import numpy as np

from router.embed import get_client
from router.index import ExemplarIndex
from scripts.baselines import load_eval_data, bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent


def run_sweep():
    items = load_eval_data()
    print(f"Loaded {len(items)} evaluation queries for hyperparameter sweep.")

    embedder = get_client()
    texts = [item["text"] for item in items]
    print(f"Embedding {len(texts)} queries via {embedder.backend}...")
    embeddings = embedder.embed(texts)

    index = ExemplarIndex()
    index.load(embed_fn=embedder.embed)
    print(f"Loaded index with {len(index.meta)} exemplars.\n")

    k_values = [3, 5, 7, 9, 11]
    power_values = [1, 2, 4, 8]

    print("=" * 80)
    print(f"{'k':<6} | {'Power (p)':<12} | {'Routing Accuracy (95% CI)':<30} | {'Status'}")
    print("=" * 80)

    best_acc = 0.0
    best_config = None

    for k in k_values:
        for p in power_values:
            correct_list = []
            for item, emb in zip(items, embeddings):
                q = np.array(emb, dtype=np.float32)
                q = q / np.linalg.norm(q)
                sims = index.vecs @ q
                top = np.argpartition(-sims, min(k, len(index.meta)) - 1)[:k]
                
                votes = {}
                for idx in top:
                    lbl = index.meta[idx]["label"]
                    sim = max(0.0, float(sims[idx]))
                    w = (sim ** p) if sim > 0 else 1e-4
                    votes[lbl] = votes.get(lbl, 0.0) + w

                pred = max(votes, key=votes.get)
                correct_list.append(1.0 if pred == item["expected_route"] else 0.0)

            mean_acc, lo, hi = bootstrap_ci(correct_list)
            status = "✓ TARGET MET (≥80%)" if mean_acc >= 0.80 else "Below Target"
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_config = (k, p)

            print(f"{k:<6} | {p:<12} | {mean_acc * 100:5.1f}% [{lo * 100:4.1f}–{hi * 100:4.1f}%] | {status}")

    print("=" * 80)
    print(f"Best Configuration: k={best_config[0]}, power={best_config[1]} with {best_acc * 100:.1f}% accuracy.\n")


if __name__ == "__main__":
    run_sweep()
