#!/usr/bin/env python3
"""Leakage & Contamination Audit Script (Step 1.4).

Audits dev.jsonl and test.jsonl against seeds.jsonl:
1. Exact string matching (case-normalized, whitespace-stripped).
2. Dense semantic similarity: computes maximum cosine similarity of each dev/test
   query against all seeds in the index.
3. Prints distribution (min, median, p95, max) and count above 0.90.
4. Fails loudly (exit 1) if any query has similarity >= 0.90 or exact duplicate.
"""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SEEDS_PATH = DATA_DIR / "seeds.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def normalize_text(t: str) -> str:
    return " ".join(t.lower().split())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("=" * 70)
    print("Checking Train/Test Leakage & Cosine Similarity Distribution")
    print("=" * 70)

    seeds = load_jsonl(SEEDS_PATH)
    dev = load_jsonl(DEV_PATH)
    test = load_jsonl(TEST_PATH)

    print(f"Loaded {len(seeds)} seeds, {len(dev)} dev queries, {len(test)} test queries.")

    # 1. Exact text duplicate check
    seed_texts_norm = {normalize_text(s["text"]): s["id"] for s in seeds}

    exact_duplicates = []
    for split_name, dataset in [("dev", dev), ("test", test)]:
        for item in dataset:
            norm = normalize_text(item["text"])
            if norm in seed_texts_norm:
                exact_duplicates.append((split_name, item["id"], seed_texts_norm[norm]))

    if exact_duplicates:
        print(f"\n❌ FATAL: Found {len(exact_duplicates)} exact duplicates between seeds and eval sets!", file=sys.stderr)
        for split, eid, sid in exact_duplicates[:10]:
            print(f"   [{split}] {eid} == [seed] {sid}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✓ Exact text duplicates check PASSED: 0 duplicates found.")

    # 2. Embedding semantic similarity check
    print("\nEmbedding queries via sentence-transformers (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    seed_texts = [s["text"] for s in seeds]
    dev_texts = [d["text"] for d in dev]
    test_texts = [t["text"] for t in test]

    seed_embs = model.encode(seed_texts, convert_to_numpy=True, normalize_embeddings=True)
    dev_embs = model.encode(dev_texts, convert_to_numpy=True, normalize_embeddings=True)
    test_embs = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)

    # Dot product since embeddings are unit normalized
    # dev_sims shape: (len(dev), len(seeds))
    dev_sims = np.dot(dev_embs, seed_embs.T)
    dev_max_sims = np.max(dev_sims, axis=1)

    test_sims = np.dot(test_embs, seed_embs.T)
    test_max_sims = np.max(test_sims, axis=1)

    print("\n" + "-" * 50)
    print("DEV SPLIT vs SEEDS Max Cosine Similarity:")
    print(f"  Min:    {np.min(dev_max_sims):.4f}")
    print(f"  Median: {np.median(dev_max_sims):.4f}")
    print(f"  p95:    {np.percentile(dev_max_sims, 95):.4f}")
    print(f"  Max:    {np.max(dev_max_sims):.4f}")
    dev_above_90 = np.sum(dev_max_sims >= 0.90)
    print(f"  Count >= 0.90: {dev_above_90}")

    print("\nTEST SPLIT vs SEEDS Max Cosine Similarity:")
    print(f"  Min:    {np.min(test_max_sims):.4f}")
    print(f"  Median: {np.median(test_max_sims):.4f}")
    print(f"  p95:    {np.percentile(test_max_sims, 95):.4f}")
    print(f"  Max:    {np.max(test_max_sims):.4f}")
    test_above_90 = np.sum(test_max_sims >= 0.90)
    print(f"  Count >= 0.90: {test_above_90}")
    print("-" * 50)

    # Flag high similarity
    leaks = []
    for idx, sim in enumerate(dev_max_sims):
        if sim >= 0.90:
            nearest_seed_idx = np.argmax(dev_sims[idx])
            leaks.append(("dev", dev[idx], seeds[nearest_seed_idx], sim))

    for idx, sim in enumerate(test_max_sims):
        if sim >= 0.90:
            nearest_seed_idx = np.argmax(test_sims[idx])
            leaks.append(("test", test[idx], seeds[nearest_seed_idx], sim))

    if leaks:
        print(f"\n❌ FATAL: Found {len(leaks)} queries with cosine similarity >= 0.90!", file=sys.stderr)
        for split, query, seed, sim in leaks:
            print(f"   [{split}] {query['id']} vs [seed] {seed['id']} (sim={sim:.4f}):\n"
                  f"      Eval: {query['text'][:80]}...\n"
                  f"      Seed: {seed['text'][:80]}...", file=sys.stderr)
        sys.exit(1)

    print("\n✓ Leakage audit PASSED: All dev and test queries have cosine similarity < 0.90 against seeds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
