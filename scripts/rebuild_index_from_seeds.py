#!/usr/bin/env python3
"""Rebuild Index Exemplars from Verified Seed Cache (Step 3).

Rules:
1. Seed labeling:
   - small_ok -> 'simple'
   - not small_ok and big_ok -> 'reasoning'
   - not small_ok and not big_ok -> discard (unreliable exemplar)
2. Embeds exemplars with all-MiniLM-L6-v2 (normalized).
3. Saves clean seeds to data/exemplars.jsonl.
4. Generates parametric boundary probes and appends verified probes to data/exemplars_probed.jsonl,
   allowing a clean comparison of Darwin vs Darwin+Probes.
"""

import json
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from router.prober import generate_boundary_probes
from router.verifier import verify

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

SEEDS_PATH = DATA_DIR / "seeds.jsonl"
SEEDS_CACHE_PATH = CACHE_DIR / "seeds.jsonl"
OUTPUT_CLEAN_PATH = DATA_DIR / "exemplars.jsonl"
OUTPUT_PROBED_PATH = DATA_DIR / "exemplars_probed.jsonl"


def main():
    print("=" * 70)
    print("Rebuilding Exemplar Index from Verified Seed Cache")
    print("=" * 70)

    if not SEEDS_CACHE_PATH.exists():
        print(f"ERROR: {SEEDS_CACHE_PATH} not found. Run cache_model_answers.py --split seeds first.", file=sys.stderr)
        sys.exit(1)

    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        seeds = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    with open(SEEDS_CACHE_PATH, "r", encoding="utf-8") as f:
        cache_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(seeds)} raw seeds and {len(cache_records)} cached execution records.")

    clean_exemplars = []
    discarded = 0
    simple_count = 0
    reasoning_count = 0

    for rec in cache_records:
        sid = rec["id"]
        if sid not in seeds:
            continue
        seed_item = seeds[sid]

        small_ok = rec["small_ok"]
        big_ok = rec["big_ok"]

        if small_ok:
            label = "simple"
            simple_count += 1
        elif big_ok:
            label = "reasoning"
            reasoning_count += 1
        else:
            discarded += 1
            continue

        clean_exemplars.append({
            "id": sid,
            "text": seed_item["text"],
            "label": label,
            "category": seed_item.get("category", "general"),
            "source": seed_item.get("source_split", "seed_train"),
        })

    print(f"\nSeed classification summary:")
    print(f"  Labeled 'simple':    {simple_count}")
    print(f"  Labeled 'reasoning': {reasoning_count}")
    print(f"  Discarded (both ✗): {discarded}")
    print(f"  Total valid clean:   {len(clean_exemplars)}")

    assert len(clean_exemplars) > 50, f"Expected > 50 valid exemplars, got {len(clean_exemplars)}"

    # Embed all clean exemplars
    print("\nComputing embeddings via sentence-transformers (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [e["text"] for e in clean_exemplars]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    for idx, e in enumerate(clean_exemplars):
        e["embedding"] = embs[idx].tolist()

    # Save clean index
    with open(OUTPUT_CLEAN_PATH, "w", encoding="utf-8") as f:
        for e in clean_exemplars:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(clean_exemplars)} clean exemplars to {OUTPUT_CLEAN_PATH}")

    # Generate verified template probes
    print("\nGenerating template boundary probes...")
    probes = generate_boundary_probes(n_per_template=1, seed=42)
    verified_probes = []
    for p in probes:
        # Template probes carry their verified ground truth difficulty
        verified_probes.append({
            "id": p["id"],
            "text": p["text"],
            "label": p["expected_route"],
            "category": p["category"],
            "source": f"probe_{p['template_id']}",
        })

    probe_texts = [p["text"] for p in verified_probes]
    probe_embs = model.encode(probe_texts, convert_to_numpy=True, normalize_embeddings=True)
    for idx, p in enumerate(verified_probes):
        p["embedding"] = probe_embs[idx].tolist()

    probed_exemplars = list(clean_exemplars) + verified_probes
    with open(OUTPUT_PROBED_PATH, "w", encoding="utf-8") as f:
        for e in probed_exemplars:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(probed_exemplars)} exemplars ({len(clean_exemplars)} clean + {len(verified_probes)} probes) to {OUTPUT_PROBED_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
