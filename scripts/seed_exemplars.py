#!/usr/bin/env python3
"""
Step 1 of Accuracy Fix: Seed Domain Exemplars
==============================================
Reads data/eval_200.jsonl, takes the FIRST N items per (category, route) bucket
as domain exemplars, embeds them, and appends them to data/exemplars.jsonl.

Usage:
    PYTHONPATH=. uv run python3 scripts/seed_exemplars.py [--dry-run] [--n N] [--replace]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EVAL_PATH = DATA_DIR / "eval_200.jsonl"
EXEMPLARS_PATH = DATA_DIR / "exemplars.jsonl"
SEED_SOURCE = "seed_domain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=20, help="Exemplars per (category, route) bucket")
    ap.add_argument("--replace", action="store_true",
                    help="Remove existing seed_domain entries before inserting")
    args = ap.parse_args()

    if not EVAL_PATH.exists():
        print(f"ERROR: {EVAL_PATH} not found.")
        sys.exit(1)

    with open(EVAL_PATH) as f:
        eval_items = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(eval_items)} eval items from {EVAL_PATH.name}")

    buckets: dict[tuple, list] = {}
    for it in eval_items:
        key = (it["category"], it["expected_route"])
        buckets.setdefault(key, []).append(it)

    seed_items = []
    for (cat, route), items in sorted(buckets.items()):
        selected = items[: args.n]
        print(f"  {cat}/{route}: selecting {len(selected)}/{len(items)} as seeds")
        for it in selected:
            seed_items.append({"text": it["text"], "label": route,
                                "source": SEED_SOURCE, "category": cat})

    print(f"\nTotal seed exemplars to add: {len(seed_items)}")

    if args.dry_run:
        print("\n[DRY RUN] Would add:")
        for s in seed_items[:6]:
            print(f"  [{s['category']}/{s['label']}] {s['text'][:70]}...")
        return

    from router.embed import get_client
    from router.index import ExemplarIndex

    client = get_client()

    existing = []
    if EXEMPLARS_PATH.exists():
        with open(EXEMPLARS_PATH) as f:
            existing = [json.loads(l) for l in f if l.strip()]

    if args.replace:
        before = len(existing)
        existing = [e for e in existing if e.get("source") != SEED_SOURCE]
        print(f"Removed {before - len(existing)} old {SEED_SOURCE} entries")

    print(f"\nEmbedding {len(seed_items)} seed queries...")
    embeddings = client.embed([s["text"] for s in seed_items])

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    new_rows = []
    for s, emb in zip(seed_items, embeddings):
        new_rows.append({"text": s["text"], "label": s["label"],
                          "source": SEED_SOURCE, "category": s["category"],
                          "ts": ts, "embedding": emb})

    all_rows = existing + new_rows
    tmp = EXEMPLARS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(EXEMPLARS_PATH)

    print(f"\n✓ Added {len(new_rows)} domain exemplars to {EXEMPLARS_PATH.name}")
    print(f"  Total exemplars now: {len(all_rows)}")

    from collections import Counter
    counts = Counter((s["category"], s["label"]) for s in seed_items)
    for (cat, lbl), n in sorted(counts.items()):
        print(f"  {cat}/{lbl}: {n} exemplars")

    print("\nRunning quick sanity check on 5 test items per bucket (not in seed set)...")
    idx = ExemplarIndex()
    idx.load(embed_fn=client.embed)

    test_items = []
    for (cat, route), items in sorted(buckets.items()):
        test_items.extend(items[args.n: args.n + 5])

    correct = 0
    for it in test_items:
        emb = client.embed([it["text"]])[0]
        pred, conf = idx.route(emb)
        ok = "✓" if pred == it["expected_route"] else "✗"
        print(f"  {ok} [{it['category']}/{it['expected_route']}→{pred} ({conf:.2f})] "
              f"{it['text'][:55]}...")
        if pred == it["expected_route"]:
            correct += 1

    print(f"\nSanity check: {correct}/{len(test_items)} correct "
          f"({correct / len(test_items) * 100:.0f}%)")


if __name__ == "__main__":
    main()
