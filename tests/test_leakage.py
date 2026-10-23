"""Unit test for train/test leakage and contamination prevention (Step 1.5).

Asserts that:
1. No dev or test query is an exact text duplicate of any seed exemplar (case/whitespace normalized).
2. No dev or test query has cosine similarity >= 0.90 against the seeds.
3. No dev query overlaps with any test query.
"""

import json
from pathlib import Path
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SEEDS_PATH = DATA_DIR / "seeds.jsonl"
DEV_PATH = DATA_DIR / "dev.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def normalize_text(t: str) -> str:
    return " ".join(t.lower().split())


def load_dataset(path: Path) -> list[dict]:
    assert path.exists(), f"Missing dataset file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_no_exact_text_duplicates_between_splits():
    seeds = load_dataset(SEEDS_PATH)
    dev = load_dataset(DEV_PATH)
    test = load_dataset(TEST_PATH)

    seed_texts = {normalize_text(s["text"]): s["id"] for s in seeds}
    dev_texts = {normalize_text(d["text"]): d["id"] for d in dev}
    test_texts = {normalize_text(t["text"]): t["id"] for t in test}

    # Dev vs Seeds
    dev_seed_overlap = set(dev_texts.keys()).intersection(set(seed_texts.keys()))
    assert not dev_seed_overlap, f"Found {len(dev_seed_overlap)} exact duplicates between Dev and Seeds: {list(dev_seed_overlap)[:3]}"

    # Test vs Seeds
    test_seed_overlap = set(test_texts.keys()).intersection(set(seed_texts.keys()))
    assert not test_seed_overlap, f"Found {len(test_seed_overlap)} exact duplicates between Test and Seeds: {list(test_seed_overlap)[:3]}"

    # Dev vs Test
    dev_test_overlap = set(dev_texts.keys()).intersection(set(test_texts.keys()))
    assert not dev_test_overlap, f"Found {len(dev_test_overlap)} exact duplicates between Dev and Test: {list(dev_test_overlap)[:3]}"


def test_no_semantic_leakage_above_threshold():
    seeds = load_dataset(SEEDS_PATH)
    dev = load_dataset(DEV_PATH)
    test = load_dataset(TEST_PATH)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    seed_embs = model.encode([s["text"] for s in seeds], convert_to_numpy=True, normalize_embeddings=True)
    dev_embs = model.encode([d["text"] for d in dev], convert_to_numpy=True, normalize_embeddings=True)
    test_embs = model.encode([t["text"] for t in test], convert_to_numpy=True, normalize_embeddings=True)

    # Check Dev vs Seeds
    dev_sims = np.dot(dev_embs, seed_embs.T)
    dev_max_sim = float(np.max(dev_sims))
    assert dev_max_sim < 0.90, f"Dev has query with cosine similarity {dev_max_sim:.4f} >= 0.90 against seeds"

    # Check Test vs Seeds
    test_sims = np.dot(test_embs, seed_embs.T)
    test_max_sim = float(np.max(test_sims))
    assert test_max_sim < 0.90, f"Test has query with cosine similarity {test_max_sim:.4f} >= 0.90 against seeds"
