#!/usr/bin/env python3
"""
Curate Code Evaluation Set (Task 1.2)
Selects 30 simple (1-3 line functions) and 30 complex (>=9 line algorithms, DP/recursion)
problems from the MBPP benchmark.
Appends the 60 code items to data/eval_200.jsonl (or updates code items in place).
"""

import json
import ssl
import sys
import urllib.request
from pathlib import Path

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_FILE = DATA_DIR / "eval_200.jsonl"
CACHE_FILE = DATA_DIR / ".mbpp_cache.jsonl"


def fetch_mbpp_data() -> list[dict]:
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 100000:
        print(f"Loading MBPP data from local cache {CACHE_FILE}...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    print(f"Fetching MBPP dataset from {MBPP_URL}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.urlopen(MBPP_URL, context=ctx, timeout=30)
    content = req.read().decode("utf-8")
    lines = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

    # Save to cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Loaded and cached {len(lines)} raw MBPP items.")
    return lines


def curate_code_set(raw_items: list[dict]) -> tuple[list[dict], list[dict]]:
    simple = []
    complex_items = []

    for item in raw_items:
        text = item.get("text", "").strip()
        code = item.get("code", "").strip()
        tests = [t.strip() for t in item.get("test_list", []) if t.strip()]

        if not text or not code or len(tests) < 2:
            continue

        # Code lines count (excluding blank lines)
        code_lines = [l for l in code.split("\n") if l.strip()]
        line_count = len(code_lines)

        # Simple criteria: <= 3 lines, straightforward utility
        if line_count <= 3 and len(simple) < 30:
            simple.append(
                {
                    "id": f"code_mbpp_simple_{len(simple)+1:02d}",
                    "text": text,
                    "category": "code",
                    "difficulty": "simple",
                    "label": "simple",
                    "expected_route": "simple",
                    "answer": code,
                    "answer_type": "code_test",
                    "test_list": tests,
                    "test_setup_code": item.get("test_setup_code", ""),
                    "lines_of_code": line_count,
                    "source": "mbpp",
                }
            )
        # Complex criteria: >= 9 lines, algorithmic
        elif line_count >= 9 and len(complex_items) < 30:
            complex_items.append(
                {
                    "id": f"code_mbpp_complex_{len(complex_items)+1:02d}",
                    "text": text,
                    "category": "code",
                    "difficulty": "complex",
                    "label": "reasoning",
                    "expected_route": "reasoning",
                    "answer": code,
                    "answer_type": "code_test",
                    "test_list": tests,
                    "test_setup_code": item.get("test_setup_code", ""),
                    "lines_of_code": line_count,
                    "source": "mbpp",
                }
            )

        if len(simple) == 30 and len(complex_items) == 30:
            break

    return simple, complex_items


def merge_into_eval_file(code_items: list[dict]):
    existing_items = []
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Filter out any old code items so we don't duplicate
                    if item.get("category") != "code":
                        existing_items.append(item)

    all_items = existing_items + code_items

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    print(f"Updated {EVAL_FILE} with {len(code_items)} code items.")
    print(f"Total items now in {EVAL_FILE}: {len(all_items)}")


def main():
    raw_items = fetch_mbpp_data()
    simple, complex_items = curate_code_set(raw_items)
    print(f"Curated {len(simple)} simple and {len(complex_items)} complex code items.")

    all_code = simple + complex_items
    merge_into_eval_file(all_code)


if __name__ == "__main__":
    main()
