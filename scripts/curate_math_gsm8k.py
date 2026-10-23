#!/usr/bin/env python3
"""
Curate Math Evaluation Set (Task 1.1)
Selects 40 simple (1-2 step) and 40 complex (>=4 step) math problems from GSM8K.
Saves the initial slice of 80 math queries to data/eval_200.jsonl.
"""

import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
)
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_200.jsonl"


def fetch_gsm8k_data():
    print(f"Fetching GSM8K test set from {GSM8K_TEST_URL}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.urlopen(GSM8K_TEST_URL, context=ctx)
    content = req.read().decode("utf-8")
    lines = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
    print(f"Loaded {len(lines)} raw GSM8K test items.")
    return lines


def curate_math_set(raw_items):
    simple = []
    complex_items = []

    for item in raw_items:
        q = item.get("question", "").strip()
        sol = item.get("answer", "").strip()

        if not q or "####" not in sol:
            continue

        raw_ans = sol.split("####")[-1].strip()
        clean_ans = raw_ans.replace(",", "").replace("$", "").strip()

        try:
            val = float(clean_ans)
            # If it is an exact integer, store as integer string (e.g., "18" instead of "18.0")
            if val.is_integer():
                clean_ans = str(int(val))
            else:
                clean_ans = str(val)
        except ValueError:
            continue

        step_tags = re.findall(r"<<.*?>>", sol)
        num_steps = len(step_tags)

        if num_steps <= 2 and len(simple) < 40:
            simple.append(
                {
                    "id": f"math_gsm8k_simple_{len(simple)+1:02d}",
                    "text": q,
                    "category": "math",
                    "difficulty": "simple",
                    "label": "simple",
                    "expected_route": "simple",
                    "answer": clean_ans,
                    "answer_type": "numeric",
                    "tolerance": 0.01,
                    "steps": num_steps,
                    "source": "gsm8k",
                }
            )
        elif num_steps >= 4 and len(complex_items) < 40:
            complex_items.append(
                {
                    "id": f"math_gsm8k_complex_{len(complex_items)+1:02d}",
                    "text": q,
                    "category": "math",
                    "difficulty": "complex",
                    "label": "reasoning",
                    "expected_route": "reasoning",
                    "answer": clean_ans,
                    "answer_type": "numeric",
                    "tolerance": 0.01,
                    "steps": num_steps,
                    "source": "gsm8k",
                }
            )

        if len(simple) == 40 and len(complex_items) == 40:
            break

    return simple, complex_items


def main():
    raw_items = fetch_gsm8k_data()
    simple, complex_items = curate_math_set(raw_items)

    total_curated = simple + complex_items
    print(f"Curated {len(simple)} simple math problems and {len(complex_items)} complex math problems.")
    assert len(simple) == 40, f"Expected 40 simple items, got {len(simple)}"
    assert len(complex_items) == 40, f"Expected 40 complex items, got {len(complex_items)}"
    assert len(total_curated) == 80, f"Expected 80 total items, got {len(total_curated)}"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in total_curated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(total_curated)} math queries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
