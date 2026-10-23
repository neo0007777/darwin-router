#!/usr/bin/env python3
"""
Curate Exact-Match QA Evaluation Set (Task 1.3)
Selects 30 simple factual questions from ARC-Easy and 30 complex science reasoning
questions from ARC-Challenge.
Merges into data/eval_200.jsonl, reaching exactly 200 total evaluation queries.
"""

import json
from pathlib import Path
import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_FILE = DATA_DIR / "eval_200.jsonl"

ARC_EASY_URL = "https://datasets-server.huggingface.co/rows?dataset=allenai/ai2_arc&config=ARC-Easy&split=test&offset=0&limit=100"
ARC_CHALLENGE_URL = "https://datasets-server.huggingface.co/rows?dataset=allenai/ai2_arc&config=ARC-Challenge&split=test&offset=0&limit=100"


def format_qa_item(row: dict, difficulty: str, idx: int) -> dict | None:
    q = row.get("question", "").strip()
    ans = str(row.get("answerKey", "")).strip().upper()
    choices = row.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])

    if not q or not ans or not labels or len(labels) != len(texts):
        return None

    # Find the correct text corresponding to the answer letter
    ans_text = ""
    choice_lines = []
    for lbl, txt in zip(labels, texts):
        choice_lines.append(f"({lbl}) {txt}")
        if lbl.upper() == ans:
            ans_text = txt.strip()

    if not ans_text:
        return None

    formatted_text = f"{q}\n" + "\n".join(choice_lines) + "\nAnswer with the correct letter."

    aliases = [ans, f"({ans})", ans_text]

    return {
        "id": f"qa_arc_{difficulty}_{idx:02d}",
        "text": formatted_text,
        "category": "qa",
        "difficulty": difficulty,
        "label": "simple" if difficulty == "simple" else "reasoning",
        "expected_route": "simple" if difficulty == "simple" else "reasoning",
        "answer": ans,
        "answer_text": ans_text,
        "answer_aliases": aliases,
        "answer_type": "exact_match",
        "source": f"arc_{difficulty}",
    }


def fetch_and_curate():
    client = httpx.Client(timeout=20)

    print("Fetching ARC-Easy rows...")
    r_easy = client.get(ARC_EASY_URL)
    r_easy.raise_for_status()
    easy_rows = [r["row"] for r in r_easy.json().get("rows", [])]

    print("Fetching ARC-Challenge rows...")
    r_chal = client.get(ARC_CHALLENGE_URL)
    r_chal.raise_for_status()
    chal_rows = [r["row"] for r in r_chal.json().get("rows", [])]

    simple_items = []
    for r in easy_rows:
        item = format_qa_item(r, "simple", len(simple_items) + 1)
        if item:
            simple_items.append(item)
        if len(simple_items) == 30:
            break

    complex_items = []
    for r in chal_rows:
        item = format_qa_item(r, "complex", len(complex_items) + 1)
        if item:
            complex_items.append(item)
        if len(complex_items) == 30:
            break

    print(f"Curated {len(simple_items)} simple and {len(complex_items)} complex QA items.")
    return simple_items + complex_items


def merge_into_eval_file(qa_items: list[dict]):
    existing_items = []
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Exclude existing qa items to prevent duplication
                    if item.get("category") != "qa":
                        existing_items.append(item)

    all_items = existing_items + qa_items

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    print(f"Updated {EVAL_FILE} with {len(qa_items)} QA items.")
    print(f"Total evaluation items now in {EVAL_FILE}: {len(all_items)}")


def main():
    qa_items = fetch_and_curate()
    merge_into_eval_file(qa_items)


if __name__ == "__main__":
    main()
