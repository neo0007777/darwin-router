#!/usr/bin/env python3
"""Build strictly disjoint data splits (seeds, dev, test) across 4 domains.

Sources:
1. Math: openai/gsm8k (Train for seeds; Test for dev & test)
2. Code: MBPP (Train tasks 11-510 for seeds; Val tasks 511-600 for dev; Test tasks 601-974 for test)
3. Science QA: allenai/ai2_arc (Train for seeds; Validation for dev; Test for test)
4. Trivia QA: mandarjoshi/trivia_qa (Train for seeds; Validation for dev & test)

Total targets:
- Seeds: 160 items (40 Math, 40 Code, 40 ARC, 40 Trivia) -> strictly from TRAIN
- Dev:   100 items (25 Math, 25 Code, 25 ARC, 25 Trivia) -> strictly from VAL/TEST
- Test:  300 items (75 Math, 75 Code, 75 ARC, 75 Trivia) -> strictly from VAL/TEST
"""

import json
import re
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MBPP_CACHE = DATA_DIR / ".mbpp_cache.jsonl"


def clean_numeric_str(s: str) -> str:
    s = s.replace("$", "").replace(",", "").strip()
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except ValueError:
        return s


# ─────────────────────────────────────────────────────────────────────────────
# 1. GSM8K (Math)
# ─────────────────────────────────────────────────────────────────────────────

import time

def fetch_with_retry(url: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.0 * (attempt + 1))


def fetch_gsm8k(split: str, limit: int = 100, offset: int = 0) -> list[dict]:
    url = f"https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split={split}&offset={offset}&limit={limit}"
    data = fetch_with_retry(url)
    return [row["row"] for row in data.get("rows", [])]


def parse_gsm8k_items(raw_rows: list[dict], split_tag: str, id_prefix: str) -> list[dict]:
    items = []
    for idx, row in enumerate(raw_rows):
        q = row.get("question", "").strip()
        ans = row.get("answer", "").strip()
        if not q or "####" not in ans:
            continue
        raw_num = ans.split("####")[-1].strip()
        clean_num = clean_numeric_str(raw_num)
        try:
            float(clean_num)
        except ValueError:
            continue

        item_text = f"{q}\nAnswer with just the final numerical value."
        items.append({
            "id": f"{id_prefix}_{idx:03d}",
            "text": item_text,
            "category": "math",
            "source_split": f"gsm8k_{split_tag}",
            "answer_spec": {
                "answer_type": "numeric",
                "answer": clean_num,
                "tolerance": 0.01,
            }
        })
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 2. MBPP (Code)
# ─────────────────────────────────────────────────────────────────────────────

def load_mbpp_splits() -> tuple[list[dict], list[dict], list[dict]]:
    if not MBPP_CACHE.exists():
        raise FileNotFoundError(f"{MBPP_CACHE} does not exist.")
    
    with open(MBPP_CACHE, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    tasks_by_id = {t["task_id"]: t for t in all_tasks}

    # Standard MBPP splits:
    # Train: 11 to 510
    # Val:   511 to 600
    # Test:  601 to 974
    train_tasks = [tasks_by_id[i] for i in range(11, 511) if i in tasks_by_id]
    val_tasks = [tasks_by_id[i] for i in range(511, 601) if i in tasks_by_id]
    test_tasks = [tasks_by_id[i] for i in range(601, 975) if i in tasks_by_id]

    return train_tasks, val_tasks, test_tasks


def format_mbpp_items(tasks: list[dict], split_tag: str, id_prefix: str, count: int) -> list[dict]:
    items = []
    for idx, t in enumerate(tasks[:count]):
        prompt = t.get("text", "").strip()
        test_list = t.get("test_list", [])
        setup_code = t.get("test_setup_code", "")

        item_text = (
            f"{prompt}\n"
            "Write a complete Python function fulfilling this specification."
        )

        items.append({
            "id": f"{id_prefix}_{t['task_id']:04d}",
            "text": item_text,
            "category": "code",
            "source_split": f"mbpp_{split_tag}",
            "answer_spec": {
                "answer_type": "code_test",
                "test_list": test_list,
                "test_setup_code": setup_code,
            }
        })
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARC (Science QA)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_arc(config: str, split: str, limit: int = 50, offset: int = 0) -> list[dict]:
    url = f"https://datasets-server.huggingface.co/rows?dataset=allenai/ai2_arc&config={config}&split={split}&offset={offset}&limit={limit}"
    data = fetch_with_retry(url)
    return [row["row"] for row in data.get("rows", [])]


def format_arc_items(raw_rows: list[dict], split_tag: str, id_prefix: str) -> list[dict]:
    items = []
    for idx, row in enumerate(raw_rows):
        q = row.get("question", "").strip()
        ans = str(row.get("answerKey", "")).strip().upper()
        choices = row.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])

        if not q or not ans or not labels or len(labels) != len(texts):
            continue

        ans_text = ""
        choice_lines = []
        for lbl, txt in zip(labels, texts):
            choice_lines.append(f"({lbl}) {txt}")
            if lbl.upper() == ans:
                ans_text = txt.strip()

        if not ans_text:
            continue

        formatted_text = f"{q}\n" + "\n".join(choice_lines) + "\nAnswer with the correct letter."
        aliases = [ans, f"({ans})", ans_text]

        items.append({
            "id": f"{id_prefix}_{idx:03d}",
            "text": formatted_text,
            "category": "qa_arc",
            "source_split": f"arc_{split_tag}",
            "answer_spec": {
                "answer_type": "exact_match",
                "answer": ans,
                "answer_aliases": aliases,
            }
        })
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 4. TriviaQA (General Trivia)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_triviaqa(split: str, limit: int = 100, offset: int = 0) -> list[dict]:
    url = f"https://datasets-server.huggingface.co/rows?dataset=mandarjoshi/trivia_qa&config=rc.nocontext&split={split}&offset={offset}&limit={limit}"
    data = fetch_with_retry(url)
    return [row["row"] for row in data.get("rows", [])]


def format_triviaqa_items(raw_rows: list[dict], split_tag: str, id_prefix: str) -> list[dict]:
    items = []
    for idx, row in enumerate(raw_rows):
        q = row.get("question", "").strip()
        ans_obj = row.get("answer", {})
        ans_val = str(ans_obj.get("value", "")).strip()
        aliases = [str(a).strip() for a in ans_obj.get("aliases", []) if str(a).strip()]

        if not q or not ans_val:
            continue

        all_aliases = list(dict.fromkeys([ans_val] + aliases))

        formatted_text = f"{q}\nAnswer with just the direct answer name or phrase."

        items.append({
            "id": f"{id_prefix}_{idx:03d}",
            "text": formatted_text,
            "category": "trivia",
            "source_split": f"triviaqa_{split_tag}",
            "answer_spec": {
                "answer_type": "exact_match",
                "answer": ans_val,
                "answer_aliases": all_aliases,
            }
        })
    return items


def main():
    print("=" * 70)
    print("Building Disjoint Splits (Seeds, Dev, Test)")
    print("=" * 70)

    # 1. Seeds (strictly TRAIN splits)
    print("\n--- 1. Generating Seeds (TRAIN splits) ---")
    gsm8k_train_raw = fetch_gsm8k("train", limit=50, offset=0)
    math_seeds = parse_gsm8k_items(gsm8k_train_raw, "train", "seed_math")[:40]
    print(f"  Math seeds (GSM8K train): {len(math_seeds)}")

    mbpp_train, mbpp_val, mbpp_test = load_mbpp_splits()
    code_seeds = format_mbpp_items(mbpp_train, "train", "seed_code", count=40)
    print(f"  Code seeds (MBPP train): {len(code_seeds)}")

    arc_easy_train = fetch_arc("ARC-Easy", "train", limit=25, offset=0)
    arc_hard_train = fetch_arc("ARC-Challenge", "train", limit=25, offset=0)
    arc_seeds = (format_arc_items(arc_easy_train, "train_easy", "seed_arc_easy")[:20] +
                 format_arc_items(arc_hard_train, "train_challenge", "seed_arc_hard")[:20])
    print(f"  ARC seeds (ARC train): {len(arc_seeds)}")

    trivia_train_raw = fetch_triviaqa("train", limit=50, offset=0)
    trivia_seeds = format_triviaqa_items(trivia_train_raw, "train", "seed_trivia")[:40]
    print(f"  Trivia seeds (TriviaQA train): {len(trivia_seeds)}")

    seeds = math_seeds + code_seeds + arc_seeds + trivia_seeds
    print(f"==> Total Seeds: {len(seeds)}")
    assert len(seeds) >= 150, f"Expected >= 150 seeds, got {len(seeds)}"

    # Embed seeds to filter any near-duplicates (sim >= 0.88)
    print("\n--- Embedding seeds for similarity guardrail ---")
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    seed_texts = [s["text"] for s in seeds]
    seed_embs = model.encode(seed_texts, convert_to_numpy=True, normalize_embeddings=True)

    def filter_leakage(candidates: list[dict], target_count: int, split_label: str) -> list[dict]:
        clean = []
        for cand in candidates:
            c_emb = model.encode([cand["text"]], convert_to_numpy=True, normalize_embeddings=True)
            max_sim = float(np.max(np.dot(c_emb, seed_embs.T)))
            if max_sim >= 0.88:
                print(f"  [FILTERED LEAK] {cand['id']} had sim {max_sim:.4f} >= 0.88 against seeds. Replacing...")
                continue
            clean.append(cand)
            if len(clean) == target_count:
                break
        if len(clean) < target_count:
            raise RuntimeError(f"Could not find {target_count} clean items for {split_label}, only got {len(clean)}")
        return clean

    # 2. Dev Set (~100 items from VAL/TEST splits)
    print("\n--- 2. Generating Dev Set (VAL/TEST splits) ---")
    gsm8k_test_raw_dev = fetch_gsm8k("test", limit=50, offset=0)
    math_dev_cands = parse_gsm8k_items(gsm8k_test_raw_dev, "test", "dev_math")
    math_dev = filter_leakage(math_dev_cands, 25, "math_dev")
    print(f"  Math dev (GSM8K test): {len(math_dev)}")

    code_dev_cands = format_mbpp_items(mbpp_val, "validation", "dev_code", count=len(mbpp_val))
    code_dev = filter_leakage(code_dev_cands, 25, "code_dev")
    print(f"  Code dev (MBPP validation): {len(code_dev)}")

    arc_easy_val = fetch_arc("ARC-Easy", "validation", limit=30, offset=0)
    arc_hard_val = fetch_arc("ARC-Challenge", "validation", limit=30, offset=0)
    arc_dev_cands = (format_arc_items(arc_easy_val, "val_easy", "dev_arc_easy") +
                     format_arc_items(arc_hard_val, "val_challenge", "dev_arc_hard"))
    arc_dev = filter_leakage(arc_dev_cands, 25, "arc_dev")
    print(f"  ARC dev (ARC validation): {len(arc_dev)}")

    trivia_val_raw_dev = fetch_triviaqa("validation", limit=50, offset=0)
    trivia_dev_cands = format_triviaqa_items(trivia_val_raw_dev, "validation", "dev_trivia")
    trivia_dev = filter_leakage(trivia_dev_cands, 25, "trivia_dev")
    print(f"  Trivia dev (TriviaQA val): {len(trivia_dev)}")

    dev = math_dev + code_dev + arc_dev + trivia_dev
    print(f"==> Total Dev: {len(dev)}")
    assert len(dev) == 100, f"Expected 100 dev items, got {len(dev)}"

    # 3. Test Set (300+ items from VAL/TEST splits)
    print("\n--- 3. Generating Test Set (VAL/TEST splits) ---")
    gsm8k_test_raw_test = fetch_gsm8k("test", limit=100, offset=30)
    math_test_cands = parse_gsm8k_items(gsm8k_test_raw_test, "test", "test_math")
    math_test = filter_leakage(math_test_cands, 75, "math_test")
    print(f"  Math test (GSM8K test): {len(math_test)}")

    code_test_cands = format_mbpp_items(mbpp_test, "test", "test_code", count=len(mbpp_test))
    code_test = filter_leakage(code_test_cands, 75, "code_test")
    print(f"  Code test (MBPP test): {len(code_test)}")

    arc_easy_test = fetch_arc("ARC-Easy", "test", limit=60, offset=0)
    arc_hard_test = fetch_arc("ARC-Challenge", "test", limit=60, offset=0)
    arc_test_cands = (format_arc_items(arc_easy_test, "test_easy", "test_arc_easy") +
                      format_arc_items(arc_hard_test, "test_challenge", "test_arc_hard"))
    arc_test = filter_leakage(arc_test_cands, 75, "arc_test")
    print(f"  ARC test (ARC test): {len(arc_test)}")

    trivia_val_raw_test = fetch_triviaqa("validation", limit=100, offset=30)
    trivia_test_cands = format_triviaqa_items(trivia_val_raw_test, "validation", "test_trivia")
    trivia_test = filter_leakage(trivia_test_cands, 75, "trivia_test")
    print(f"  Trivia test (TriviaQA val): {len(trivia_test)}")

    test = math_test + code_test + arc_test + trivia_test
    print(f"==> Total Test: {len(test)}")
    assert len(test) == 300, f"Expected 300 test items, got {len(test)}"

    # Save to disk
    seeds_path = DATA_DIR / "seeds.jsonl"
    dev_path = DATA_DIR / "dev.jsonl"
    test_path = DATA_DIR / "test.jsonl"

    with open(seeds_path, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(seeds)} seeds to {seeds_path}")

    with open(dev_path, "w", encoding="utf-8") as f:
        for d in dev:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dev)} dev queries to {dev_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        for t in test:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(test)} test queries to {test_path}")

    print("\n✓ Split generation complete. Next: run check_leakage.py")


if __name__ == "__main__":
    main()
