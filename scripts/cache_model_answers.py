#!/usr/bin/env python3
"""Cache model answers for Dev, Test, and Seed splits (Steps 2 & 3).

Runs:
1. Small model (Ollama llama3.2:3b, temperature 0)
2. Large model (Gemini gemini-3.5-flash, temperature 0)

Verifies each answer with router.verifier.verify().
Saves to cache/{dev,test,seeds}.jsonl with:
- id
- small_ok, big_ok
- small_in_tokens, small_out_tokens
- big_in_tokens, big_out_tokens (includes thought/reasoning tokens)
- small_raw_response, big_raw_response
- small_model, big_model

Resumable: reads existing records and skips already cached IDs.
Handles Gemini rate limits with exponential backoff.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from router import config
from router.verifier import verify

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def load_dataset(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cached_ids(cache_file: Path) -> dict[str, dict]:
    cached = {}
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        # Only consider cached if BOTH responses succeeded
                        if record.get("small_raw_response") and record.get("big_raw_response"):
                            cached[record["id"]] = record
                    except json.JSONDecodeError:
                        continue
    return cached


async def call_small_model(http_client: httpx.AsyncClient, text: str, max_retries: int = 3) -> tuple[str, int, int]:
    """Call Ollama llama3.2:3b at temperature 0."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }
    for attempt in range(max_retries):
        try:
            r = await http_client.post(
                f"{config.OLLAMA_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=60.0,
            )
            r.raise_for_status()
            res = r.json()
            content = res["choices"][0]["message"]["content"]
            usage = res.get("usage", {})
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            return content, in_tok, out_tok
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2.0)


async def call_large_model(
    http_client: httpx.AsyncClient,
    text: str,
    max_retries: int = 8,
) -> tuple[str, int, int]:
    """Call Gemini using native generateContent with exponential backoff on errors."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.0},
    }

    backoff = 4.0
    for attempt in range(max_retries):
        try:
            r = await http_client.post(url, json=payload, timeout=60.0)
            if r.status_code in (429, 500, 502, 503, 504):
                wait_time = max(10.0, backoff) if r.status_code == 429 else backoff
                err_preview = r.text.replace("\n", " ")[:120]
                print(f"    [Gemini HTTP {r.status_code}] {err_preview} Backing off for {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(wait_time)
                backoff = min(60.0, backoff * 2.0)
                continue

            r.raise_for_status()
            res = r.json()
            candidates = res.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"No candidates returned: {res}")
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)
            usage = res.get("usageMetadata", {})
            in_tok = usage.get("promptTokenCount", 0)
            out_tok = usage.get("candidatesTokenCount", 0)
            return content, in_tok, out_tok
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            wait_time = backoff
            print(f"    [Gemini Exception {type(e).__name__}] Backing off for {wait_time:.1f}s (attempt {attempt+1}/{max_retries})...")
            await asyncio.sleep(wait_time)
            backoff = min(60.0, backoff * 2.0)

    raise RuntimeError(f"Exceeded max retries calling Gemini for prompt: {text[:50]}...")


async def process_split(split_name: str, delay_between_calls: float = 1.0):
    cache_file = CACHE_DIR / f"{split_name}.jsonl"
    dataset = load_dataset(split_name)
    cached = load_cached_ids(cache_file)

    print(f"\n=======================================================")
    print(f"Caching Split: {split_name.upper()} ({len(dataset)} items)")
    print(f"Existing in cache: {len(cached)} / {len(dataset)}")
    print(f"=======================================================")

    uncached = [item for item in dataset if item["id"] not in cached]
    if not uncached:
        print(f"✓ Split '{split_name}' is already 100% cached ({len(cached)} items).")
        return

    print(f"Processing remaining {len(uncached)} queries for '{split_name}'...")

    async with httpx.AsyncClient() as http_client:
        for idx, item in enumerate(uncached):
            item_id = item["id"]
            text = item["text"]
            spec = item.get("answer_spec", item)

            t0 = time.time()

            # Run Small model (local Ollama) and Large model (Gemini API) concurrently
            small_resp, small_in, small_out, small_ok = "", 0, 0, False
            big_resp, big_in, big_out, big_ok = "", 0, 0, False

            small_task = call_small_model(http_client, text)
            big_task = call_large_model(http_client, text)
            results = await asyncio.gather(small_task, big_task, return_exceptions=True)

            if isinstance(results[0], Exception):
                print(f"  ❌ Error calling small model on {item_id}: {results[0]}")
            else:
                small_resp, small_in, small_out = results[0]
                small_ok = verify(small_resp, spec)

            if isinstance(results[1], Exception):
                print(f"  ❌ Error calling large model on {item_id}: {results[1]}")
            else:
                big_resp, big_in, big_out = results[1]
                big_ok = verify(big_resp, spec)

            if not small_resp or not big_resp:
                print(f"  ⚠️ Skipping saving incomplete response for {item_id}")
                continue

            record = {
                "id": item_id,
                "small_ok": small_ok,
                "big_ok": big_ok,
                "small_in_tokens": small_in,
                "small_out_tokens": small_out,
                "big_in_tokens": big_in,
                "big_out_tokens": big_out,
                "small_raw_response": small_resp,
                "big_raw_response": big_resp,
                "small_model": config.OLLAMA_MODEL,
                "big_model": config.GEMINI_MODEL,
            }

            # Append immediately to cache file
            with open(cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            cached[item_id] = record
            elapsed = time.time() - t0

            status_str = f"S:{'✓' if small_ok else '✗'} B:{'✓' if big_ok else '✗'}"
            print(f"  [{len(cached):3d}/{len(dataset)}] {item_id:<18} {status_str} ({elapsed:.1f}s) [toks: S_out={small_out}, B_out={big_out}]")

            # Rate pacing for Gemini free tier
            await asyncio.sleep(delay_between_calls)

    print(f"\n✓ Finished caching '{split_name}'. Total cached: {len(cached)} / {len(dataset)} items.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "seeds", "all"], default="all",
                        help="Which dataset split to cache")
    parser.add_argument("--delay", type=float, default=12.5,
                        help="Seconds to wait between queries for Gemini rate limits")
    args = parser.parse_args()

    if args.split == "all":
        # Cache in order: dev (100) -> seeds (160) -> test (300)
        for s in ["dev", "seeds", "test"]:
            await process_split(s, delay_between_calls=args.delay)
    else:
        await process_split(args.split, delay_between_calls=args.delay)


if __name__ == "__main__":
    asyncio.run(main())
