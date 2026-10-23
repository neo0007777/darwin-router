"""Unit test suite for router/verifier.py (Phase 2).
Tests deterministic numeric, code, and exact-match verification edge cases.
"""

import pytest
from router.verifier import (
    verify_numeric,
    verify_code,
    verify_exact_match,
    verify,
)


# ── 1. Numeric Verifier Tests ──────────────────────────────────────────────

def test_numeric_gsm8k_format():
    response = "The calculation is 5 * 10 = 50. Then subtract 8.\n#### 42"
    assert verify_numeric(response, "42") is True
    assert verify_numeric(response, "40") is False


def test_numeric_natural_language_conclusion():
    response = "Therefore, the total amount of water remaining is exactly 125.5 liters."
    assert verify_numeric(response, 125.5) is True
    assert verify_numeric(response, 125.0, tolerance=0.01) is False


def test_numeric_currency_and_formatting():
    response = "She made **$1,250.00** in revenue."
    assert verify_numeric(response, "1250") is True


def test_numeric_tolerance():
    # 33.33 vs 33.3333 with 0.05 tolerance
    assert verify_numeric("The answer is 33.33 hours.", 33.3333, tolerance=0.05) is True
    # Off by too much
    assert verify_numeric("The answer is 33.10 hours.", 33.3333, tolerance=0.001) is False


def test_numeric_empty_or_no_numbers():
    assert verify_numeric("", 42) is False
    assert verify_numeric("I don't know the answer.", 42) is False


# ── 2. Code Verifier Tests ─────────────────────────────────────────────────

def test_code_correct_function():
    code_resp = """```python
def similar_elements(test_tup1, test_tup2):
    return tuple(set(test_tup1) & set(test_tup2))
```"""
    tests = [
        "assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)",
        "assert similar_elements((1, 2, 3, 4), (5, 4, 3, 7)) == (3, 4)",
    ]
    assert verify_code(code_resp, tests) is True


def test_code_wrong_logic():
    code_resp = """```python
def similar_elements(test_tup1, test_tup2):
    return (1, 2)
```"""
    tests = [
        "assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)",
    ]
    assert verify_code(code_resp, tests) is False


def test_code_ast_security_blocks_os_import():
    hostile_resp = """```python
import os
def dangerous():
    os.system("echo hacked")
    return 42
```"""
    tests = ["assert dangerous() == 42"]
    assert verify_code(hostile_resp, tests) is False


def test_code_ast_security_blocks_eval():
    hostile_resp = """```python
def sneaky():
    eval("__import__('os').system('ls')")
    return True
```"""
    tests = ["assert sneaky() is True"]
    assert verify_code(hostile_resp, tests) is False


def test_code_timeout():
    infinite_loop = """```python
def loop_forever():
    while True:
        pass
    return 1
```"""
    tests = ["assert loop_forever() == 1"]
    # Should safely timeout and return False
    assert verify_code(infinite_loop, tests, timeout_sec=0.5) is False


# ── 3. Exact-Match QA Verifier Tests ───────────────────────────────────────

def test_exact_match_mcq_letter():
    resp1 = "The correct answer is (B)."
    assert verify_exact_match(resp1, "B") is True

    resp2 = "Option C"
    assert verify_exact_match(resp2, "C") is True
    assert verify_exact_match(resp2, "A") is False


def test_exact_match_normalized_phrase():
    resp = "Photosynthesis is the process that powers the food web."
    assert verify_exact_match(resp, "photosynthesis") is True


def test_exact_match_with_aliases():
    resp = "The United States of America won the tournament."
    assert verify_exact_match(resp, "USA", aliases=["United States", "U.S.", "United States of America"]) is True


# ── 4. Master Dispatch Verifier Tests ──────────────────────────────────────

def test_master_dispatch_numeric():
    spec = {"answer_type": "numeric", "answer": "100", "tolerance": 0.01}
    assert verify("Final count: 100", spec) is True
    assert verify("Final count: 50", spec) is False


def test_master_dispatch_code():
    spec = {
        "answer_type": "code_test",
        "test_list": ["assert add(2, 3) == 5"],
        "test_setup_code": "",
    }
    assert verify("def add(a, b):\n    return a + b", spec) is True
    assert verify("def add(a, b):\n    return a - b", spec) is False


def test_master_dispatch_qa():
    spec = {
        "answer_type": "exact_match",
        "answer": "D",
        "answer_aliases": ["D", "(D)", "Mitochondria"],
    }
    assert verify("Answer: D", spec) is True
    assert verify("The powerhouse is Mitochondria.", spec) is True
    assert verify("Answer: A", spec) is False
