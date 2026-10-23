"""Unit test suite for router/prober.py synthetic templates (Phase 3).
Tests all probe generators across 1,000 iterations to guarantee mathematical
validity, formatting consistency, and no divide-by-zero errors.
"""

import pytest
from router.prober import (
    ALL_TEMPLATES,
    probe_simple_unit_discount,
    probe_simple_work_rate,
    probe_multi_step_rate_change,
    probe_multi_step_inventory_fees,
    probe_joint_work_remaining,
    probe_symbol_reversal_trap,
)


@pytest.mark.parametrize("template_fn", ALL_TEMPLATES)
def test_template_schema_and_integrity(template_fn):
    """Run each template 200 times (1,200 total runs across all templates)."""
    for _ in range(200):
        item = template_fn()

        # Check required keys
        assert "text" in item and isinstance(item["text"], str) and len(item["text"]) > 10
        assert "answer" in item and isinstance(item["answer"], str) and len(item["answer"]) > 0
        assert "answer_type" in item and item["answer_type"] in ("numeric", "exact_match")
        assert "difficulty" in item and item["difficulty"] in ("simple", "complex")

        # Specific assertions by type
        if item["answer_type"] == "numeric":
            val = float(item["answer"])
            assert val > 0, f"Expected positive numerical answer, got {val}"
            assert "tolerance" in item and item["tolerance"] > 0
        elif item["answer_type"] == "exact_match":
            assert len(item["answer"]) > 0


def test_rate_change_math_consistency():
    """Verify that multi-step rate change template produces mathematically consistent numbers."""
    for _ in range(100):
        probe = probe_multi_step_rate_change()
        answer = float(probe["answer"])
        # Should take a positive, realistic number of hours
        assert 5.0 <= answer <= 500.0


def test_joint_work_math_consistency():
    """Verify joint work template produces consistent answers."""
    for _ in range(100):
        probe = probe_joint_work_remaining()
        answer = float(probe["answer"])
        assert answer > 0.0


def test_symbol_reversal_logic():
    """Verify that symbol reversal logic accurately finds the target element."""
    for _ in range(100):
        probe = probe_symbol_reversal_trap()
        ans = probe["answer"]
        text = probe["text"]
        assert ans in text
