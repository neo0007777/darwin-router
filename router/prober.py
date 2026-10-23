"""Active Boundary Probing Engine (Phase 3).
Generates parameter-randomized synthetic boundary probes with mathematically computed ground truth.
Runs local models at temperature=0, programmatically verifies answers,
and discovers clean boundary exemplars without paying for an LLM judge.
"""

import json
import random
import time
from typing import Callable

from router import config
from router.embed import get_client
from router.index import get_index
from router.verifier import verify


# ─────────────────────────────────────────────────────────────────────────────
# Parametric Probe Templates (Deterministic Ground Truth)
# ─────────────────────────────────────────────────────────────────────────────

def probe_simple_unit_discount() -> dict:
    """Simple 1-step arithmetic: item cost minus flat or percentage discount."""
    price = random.choice([20, 30, 40, 50, 60, 80, 100])
    discount_pct = random.choice([10, 20, 25, 50])
    discount_amount = (price * discount_pct) // 100
    final_price = price - discount_amount

    items = ["jacket", "backpack", "pair of shoes", "smartwatch", "guitar pedal"]
    item = random.choice(items)

    text = f"A {item} normally costs ${price}. During a weekend sale, it is discounted by {discount_pct}%. What is the final price in dollars?"
    return {
        "text": text,
        "answer": str(final_price),
        "answer_type": "numeric",
        "tolerance": 0.01,
        "template_name": "simple_unit_discount",
        "difficulty": "simple",
    }


def probe_simple_work_rate() -> dict:
    """Simple 1-step rate: speed * time = distance."""
    speed = random.choice([40, 50, 60, 70, 80])
    hours = random.choice([2, 3, 4, 5])
    dist = speed * hours
    text = f"A train travels at a constant speed of {speed} miles per hour. How many miles does it travel in {hours} hours?"
    return {
        "text": text,
        "answer": str(dist),
        "answer_type": "numeric",
        "tolerance": 0.01,
        "template_name": "simple_work_rate",
        "difficulty": "simple",
    }


def probe_multi_step_rate_change() -> dict:
    """Complex multi-step: filling a tank where the rate changes halfway."""
    # Ensure rate1 is even and divides capacity / 2 evenly
    rate1 = random.choice([4, 6, 8, 10, 12])
    rate2 = rate1 // 2
    # Capacity chosen so half-capacity is integer multiple of rate1
    multiplier = random.choice([10, 15, 20, 25])
    capacity = multiplier * 2 * rate1

    time_first_half = (capacity / 2) / rate1
    time_second_half = (capacity / 2) / rate2
    total_time = round(time_first_half + time_second_half, 2)

    text = (
        f"A water tank holds {capacity} liters. An intake valve fills it at {rate1} liters per hour, "
        f"but to prevent overflow, the flow rate halves when the tank reaches half capacity. "
        f"How many hours does it take to fill the tank completely from empty?"
    )
    return {
        "text": text,
        "answer": str(total_time),
        "answer_type": "numeric",
        "tolerance": 0.05,
        "template_name": "multi_step_rate_change",
        "difficulty": "complex",
    }


def probe_multi_step_inventory_fees() -> dict:
    """Complex multi-step: multiple sales, returns, and commission/storage fees."""
    initial_stock = random.choice([100, 200, 300])
    sold = random.choice([30, 40, 50])
    returned = random.choice([5, 10])
    unit_price = random.choice([10, 20])
    fee_per_unit = 2

    remaining_stock = initial_stock - sold + returned
    net_sold = sold - returned
    revenue = (net_sold * unit_price) - (net_sold * fee_per_unit)

    text = (
        f"A store starts with {initial_stock} gadgets. In week one, they sell {sold} gadgets. "
        f"However, {returned} unsatisfied customers return their gadgets for full refunds. "
        f"If each gadget sells for ${unit_price} and the payment processor takes a flat fee of "
        f"${fee_per_unit} per retained sale, what is the store's net profit in dollars from these sales?"
    )
    return {
        "text": text,
        "answer": str(revenue),
        "answer_type": "numeric",
        "tolerance": 0.01,
        "template_name": "multi_step_inventory_fees",
        "difficulty": "complex",
    }


def probe_joint_work_remaining() -> dict:
    """Complex work-rate: two workers work together, then one leaves."""
    # Worker A takes time A, Worker B takes time B
    # Let rateA = 1/a, rateB = 1/b
    # Let common units: work = LCM(a, b) * k
    rate_a_units = random.choice([3, 4, 5])
    rate_b_units = random.choice([2, 3])
    # Together they do (rate_a_units + rate_b_units) units per hour
    hours_together = random.choice([2, 3])
    remaining_hours_b = random.choice([3, 4, 5])

    total_work = (rate_a_units + rate_b_units) * hours_together + rate_b_units * remaining_hours_b
    time_a_alone = round(total_work / rate_a_units, 2)

    text = (
        f"Alice and Bob are painting a house. Together, Alice paints at {rate_a_units} rooms per day "
        f"and Bob paints at {rate_b_units} rooms per day. They work together for {hours_together} days, "
        f"after which Alice leaves and Bob finishes the remaining rooms in {remaining_hours_b} days alone. "
        f"If Alice had painted the entire house by herself from the start, how many days would it take her?"
    )
    return {
        "text": text,
        "answer": str(time_a_alone),
        "answer_type": "numeric",
        "tolerance": 0.05,
        "template_name": "joint_work_remaining",
        "difficulty": "complex",
    }


def probe_symbol_reversal_trap() -> dict:
    """Complex sequence trap: reversing a punctuated sequence and asking for index/length."""
    symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    k = random.choice([4, 5, 6])
    sub = symbols[:k]
    # Reverse the list
    rev = list(reversed(sub))
    target_idx = random.randint(1, k)
    target_item = rev[target_idx - 1]

    text = (
        f"Consider the ordered list of code words: {', '.join(sub)}. "
        f"If the entire list is reversed, which code word appears at position {target_idx} (1-indexed)? "
        f"Answer with only the exact code word."
    )
    return {
        "text": text,
        "answer": target_item,
        "answer_type": "exact_match",
        "template_name": "symbol_reversal_trap",
        "difficulty": "complex",
    }


def probe_simple_code_slice() -> dict:
    """Simple 1-line Python coding task: string slicing or reversing."""
    word = random.choice(["python", "algorithm", "routing", "vector", "matrix"])
    ans = word[::-1]
    text = f"What is the output of the Python expression '{word}'[::-1]? Answer with only the resulting string."
    return {
        "text": text,
        "answer": ans,
        "answer_type": "exact_match",
        "template_name": "simple_code_slice",
        "difficulty": "simple",
    }


def probe_complex_code_recursion() -> dict:
    """Complex multi-step algorithmic trace: recurrence relation or fibonacci."""
    n = random.choice([5, 6, 7])
    # T(0) = 1, T(n) = 2*T(n-1) + 1
    val = (2 ** (n + 1)) - 1
    text = (
        f"For the recurrence relation T(0) = 1 and T(n) = 2*T(n-1) + 1, "
        f"what is the exact integer value of T({n})? Answer with only the integer."
    )
    return {
        "text": text,
        "answer": str(val),
        "answer_type": "numeric",
        "tolerance": 0.01,
        "template_name": "complex_code_recursion",
        "difficulty": "complex",
    }


def probe_simple_science_recall() -> dict:
    """Simple direct science recall: state changes or celestial facts."""
    pairs = [
        ("water boils at sea level", "100", "in degrees Celsius"),
        ("pure ethanol boils at standard pressure", "78", "in degrees Celsius"),
        ("normal human body temperature", "37", "in degrees Celsius"),
    ]
    fact, ans, unit = random.choice(pairs)
    text = f"At what temperature ({unit}) does {fact}? Answer with only the number."
    return {
        "text": text,
        "answer": ans,
        "answer_type": "numeric",
        "tolerance": 0.1,
        "template_name": "simple_science_recall",
        "difficulty": "simple",
    }


def probe_complex_science_cause_effect() -> dict:
    """Complex science multi-hop: conservation of momentum and energy."""
    m1 = random.choice([2, 4])
    v1 = random.choice([6, 8, 10])
    m2 = random.choice([2, 4])
    # Inelastic collision: m1*v1 = (m1+m2)*v_f
    vf = round((m1 * v1) / (m1 + m2), 2)
    text = (
        f"A cart of mass {m1} kg moving at {v1} m/s collides with and sticks to a stationary cart of mass {m2} kg. "
        f"Assuming a frictionless track, what is their final combined velocity in m/s? "
        f"Answer with only the numeric value."
    )
    return {
        "text": text,
        "answer": str(vf),
        "answer_type": "numeric",
        "tolerance": 0.05,
        "template_name": "complex_science_cause_effect",
        "difficulty": "complex",
    }


ALL_TEMPLATES: list[Callable[[], dict]] = [
    probe_simple_unit_discount,
    probe_simple_work_rate,
    probe_multi_step_rate_change,
    probe_multi_step_inventory_fees,
    probe_joint_work_remaining,
    probe_symbol_reversal_trap,
    probe_simple_code_slice,
    probe_complex_code_recursion,
    probe_simple_science_recall,
    probe_complex_science_cause_effect,
]


def generate_probe() -> dict:
    template_fn = random.choice(ALL_TEMPLATES)
    return template_fn()


def generate_boundary_probes(n_per_template: int = 1, seed: int = 42) -> list[dict]:
    """Generate deterministic parameter-randomized boundary probes from ALL_TEMPLATES."""
    rng = random.Random(seed)
    # Save global state, set seed, restore
    old_state = random.getstate()
    random.setstate(rng.getstate())
    try:
        probes = []
        for idx, template_fn in enumerate(ALL_TEMPLATES):
            for rep in range(n_per_template):
                item = template_fn()
                route = "simple" if item.get("difficulty") == "simple" else "reasoning"
                tname = item.get("template_name", f"template_{idx}")
                if "code" in tname:
                    category = "code"
                elif "science" in tname:
                    category = "arc"
                else:
                    category = "math"
                probes.append({
                    "id": f"probe_{tname}_{rep}",
                    "text": item["text"],
                    "expected_route": route,
                    "category": category,
                    "template_id": tname,
                    "answer_spec": item,
                })
        return probes
    finally:
        random.setstate(old_state)


# ─────────────────────────────────────────────────────────────────────────────
# Probing Loop (Active Boundary Refinement)
# ─────────────────────────────────────────────────────────────────────────────

def run_probing_cycle(
    num_probes: int = 40,
    dedup_threshold: float = 0.95,
) -> dict:
    """Run an active probing loop to discover boundary exemplars.
    Small model tested via Ollama at temperature=0.
    Verified deterministically with verifier.py.
    """
    import httpx

    index = get_index()
    embedder = get_client()
    if len(index.meta) == 0:
        index.load(embed_fn=embedder.embed)

    probes_generated = 0
    probes_inserted = 0
    small_passed = 0
    small_failed = 0
    dedup_skipped = 0

    print(f"[prober] Starting active boundary probing cycle (target: {num_probes} probes)...")

    for i in range(num_probes):
        probe = generate_probe()
        probes_generated += 1
        q_text = probe["text"]

        # 1. Embed query
        emb = embedder.embed([q_text])[0]

        # 2. Query cheap lane (Ollama llama3.2:3b) at temperature 0
        cheap_model = config.MODEL_FOR_ROUTE.get("simple", "llama3.2:3b")
        try:
            r = httpx.post(
                f"{config.OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": cheap_model,
                    "messages": [{"role": "user", "content": q_text}],
                    "temperature": 0.0,
                },
                timeout=30,
            )
            r.raise_for_status()
            cheap_resp = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[prober] Cheap lane call failed ({e}), skipping probe.")
            continue

        # 3. Deterministic verification
        is_correct = verify(cheap_resp, probe)

        if is_correct:
            small_passed += 1
            # Cheap model succeeds -> correct route is 'simple' (adequacy rule)
            label = "simple"
        else:
            small_failed += 1
            # Cheap model fails on this probe -> boundary query requires 'reasoning'
            label = "reasoning"

        # 4. Poisoning & Dedup guard: check cosine similarity against existing exemplars
        max_sim = index.max_sim(emb, label)
        if max_sim >= dedup_threshold:
            dedup_skipped += 1
            continue

        # 5. Insert verified boundary exemplar into index
        index.add(q_text, label, source="probe", embedding=emb)
        probes_inserted += 1
        print(f"[prober #{i+1:02d}] '{label}' exemplar inserted (sim={max_sim:.3f}): {q_text[:70]}...")

    summary = {
        "generated": probes_generated,
        "inserted": probes_inserted,
        "small_passed": small_passed,
        "small_failed": small_failed,
        "dedup_skipped": dedup_skipped,
        "total_exemplars": len(index.meta),
    }
    print(f"\n[prober] Cycle complete: {summary}")
    return summary


if __name__ == "__main__":
    run_probing_cycle(num_probes=30)
