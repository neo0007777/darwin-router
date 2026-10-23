#!/usr/bin/env python3
"""End-to-End Evaluation on TEST Split (Steps 5 & 6).

Evaluates 9 policies purely by cache lookup:
1. always-small (Llama 3.2 3B)
2. always-big (Gemini 3.5 Flash)
3. category-rule (Math/Code -> Big; Trivia/ARC -> Small)
4. static-knn (Bootstrap seeds only)
5. naive-knn (Uniform k=5 on exemplars)
6. confidence-gated (Threshold-based escalation)
7. darwin-router (Frozen dev-tuned parameters on clean exemplars)
8. darwin-verifier-probed (Frozen parameters on probed exemplars)
9. oracle-router (Cheapest model that is correct; False if both fail)

Computes:
- End-to-end accuracy with 95% bootstrap CI (10,000 resamples)
- Proxy cost per query
- Cost reduction vs always-big
- Accuracy retention vs always-big (policy_acc / always_big_acc)
- Paired bootstrap differences with 95% CIs (no p-values)
- Route-label match accuracy as secondary diagnostic
- Lambda sweep (0.0 - 1.0, 11 points)
- Saves assets/quality_vs_cost.png and assets/results.json
- Runs sanity checks (fails loudly if any check fails)
"""

import json
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from router import config
from router.costs import calculate_cost

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
CONFIG_DIR = ROOT / "config"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = DATA_DIR / "test.jsonl"
TEST_CACHE_PATH = CACHE_DIR / "test.jsonl"
EXEMPLARS_CLEAN_PATH = DATA_DIR / "exemplars.jsonl"
EXEMPLARS_PROBED_PATH = DATA_DIR / "exemplars_probed.jsonl"
FROZEN_CONFIG_PATH = CONFIG_DIR / "frozen.json"
RESULTS_JSON_PATH = ASSETS_DIR / "results.json"
PLOT_PNG_PATH = ASSETS_DIR / "quality_vs_cost.png"


def load_jsonl(p: Path) -> list[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap_ci(values: list[float], n_resamples: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(42)
    sample_indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    resampled_means = np.mean(arr[sample_indices], axis=1)
    low_pct = (1.0 - ci) / 2.0 * 100.0
    high_pct = (1.0 + ci) / 2.0 * 100.0
    return float(np.mean(arr)), float(np.percentile(resampled_means, low_pct)), float(np.percentile(resampled_means, high_pct))


def paired_bootstrap_diff(a: list[float], b: list[float], n_resamples: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    assert len(arr_a) == len(arr_b)
    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    rng = np.random.default_rng(42)
    sample_indices = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    resampled_diff_means = np.mean(diff[sample_indices], axis=1)
    low_pct = (1.0 - ci) / 2.0 * 100.0
    high_pct = (1.0 + ci) / 2.0 * 100.0
    return mean_diff, float(np.percentile(resampled_diff_means, low_pct)), float(np.percentile(resampled_diff_means, high_pct))


def main():
    print("=" * 70)
    print("END-TO-END EVALUATION ON TEST SPLIT (LOOKUP ONLY)")
    print("=" * 70)

    if not TEST_CACHE_PATH.exists():
        print(f"ERROR: {TEST_CACHE_PATH} not found. Run cache_model_answers.py --split test first.", file=sys.stderr)
        sys.exit(1)
    if not FROZEN_CONFIG_PATH.exists():
        print(f"ERROR: {FROZEN_CONFIG_PATH} not found. Run tune_dev.py first.", file=sys.stderr)
        sys.exit(1)

    with open(FROZEN_CONFIG_PATH, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)

    print("Loaded Frozen Hyperparameters (tuned on dev):")
    print(f"  k:                     {frozen_cfg['k']}")
    print(f"  sim_exponent:          {frozen_cfg['sim_exponent']}")
    print(f"  margin_threshold:      {frozen_cfg['margin_threshold']}")
    print(f"  near_duplicate_cutoff: {frozen_cfg['near_duplicate_cutoff']}")

    test_items = load_jsonl(TEST_PATH)
    test_cache = {r["id"]: r for r in load_jsonl(TEST_CACHE_PATH)}
    clean_exemplars = load_jsonl(EXEMPLARS_CLEAN_PATH)
    probed_exemplars = load_jsonl(EXEMPLARS_PROBED_PATH)

    print(f"\nLoaded {len(test_items)} test items, {len(clean_exemplars)} clean exemplars, {len(probed_exemplars)} probed exemplars.")

    # Validate test cache completeness
    for item in test_items:
        if item["id"] not in test_cache:
            print(f"ERROR: Incomplete test cache. Missing {item['id']}", file=sys.stderr)
            sys.exit(1)

    # Embed test queries and exemplars
    model = SentenceTransformer("all-MiniLM-L6-v2")
    test_texts = [t["text"] for t in test_items]
    test_embs = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)

    clean_embs = np.array([e["embedding"] for e in clean_exemplars], dtype=np.float32)
    clean_labels = np.array([e["label"] for e in clean_exemplars])

    probed_embs = np.array([e["embedding"] for e in probed_exemplars], dtype=np.float32)
    probed_labels = np.array([e["label"] for e in probed_exemplars])

    sim_clean = np.dot(test_embs, clean_embs.T)
    sim_probed = np.dot(test_embs, probed_embs.T)

    # Data structures to store per-query results for each policy
    policy_correct: dict[str, list[float]] = {}
    policy_costs: dict[str, list[float]] = {}
    policy_routes: dict[str, list[str]] = {}
    policy_label_matches: dict[str, list[float]] = {}

    policies = [
        "always-small",
        "always-big",
        "category-rule",
        "static-knn",
        "naive-knn",
        "confidence-gated",
        "darwin-router",
        "darwin-verifier-probed",
        "oracle-router",
    ]

    for p in policies:
        policy_correct[p] = []
        policy_costs[p] = []
        policy_routes[p] = []
        policy_label_matches[p] = []

    # Run routing lookup per test query
    for q_idx, item in enumerate(test_items):
        qid = item["id"]
        rec = test_cache[qid]
        cat = item.get("category", "")

        small_ok = rec["small_ok"]
        big_ok = rec["big_ok"]
        small_cost = calculate_cost(rec["small_model"], rec["small_in_tokens"], rec["small_out_tokens"])
        big_cost = calculate_cost(rec["big_model"], rec["big_in_tokens"], rec["big_out_tokens"])

        # Ground truth route label for diagnostic
        gt_route = "simple" if small_ok else ("reasoning" if big_ok else "reasoning")

        # 1. always-small
        policy_routes["always-small"].append("simple")
        policy_correct["always-small"].append(1.0 if small_ok else 0.0)
        policy_costs["always-small"].append(small_cost)
        policy_label_matches["always-small"].append(1.0 if gt_route == "simple" else 0.0)

        # 2. always-big
        policy_routes["always-big"].append("reasoning")
        policy_correct["always-big"].append(1.0 if big_ok else 0.0)
        policy_costs["always-big"].append(big_cost)
        policy_label_matches["always-big"].append(1.0 if gt_route == "reasoning" else 0.0)

        # 3. category-rule: math/code -> reasoning; trivia/arc -> simple
        cat_route = "reasoning" if cat in ("math", "code") else "simple"
        policy_routes["category-rule"].append(cat_route)
        policy_correct["category-rule"].append(1.0 if (big_ok if cat_route == "reasoning" else small_ok) else 0.0)
        policy_costs["category-rule"].append(big_cost if cat_route == "reasoning" else small_cost)
        policy_label_matches["category-rule"].append(1.0 if cat_route == gt_route else 0.0)

        # 4. static-knn: top-5 uniform on first 20 seeds
        sub_sims = sim_clean[q_idx, :20]
        top_k_sub = np.argsort(sub_sims)[-5:]
        sub_labels = clean_labels[:20][top_k_sub]
        static_route = "simple" if np.sum(sub_labels == "simple") >= np.sum(sub_labels == "reasoning") else "reasoning"
        policy_routes["static-knn"].append(static_route)
        policy_correct["static-knn"].append(1.0 if (small_ok if static_route == "simple" else big_ok) else 0.0)
        policy_costs["static-knn"].append(small_cost if static_route == "simple" else big_cost)
        policy_label_matches["static-knn"].append(1.0 if static_route == gt_route else 0.0)

        # 5. naive-knn: top-5 uniform vote on clean exemplars
        top_5_idx = np.argsort(sim_clean[q_idx])[-5:]
        top_5_labels = clean_labels[top_5_idx]
        naive_route = "simple" if np.sum(top_5_labels == "simple") >= np.sum(top_5_labels == "reasoning") else "reasoning"
        policy_routes["naive-knn"].append(naive_route)
        policy_correct["naive-knn"].append(1.0 if (small_ok if naive_route == "simple" else big_ok) else 0.0)
        policy_costs["naive-knn"].append(small_cost if naive_route == "simple" else big_cost)
        policy_label_matches["naive-knn"].append(1.0 if naive_route == gt_route else 0.0)

        # 6. confidence-gated: if max similarity < 0.60 or margin < 0.50 -> reasoning
        max_s = float(np.max(sim_clean[q_idx]))
        top_k_cg = np.argsort(sim_clean[q_idx])[-5:]
        cg_sims = np.maximum(0.0, sim_clean[q_idx][top_k_cg])
        cg_labels = clean_labels[top_k_cg]
        s_score = float(np.sum(cg_sims[cg_labels == "simple"]))
        r_score = float(np.sum(cg_sims[cg_labels == "reasoning"]))
        margin_cg = abs(s_score - r_score) / (s_score + r_score + 1e-9)
        if max_s < 0.60 or margin_cg < 0.50:
            cg_route = "reasoning"
        else:
            cg_route = "simple" if s_score > r_score else "reasoning"
        policy_routes["confidence-gated"].append(cg_route)
        policy_correct["confidence-gated"].append(1.0 if (small_ok if cg_route == "simple" else big_ok) else 0.0)
        policy_costs["confidence-gated"].append(small_cost if cg_route == "simple" else big_cost)
        policy_label_matches["confidence-gated"].append(1.0 if cg_route == gt_route else 0.0)

        # 7. darwin-router (frozen dev hyperparameters on clean index)
        k_d = frozen_cfg["k"]
        exp_d = frozen_cfg["sim_exponent"]
        m_th = frozen_cfg["margin_threshold"]
        dup_th = frozen_cfg["near_duplicate_cutoff"]

        sims_d = sim_clean[q_idx]
        max_idx_d = int(np.argmax(sims_d))
        max_sim_d = float(sims_d[max_idx_d])

        if max_sim_d >= dup_th:
            d_route = clean_labels[max_idx_d]
        else:
            top_k_d = np.argsort(sims_d)[-k_d:][::-1]
            weights_d = np.maximum(0.0, sims_d[top_k_d]) ** exp_d
            labels_d = clean_labels[top_k_d]

            s_w = float(np.sum(weights_d[labels_d == "simple"]))
            r_w = float(np.sum(weights_d[labels_d == "reasoning"]))
            tot_w = s_w + r_w

            if tot_w == 0:
                d_route = "reasoning"
            else:
                margin_d = abs(s_w - r_w) / tot_w
                raw_d = "simple" if s_w > r_w else "reasoning"
                d_route = "reasoning" if margin_d < m_th else raw_d

        policy_routes["darwin-router"].append(d_route)
        policy_correct["darwin-router"].append(1.0 if (small_ok if d_route == "simple" else big_ok) else 0.0)
        policy_costs["darwin-router"].append(small_cost if d_route == "simple" else big_cost)
        policy_label_matches["darwin-router"].append(1.0 if d_route == gt_route else 0.0)

        # 8. darwin-verifier-probed (frozen parameters on probed index)
        sims_p = sim_probed[q_idx]
        max_idx_p = int(np.argmax(sims_p))
        max_sim_p = float(sims_p[max_idx_p])

        if max_sim_p >= dup_th:
            p_route = probed_labels[max_idx_p]
        else:
            top_k_p = np.argsort(sims_p)[-k_d:][::-1]
            weights_p = np.maximum(0.0, sims_p[top_k_p]) ** exp_d
            labels_p = probed_labels[top_k_p]

            s_wp = float(np.sum(weights_p[labels_p == "simple"]))
            r_wp = float(np.sum(weights_p[labels_p == "reasoning"]))
            tot_wp = s_wp + r_wp

            if tot_wp == 0:
                p_route = "reasoning"
            else:
                margin_p = abs(s_wp - r_wp) / tot_wp
                raw_p = "simple" if s_wp > r_wp else "reasoning"
                p_route = "reasoning" if margin_p < m_th else raw_p

        policy_routes["darwin-verifier-probed"].append(p_route)
        policy_correct["darwin-verifier-probed"].append(1.0 if (small_ok if p_route == "simple" else big_ok) else 0.0)
        policy_costs["darwin-verifier-probed"].append(small_cost if p_route == "simple" else big_cost)
        policy_label_matches["darwin-verifier-probed"].append(1.0 if p_route == gt_route else 0.0)

        # 9. oracle-router: cheapest model that answers correctly; False if both fail
        if not small_ok and not big_ok:
            o_correct = 0.0
            o_cost = small_cost
            o_route = "simple"
        elif small_ok:
            o_correct = 1.0
            o_cost = small_cost
            o_route = "simple"
        else:
            o_correct = 1.0
            o_cost = big_cost
            o_route = "reasoning"

        policy_routes["oracle-router"].append(o_route)
        policy_correct["oracle-router"].append(o_correct)
        policy_costs["oracle-router"].append(o_cost)
        policy_label_matches["oracle-router"].append(1.0)

    # Compute Metrics & CIs per policy
    base_big_cost = float(np.mean(policy_costs["always-big"]))
    base_big_acc = float(np.mean(policy_correct["always-big"]))

    summary_results = {}
    print("\n" + "=" * 90)
    print(f"{'Policy':<24} {'End-to-End Acc (95% CI)':<26} {'Proxy Cost/Req':<16} {'Cost Red':<10} {'Acc Ret':<8} {'Route Match':<10}")
    print("=" * 90)

    for p in policies:
        acc_mean, acc_low, acc_high = bootstrap_ci(policy_correct[p])
        cost_mean, cost_low, cost_high = bootstrap_ci(policy_costs[p])
        match_mean, _, _ = bootstrap_ci(policy_label_matches[p])

        cost_red = (base_big_cost - cost_mean) / base_big_cost
        acc_ret = acc_mean / base_big_acc if base_big_acc > 0 else 0.0

        summary_results[p] = {
            "accuracy_mean": acc_mean,
            "accuracy_ci": [acc_low, acc_high],
            "cost_mean": cost_mean,
            "cost_ci": [cost_low, cost_high],
            "cost_reduction_vs_big": cost_red,
            "accuracy_retention_vs_big": acc_ret,
            "route_label_match_acc": match_mean,
        }

        acc_ci_str = f"{acc_mean*100:5.1f}% [{acc_low*100:4.1f}%–{acc_high*100:4.1f}%]"
        cost_str = f"${cost_mean:.6f}"
        red_str = f"{-cost_red*100:5.1f}%" if abs(cost_red) >= 0.001 else "Baseline"
        ret_str = f"{acc_ret*100:5.1f}%"
        match_str = f"{match_mean*100:4.1f}%"

        print(f"{p:<24} {acc_ci_str:<26} {cost_str:<16} {red_str:<10} {ret_str:<8} {match_str:<10}")

    print("=" * 90)

    # Paired Bootstrap Differences
    paired_comparisons = [
        ("darwin-router", "always-small"),
        ("darwin-router", "always-big"),
        ("darwin-router", "category-rule"),
        ("darwin-router", "naive-knn"),
        ("darwin-verifier-probed", "darwin-router"),
    ]

    paired_results = {}
    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP DIFFERENCES (10,000 RESAMPLES, 95% CIs)")
    print("=" * 80)
    print(f"{'Comparison (A vs B)':<40} {'Mean Diff (pp)':<16} {'95% Bootstrap CI':<24}")
    print("-" * 80)

    for pA, pB in paired_comparisons:
        diff_mean, diff_low, diff_high = paired_bootstrap_diff(policy_correct[pA], policy_correct[pB])
        key = f"{pA}__vs__{pB}"
        paired_results[key] = {
            "policy_a": pA,
            "policy_b": pB,
            "mean_difference_pp": diff_mean * 100.0,
            "ci_95": [diff_low * 100.0, diff_high * 100.0],
        }
        ci_str = f"[{diff_low*100:+.2f}pp, {diff_high*100:+.2f}pp]"
        print(f"{f'{pA} vs {pB}':<40} {diff_mean*100:+6.2f} pp       {ci_str:<24}")
    print("=" * 80)

    # Lambda Sweep (0.0 - 1.0, 11 points)
    print("\nRunning Lambda Cost-Weight Sweep (0.0 to 1.0)...")
    lambda_vals = np.linspace(0.0, 1.0, 11)
    lambda_points = []

    k_d = frozen_cfg["k"]
    exp_d = frozen_cfg["sim_exponent"]

    # Normalize costs by big cost
    norm_small_cost = float(np.mean([calculate_cost(test_cache[t["id"]]["small_model"], test_cache[t["id"]]["small_in_tokens"], test_cache[t["id"]]["small_out_tokens"]) for t in test_items])) / base_big_cost
    norm_big_cost = 1.0

    for lam in lambda_vals:
        lam_correct = []
        lam_costs = []
        for q_idx, item in enumerate(test_items):
            qid = item["id"]
            rec = test_cache[qid]
            sims = sim_clean[q_idx]

            top_k_indices = np.argsort(sims)[-k_d:][::-1]
            weights = np.maximum(0.0, sims[top_k_indices]) ** exp_d
            labels = clean_labels[top_k_indices]

            s_w = float(np.sum(weights[labels == "simple"]))
            r_w = float(np.sum(weights[labels == "reasoning"]))
            tot = s_w + r_w

            if tot == 0:
                route = "reasoning"
            else:
                # Margin in favor of simple: (s_w - r_w) / tot
                simple_margin = (s_w - r_w) / tot
                # Objective for routing to simple vs reasoning:
                # Utility(simple) = (1 - lam) * simple_margin - lam * norm_small_cost
                # Utility(reasoning) = (1 - lam) * (-simple_margin) - lam * norm_big_cost
                # Simple wins if Utility(simple) > Utility(reasoning):
                # 2 * (1 - lam) * simple_margin + lam * (norm_big_cost - norm_small_cost) > 0
                threshold = -lam * (norm_big_cost - norm_small_cost) / (2.0 * max(1e-6, 1.0 - lam))
                route = "simple" if simple_margin > threshold else "reasoning"

            is_corr = rec["small_ok"] if route == "simple" else rec["big_ok"]
            c = calculate_cost(rec["small_model"], rec["small_in_tokens"], rec["small_out_tokens"]) if route == "simple" else calculate_cost(rec["big_model"], rec["big_in_tokens"], rec["big_out_tokens"])
            lam_correct.append(1.0 if is_corr else 0.0)
            lam_costs.append(c)

        lambda_points.append({
            "lambda": float(lam),
            "accuracy": float(np.mean(lam_correct)),
            "cost": float(np.mean(lam_costs)),
        })

    # Plot Quality vs Cost
    plt.figure(figsize=(9, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot lambda sweep curve
    sweep_costs = [p["cost"] * 1e5 for p in lambda_points]
    sweep_accs = [p["accuracy"] * 100.0 for p in lambda_points]
    plt.plot(sweep_costs, sweep_accs, "o-", color="#3b82f6", linewidth=2.0, markersize=5, label="Darwin Router (Lambda Cost Sweep)")

    # Plot landmark points
    landmarks = [
        ("Oracle Router", summary_results["oracle-router"]["cost_mean"]*1e5, summary_results["oracle-router"]["accuracy_mean"]*100, "#10b981", "*", 14),
        ("Darwin Router (Frozen)", summary_results["darwin-router"]["cost_mean"]*1e5, summary_results["darwin-router"]["accuracy_mean"]*100, "#6366f1", "D", 9),
        ("Darwin + Probes", summary_results["darwin-verifier-probed"]["cost_mean"]*1e5, summary_results["darwin-verifier-probed"]["accuracy_mean"]*100, "#8b5cf6", "^", 8),
        ("Category Rule", summary_results["category-rule"]["cost_mean"]*1e5, summary_results["category-rule"]["accuracy_mean"]*100, "#f59e0b", "s", 8),
        ("Naive kNN", summary_results["naive-knn"]["cost_mean"]*1e5, summary_results["naive-knn"]["accuracy_mean"]*100, "#06b6d4", "v", 8),
        ("Always-Big (Flash)", summary_results["always-big"]["cost_mean"]*1e5, summary_results["always-big"]["accuracy_mean"]*100, "#ef4444", "P", 10),
        ("Always-Small (3B)", summary_results["always-small"]["cost_mean"]*1e5, summary_results["always-small"]["accuracy_mean"]*100, "#64748b", "X", 10),
    ]

    for name, c_val, a_val, color, marker, size in landmarks:
        plt.plot(c_val, a_val, marker=marker, color=color, markersize=size, linestyle="None", label=name)

    plt.xlabel("Proxy Cost per Request ($ × 10⁻⁵)", fontsize=11, fontweight="600")
    plt.ylabel("End-to-End Test Accuracy (%)", fontsize=11, fontweight="600")
    plt.title("Quality vs. Cost Trade-off: End-to-End Correctness on Held-Out Test Set (N=300)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_PNG_PATH, dpi=150)
    plt.close()
    print(f"✓ Saved trade-off plot to {PLOT_PNG_PATH}")

    # Save results.json
    final_output = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_items_count": len(test_items),
        "clean_exemplars_count": len(clean_exemplars),
        "probed_exemplars_count": len(probed_exemplars),
        "frozen_hyperparameters": frozen_cfg,
        "policies": summary_results,
        "paired_differences": paired_results,
        "lambda_sweep": lambda_points,
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"✓ Saved results to {RESULTS_JSON_PATH}")

    # =========================================================================
    # Step 6: Sanity Checks (Fail loudly if any fails)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING STEP 6 SANITY CHECKS")
    print("=" * 70)

    # 1. Oracle accuracy >= every other policy
    oracle_acc = summary_results["oracle-router"]["accuracy_mean"]
    for p, stats in summary_results.items():
        if p != "oracle-router":
            assert oracle_acc >= stats["accuracy_mean"] - 1e-6, \
                f"SANITY CHECK FAILED: Oracle acc ({oracle_acc}) < {p} acc ({stats['accuracy_mean']})"
    print("✓ Sanity Check 1 Passed: Oracle accuracy >= every other policy.")

    # 2. Always-Big end-to-end accuracy > Always-Small accuracy
    big_acc = summary_results["always-big"]["accuracy_mean"]
    small_acc = summary_results["always-small"]["accuracy_mean"]
    assert big_acc > small_acc, \
        f"SANITY CHECK FAILED: Always-Big acc ({big_acc}) <= Always-Small acc ({small_acc})"
    print(f"✓ Sanity Check 2 Passed: Always-Big ({big_acc*100:.1f}%) > Always-Small ({small_acc*100:.1f}%).")

    # 3. Darwin+probes must not have the same cost as oracle unless it also has the same accuracy
    probes_cost = summary_results["darwin-verifier-probed"]["cost_mean"]
    oracle_cost = summary_results["oracle-router"]["cost_mean"]
    probes_acc = summary_results["darwin-verifier-probed"]["accuracy_mean"]
    if abs(probes_cost - oracle_cost) < 1e-9:
        assert abs(probes_acc - oracle_acc) < 1e-9, \
            "SANITY CHECK FAILED: Darwin+probes has identical cost to Oracle without identical accuracy."
    print("✓ Sanity Check 3 Passed: Darwin+probes cost distinction verified.")

    # 4. Max dev/test cosine similarity against seeds < 0.90
    max_test_sim = float(np.max(sim_clean))
    assert max_test_sim < 0.90, f"SANITY CHECK FAILED: Max test-to-seed cosine {max_test_sim:.4f} >= 0.90"
    print(f"✓ Sanity Check 4 Passed: Max test-to-seed cosine similarity is {max_test_sim:.4f} (< 0.90).")

    print("\n✓ ALL SANITY CHECKS PASSED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
