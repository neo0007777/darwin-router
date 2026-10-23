#!/usr/bin/env python3
"""Master Autonomous Pipeline Runner (Steps 2 through 7).

Executes the full evaluation rebuild end-to-end:
1. Cache dev queries (100 items)
2. Cache seed queries (160 items)
3. Rebuild exemplar index from verified seeds only
4. Tune hyperparameters on dev split and freeze to config/frozen.json
5. Cache test queries (300 items)
6. Evaluate test split end-to-end, compute bootstrap CIs and paired differences,
   run lambda sweep, generate plots and assets/results.json
7. Run sanity checks and full pytest suite
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_step(cmd: list[str], step_name: str):
    print("\n" + "=" * 75)
    print(f"PIPELINE STEP: {step_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n❌ FAILED: {step_name} exited with code {res.returncode} ({elapsed:.1f}s)", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"✓ COMPLETED: {step_name} in {elapsed:.1f}s")


def main():
    print("=" * 75)
    print("STARTING AUTONOMOUS END-TO-END EVALUATION PIPELINE")
    print("=" * 75)

    # 1. Cache DEV split (100 items)
    run_step([sys.executable, "scripts/cache_model_answers.py", "--split", "dev"],
             "Cache Dev Model Answers")

    # 2. Cache SEEDS split (160 items)
    run_step([sys.executable, "scripts/cache_model_answers.py", "--split", "seeds"],
             "Cache Seeds Model Answers")

    # 3. Rebuild exemplar index from verified seeds
    run_step([sys.executable, "scripts/rebuild_index_from_seeds.py"],
             "Rebuild Clean & Probed Exemplar Indices")

    # 4. Tune hyperparameters on DEV split and freeze
    run_step([sys.executable, "scripts/tune_dev.py"],
             "Tune Hyperparameters on Dev Split & Freeze Config")

    # 5. Cache TEST split (300 items)
    run_step([sys.executable, "scripts/cache_model_answers.py", "--split", "test"],
             "Cache Test Model Answers")

    # 6. Evaluate on TEST split end-to-end
    run_step([sys.executable, "scripts/eval_test_end_to_end.py"],
             "Evaluate on Test Split (9 Policies, CIs, Paired Diffs, Plots)")

    # 7. Run Pytest Suite
    run_step([sys.executable, "-m", "pytest", "tests/", "-v"],
             "Run Complete Pytest Verification Suite")

    print("\n" + "=" * 75)
    print("🎉 FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
