#!/usr/bin/env python3
"""Autonomous Evaluation Pipeline Runner.

Chains all steps cleanly:
1. Waits for PID of active dev caching task if running.
2. Verifies dev split cache is 100% complete.
3. Caches seeds split (160 items).
4. Rebuilds index exemplars from verified seeds (clean and probed).
5. Tunes hyperparameters on dev split and freezes config to config/frozen.json.
6. Caches test split (300 items).
7. Evaluates test split end-to-end (9 policies, 10,000 bootstrap CIs, paired diffs, sanity checks, plot).
8. Runs pytest verification suite.
Logs progress cleanly.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "pipeline.log"


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def wait_for_pid(pid: int):
    log(f"Waiting for process {pid} to complete...")
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(5)
    log(f"Process {pid} has completed.")


def run_command(cmd: list[str], step_name: str):
    log("=" * 70)
    log(f"STARTING: {step_name}")
    log(f"COMMAND: {' '.join(cmd)}")
    log("=" * 70)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        log(f"❌ FAILED: {step_name} exited with code {res.returncode} ({elapsed:.1f}s)")
        sys.exit(res.returncode)
    log(f"✓ COMPLETED: {step_name} in {elapsed:.1f}s")


def main():
    log("=====================================================================")
    log("AUTONOMOUS MASTER PIPELINE STARTED")
    log("=====================================================================")

    # 1. Wait for active PID if passed
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        wait_for_pid(int(sys.argv[1]))

    # 2. Ensure DEV caching is complete
    run_command([sys.executable, "scripts/cache_model_answers.py", "--split", "dev"],
                "Step 2A: Cache DEV Model Answers (100 items)")

    # 3. Cache SEEDS split (160 items)
    run_command([sys.executable, "scripts/cache_model_answers.py", "--split", "seeds"],
                "Step 2B: Cache SEEDS Model Answers (160 items)")

    # 4. Rebuild exemplar index from verified seeds
    run_command([sys.executable, "scripts/rebuild_index_from_seeds.py"],
                "Step 3: Rebuild Clean & Probed Exemplar Indices")

    # 5. Tune hyperparameters on DEV split and freeze
    run_command([sys.executable, "scripts/tune_dev.py"],
                "Step 4: Tune Hyperparameters on Dev Split & Freeze Config")

    # 6. Cache TEST split (300 items)
    run_command([sys.executable, "scripts/cache_model_answers.py", "--split", "test"],
                "Step 5A: Cache TEST Model Answers (300 items)")

    # 7. Evaluate on TEST split end-to-end
    run_command([sys.executable, "scripts/eval_test_end_to_end.py"],
                "Step 5B & 6: Evaluate on Test Split (9 Policies, Bootstrap CIs, Paired Diffs, Sanity Checks)")

    # 8. Full Pytest Suite
    run_command([sys.executable, "-m", "pytest", "tests/", "-v"],
                "Step 7: Complete Pytest Verification Suite")

    log("=====================================================================")
    log("🎉 ALL EVALUATION PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    log("=====================================================================")


if __name__ == "__main__":
    main()
