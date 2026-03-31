# scripts/Function_based/train_all.py
# Train all RL algorithms sequentially for fair comparison.
#
# Each algorithm runs in its own subprocess (Isaac Lab requires one
# AppLauncher per process). All runs share a timestamp so TensorBoard
# logs align for easy overlay comparison.
#
# Usage:
#   python train_all.py                          # train all 8 algorithms
#   python train_all.py --algorithms DQN PPO SAC # train only these
#   python train_all.py --num_envs 128           # override env count
#
# TensorBoard:
#   tensorboard --logdir=../../runs --port=6006

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train.py")
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

ALL_ALGORITHMS = [
    "Linear_Q",
    "DQN",
    "MC_REINFORCE",
    "AC",
    "A2C",
    "PPO",
    "SAC",
    "TD3",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def run_algorithm(algo: str, num_envs: int, tb_dir: str, log_file: str) -> tuple[bool, float]:
    """
    Launch train.py for one algorithm in a subprocess.
    Output goes to both console and log file.

    Returns:
        (success: bool, elapsed_seconds: float)
    """
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--algorithm", algo,
        "--num_envs", str(num_envs),
        "--headless",
        "--tb_log_dir", tb_dir,
    ]

    print(f"\n  Command: {' '.join(cmd)}\n")

    t0 = time.time()
    # Use PYTHONUNBUFFERED so tqdm lines are flushed immediately.
    # Pipe through 'tee' to get both console output and log file.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    elapsed = time.time() - t0

    return result.returncode == 0, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train all RL algorithms sequentially for fair comparison"
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help=f"Algorithms to train (default: all). Choices: {ALL_ALGORITHMS}",
    )
    parser.add_argument(
        "--num_envs", type=int, default=256,
        help="Number of parallel environments (default: 256)",
    )
    args = parser.parse_args()

    algorithms = args.algorithms if args.algorithms else ALL_ALGORITHMS

    # Validate names
    for algo in algorithms:
        if algo not in ALL_ALGORITHMS:
            print(f"ERROR: Unknown algorithm '{algo}'")
            print(f"Valid options: {ALL_ALGORITHMS}")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(PROJECT_ROOT, f"train_all_{timestamp}.log")

    # ---- Banner ----
    W = 70
    banner = (
        f"\n{'=' * W}\n"
        f"  TRAIN ALL ALGORITHMS  -  Fair Comparison\n"
        f"{'=' * W}\n"
        f"  Algorithms : {', '.join(algorithms)}\n"
        f"  Num Envs   : {args.num_envs}\n"
        f"  Timestamp  : {timestamp}\n"
        f"  Log File   : {log_file}\n"
        f"  TensorBoard: tensorboard --logdir={RUNS_DIR} --port=6006\n"
        f"{'=' * W}\n"
    )
    print(banner)
    with open(log_file, "w") as lf:
        lf.write(banner)

    total = len(algorithms)
    results = []
    t_start = time.time()

    for idx, algo in enumerate(algorithms, 1):
        tb_dir = os.path.join(RUNS_DIR, f"{algo}_{timestamp}")

        print(f"{'=' * W}")
        print(f"  [{idx}/{total}]  Training: {algo}")
        print(f"{'=' * W}")

        success, elapsed = run_algorithm(algo, args.num_envs, tb_dir, log_file)

        status = "PASS" if success else "FAIL"
        results.append((algo, status, elapsed))

        print(f"\n  >> {algo}  [{status}]  {fmt_duration(elapsed)}")

    total_elapsed = time.time() - t_start

    # ---- Summary ----
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = [a for a, s, _ in results if s == "FAIL"]

    summary_lines = [
        f"\n{'=' * W}",
        f"  TRAINING COMPLETE",
        f"{'=' * W}",
        f"  Total Time : {fmt_duration(total_elapsed)}",
        f"  Passed     : {passed} / {total}",
    ]
    if failed:
        summary_lines.append(f"  Failed     : {', '.join(failed)}")
    summary_lines.append("")
    for algo, status, elapsed in results:
        marker = "v" if status == "PASS" else "X"
        summary_lines.append(f"    [{marker}] {algo:<15s}  {fmt_duration(elapsed)}")
    summary_lines.extend([
        "",
        f"  Compare results:",
        f"    tensorboard --logdir={RUNS_DIR} --port=6006",
        f"{'=' * W}\n",
    ])

    summary = "\n".join(summary_lines)
    print(summary)
    with open(log_file, "a") as lf:
        lf.write(summary)
    print(f"  Log saved to: {log_file}")


if __name__ == "__main__":
    main()
