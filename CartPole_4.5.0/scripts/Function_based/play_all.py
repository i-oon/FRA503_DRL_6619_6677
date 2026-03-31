# scripts/Function_based/play_all.py
# Evaluate all trained algorithms and produce a comparison summary.
#
# Prefers _best.pth models (peak performance), falls back to _final.pth.
# Each algorithm runs in its own subprocess (Isaac Lab requires one
# AppLauncher per process).
#
# Usage:
#   python play_all.py                             # evaluate all 8 algorithms
#   python play_all.py --algorithms DQN PPO SAC    # evaluate only these
#   python play_all.py --n_episodes 20             # more episodes per algo
#   python play_all.py --use_final                 # force _final instead of _best

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAY_SCRIPT = os.path.join(SCRIPT_DIR, "play.py")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

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

# Model file extensions per algorithm
MODEL_EXT = {
    "Linear_Q": ".npy",
}
DEFAULT_EXT = ".pth"


def get_model_path(algo: str, use_final: bool = False) -> str | None:
    """Find the best or final model for an algorithm."""
    ext = MODEL_EXT.get(algo, DEFAULT_EXT)
    algo_dir = os.path.join(MODEL_DIR, algo)

    if not use_final:
        best = os.path.join(algo_dir, f"{algo}_best{ext}")
        if os.path.exists(best):
            return best

    final = os.path.join(algo_dir, f"{algo}_final{ext}")
    if os.path.exists(final):
        return final

    return None


def parse_play_output(output: str) -> dict:
    """Extract metrics from play.py output."""
    stats = {}

    m = re.search(r"Mean Return\s*:\s*([\d.]+)\s*\+/-\s*([\d.]+)", output)
    if m:
        stats["mean_return"] = float(m.group(1))
        stats["std_return"] = float(m.group(2))

    m = re.search(r"Median Return\s*:\s*([\d.]+)", output)
    if m:
        stats["median_return"] = float(m.group(1))

    m = re.search(r"Min / Max\s*:\s*([\d.]+)\s*/\s*([\d.]+)", output)
    if m:
        stats["min_return"] = float(m.group(1))
        stats["max_return"] = float(m.group(2))

    m = re.search(r"Mean Length\s*:\s*([\d.]+)\s*\+/-\s*([\d.]+)", output)
    if m:
        stats["mean_length"] = float(m.group(1))
        stats["std_length"] = float(m.group(2))

    m = re.search(r"Success Rate\s*:\s*(\d+)/(\d+)\s*\((\d+)%\)", output)
    if m:
        stats["success_count"] = int(m.group(1))
        stats["n_episodes"] = int(m.group(2))
        stats["success_rate"] = float(m.group(3))

    m = re.search(r"Max Length Rate\s*:\s*(\d+)/(\d+)\s*\((\d+)%\)", output)
    if m:
        stats["max_length_count"] = int(m.group(1))
        stats["max_length_rate"] = float(m.group(3))

    return stats


def run_play(algo: str, model_path: str, n_episodes: int) -> tuple[dict, bool]:
    """Run play.py for one algorithm and parse results."""
    cmd = [
        sys.executable, PLAY_SCRIPT,
        "--algorithm", algo,
        "--model_path", model_path,
        "--n_episodes", str(n_episodes),
        "--headless",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd, cwd=SCRIPT_DIR,
        capture_output=True, text=True, env=env,
    )

    # Print stdout live
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        # Only print stderr on failure (skip Isaac Lab warnings)
        for line in result.stderr.splitlines():
            if "Error" in line or "Traceback" in line:
                print(f"  STDERR: {line}")

    stats = parse_play_output(result.stdout or "")
    success = result.returncode == 0 and "mean_return" in stats

    return stats, success


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained RL algorithms and produce a comparison summary"
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help=f"Algorithms to evaluate (default: all). Choices: {ALL_ALGORITHMS}",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10,
        help="Number of evaluation episodes per algorithm (default: 10)",
    )
    parser.add_argument(
        "--use_final", action="store_true", default=False,
        help="Use _final model instead of _best (default: prefer _best)",
    )
    args = parser.parse_args()

    algorithms = args.algorithms if args.algorithms else ALL_ALGORITHMS

    # Validate
    for algo in algorithms:
        if algo not in ALL_ALGORITHMS:
            print(f"ERROR: Unknown algorithm '{algo}'")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "final" if args.use_final else "best"

    W = 70
    print(f"\n{'=' * W}")
    print(f"  PLAY ALL ALGORITHMS  -  Comparison Evaluation")
    print(f"{'=' * W}")
    print(f"  Algorithms : {', '.join(algorithms)}")
    print(f"  Episodes   : {args.n_episodes} per algorithm")
    print(f"  Model pref : _{model_type}")
    print(f"{'=' * W}\n")

    all_results = {}
    total = len(algorithms)
    t_start = time.time()

    for idx, algo in enumerate(algorithms, 1):
        model_path = get_model_path(algo, args.use_final)

        print(f"{'=' * W}")
        print(f"  [{idx}/{total}]  Evaluating: {algo}")

        if model_path is None:
            print(f"  SKIPPED — no trained model found in {MODEL_DIR}/{algo}/")
            print(f"{'=' * W}\n")
            all_results[algo] = {"status": "SKIP", "reason": "no model"}
            continue

        which = "best" if "_best" in model_path else "final"
        print(f"  Model: {model_path}  ({which})")
        print(f"{'=' * W}")

        t0 = time.time()
        stats, success = run_play(algo, model_path, args.n_episodes)
        elapsed = time.time() - t0

        if success:
            stats["status"] = "OK"
            stats["model"] = which
            stats["time"] = elapsed
        else:
            stats = {"status": "FAIL", "model": which, "time": elapsed}

        all_results[algo] = stats

    total_elapsed = time.time() - t_start

    # ---- Comparison Table ----
    print(f"\n{'=' * W}")
    print(f"{'COMPARISON SUMMARY':^{W}}")
    print(f"{'=' * W}")
    print(f"  {'Algorithm':<15s} {'Return':>14s} {'Length':>8s} {'Success':>9s} {'Model':>7s}")
    print(f"  {'-'*15} {'-'*14} {'-'*8} {'-'*9} {'-'*7}")

    for algo in algorithms:
        r = all_results.get(algo, {})
        if r.get("status") == "OK":
            ret_str = f"{r['mean_return']:.1f}+/-{r['std_return']:.1f}"
            len_str = f"{r['mean_length']:.0f}"
            suc_str = f"{r.get('success_rate', 0):.0f}%"
            mdl_str = r.get("model", "?")
        elif r.get("status") == "SKIP":
            ret_str = "---"
            len_str = "---"
            suc_str = "---"
            mdl_str = "none"
        else:
            ret_str = "FAIL"
            len_str = "---"
            suc_str = "---"
            mdl_str = r.get("model", "?")

        print(f"  {algo:<15s} {ret_str:>14s} {len_str:>8s} {suc_str:>9s} {mdl_str:>7s}")

    print(f"\n  Total Time: {fmt_duration(total_elapsed)}")
    print(f"{'=' * W}\n")

    # ---- Save results to file ----
    log_dir = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "runs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"play_all_{timestamp}.txt")

    with open(log_file, "w") as f:
        f.write(f"Play All Results - {timestamp}\n")
        f.write(f"Episodes per algorithm: {args.n_episodes}\n")
        f.write(f"Model preference: _{model_type}\n\n")
        f.write(f"{'Algorithm':<15s} {'Mean Return':>12s} {'Std':>8s} {'Length':>8s} {'Success%':>10s} {'Model':>7s}\n")
        f.write(f"{'-'*60}\n")
        for algo in algorithms:
            r = all_results.get(algo, {})
            if r.get("status") == "OK":
                f.write(f"{algo:<15s} {r['mean_return']:>12.2f} {r['std_return']:>8.2f} "
                        f"{r['mean_length']:>8.1f} {r.get('success_rate', 0):>9.0f}% "
                        f"{r.get('model', '?'):>7s}\n")
            elif r.get("status") == "SKIP":
                f.write(f"{algo:<15s} {'SKIPPED':>12s}\n")
            else:
                f.write(f"{algo:<15s} {'FAILED':>12s}\n")

    print(f"  Results saved to: {log_file}")


if __name__ == "__main__":
    main()
