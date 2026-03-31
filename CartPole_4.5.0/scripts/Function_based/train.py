# scripts/Function_based/train.py
# HW3 Training Script - Function Approximation Algorithms
# Usage: python train.py --algorithm DQN --num_envs 256

import argparse
import os
import sys
import time
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train RL agent on CartPole")
parser.add_argument("--algorithm",     type=str,  default=None)
parser.add_argument("--task",          type=str,  default=None,
                    help="Task: Stabilize-Isaac-Cartpole-v0 or SwingUp-Isaac-Cartpole-v0")
parser.add_argument("--num_envs",      type=int,  default=256)
parser.add_argument("--headless",      action="store_true", default=True)
parser.add_argument("--load_model",    type=str,  default=None)
parser.add_argument("--save_interval", type=int,  default=100)
parser.add_argument("--tb_log_dir",    type=str,  default=None,
                    help="TensorBoard log dir (default: runs/<algo>_<timestamp>)")
args, remaining = parser.parse_known_args()

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from isaaclab_tasks.utils import parse_env_cfg
import CartPole.tasks  # noqa: F401
from config import get_config, create_agent, print_config, validate_config, ALGORITHM, NUM_ENVS

if args.algorithm:
    import config; config.ALGORITHM = args.algorithm
if args.task:
    import config; config.TASK = args.task
if args.num_envs:
    import config; config.NUM_ENVS  = args.num_envs


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard
# ─────────────────────────────────────────────────────────────────────────────

def make_writer(algo_name: str, log_dir) -> SummaryWriter:
    from datetime import datetime
    if log_dir is None:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(project_root, "runs", f"{algo_name}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    runs_dir = os.path.join(project_root, "runs")
    print(f"\n📊 TensorBoard → {log_dir}")
    print(f"   Open with : tensorboard --logdir={runs_dir} --port=6006\n")
    return SummaryWriter(log_dir)


def tb_log(writer, agent, step, avg_return, recent_avg, total_env_steps=None, extra=None):
    """Write standard scalars every episode/iteration."""
    writer.add_scalar("Return/episode",  avg_return,  step)
    writer.add_scalar("Return/avg_100",  recent_avg,  step)
    if hasattr(agent, 'epsilon'):
        writer.add_scalar("Explore/epsilon", agent.epsilon, step)

    # Log against total environment steps for fair cross-algorithm comparison
    if total_env_steps is not None:
        writer.add_scalar("Return/episode_vs_steps",  avg_return,  total_env_steps)
        writer.add_scalar("Return/avg_100_vs_steps",  recent_avg,  total_env_steps)

    # Per-episode stats populated by each algorithm's learn()
    stats = getattr(agent, '_episode_stats', {})
    writer.add_scalar("Rollout/ep_rew_mean", stats.get('ep_rew_mean', avg_return), step)
    writer.add_scalar("Rollout/ep_len_mean", stats.get('ep_len_mean', avg_return), step)
    if 'value_loss' in stats:
        writer.add_scalar("Loss/value_loss",           stats['value_loss'],           step)
    if 'policy_gradient_loss' in stats:
        writer.add_scalar("Loss/policy_gradient_loss", stats['policy_gradient_loss'], step)

    if extra:
        for tag, val in extra.items():
            if val is not None:
                writer.add_scalar(tag, float(val), step)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colour helpers
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m";  BOLD    = "\033[1m"
    RED     = "\033[91m"; YELLOW  = "\033[93m"
    GREEN   = "\033[92m"; CYAN    = "\033[96m"
    MAGENTA = "\033[95m"; GREY    = "\033[90m"

def status_dot(avg):
    if avg >= 475: return f"{C.GREEN}{C.BOLD}🟢 SOLVED{C.RESET}"
    if avg >= 350: return f"{C.YELLOW}{C.BOLD}🟠 GOOD  {C.RESET}"
    if avg >= 150: return f"{C.YELLOW}🟡 FAIR  {C.RESET}"
    return             f"{C.RED}🔴 POOR  {C.RESET}"

def print_summary(algo, returns, agent, config, start_time):
    """Print a detailed training summary after all loops complete."""
    elapsed   = (time.time() - start_time) / 60
    n         = len(returns)
    if n == 0:
        return

    ret = np.array(returns, dtype=np.float64)
    final_avg = float(np.mean(ret[-100:])) if n >= 100 else float(np.mean(ret))
    final_std = float(np.std(ret[-100:])) if n >= 100 else float(np.std(ret))
    best_ep   = float(np.max(ret))
    worst_ep  = float(np.min(ret))
    peak_idx  = int(np.argmax(ret)) + 1

    # Milestone: first episode where 100-ep avg >= 475
    solved_ep = None
    for i in range(99, n):
        if np.mean(ret[i-99:i+1]) >= 475:
            solved_ep = i + 1
            break

    # --- Training Performance Metrics ---
    # Convergence speed: first episode where 100-ep avg >= 400
    converge_ep = None
    for i in range(99, n):
        if np.mean(ret[i-99:i+1]) >= 400:
            converge_ep = i + 1
            break

    # Stability: rolling std (window=100) and coefficient of variation
    if n >= 100:
        rolling_stds = [float(np.std(ret[i-99:i+1])) for i in range(99, n)]
        avg_rolling_std = float(np.mean(rolling_stds))
        coeff_of_variation = final_std / max(abs(final_avg), 1e-8)
    else:
        avg_rolling_std = float(np.std(ret))
        coeff_of_variation = final_std / max(abs(final_avg), 1e-8)

    # Sample efficiency: AUC (area under learning curve) normalized by total steps
    auc = float(np.sum(ret))
    total_steps = n * config.get('max_steps', 0) * config['num_envs']
    if config.get('num_transitions_per_env'):
        total_steps = n * config['num_transitions_per_env'] * config['num_envs']
    normalized_auc = auc / max(total_steps, 1) * 1000  # per 1000 steps

    W = 80
    print(f"\n{C.BOLD}{'═'*W}{C.RESET}")
    print(f"  {'TRAINING SUMMARY':^{W-4}}")
    print(f"{C.BOLD}{'═'*W}{C.RESET}")
    print(f"  Algorithm       : {C.BOLD}{C.CYAN}{algo}{C.RESET}")
    print(f"  Task            : {config['task']}")
    print(f"  Device          : {config['device']}")
    print(f"  Parallel Envs   : {config['num_envs']}")
    print(f"{C.GREY}  {'─'*76}{C.RESET}")
    print(f"  Episodes Done   : {n:,}")
    print(f"  Total Steps     : ~{total_steps:,}")
    print(f"  Training Time   : {elapsed:.1f} min  ({elapsed/max(n,1)*60:.1f} sec/ep)")
    print(f"{C.GREY}  {'─'*76}{C.RESET}")
    print(f"  Final Avg-100   : {C.BOLD}{final_avg:7.2f}{C.RESET}  {status_dot(final_avg)}")
    print(f"  Final Std-100   : {final_std:7.2f}")
    print(f"  Best Episode    : {C.GREEN}{best_ep:7.2f}{C.RESET}  (ep {peak_idx})")
    print(f"  Worst Episode   : {C.RED}{worst_ep:7.2f}{C.RESET}")
    if solved_ep:
        print(f"  {C.GREEN}{C.BOLD}SOLVED at episode {solved_ep}{C.RESET}  (avg-100 >= 475)")
    else:
        remaining = 475 - final_avg
        print(f"  {C.YELLOW}Not solved{C.RESET}  (need +{remaining:.1f} more avg reward)")
    print(f"{C.GREY}  {'─'*76}{C.RESET}")
    print(f"  {C.BOLD}Training Performance{C.RESET}")
    print(f"  Convergence     : {'ep ' + str(converge_ep) if converge_ep else 'did not reach 400'}")
    print(f"  Stability (std) : {avg_rolling_std:.2f}  (rolling-100 avg)")
    print(f"  Coeff of Var    : {coeff_of_variation:.3f}")
    print(f"  Sample Eff (AUC): {normalized_auc:.4f}  (per 1000 env steps)")
    print(f"{C.GREY}  {'─'*76}{C.RESET}")
    if hasattr(agent, 'epsilon'):
        print(f"  Final Epsilon   : {agent.epsilon:.4f}")
    if hasattr(agent, '_episode_stats'):
        s = agent._episode_stats
        if 'value_loss' in s and s['value_loss'] > 0:
            print(f"  Last Value Loss : {s['value_loss']:.5f}")
        if 'policy_gradient_loss' in s and s['policy_gradient_loss'] != 0:
            print(f"  Last PG Loss    : {s['policy_gradient_loss']:.5f}")
    print(f"{C.GREY}  {'─'*76}{C.RESET}")
    model_dir = os.path.join(config['model_dir'], algo)
    print(f"  Model saved to  : {model_dir}/{algo}_final.pth")
    print(f"  TensorBoard     : tensorboard --logdir={os.path.join(project_root, 'runs')} --port=6006")
    print(f"{C.BOLD}{'═'*W}{C.RESET}\n")


def print_footer(algo, n_done, elapsed_min, final_avg, unit="Episodes"):
    print(f"\n{C.BOLD}{'─'*80}{C.RESET}")
    print(f"  ✅ Training Complete — {C.BOLD}{algo}{C.RESET}")
    print(f"  Total {unit}  : {n_done}")
    print(f"  Total Time   : {elapsed_min:.1f} min")
    print(f"  Final Avg-100: {C.CYAN}{C.BOLD}{final_avg:.2f}{C.RESET}  {status_dot(final_avg)}")
    print(f"{C.BOLD}{'─'*80}{C.RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training loops
# ─────────────────────────────────────────────────────────────────────────────

def train_off_policy(agent, env, config, writer):
    algo   = config['algorithm_name']
    n_ep   = config['n_episodes']
    m_step = config['max_steps']
    n_envs = config['num_envs']
    log_iv = config['log_interval']
    sav_iv = config['save_interval']

    total_returns = []
    start_time    = time.time()
    solved        = False
    best_avg      = -float('inf')
    total_env_steps = 0

    pbar = tqdm(range(n_ep), desc=f"{C.BOLD}{algo}{C.RESET}", unit="ep",
                dynamic_ncols=True, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for episode in pbar:
        avg_return, steps = agent.learn(env, max_steps=m_step)
        total_returns.append(avg_return)
        total_env_steps += m_step * n_envs

        recent_avg = (np.mean(total_returns[-100:]) if len(total_returns) >= 100
                      else np.mean(total_returns))
        elapsed    = (time.time() - start_time) / 60

        # Save best model
        if recent_avg > best_avg:
            best_avg = recent_avg
            agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_best.pth")

        # TensorBoard
        tb_log(writer, agent, episode + 1, avg_return, recent_avg, total_env_steps=total_env_steps)
        writer.add_scalar("Time/elapsed_min", elapsed, episode + 1)

        # tqdm postfix
        pbar.set_postfix({
            "ret"   : f"{avg_return:6.1f}",
            "avg100": f"{recent_avg:6.1f}",
            "ε"     : f"{agent.epsilon:.4f}" if hasattr(agent, 'epsilon') else "—",
            "status": ("🟢" if recent_avg >= 475 else
                       "🟠" if recent_avg >= 350 else
                       "🟡" if recent_avg >= 150 else "🔴"),
        }, refresh=True)

        # Periodic terminal row
        if (episode + 1) % log_iv == 0:
            eps_str = f"ε={agent.epsilon:.4f}  " if hasattr(agent, 'epsilon') else ""
            tqdm.write(f"  Ep {episode+1:>5}/{n_ep}  ret={avg_return:6.1f}  "
                       f"avg100={recent_avg:6.1f}  {eps_str}{status_dot(recent_avg)}  {elapsed:.1f}min")

        # Checkpoint
        if (episode + 1) % sav_iv == 0:
            save_path = os.path.join(config['model_dir'], algo)
            agent.save_model(save_path, f"{algo}_ep{episode+1}.pth")
            tqdm.write(f"  {C.MAGENTA}💾 Checkpoint → {algo}_ep{episode+1}.pth{C.RESET}")

        # Solved?
        if len(total_returns) >= 100 and recent_avg >= 475 and not solved:
            solved = True
            writer.add_scalar("Event/solved_episode", episode + 1, episode + 1)
            tqdm.write(f"\n{C.GREEN}{C.BOLD}  🎉 SOLVED at episode {episode+1}! "
                       f"Avg-100={recent_avg:.2f}  ({elapsed:.1f}min){C.RESET}\n")
            break

    pbar.close()
    agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_final.pth")
    tqdm.write(f"  {C.CYAN}⭐ Best avg-100: {best_avg:.2f} → {algo}_best.pth{C.RESET}")

    final_avg  = np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns)
    print_footer(algo, len(total_returns), (time.time()-start_time)/60, final_avg)
    return total_returns


def train_on_policy_rollout(agent, env, config, writer):
    algo    = config['algorithm_name']
    max_it  = config['max_iterations']
    n_envs  = config['num_envs']
    n_trans = config['num_transitions_per_env']
    log_iv  = config['log_interval']
    sav_iv  = config['save_interval']

    total_returns = []
    start_time    = time.time()
    solved        = False
    best_avg      = -float('inf')
    total_env_steps = 0

    pbar = tqdm(range(max_it), desc=f"{C.BOLD}{algo}{C.RESET}", unit="iter",
                dynamic_ncols=True, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for iteration in pbar:
        avg_return, _ = agent.learn(env=env, num_envs=n_envs, num_transitions_per_env=n_trans)
        total_returns.append(avg_return)
        total_env_steps += n_trans * n_envs

        recent_avg = (np.mean(total_returns[-100:]) if len(total_returns) >= 100
                      else np.mean(total_returns))
        elapsed    = (time.time() - start_time) / 60

        # Save best model
        if recent_avg > best_avg:
            best_avg = recent_avg
            agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_best.pth")

        # TensorBoard — live per iteration
        tb_log(writer, agent, iteration + 1, avg_return, recent_avg, total_env_steps=total_env_steps)
        writer.add_scalar("Time/elapsed_min", elapsed, iteration + 1)

        pbar.set_postfix({
            "ret"   : f"{avg_return:6.1f}",
            "avg100": f"{recent_avg:6.1f}",
            "status": ("🟢" if recent_avg >= 475 else
                       "🟠" if recent_avg >= 350 else
                       "🟡" if recent_avg >= 150 else "🔴"),
        }, refresh=True)

        if (iteration + 1) % log_iv == 0:
            tqdm.write(f"  It {iteration+1:>5}/{max_it}  ret={avg_return:6.1f}  "
                       f"avg100={recent_avg:6.1f}  {status_dot(recent_avg)}  {elapsed:.1f}min")

        if (iteration + 1) % sav_iv == 0:
            save_path = os.path.join(config['model_dir'], algo)
            agent.save_model(save_path, f"{algo}_iter{iteration+1}.pth")
            tqdm.write(f"  {C.MAGENTA}💾 Checkpoint → {algo}_iter{iteration+1}.pth{C.RESET}")

        if len(total_returns) >= 100 and recent_avg >= 475 and not solved:
            solved = True
            writer.add_scalar("Event/solved_iteration", iteration + 1, iteration + 1)
            tqdm.write(f"\n{C.GREEN}{C.BOLD}  🎉 SOLVED at iteration {iteration+1}! "
                       f"Avg-100={recent_avg:.2f}  ({elapsed:.1f}min){C.RESET}\n")
            break

    pbar.close()
    agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_final.pth")
    tqdm.write(f"  {C.CYAN}⭐ Best avg-100: {best_avg:.2f} → {algo}_best.pth{C.RESET}")

    final_avg = (np.mean(total_returns[-100:]) if len(total_returns) >= 100
                 else np.mean(total_returns) if total_returns else 0.0)
    print_footer(algo, len(total_returns), (time.time()-start_time)/60, final_avg, "Iterations")
    return total_returns


def train_reinforce(agent, env, config, writer):
    algo   = config['algorithm_name']
    n_it   = config['n_iterations']
    m_step = config['max_steps']
    n_envs = config['num_envs']
    log_iv = config['log_interval']
    sav_iv = config['save_interval']

    total_returns = []
    start_time    = time.time()
    solved        = False
    best_avg      = -float('inf')
    total_env_steps = 0

    pbar = tqdm(range(n_it), desc=f"{C.BOLD}{algo}{C.RESET}", unit="iter",
                dynamic_ncols=True, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for iteration in pbar:
        avg_return, loss, num_episodes = agent.learn(env=env, max_steps=m_step, num_agents=n_envs)
        if num_episodes > 0:
            total_returns.append(avg_return)
        elif total_returns:
            avg_return = total_returns[-1]  # show last known return, not 0.0
        total_env_steps += m_step * n_envs

        recent_avg = (np.mean(total_returns[-100:]) if len(total_returns) >= 100
                      else np.mean(total_returns) if total_returns else 0.0)
        elapsed    = (time.time() - start_time) / 60

        # Save best model
        if recent_avg > best_avg:
            best_avg = recent_avg
            agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_best.pth")

        # TensorBoard
        tb_log(writer, agent, iteration + 1, avg_return, recent_avg,
               total_env_steps=total_env_steps,
               extra={"Loss/policy_gradient_loss": loss, "Misc/episodes_per_iter": num_episodes})
        writer.add_scalar("Time/elapsed_min", elapsed, iteration + 1)

        pbar.set_postfix({
            "ret"   : f"{avg_return:6.1f}",
            "avg100": f"{recent_avg:6.1f}",
            "loss"  : f"{loss:.4f}",
            "eps"   : num_episodes,
            "status": ("🟢" if recent_avg >= 475 else
                       "🟠" if recent_avg >= 350 else
                       "🟡" if recent_avg >= 150 else "🔴"),
        }, refresh=True)

        if (iteration + 1) % log_iv == 0:
            tqdm.write(f"  It {iteration+1:>5}/{n_it}  ret={avg_return:6.1f}  "
                       f"avg100={recent_avg:6.1f}  loss={loss:.4f}  "
                       f"{status_dot(recent_avg)}  {elapsed:.1f}min")

        if (iteration + 1) % sav_iv == 0:
            save_path = os.path.join(config['model_dir'], algo)
            agent.save_model(save_path, f"{algo}_iter{iteration+1}.pth")
            tqdm.write(f"  {C.MAGENTA}💾 Checkpoint → {algo}_iter{iteration+1}.pth{C.RESET}")

        if len(total_returns) >= 100 and recent_avg >= 475 and not solved:
            solved = True
            writer.add_scalar("Event/solved_iteration", iteration + 1, iteration + 1)
            tqdm.write(f"\n{C.GREEN}{C.BOLD}  🎉 SOLVED at iteration {iteration+1}! "
                       f"Avg-100={recent_avg:.2f}  ({elapsed:.1f}min){C.RESET}\n")
            break

    pbar.close()
    agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_final.pth")
    tqdm.write(f"  {C.CYAN}⭐ Best avg-100: {best_avg:.2f} → {algo}_best.pth{C.RESET}")

    final_avg  = (np.mean(total_returns[-100:]) if len(total_returns) >= 100
                  else np.mean(total_returns) if total_returns else 0.0)
    print_footer(algo, len(total_returns), (time.time()-start_time)/60, final_avg, "Iterations")
    return total_returns


def train_linear_q(agent, env, config, writer):
    algo   = config['algorithm_name']
    n_ep   = config['n_episodes']
    m_step = config['max_steps']
    n_envs = config['num_envs']
    log_iv = config['log_interval']
    sav_iv = config['save_interval']

    total_returns = []
    start_time    = time.time()
    solved        = False
    best_avg      = -float('inf')
    total_env_steps = 0

    pbar = tqdm(range(n_ep), desc=f"{C.BOLD}{algo}{C.RESET}", unit="ep",
                dynamic_ncols=True, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for episode in pbar:
        avg_return, steps = agent.learn(env, max_steps=m_step)
        total_returns.append(avg_return)
        total_env_steps += steps  # Linear_Q uses only env[0], so steps = actual steps taken

        recent_avg = (np.mean(total_returns[-100:]) if len(total_returns) >= 100
                      else np.mean(total_returns))
        elapsed    = (time.time() - start_time) / 60

        # Save best model
        if recent_avg > best_avg:
            best_avg = recent_avg
            agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_best.npy")

        # TensorBoard
        tb_log(writer, agent, episode + 1, avg_return, recent_avg, total_env_steps=total_env_steps)
        writer.add_scalar("Time/elapsed_min", elapsed, episode + 1)

        pbar.set_postfix({
            "ret"   : f"{avg_return:6.1f}",
            "avg100": f"{recent_avg:6.1f}",
            "ε"     : f"{agent.epsilon:.4f}",
            "status": ("🟢" if recent_avg >= 475 else
                       "🟠" if recent_avg >= 350 else
                       "🟡" if recent_avg >= 150 else "🔴"),
        }, refresh=True)

        if (episode + 1) % log_iv == 0:
            tqdm.write(f"  Ep {episode+1:>5}/{n_ep}  ret={avg_return:6.1f}  "
                       f"avg100={recent_avg:6.1f}  ε={agent.epsilon:.4f}  "
                       f"{status_dot(recent_avg)}  {elapsed:.1f}min")

        if (episode + 1) % sav_iv == 0:
            save_path = os.path.join(config['model_dir'], algo)
            agent.save_model(save_path, f"{algo}_ep{episode+1}.npy")
            tqdm.write(f"  {C.MAGENTA}💾 Checkpoint → {algo}_ep{episode+1}.npy{C.RESET}")

        if len(total_returns) >= 100 and recent_avg >= 475 and not solved:
            solved = True
            writer.add_scalar("Event/solved_episode", episode + 1, episode + 1)
            tqdm.write(f"\n{C.GREEN}{C.BOLD}  🎉 SOLVED at episode {episode+1}! "
                       f"Avg-100={recent_avg:.2f}  ({elapsed:.1f}min){C.RESET}\n")
            break

    pbar.close()
    agent.save_model(os.path.join(config['model_dir'], algo), f"{algo}_final.npy")
    tqdm.write(f"  {C.CYAN}⭐ Best avg-100: {best_avg:.2f} → {algo}_best.npy{C.RESET}")

    final_avg  = np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns)
    print_footer(algo, len(total_returns), (time.time()-start_time)/60, final_avg)
    return total_returns


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_config()
    if not validate_config():
        print("❌ Configuration validation failed. Exiting.")
        return

    config    = get_config()
    algo_name = config['algorithm_name']

    # ── Fix all random seeds for reproducibility ─────────────────────── #
    import random
    SEED = 42
    random.seed(SEED)                          # Python stdlib random
    np.random.seed(SEED)                       # NumPy random
    torch.manual_seed(SEED)                    # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)           # PyTorch current GPU
        torch.cuda.manual_seed_all(SEED)       # PyTorch all GPUs
    torch.backends.cudnn.deterministic = True  # deterministic convolutions
    torch.backends.cudnn.benchmark = False     # disable auto-tuner (non-deterministic)
    print(f"  Random seed: {SEED}")

    print(f"\n🌍 Creating environment: {config['task']}")
    env_cfg = parse_env_cfg(config['task'], device=str(config['device']),
                            num_envs=config['num_envs'])
    env_cfg.seed = SEED                        # Isaac Lab environment seed
    env = gym.make(config['task'], cfg=env_cfg)
    print(f"✅ Environment created successfully")

    print(f"\n🤖 Creating agent: {algo_name}")
    agent = create_agent()

    if args.load_model:
        print(f"\n📂 Loading model from: {args.load_model}")
        agent.load_model(os.path.dirname(args.load_model),
                         os.path.basename(args.load_model))

    # TensorBoard writer
    writer = make_writer(algo_name, args.tb_log_dir)

    # Log hyperparameters as readable text in TB
    hparam_lines = "\n".join(f"    {k}: {v}" for k, v in config.items()
                              if not callable(v))
    writer.add_text("Hyperparameters", f"```\n{hparam_lines}\n```", 0)

    print(f"\n🚀 Starting training...\n")
    train_start = time.time()

    if algo_name == 'Linear_Q':
        returns = train_linear_q(agent, env, config, writer)
    elif algo_name in ['DQN', 'SAC', 'TD3']:
        returns = train_off_policy(agent, env, config, writer)
    elif algo_name in ['AC', 'A2C', 'PPO']:
        returns = train_on_policy_rollout(agent, env, config, writer)
    elif algo_name == 'MC_REINFORCE':
        returns = train_reinforce(agent, env, config, writer)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    print_summary(algo_name, returns, agent, config, train_start)

    writer.flush()
    writer.close()
    print(f"📊 TensorBoard data saved.  Run:  tensorboard --logdir={os.path.join(project_root, 'runs')} --port=6006")

    try:
        agent.plot_durations(show_result=True)
    except Exception as e:
        print(f"⚠️  Could not display plot: {e}")

    env.close()
    print("\n✅ Training session complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()