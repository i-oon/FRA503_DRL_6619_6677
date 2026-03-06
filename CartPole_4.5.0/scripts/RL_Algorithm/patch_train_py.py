"""Script to train RL agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Add config directory to path
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

# Import config FIRST
from config import get_config, print_config, create_agent

from RL_Algorithm.RL_base import ControlType
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random
import numpy as np
import csv
from collections import deque

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def save_metrics_to_csv(csv_writer, episode, reward, length, epsilon, reward_window, length_window, timestamp):
    """Save episode metrics to CSV file."""
    avg_reward_100 = np.mean(reward_window) if len(reward_window) > 0 else 0.0
    avg_length_100 = np.mean(length_window) if len(length_window) > 0 else 0.0
    csv_writer.writerow([
        episode,
        f"{reward:.4f}",
        length,
        f"{epsilon:.6f}",
        f"{avg_reward_100:.4f}",
        f"{avg_length_100:.4f}",
        timestamp
    ])


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RL agent."""
    
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ==================== GET CONFIGURATION FROM config.py ============== #
    # ==================================================================== #
    
    # Get ALL parameters from config.py (NO hardcoded values!)
    config = get_config()  # Uses global ALGORITHM setting
    print_config()         # Print configuration
    
    # ===== MODIFICATION 1: Get experiment suffix ===== #
    # Get experiment suffix for separate file saving (experimental mode)
    experiment_suffix = config.get('experiment_suffix', '')
    if experiment_suffix:
        print(f"🔬 Experimental Mode: {experiment_suffix}")
    # ================================================= #
    
    # Extract task name from args
    task_name = str(args_cli.task).split('-')[0]
    
    # Get parameters
    Algorithm_name = config['algorithm_name']
    n_episodes = config['n_episodes']
    discretize_state_weight = config['discretize_state_weight']
    num_of_action = config['num_of_action']
    action_range = config['action_range']
    
    # Create agent using config (automatically selects correct class!)
    agent = create_agent(testing=False)
    
    print(f"Created {Algorithm_name} agent")
    print(f"Epsilon: {agent.epsilon}")
    print(f" Q-table size: {len(agent.q_values)}\n")
    
    # ==================================================================== #
    # ==================== Directory Setup =============================== #
    # ==================================================================== #
    
    script_dir = os.path.dirname(__file__)
    project_root = os.path.join(script_dir, "..", "..")

    # ===== MODIFICATION 2 & 3: Add experiment suffix to directory names ===== #
    # Add experiment suffix to directory name for separate file storage
    if experiment_suffix:
        logs_dir = os.path.join(project_root, "logs", task_name, f"{Algorithm_name}{experiment_suffix}")
        q_value_dir = os.path.join(project_root, "q_value", task_name, f"{Algorithm_name}{experiment_suffix}")
    else:
        logs_dir = os.path.join(project_root, "logs", task_name, Algorithm_name)
        q_value_dir = os.path.join(project_root, "q_value", task_name, Algorithm_name)
    # ======================================================================= #

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(q_value_dir, exist_ok=True)

    print(f"Training: {Algorithm_name} on {task_name} task")
    print(f"Logs: {logs_dir}")
    print(f"Q-values: {q_value_dir}\n")

    # ==================================================================== #
    # ======================= CSV File Setup ============================= #
    # ==================================================================== #
    
    csv_filename = os.path.join(logs_dir, "training_metrics.csv")
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'length', 'epsilon', 'avg_reward_100', 'avg_length_100', 'timestamp'])

    reward_window = deque(maxlen=100)
    length_window = deque(maxlen=100)

    # ==================================================================== #
    # ========================= Vectorized Training Loop ================= #
    # ==================================================================== #

    obs, _ = env.reset()
    timestep = 0
    
    episode_rewards = []  # Track all episode rewards
    episode_lengths = []  # Track all episode lengths

    # Arrays to track cumulative rewards and lengths for all environments
    env_cumulative_rewards = np.zeros(args_cli.num_envs)
    env_episode_lengths = np.zeros(args_cli.num_envs)

    best_avg_reward = -float('inf')
    best_episode = 0
    patience = 15000
    episodes_without_improvement = 0
    episodes_completed = 0

    pbar = tqdm(total=n_episodes, desc=f"Training {Algorithm_name}")
    action_tensor, action_idx = agent.get_action(obs)

   # simulate environment
    while simulation_app.is_running() and episodes_completed < n_episodes:
        with torch.inference_mode():
            
            # Step ALL environments
            next_obs, rewards, terminated, truncated, _ = env.step(action_tensor)

            # Move tensors to CPU numpy arrays for Q-table operations
            rewards_np = rewards.cpu().numpy()
            terminated_np = terminated.cpu().numpy()
            truncated_np = truncated.cpu().numpy()
            dones_np = terminated_np | truncated_np

            env_cumulative_rewards += rewards_np
            env_episode_lengths += 1

            # Get next actions (but may be overridden by MC)
            next_action_tensor, next_action_idx = agent.get_action(next_obs)

            # --- ALGORITHM BATCH UPDATES ---
            if agent.control_type == ControlType.SARSA:
                agent.update_batch(
                    obs=obs, action=action_idx, reward=rewards_np, 
                    terminated=terminated_np, next_obs=next_obs, next_action=next_action_idx
                )
                # SARSA uses the computed next_action
                action_tensor = next_action_tensor
                action_idx = next_action_idx
                
            elif agent.control_type in [ControlType.Q_LEARNING, ControlType.DOUBLE_Q_LEARNING]:
                agent.update_batch(
                    obs=obs, action=action_idx, reward=rewards_np, 
                    terminated=terminated_np, next_obs=next_obs
                )
                # Q-Learning uses the computed next_action
                action_tensor = next_action_tensor
                action_idx = next_action_idx
                
            elif agent.control_type == ControlType.MONTE_CARLO:
                agent.update_batch(
                    obs=obs,
                    action=action_idx,
                    reward=rewards_np,
                    terminated=terminated_np,
                    next_obs=next_obs
                )
                # MC gets its own next action (DON'T use next_action_tensor!)
                action_tensor, action_idx = agent.get_action(next_obs)

            # Advance state for the next loop
            obs = next_obs

            # Check if any environments just finished an episode
            for i in range(args_cli.num_envs):
                if dones_np[i]:
                    episodes_completed += 1
                    pbar.update(1)

                    cumulative_reward = env_cumulative_rewards[i]
                    episode_length = env_episode_lengths[i]

                    episode_rewards.append(cumulative_reward)
                    episode_lengths.append(episode_length)
                    reward_window.append(cumulative_reward)
                    length_window.append(episode_length)

                    # Reset tracking arrays for this specific environment
                    env_cumulative_rewards[i] = 0
                    env_episode_lengths[i] = 0

                    agent.decay_epsilon()
                    # Save metrics and Q-table every 100 total episodes across all envs
                    if episodes_completed % 100 == 0:
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        save_metrics_to_csv(
                            csv_writer, episodes_completed, cumulative_reward, episode_length,
                            agent.epsilon, reward_window, length_window, timestamp
                        )
                        csv_file.flush()

                        q_value_file = f"{Algorithm_name}_{episodes_completed}_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                        agent.save_q_value(q_value_dir, q_value_file)

                        # Check for improvements
                        if len(episode_rewards) >= 100:
                            avg_reward = np.mean(episode_rewards[-100:])
                            
                            if avg_reward > best_avg_reward:
                                best_avg_reward = avg_reward
                                best_episode = episodes_completed
                                episodes_without_improvement = 0
                                
                                best_q_filename = f"{Algorithm_name}_BEST_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                                agent.save_q_value(q_value_dir, best_q_filename)
                                print(f"\n💾 NEW BEST at episode {episodes_completed}: {avg_reward:.2f} reward")
                            else:
                                # ✅ FIX: Only increment when we actually checked (every 100 episodes)
                                episodes_without_improvement += 100

                        # Early stopping trigger
                        if episodes_without_improvement >= patience:
                            break
            
            # Break completely if early stopping was triggered
            if episodes_without_improvement >= patience:
                print(f"\n🛑 EARLY STOPPING at episode {episodes_completed}")
                print(f"   Best performance: {best_avg_reward:.2f} at episode {best_episode}")
                print(f"   Final Q-table saved as: {best_q_filename}")
                break

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
    print("\n!!! Training is complete !!!")
    print(f"CSV saved to: {csv_filename}")

    csv_file.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()