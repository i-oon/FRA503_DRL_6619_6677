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

    logs_dir = os.path.join(project_root, "logs", task_name, Algorithm_name)
    q_value_dir = os.path.join(project_root, "q_value", task_name, Algorithm_name)

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
    # ========================= Training Loop ============================ #
    # ==================================================================== #

    obs, _ = env.reset()
    timestep = 0
    
    episode_rewards = []  # Track all episode rewards
    episode_lengths = []  # Track all episode lengths

    # Initialize early stopping variables BEFORE the loop
    best_avg_reward = -float('inf')
    best_episode = 0
    patience = 500
    episodes_without_improvement = 0

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes), desc=f"Training {Algorithm_name}"):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0
                episode_length = 0

                action_tensor, action_idx = agent.get_action(obs)

                while not done:
                    next_obs, reward, terminated, truncated, _ = env.step(action_tensor)

                    reward_value = reward.item()
                    terminated_value = terminated.item() 
                    cumulative_reward += reward_value
                    episode_length += 1

                    # Algorithm-specific update logic
                    if agent.control_type == ControlType.SARSA:
                        next_action_tensor, next_action_idx = agent.get_action(next_obs)
                        agent.update(
                            obs=obs,
                            action=action_idx,
                            reward=reward_value,
                            terminated=terminated_value,
                            next_obs=next_obs,
                            next_action=next_action_idx
                        )
                        action_tensor = next_action_tensor
                        action_idx = next_action_idx

                    elif agent.control_type in [ControlType.Q_LEARNING, ControlType.DOUBLE_Q_LEARNING]:
                        agent.update(
                            obs=obs,
                            action=action_idx,
                            reward=reward_value,
                            terminated=terminated_value,
                            next_obs=next_obs
                        )
                        action_tensor, action_idx = agent.get_action(next_obs)

                    elif agent.control_type == ControlType.MONTE_CARLO:
                        agent.obs_hist.append(obs)
                        agent.action_hist.append(action_idx)
                        agent.reward_hist.append(reward_value)
                        action_tensor, action_idx = agent.get_action(next_obs)

                    done = terminated or truncated
                    obs = next_obs

                # Episode finished - episode_length = total timesteps in this episode
                print(f"Episode {episode}: {episode_length} timesteps, reward = {cumulative_reward}")

                episode_rewards.append(cumulative_reward)
                episode_lengths.append(episode_length)

                # End of episode updates
                if agent.control_type == ControlType.MONTE_CARLO:
                    agent.update()

                agent.decay_epsilon()
                
                # CSV Logging
                reward_window.append(cumulative_reward)
                length_window.append(episode_length)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_metrics_to_csv(
                    csv_writer,
                    episode,
                    cumulative_reward,
                    episode_length,
                    agent.epsilon,
                    reward_window,
                    length_window,
                    timestamp
                )
                
                best_avg_reward = -float('inf')
                best_episode = 0
                patience = 500  # Stop if no improvement for 500 episodes
                episodes_without_improvement = 0

                if (episode + 1) % 10 == 0:
                    csv_file.flush()
                
                # Save Q-values every 100 episodes
                if episode % 100 == 0:
                    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    agent.save_q_value(q_value_dir, q_value_file)
                    
                    if len(episode_rewards) >= 100:
                        avg_reward = np.mean(episode_rewards[-100:])
                        
                        if avg_reward > best_avg_reward:
                            best_avg_reward = avg_reward
                            best_episode = episode
                            episodes_without_improvement = 0
                            
                            # Save BEST Q-table
                            best_q_filename = f"{Algorithm_name}_BEST_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                            agent.save_q_value(q_value_dir, best_q_filename)
                            print(f"💾 NEW BEST at episode {episode}: {avg_reward:.2f} reward")
                        else:
                            episodes_without_improvement += 100
                            print(f"⚠️  No improvement for {episodes_without_improvement} episodes")
                        
                        # Early stopping
                        if episodes_without_improvement >= patience:
                            print(f"\n🛑 EARLY STOPPING at episode {episode}")
                            print(f"   Best performance: {best_avg_reward:.2f} at episode {best_episode}")
                            print(f"   Final Q-table saved as: {best_q_filename}")
                            break

        
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
        print("\n!!! Training is complete !!!")
        print(f"CSV saved to: {csv_filename}")
        break

    csv_file.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()