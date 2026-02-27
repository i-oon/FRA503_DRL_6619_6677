"""Script to play RL agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import glob
from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Add config directory to path
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

# Import config FIRST
from config import get_config, print_config, create_agent

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
import numpy as np

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import CartPole.tasks

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Play with trained RL agent."""
    
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs,
    )

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ==================== GET CONFIGURATION FROM config.py ============== #
    # ==================================================================== #

    # Get ALL parameters from config.py (NO hardcoded values!)
    config = get_config()  # Uses global ALGORITHM setting
    print_config()         # Print configuration

    # Extract parameters
    task_name = str(args_cli.task).split('-')[0]
    Algorithm_name = config['algorithm_name']
    num_of_action = config['num_of_action']
    action_range = config['action_range']
    discretize_state_weight = config['discretize_state_weight']
    n_test_episodes = config['n_test_episodes']

    # Create agent for TESTING (epsilon=0, no exploration!)
    agent = create_agent(testing=True)
    
    print(f"Created {Algorithm_name} agent for testing")
    print(f"Epsilon: {agent.epsilon} (should be 0.0)")
    print(f"Q-table size: {len(agent.q_values)} (empty until loaded)\n")

    # ==================================================================== #
    # ==================== LOAD Q-VALUES ================================= #
    # ==================================================================== #

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    q_value_dir = os.path.join(project_root, "q_value", task_name, Algorithm_name)

    print("="*70)
    print("🔍 LOADING Q-VALUES")
    print("="*70)
    print(f"Algorithm: {Algorithm_name}")
    print(f"Task: {task_name}")
    print(f"Config: {discretize_state_weight}")
    print(f"Directory: {q_value_dir}")

    if not os.path.exists(q_value_dir):
        print(f"\n❌ ERROR: Q-value directory does not exist!")
        print(f"   Path: {q_value_dir}")
        print(f"   Did you train this algorithm yet?")
        env.close()
        simulation_app.close()
        sys.exit(1)

    # Build pattern matching CURRENT config
    pattern = f"{Algorithm_name}_*_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
    
    print(f"\n🔍 Searching for: {pattern}")
    
    q_value_files = glob.glob(os.path.join(q_value_dir, pattern))

    if len(q_value_files) == 0:
        print(f"\nERROR: No Q-value files found matching this config!")
        print(f"   Pattern: {pattern}")
        print(f"   Config: {discretize_state_weight}")
        print(f"\n   Available files in directory:")
        all_files = glob.glob(os.path.join(q_value_dir, "*.json"))
        if all_files:
            for f in sorted(all_files)[:10]:  # Show first 10
                print(f"     - {os.path.basename(f)}")
        else:
            print(f"     (none)")
        print(f"\n   Make sure config.py has: DISCRETIZE_STATE_WEIGHT = {discretize_state_weight}")
        env.close()
        simulation_app.close()
        sys.exit(1)

    print(f"\nFound {len(q_value_files)} Q-value file(s) for this config:")
    
    # Sort files by episode number
    file_info = []
    for f in q_value_files:
        basename = os.path.basename(f)
        # Extract episode: remove algorithm name prefix, then get first number
        episode_str = basename.replace(f"{Algorithm_name}_", "").split('_')[0]
        try:
            episode = int(episode_str)
            file_info.append((episode, basename))
        except ValueError:
            print(f"   ⚠️  Warning: Could not parse episode from {basename}")
            continue
    
    file_info.sort()
    
    for episode, basename in file_info:
        print(f"   Episode {episode:4d}: {basename}")

    if not file_info:
        print(f"\nERROR: Could not parse any episode numbers from filenames!")
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    latest_episode, latest_filename = file_info[-1]
    
    print(f"\n Loading Q-values from EPISODE {latest_episode}")
    print(f"   File: {latest_filename}")

    # Load Q-values
    agent.load_q_value(q_value_dir, latest_filename)

    print(f"Successfully loaded Q-table")
    print(f"States in Q-table: {len(agent.q_values)}")
    
    if len(agent.q_values) == 0:
        print("   ERROR: Q-table is EMPTY!")
        env.close()
        simulation_app.close()
        sys.exit(1)
    
    # Sample Q-values
    sample_states = list(agent.q_values.items())[:5]
    max_q = max([max(q_vals) for _, q_vals in sample_states])
    min_q = min([min(q_vals) for _, q_vals in sample_states])
    
    print(f"   Q-value range (sample): [{min_q:.2f}, {max_q:.2f}]")
    
    if max_q < 0.1:
        print("   ⚠️  WARNING: Q-values seem very low. Training might have failed.")
    else:
        print("   Q-values look reasonable")
    
    print("="*70 + "\n")

    # ==================================================================== #
    # ==================== PLAY EPISODES ================================= #
    # ==================================================================== #

    print(f"  Running {n_test_episodes} test episodes...\n")

    episode_rewards = []
    episode_lengths = []

    obs, _ = env.reset()
    timestep = 0
    
    while simulation_app.is_running():
        with torch.inference_mode():
        
            for episode in range(n_test_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0

                while not done:
                    # Agent stepping (no exploration!)
                    action, action_idx = agent.get_action(obs)

                    # Environment stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    episode_reward += reward.item()
                    episode_length += 1

                    done = terminated or truncated
                    obs = next_obs
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                print(f"Episode {episode+1}/{n_test_episodes}: "
                      f"Reward = {episode_reward:.2f}, Length = {episode_length} steps")

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Best Reward:    {np.max(episode_rewards):.2f} (Episode {np.argmax(episode_rewards)+1})")
        print(f"Worst Reward:   {np.min(episode_rewards):.2f} (Episode {np.argmin(episode_rewards)+1})")
        print("="*70)
        
        break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()