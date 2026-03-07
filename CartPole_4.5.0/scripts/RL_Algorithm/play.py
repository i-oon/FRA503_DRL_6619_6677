"""Script to evaluate trained tabular RL agent on CartPole stabilization."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import glob
import csv
from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Add config directory to path
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir)

# Import config FIRST
from config import get_config, print_config, create_agent

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate trained RL agent on CartPole.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results.")

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
from datetime import datetime

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


def safe_tensor_to_bool(tensor_value):
    """Safely convert tensor or scalar to boolean."""
    if torch.is_tensor(tensor_value):
        return tensor_value.item() if tensor_value.numel() == 1 else bool(tensor_value.any())
    return bool(tensor_value)


def main():
    """Evaluate trained RL agent with survival time analysis."""
    
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs,
    )

    # Calculate timing parameters from environment config
    step_time = env_cfg.sim.dt * env_cfg.decimation  # Time per RL step (should be 0.01s)
    max_episode_steps = int(env_cfg.episode_length_s / step_time)  # Should be 1000 steps
    
    print(f"Environment timing:")
    print(f"  sim.dt = {env_cfg.sim.dt} s")
    print(f"  decimation = {env_cfg.decimation}")
    print(f"  episode_length_s = {env_cfg.episode_length_s} s")
    print(f"  step_time = {step_time} s")
    print(f"  max_episode_steps = {max_episode_steps}")

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
    
    print(f"Created {Algorithm_name} agent for evaluation")
    print(f"Epsilon: {agent.epsilon} (should be 0.0 for greedy evaluation)")
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
    
    print(f"\n🔄 Loading Q-values from EPISODE {latest_episode}")
    print(f"   File: {latest_filename}")

    # Load Q-values
    agent.load_q_value(q_value_dir, latest_filename)

    print(f"✅ Successfully loaded Q-table")
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
        print("   ✅ Q-values look reasonable")
    
    print("="*70 + "\n")

    # ==================================================================== #
    # ==================== PREPARE OUTPUT DIRECTORY ==================== #
    # ==================================================================== #
    
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{Algorithm_name}_{task_name}_evaluation_{timestamp}.csv"
    csv_path = os.path.join(args_cli.output_dir, csv_filename)

    # ==================================================================== #
    # ==================== RUN EVALUATION EPISODES ====================== #
    # ==================================================================== #

    print(f"🚀 Running {n_test_episodes} evaluation episodes...\n")

    episode_results = []
    
    obs, _ = env.reset()
    timestep = 0
    
    while simulation_app.is_running():
        with torch.inference_mode():
        
            for episode in range(n_test_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0

                while True:
                    # Agent stepping (greedy action, no exploration!)
                    action, action_idx = agent.get_action(obs)

                    # Environment stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    # Safe tensor-to-boolean conversion
                    terminated_flag = safe_tensor_to_bool(terminated)
                    truncated_flag = safe_tensor_to_bool(truncated)
                    done = terminated_flag or truncated_flag

                    episode_reward += reward.item() if torch.is_tensor(reward) else reward
                    episode_length += 1

                    if done:
                        break
                    
                    obs = next_obs
                
                # Calculate survival metrics
                episode_time = episode_length * step_time
                success = episode_length >= max_episode_steps
                termination_reason = "time_limit" if truncated_flag else "failure"
                
                # Store results
                result = {
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'length_steps': episode_length,
                    'length_seconds': episode_time,
                    'success': success,
                    'termination_reason': termination_reason
                }
                episode_results.append(result)
                
                # Print episode summary
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"Episode {episode+1:3d}/{n_test_episodes}: "
                      f"Reward = {episode_reward:6.2f}, "
                      f"Length = {episode_length:4d} steps ({episode_time:5.2f}s), "
                      f"{status}")

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
        # ==================================================================== #
        # ==================== SAVE RESULTS TO CSV ========================== #
        # ==================================================================== #
        
        print(f"\n💾 Saving detailed results to: {csv_path}")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'reward', 'length_steps', 'length_seconds', 'success', 'termination_reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_results)
        
        # ==================================================================== #
        # ==================== COMPUTE SUMMARY STATISTICS =================== #
        # ==================================================================== #
        
        rewards = [r['reward'] for r in episode_results]
        lengths_steps = [r['length_steps'] for r in episode_results]
        lengths_seconds = [r['length_seconds'] for r in episode_results]
        successes = [r['success'] for r in episode_results]
        
        success_rate = np.mean(successes) * 100
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)
        
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        print(f"Algorithm: {Algorithm_name} (Episode {latest_episode})")
        print(f"Episodes:  {n_test_episodes}")
        print(f"")
        print(f"Reward Statistics:")
        print(f"  Mean:    {np.mean(rewards):7.2f} ± {np.std(rewards):.2f}")
        print(f"  Best:    {np.max(rewards):7.2f} (Episode {best_idx + 1})")
        print(f"  Worst:   {np.min(rewards):7.2f} (Episode {worst_idx + 1})")
        print(f"")
        print(f"Survival Statistics:")
        print(f"  Steps:   {np.mean(lengths_steps):7.1f} ± {np.std(lengths_steps):.1f}")
        print(f"  Time:    {np.mean(lengths_seconds):7.2f} ± {np.std(lengths_seconds):.2f} seconds")
        print(f"  Success: {success_rate:7.1f}% ({np.sum(successes)}/{n_test_episodes} episodes)")
        print(f"  Max possible: {max_episode_steps} steps ({env_cfg.episode_length_s:.1f}s)")
        print(f"")
        print(f"Agent Configuration:")
        print(f"  Epsilon: {agent.epsilon:.3f} (greedy evaluation)")
        print(f"  Q-table: {len(agent.q_values)} states")
        print("="*70)
        
        break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()