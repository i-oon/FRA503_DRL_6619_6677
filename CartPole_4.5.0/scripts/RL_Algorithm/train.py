# scripts/Function_based/train.py
# HW3 Training Script - Function Approximation Algorithms
# Usage: python train.py --algorithm DQN --num_envs 256

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Isaac Lab environment
from omni.isaac.lab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train RL agent on CartPole")
parser.add_argument("--algorithm", type=str, default=None, help="Algorithm to train")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI")
parser.add_argument("--load_model", type=str, default=None, help="Path to load model from")
parser.add_argument("--save_interval", type=int, default=100, help="Save model every N iterations")
args, remaining = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after launching
import gymnasium as gym
from config_hw3 import (
    get_config, create_agent, print_config, validate_config,
    ALGORITHM, NUM_ENVS
)

# Override config if command-line args provided
if args.algorithm:
    import config_hw3
    config_hw3.ALGORITHM = args.algorithm
    
if args.num_envs:
    import config_hw3
    config_hw3.NUM_ENVS = args.num_envs


def train_off_policy(agent, env, config):
    """
    Training loop for off-policy algorithms (DQN, SAC, TD3).
    
    These algorithms:
    - Use replay buffer
    - Update every step
    - Don't need episode boundaries for updates
    """
    print(f"\n{'='*80}")
    print(f"🎯 Training {config['algorithm_name']} (Off-Policy)")
    print(f"{'='*80}\n")
    
    n_episodes = config['n_episodes']
    max_steps = config['max_steps']
    num_envs = config['num_envs']
    
    total_returns = []
    start_time = time.time()
    
    for episode in range(n_episodes):
        avg_return, steps = agent.learn(env, max_steps=max_steps, num_agents=num_envs)
        total_returns.append(avg_return)
        
        # Logging
        if (episode + 1) % config['log_interval'] == 0:
            elapsed = (time.time() - start_time) / 60
            recent_avg = np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns)
            
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Return: {avg_return:.2f} | "
                  f"Recent 100: {recent_avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {elapsed:.1f}min")
        
        # Save checkpoint
        if (episode + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['model_dir'], config['algorithm_name'])
            filename = f"{config['algorithm_name']}_ep{episode + 1}.pth"
            agent.save_model(save_path, filename)
            print(f"💾 Saved checkpoint: {filename}")
        
        # Check if solved (CartPole: avg 475+ over 100 episodes)
        if len(total_returns) >= 100:
            recent_avg = np.mean(total_returns[-100:])
            if recent_avg >= 475:
                print(f"\n🎉 SOLVED! Average return {recent_avg:.2f} over last 100 episodes")
                print(f"   Solved in {episode + 1} episodes ({(time.time() - start_time) / 60:.1f} min)")
                break
    
    # Final save
    save_path = os.path.join(config['model_dir'], config['algorithm_name'])
    filename = f"{config['algorithm_name']}_final.pth"
    agent.save_model(save_path, filename)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"   Total Episodes: {episode + 1}")
    print(f"   Total Time: {total_time:.1f} minutes")
    print(f"   Final Avg Return: {np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns):.2f}")
    print(f"{'='*80}\n")
    
    return total_returns


def train_on_policy_rollout(agent, env, config):
    """
    Training loop for on-policy rollout-based algorithms (A2C, PPO).
    
    These algorithms:
    - Collect fixed-length rollouts
    - Update after each rollout
    - Use parallel environments efficiently
    """
    print(f"\n{'='*80}")
    print(f"🎯 Training {config['algorithm_name']} (On-Policy Rollout)")
    print(f"{'='*80}\n")
    
    max_iterations = config['max_iterations']
    num_envs = config['num_envs']
    num_transitions_per_env = config['num_transitions_per_env']
    
    start_time = time.time()
    
    # Train using agent's built-in learn method
    mean_return, losses, iterations = agent.learn(
        env=env,
        num_envs=num_envs,
        num_transitions_per_env=num_transitions_per_env,
        max_iterations=max_iterations,
    )
    
    total_time = (time.time() - start_time) / 60
    
    # Save final model
    save_path = os.path.join(config['model_dir'], config['algorithm_name'])
    filename = f"{config['algorithm_name']}_final.pth"
    agent.save_model(save_path, filename)
    
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"   Total Iterations: {iterations}")
    print(f"   Total Time: {total_time:.1f} minutes")
    print(f"   Final Mean Return: {mean_return:.2f}")
    print(f"{'='*80}\n")
    
    return agent.episode_durations


def train_reinforce(agent, env, config):
    """
    Training loop for REINFORCE (episodic policy gradient).
    
    REINFORCE:
    - Collects complete episodes from parallel envs
    - Updates after collecting multiple episodes
    - Handles episodes finishing at different times
    """
    print(f"\n{'='*80}")
    print(f"🎯 Training {config['algorithm_name']} (Policy Gradient)")
    print(f"{'='*80}\n")
    
    n_iterations = config['n_iterations']
    max_steps = config['max_steps']
    num_envs = config['num_envs']
    
    total_returns = []
    start_time = time.time()
    
    for iteration in range(n_iterations):
        avg_return, loss, num_episodes = agent.learn(
            env=env,
            max_steps=max_steps,
            num_agents=num_envs
        )
        
        if num_episodes > 0:
            total_returns.append(avg_return)
        
        # Logging
        if (iteration + 1) % config['log_interval'] == 0:
            elapsed = (time.time() - start_time) / 60
            recent_avg = np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns) if total_returns else 0.0
            
            print(f"Iteration {iteration + 1}/{n_iterations} | "
                  f"Avg Return: {avg_return:.2f} | "
                  f"Episodes: {num_episodes} | "
                  f"Loss: {loss:.4f} | "
                  f"Recent 100: {recent_avg:.2f} | "
                  f"Time: {elapsed:.1f}min")
        
        # Save checkpoint
        if (iteration + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['model_dir'], config['algorithm_name'])
            filename = f"{config['algorithm_name']}_iter{iteration + 1}.pth"
            agent.save_model(save_path, filename)
            print(f"💾 Saved checkpoint: {filename}")
        
        # Check if solved
        if len(total_returns) >= 100:
            recent_avg = np.mean(total_returns[-100:])
            if recent_avg >= 475:
                print(f"\n🎉 SOLVED! Average return {recent_avg:.2f} over last 100 iterations")
                print(f"   Solved in {iteration + 1} iterations ({(time.time() - start_time) / 60:.1f} min)")
                break
    
    # Final save
    save_path = os.path.join(config['model_dir'], config['algorithm_name'])
    filename = f"{config['algorithm_name']}_final.pth"
    agent.save_model(save_path, filename)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"   Total Iterations: {iteration + 1}")
    print(f"   Total Time: {total_time:.1f} minutes")
    print(f"   Final Avg Return: {np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns) if total_returns else 0.0:.2f}")
    print(f"{'='*80}\n")
    
    return total_returns


def train_linear_q(agent, env, config):
    """
    Training loop for Linear Q-Learning.
    
    Linear Q:
    - Updates weights after each step
    - Works with parallel environments
    - Simple and fast
    """
    print(f"\n{'='*80}")
    print(f"🎯 Training {config['algorithm_name']} (Linear Function Approximation)")
    print(f"{'='*80}\n")
    
    n_episodes = config['n_episodes']
    max_steps = config['max_steps']
    num_envs = config['num_envs']
    
    total_returns = []
    start_time = time.time()
    
    for episode in range(n_episodes):
        avg_return, steps = agent.learn(env, max_steps=max_steps, num_agents=num_envs)
        total_returns.append(avg_return)
        
        # Logging
        if (episode + 1) % config['log_interval'] == 0:
            elapsed = (time.time() - start_time) / 60
            recent_avg = np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns)
            
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Return: {avg_return:.2f} | "
                  f"Recent 100: {recent_avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {elapsed:.1f}min")
        
        # Save checkpoint
        if (episode + 1) % config['save_interval'] == 0:
            save_path = os.path.join(config['model_dir'], config['algorithm_name'])
            filename = f"{config['algorithm_name']}_ep{episode + 1}.npy"
            agent.save_model(save_path, filename)
            print(f"💾 Saved checkpoint: {filename}")
        
        # Check if solved
        if len(total_returns) >= 100:
            recent_avg = np.mean(total_returns[-100:])
            if recent_avg >= 475:
                print(f"\n🎉 SOLVED! Average return {recent_avg:.2f} over last 100 episodes")
                break
    
    # Final save
    save_path = os.path.join(config['model_dir'], config['algorithm_name'])
    filename = f"{config['algorithm_name']}_final.npy"
    agent.save_model(save_path, filename)
    
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"   Total Episodes: {episode + 1}")
    print(f"   Total Time: {total_time:.1f} minutes")
    print(f"   Final Avg Return: {np.mean(total_returns[-100:]) if len(total_returns) >= 100 else np.mean(total_returns):.2f}")
    print(f"{'='*80}\n")
    
    return total_returns


def main():
    """Main training function."""
    # Print configuration
    print_config()
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed. Exiting.")
        return
    
    # Get configuration
    config = get_config()
    
    # Create environment
    print(f"\n🌍 Creating environment: {config['task']}")
    print(f"   Parallel environments: {config['num_envs']}")
    print(f"   Device: {config['device']}")
    
    env = gym.make(
        config['task'],
        num_envs=config['num_envs'],
        device=str(config['device']),
    )
    print(f"✅ Environment created successfully")
    
    # Create agent
    print(f"\n🤖 Creating agent: {config['algorithm_name']}")
    agent = create_agent()
    
    # Load model if specified
    if args.load_model:
        print(f"\n📂 Loading model from: {args.load_model}")
        agent.load_model(os.path.dirname(args.load_model), os.path.basename(args.load_model))
    
    # Select appropriate training loop
    algo_name = config['algorithm_name']
    
    print(f"\n🚀 Starting training...")
    
    if algo_name == 'Linear_Q':
        returns = train_linear_q(agent, env, config)
    elif algo_name in ['DQN', 'SAC', 'TD3']:
        returns = train_off_policy(agent, env, config)
    elif algo_name in ['A2C', 'PPO']:
        returns = train_on_policy_rollout(agent, env, config)
    elif algo_name == 'MC_REINFORCE':
        returns = train_reinforce(agent, env, config)
    else:
        raise ValueError(f"Unknown training loop for algorithm: {algo_name}")
    
    # Plot results
    try:
        agent.plot_durations(show_result=True)
        print("📊 Training curve displayed")
    except Exception as e:
        print(f"⚠️  Could not display plot: {e}")
    
    # Close environment
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