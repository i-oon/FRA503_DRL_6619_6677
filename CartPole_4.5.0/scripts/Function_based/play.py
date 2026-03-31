"""Script to play/test RL agent - HW3 Function-Based Algorithms."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Add project root to path (go up 2 levels: Function_based -> scripts -> CartPole_4.5.0)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Parse arguments
parser = argparse.ArgumentParser(description="Play/test trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0", help="Name of the task.")
parser.add_argument("--algorithm", type=str, default=None, help="Algorithm to use (overrides config).")
parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file.")
parser.add_argument("--n_episodes", type=int, default=10, help="Number of test episodes.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

# Import Isaac Lab utilities for env parsing
from isaaclab_tasks.utils import parse_env_cfg

# Import config system
from config import get_config, create_agent, get_device

# Import environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def play_episode(env, agent, max_steps=1000, render=False):
    """
    Play one episode with the trained agent.
    
    Args:
        env: Isaac Lab environment
        agent: Trained RL agent
        max_steps (int): Maximum steps per episode
        render (bool): Whether to render (for video recording)
    
    Returns:
        tuple: (episode_return, episode_length)
    """
    obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    
    for step in range(max_steps):
        # Get observation for policy
        obs_tensor = obs["policy"]
        
        # Select action (deterministic for testing)
        with torch.no_grad():
            action = agent.select_action(obs_tensor[0:1])
            
            # Handle different return types
            if isinstance(action, tuple):
                action = action[0]  # (action_tensor, action_idx)
            
            # Ensure action is a tensor
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            
            # Add batch dimension if needed
            if action.dim() == 0:
                action = action.unsqueeze(0)
            
            # Repeat for all environments
            num_envs = obs_tensor.shape[0]
            action_batched = action.unsqueeze(0).repeat(num_envs, 1).to(obs_tensor.device)
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action_batched)
        
        # Accumulate return (first environment only for single-env testing)
        episode_return += float(reward[0].item())
        episode_length += 1
        
        # Check if done (first environment)
        if terminated[0] or truncated[0]:
            break
        
        obs = next_obs
    
    return episode_return, episode_length


def evaluate_agent(env, agent, n_episodes=10, max_steps=1000):
    """
    Evaluate agent over multiple episodes.
    
    Args:
        env: Isaac Lab environment
        agent: Trained RL agent
        n_episodes (int): Number of evaluation episodes
        max_steps (int): Maximum steps per episode
    
    Returns:
        dict: Evaluation statistics
    """
    print(f"\n{'='*80}")
    print(f"🎮 Evaluating Agent")
    print(f"{'='*80}")
    
    returns = []
    lengths = []
    
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        episode_return, episode_length = play_episode(env, agent, max_steps)
        returns.append(episode_return)
        lengths.append(episode_length)
        
        print(f"  Episode {episode + 1}/{n_episodes}: "
              f"Return = {episode_return:.2f}, Length = {episode_length}")
    
    # Calculate statistics
    ret = np.array(returns, dtype=np.float64)
    lng = np.array(lengths, dtype=np.float64)
    max_length = 1000  # env max episode length
    success_count = int(np.sum(lng >= max_length))
    success_rate = success_count / len(lng) * 100

    stats = {
        'mean_return': float(np.mean(ret)),
        'std_return': float(np.std(ret)),
        'min_return': float(np.min(ret)),
        'max_return': float(np.max(ret)),
        'median_return': float(np.median(ret)),
        'mean_length': float(np.mean(lng)),
        'std_length': float(np.std(lng)),
        'success_rate': success_rate,
        'success_count': success_count,
        'n_episodes': len(ret),
    }

    W = 80
    print(f"\n{'='*W}")
    print(f"{'Playing Performance':^{W}}")
    print(f"{'='*W}")
    print(f"  Mean Return     : {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")
    print(f"  Median Return   : {stats['median_return']:.2f}")
    print(f"  Min / Max       : {stats['min_return']:.2f} / {stats['max_return']:.2f}")
    print(f"  Mean Length     : {stats['mean_length']:.1f} +/- {stats['std_length']:.1f}")
    print(f"  Success Rate    : {success_count}/{len(lng)} ({success_rate:.0f}%)  (survived {max_length} steps)")
    print(f"{'='*W}\n")

    return stats


def main():
    """Main play/test function."""
    print(f"\n{'='*80}")
    print(f"🎮 HW3 Agent Testing - Function-Based Algorithms")
    print(f"{'='*80}\n")
    
    # Determine algorithm from model path if not specified
    if args_cli.algorithm is None:
        # Try to infer from model path
        model_filename = os.path.basename(args_cli.model_path)
        for algo in ['Linear_Q', 'DQN', 'MC_REINFORCE', 'A2C', 'PPO', 'SAC', 'TD3']:
            if algo.lower() in model_filename.lower():
                args_cli.algorithm = algo
                break
        
        if args_cli.algorithm is None:
            print("❌ Could not determine algorithm from model path.")
            print("   Please specify --algorithm explicitly.")
            simulation_app.close()
            return
    
    print(f"Algorithm: {args_cli.algorithm}")
    print(f"Model Path: {args_cli.model_path}")
    print(f"Task: {args_cli.task}")
    print(f"Number of Test Episodes: {args_cli.n_episodes}")
    print(f"Random Seed: {args_cli.seed}")
    
    # Set random seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    
    # Get configuration
    try:
        config = get_config(args_cli.algorithm)
    except ValueError as e:
        print(f"❌ Error: {e}")
        simulation_app.close()
        return
    
    # Create environment
    print(f"\n🌍 Creating environment...")
    
    # Parse environment config (Isaac Lab specific)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs
    )
    
    # Create environment with parsed config
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None
    )
    print(f"✅ Environment created: {args_cli.task}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Create agent (in testing mode - no exploration)
    print(f"\n🤖 Creating agent...")
    agent = create_agent(algorithm=args_cli.algorithm, testing=True)
    print(f"✅ Agent created: {args_cli.algorithm}")
    
    # Load trained model
    print(f"\n📂 Loading trained model...")
    try:
        model_dir = os.path.dirname(args_cli.model_path)
        model_file = os.path.basename(args_cli.model_path)
        agent.load_model(model_dir, model_file)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        simulation_app.close()
        return
    
    # Evaluate agent
    if args_cli.video:
        # Record video mode - just play one episode
        print(f"\n🎥 Recording video...")
        
        obs, _ = env.reset()
        timestep = 0
        episode_return = 0.0
        
        while simulation_app.is_running() and timestep < args_cli.video_length:
            with torch.no_grad():
                # Get action
                obs_tensor = obs["policy"]
                action = agent.select_action(obs_tensor[0:1])
                
                # Handle different return types
                if isinstance(action, tuple):
                    action = action[0]
                
                # Ensure proper format
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action, dtype=torch.float32)
                
                if action.dim() == 0:
                    action = action.unsqueeze(0)
                
                num_envs = obs_tensor.shape[0]
                action_batched = action.unsqueeze(0).repeat(num_envs, 1).to(obs_tensor.device)

                # Step environment
                next_obs, reward, terminated, truncated, _ = env.step(action_batched)
                
                episode_return += float(reward[0].item())
                obs = next_obs
                timestep += 1
                
                if terminated[0] or truncated[0]:
                    print(f"✅ Episode completed!")
                    print(f"   Return: {episode_return:.2f}")
                    print(f"   Length: {timestep} steps")
                    break
        
        print(f"🎥 Video recording complete!")
    
    else:
        # Evaluation mode - play multiple episodes
        stats = evaluate_agent(
            env=env,
            agent=agent,
            n_episodes=args_cli.n_episodes,
            max_steps=1000
        )
        
        # Check if agent is good
        if stats['mean_return'] >= 475:
            print("🎉 EXCELLENT! Agent is solving the task consistently!")
        elif stats['mean_return'] >= 400:
            print("✅ GOOD! Agent performs well.")
        elif stats['mean_return'] >= 300:
            print("⚠️  FAIR. Agent needs more training.")
        else:
            print("❌ POOR. Agent is not performing well.")
    
    # Close environment
    env.close()
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()