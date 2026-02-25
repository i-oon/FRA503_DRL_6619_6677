"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.RL_base import ControlType
from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
from RL_Algorithm.Algorithm.SARSA import SARSA
# from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
# from RL_Algorithm.Algorithm.Monte_Carlo import Monte_Carlo
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


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
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
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
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 5
    action_range = [-10.0, 10.0]  # [min, max]
    discretize_state_weight = [10, 10, 10, 10]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = 500
    start_epsilon = 1.0
    epsilon_decay = 0.995  # reduce the exploration over time
    final_epsilon = 0.01
    discount = 0.99

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "Q_Learning"
    agent = Q_Learning(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    sum_reward = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes)):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0

                # 1. Get the INITIAL action before the loop starts
                action_tensor, action_idx = agent.get_action(obs)

                while not done:
                    # 2. Step the environment
                    next_obs, reward, terminated, truncated, _ = env.step(action_tensor)

                    reward_value = reward.item()
                    terminated_value = terminated.item() 
                    cumulative_reward += reward_value

                    # ========================================================== #
                    # ================= ALGORITHM LOGIC SWITCH ================= #
                    
                    if agent.control_type == ControlType.SARSA:
                        # SARSA: Pick next action now, use it to update, and save it for the next loop
                        next_action_tensor, next_action_idx = agent.get_action(next_obs)
                        agent.update(
                            obs=obs,
                            action=action_idx,
                            reward=reward_value,
                            terminated=terminated_value,
                            next_obs=next_obs,
                            next_action=next_action_idx
                        )
                        # Carry the action forward to the next step
                        action_tensor = next_action_tensor
                        action_idx = next_action_idx

                    elif agent.control_type in [ControlType.Q_LEARNING, ControlType.DOUBLE_Q_LEARNING]:
                        # Q-Learning / Double Q: Update based on max future Q, then pick a new action
                        agent.update(
                            obs=obs,
                            action=action_idx,
                            reward=reward_value,
                            terminated=terminated_value,
                            next_obs=next_obs
                        )
                        # Pick a fresh action for the next step
                        action_tensor, action_idx = agent.get_action(next_obs)

                    elif agent.control_type == ControlType.MONTE_CARLO:
                        # Monte Carlo: Just store the history, do not update yet
                        agent.obs_hist.append(obs)
                        agent.action_hist.append(action_idx)
                        agent.reward_hist.append(reward_value)
                        
                        # Pick a fresh action for the next step
                        action_tensor, action_idx = agent.get_action(next_obs)

                    # ========================================================== #

                    # 3. Transition to the next state
                    done = terminated or truncated
                    obs = next_obs
                
                # ================= END OF EPISODE UPDATES ===================== #
                
                # If Monte Carlo, we process the history arrays and update the Q-table NOW
                if agent.control_type == ControlType.MONTE_CARLO:
                    agent.update()

                # Decay epsilon at the end of every episode for all algorithms
                agent.decay_epsilon()
                
                # ============================================================== #

                # Logging and Saving
                sum_reward += cumulative_reward
                if episode % 100 == 0:
                    print("avg_score: ", sum_reward / 100.0)
                    sum_reward = 0
                    print(f"Epsilon: {agent.epsilon}")

                    # Save agent
                    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)
                    agent.save_q_value(full_path, q_value_file)
             
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        print("!!! Training is complete !!!")
        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()