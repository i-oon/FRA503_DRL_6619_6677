import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


class BaseAlgorithm:
    """
    Base class for all function approximation-based RL algorithms.

    Provides common functionality for:
    - Action scaling (discrete -> continuous)
    - Epsilon-greedy exploration with decay
    - Visualization (matplotlib plots)
    - TensorBoard logging
    - Model persistence (save/load)

    Args:
        num_of_action (int): Number of discrete actions available.
        action_range (list): [action_min, action_max] for continuous scaling.
        learning_rate (float): Learning rate for updates.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step decay applied to epsilon.
        final_epsilon (float): Floor value for epsilon.
        discount_factor (float): Discount factor γ for future rewards.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
    ):
        # Learning parameters
        self.lr = learning_rate
        self.discount_factor = discount_factor
        
        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        # Action space
        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]
        
        # Training tracking
        self.training_error = []
        self.episode_durations = []
        
        # Matplotlib / plotting (shared by all subclasses)
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()
        
        # TensorBoard (optional)
        self.use_tensorboard = False
        self.writer = None
        self.global_step = 0

    # ------------------------------------------------------------------ #
    # Linear Q-value estimation (for Linear Q-Learning)                  #
    # ------------------------------------------------------------------ #

    def q(self, obs, a=None):
        """
        Returns the linearly-estimated Q-value for a given state and action.
        
        This is used by Linear Q-Learning: Q(s,a) = w^T * s
        where w is the weight matrix.
        
        Args:
            obs: State observation (numpy array or tensor).
            a (int | None): Action index. If None, returns Q-values for all actions.
        
        Returns:
            float | ndarray: Q-value(s) for the given state and action(s).
        """
        # Convert observation to numpy if needed
        if isinstance(obs, torch.Tensor):
            state = obs.cpu().numpy()
        else:
            state = np.array(obs)
        
        # Linear approximation: Q(s,a) = state @ w[:, a]
        if hasattr(self, 'w'):
            if a is None:
                # Get Q-values for all actions: state @ w
                # state shape: (state_dim,), w shape: (state_dim, num_actions)
                # result shape: (num_actions,)
                return state @ self.w
            else:
                # Get Q-value for specific action
                return (state @ self.w[:, a]).item() if hasattr(self.w[:, a], 'item') else float(state @ self.w[:, a])
        else:
            # Fallback for non-linear Q algorithms
            if a is None:
                return np.zeros(self.num_of_action)
            else:
                return 0.0

    # ------------------------------------------------------------------ #
    # Action utilities                                                     #
    # ------------------------------------------------------------------ #

    def scale_action(self, action: int) -> torch.Tensor:
        """
        Map a discrete action index [0, num_of_action-1] to a continuous value
        in [action_min, action_max].
        
        Uses linear interpolation:
            scaled_action = action_min + (action_max - action_min) * action / (n - 1)

        Args:
            action (int): Discrete action index in [0, num_of_action - 1].

        Returns:
            torch.Tensor: Scaled continuous action tensor.
        """
        # Check if action range is defined
        if self.action_range[0] is None or self.action_range[1] is None:
            # No scaling needed - return action as-is
            return torch.tensor(action, dtype=torch.float32)
        
        # Handle single action case
        if self.num_of_action <= 1:
            return torch.tensor(self.action_range[0], dtype=torch.float32)
        
        # Linear interpolation from [0, num_of_action-1] to [action_min, action_max]
        action_min, action_max = self.action_range
        scaled = action_min + (action_max - action_min) * action / (self.num_of_action - 1)
        
        return torch.tensor(scaled, dtype=torch.float32)

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate by ``epsilon_decay``, floored at
        ``final_epsilon``.

        Call once per environment step during training.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Model persistence                                                    #
    # ------------------------------------------------------------------ #

    def save_w(self, path: str, filename: str) -> None:
        """
        Save weight parameters (for Linear Q-Learning).

        Args:
            path (str): Directory path to save weights.
            filename (str): Filename for the weights (e.g., 'weights.npy').
        """
        if not hasattr(self, 'w'):
            print("⚠️  No weights to save (self.w not defined)")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save weights as numpy array
        filepath = os.path.join(path, filename)
        np.save(filepath, self.w)
        
        print(f"✅ Weights saved to {filepath}")
        print(f"   Shape: {self.w.shape}")

    def load_w(self, path: str, filename: str) -> None:
        """
        Load weight parameters (for Linear Q-Learning).

        Args:
            path (str): Directory path containing weights.
            filename (str): Filename of the weights (e.g., 'weights.npy').
        """
        filepath = os.path.join(path, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"❌ Weights file not found: {filepath}")
            return
        
        # Load weights
        self.w = np.load(filepath)
        
        print(f"✅ Weights loaded from {filepath}")
        print(f"   Shape: {self.w.shape}")

    # ------------------------------------------------------------------ #
    # TensorBoard Integration                                              #
    # ------------------------------------------------------------------ #

    def init_tensorboard(self, log_dir: str = None, comment: str = '') -> None:
        """
        Initialize TensorBoard writer.

        Args:
            log_dir (str | None): Directory for logs. If None, creates timestamped dir.
            comment (str): Additional comment for the run name.
        """
        # Create log directory with timestamp if not provided
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = f'runs/{self.__class__.__name__}_{comment}_{timestamp}'
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        self.use_tensorboard = True
        
        print(f"📊 TensorBoard logging to: {log_dir}")
        print(f"   View with: tensorboard --logdir=runs --port=6006")

    def log_scalar(self, tag: str, value: float, step: int = None) -> None:
        """
        Log a scalar value to TensorBoard.

        Args:
            tag (str): Name of the scalar (e.g., 'Loss/Critic').
            value (float): Value to log.
            step (int | None): Training step. Uses self.global_step if None.
        """
        if self.use_tensorboard and self.writer:
            step = step if step is not None else self.global_step
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int = None) -> None:
        """
        Log a histogram to TensorBoard.

        Args:
            tag (str): Name of the histogram (e.g., 'Weights/Actor').
            values (Tensor): Values to histogram.
            step (int | None): Training step. Uses self.global_step if None.
        """
        if self.use_tensorboard and self.writer:
            step = step if step is not None else self.global_step
            self.writer.add_histogram(tag, values, step)

    def close_tensorboard(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            print("📊 TensorBoard writer closed.")

    # ------------------------------------------------------------------ #
    # Visualization                                                        #
    # ------------------------------------------------------------------ #

    def plot_durations(self, timestep=None, show_result=False):
        """
        Plot episode durations with a 100-episode running average.

        Args:
            timestep (int | None): Episode length to record. Pass None to
                                   redraw without adding a new data point.
            show_result (bool): If True titles the plot 'Result',
                                otherwise 'Training...'.
        """
        # Add new data point if provided
        if timestep is not None:
            self.episode_durations.append(timestep)

        # Create plot
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        
        # Plot running average if we have enough data
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        
        # Handle IPython display
        if self.is_ipython:
            from IPython import display
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    # ------------------------------------------------------------------ #
    # Utility methods                                                      #
    # ------------------------------------------------------------------ #

    def get_training_stats(self) -> dict:
        """
        Get training statistics summary.
        
        Returns:
            dict: Dictionary containing training metrics.
        """
        stats = {
            'epsilon': self.epsilon,
            'total_episodes': len(self.episode_durations),
            'avg_duration_last_100': np.mean(self.episode_durations[-100:]) if len(self.episode_durations) >= 100 else np.mean(self.episode_durations) if self.episode_durations else 0.0,
            'training_errors': len(self.training_error),
        }
        
        return stats

    def reset_training_stats(self) -> None:
        """Reset all training statistics."""
        self.training_error = []
        self.episode_durations = []
        self.global_step = 0
        print("✅ Training statistics reset")