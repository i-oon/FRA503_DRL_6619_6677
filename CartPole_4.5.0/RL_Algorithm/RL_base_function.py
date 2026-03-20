import numpy as np
from collections import defaultdict
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class BaseAlgorithm:
    """
    Base class for all function approximation-based RL algorithms.

    Args:
        num_of_action (int): Number of discrete actions available.
        action_range (list): [action_min, action_max] for continuous scaling.
        learning_rate (float): Learning rate (stored as self.lr).
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step decay applied to epsilon.
        final_epsilon (float): Floor value for epsilon.
        discount_factor (float): Discount factor γ for future rewards.
    """

    def __init__(
        self,
        num_of_action: int = None,
        action_range: list = [None, None],
        learning_rate: float = None,
        initial_epsilon: float = None,
        epsilon_decay: float = None,
        final_epsilon: float = None,
        discount_factor: float = None,
    ):
        self.lr              = learning_rate
        self.discount_factor = discount_factor
        self.epsilon         = initial_epsilon
        self.epsilon_decay   = epsilon_decay
        self.final_epsilon   = final_epsilon
        self.num_of_action   = num_of_action
        self.action_range    = action_range   # [action_min, action_max]
        self.training_error  = []

        # ===== Matplotlib / plotting (shared by all subclasses) ===== #
        self.episode_durations = []
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display
        plt.ion()

        # ===== TensorBoard (optional) ===== #
        self.use_tensorboard = False
        self.writer = None
        self.global_step = 0

    def scale_action(self, action: int) -> torch.Tensor:
        """
        Map a discrete action index [0, n-1] to a continuous value in
        [action_min, action_max].

        Args:
            action (int): Discrete action index in [0, num_of_action - 1].

        Returns:
            torch.Tensor: Scaled continuous action tensor.
        """
        # ========= put your code here ========= #
        if self.action_range[0] is None or self.action_range[1] is None:
            # No scaling needed - return action as-is
            return torch.tensor(action, dtype=torch.float32)
        
        # Linear interpolation from [0, num_of_action-1] to [action_min, action_max]
        action_min, action_max = self.action_range
        scaled = action_min + (action_max - action_min) * action / (self.num_of_action - 1)
        return torch.tensor(scaled, dtype=torch.float32)
        # ====================================== #

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate by ``epsilon_decay``, floored at
        ``final_epsilon``.

        Call once per environment step during training.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        # ====================================== #

    # ------------------------------------------------------------------ #
    # TensorBoard Integration                                            #
    # ------------------------------------------------------------------ #
 
    def init_tensorboard(self, log_dir: str = None, comment: str = '') -> None:
        """
        Initialize TensorBoard writer.
 
        Args:
            log_dir (str | None): Directory for logs. If None, creates timestamped dir.
            comment (str): Additional comment for the run name.
        """
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = f'runs/{self.__class__.__name__}_{comment}_{timestamp}'
        
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
    # Visualisation                                                      #
    # ------------------------------------------------------------------ #

    # Modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        """
        Plot episode durations with a 100-episode running average.

        Args:
            timestep (int | None): Episode length to record. Pass None to
                                   redraw without adding a new data point.
            show_result (bool): If True titles the plot 'Result',
                                otherwise 'Training...'.
        """
        if timestep is not None:
            self.episode_durations.append(timestep)

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
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if self.is_ipython:
            from IPython import display
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #