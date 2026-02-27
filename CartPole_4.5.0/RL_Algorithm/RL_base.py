import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch
import numpy as np
import ast

class ControlType(Enum):
    """
    Enum representing different control algorithms.
    """
    MONTE_CARLO = 1
    TEMPORAL_DIFFERENCE = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4
    SARSA = 5


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        control_type (ControlType): The type of control algorithm used.
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):

        # Hyperparameters
        self.control_type = control_type # enum 1, 2, 3, 4
        self.lr = learning_rate    # step size
        self.discount_factor = discount_factor # gramma
        self.epsilon = initial_epsilon 
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.initial_epsilon = initial_epsilon 
        # Action Space Configuration
        self.num_of_action = num_of_action
        self.action_range = action_range

        # State Discretization Configuration [x_cart, theta_pole, x_dot, theta_dot]
        self.discretize_state_weight = discretize_state_weight

        # Q Table Storage Q(s, :) stored as vector
        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))

        # Counting visits for Monte Carlo method
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))

        self.training_error = []

        # Monte Carlo Buffers
        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        # Double Q-Learning Storage
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))

    # Convert continuous observation → finite tuple key.
    def discretize_state(self, obs: dict):
        """
        Discretize the observation state.

        Args:
            obs (dict): Observation dictionary containing policy states.

        Returns:
            Tuple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
        """

        state = obs["policy"]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.squeeze(state)  # ensure shape (4,)

        if not hasattr(self, '_discretize_calls'):
            self._discretize_calls = 0
        self._discretize_calls += 1
        
        if self._discretize_calls <= 5:
            print(f"Discretize call #{self._discretize_calls}:")
            print(f"  Raw state: {state}")


        # Apply scaling for discretization
        scaled = state * np.array(self.discretize_state_weight)
        # Convert to integer bins
        discretized = tuple(int(np.round(val)) for val in scaled)
        return tuple(discretized)

    # Epsilon-greedy action selection
    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_action)
        if self.control_type == ControlType.DOUBLE_Q_LEARNING:
            q_sum = self.qa_values[obs_dis] + self.qb_values[obs_dis]
            return int(np.argmax(q_sum))
        else:
            return int(np.argmax(self.q_values[obs_dis]))
    
    # Convert discrete index → continuous force tensor.
    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n]
            n (int): Number of discrete actions
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        action_min, action_max = self.action_range
        if self.num_of_action == 1:
            scaled_value = action_min
        else:
            scaled_value = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)
        return torch.tensor([[scaled_value]], dtype=torch.float32)

    def get_action(self, obs) -> torch.tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = self.get_discretize_action(obs_dis)
        action_tensor = self.mapping_action(action_idx)
        return action_tensor, action_idx
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        # Convert tuple keys to strings and numpy arrays to lists
        q_values_str_keys = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.q_values.items()}
        
        model_params = {'q_values': q_values_str_keys}
        
        # Add n_values if it's Monte Carlo
        if self.control_type == ControlType.MONTE_CARLO:
            n_values_str_keys = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.n_values.items()}
            model_params['n_values'] = n_values_str_keys

        # --- THE FIX IS HERE ---
        # Ensure the target directory exists before trying to save a file inside it
        os.makedirs(path, exist_ok=True)
        # -----------------------

        full_path = os.path.join(path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(model_params, f, indent=4) # indent=4 makes the JSON readable for humans!

            
    def load_q_value(self, path, filename):
        """
        Load model parameters from a JSON file.
        """
        full_path = os.path.join(path, filename)        
        with open(full_path, 'r') as file:
            data = json.load(file)
            
            # Load Q-Values
            data_q_values = data['q_values']
            for state_str, action_values in data_q_values.items():
                # Safely parse the string tuple back into actual integers
                tuple_state = tuple(int(x) for x in ast.literal_eval(state_str))
                
                # IMPORTANT: Convert the loaded list back to a numpy array
                self.q_values[tuple_state] = np.array(action_values)
                
                if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                    self.qa_values[tuple_state] = np.array(action_values)
                    self.qb_values[tuple_state] = np.array(action_values)
                    
            # Load N-Values for Monte Carlo
            if self.control_type == ControlType.MONTE_CARLO and 'n_values' in data:
                data_n_values = data['n_values']
                for state_str, n_values in data_n_values.items():
                    tuple_state = tuple(int(x) for x in ast.literal_eval(state_str))
                    self.n_values[tuple_state] = np.array(n_values)
                    
            return self.q_values

