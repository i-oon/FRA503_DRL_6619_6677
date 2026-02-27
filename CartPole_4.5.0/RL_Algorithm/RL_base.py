import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch
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
        self.discount_factor = discount_factor # gamma
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

    def discretize_state(self, obs: dict) -> list:
        """
        Discretize the observation state for a batch of environments.
        """
        state = obs["policy"]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
            
        # If the state is a flat array (single env), reshape it to a 2D array (1, features)
        if state.ndim == 1:
            state = state[np.newaxis, :]

        if not hasattr(self, '_discretize_calls'):
            self._discretize_calls = 0
            
        self._discretize_calls += 1
        if self._discretize_calls <= 5:
            print(f"Discretize call #{self._discretize_calls}:")
            print(f"  Raw state shape: {state.shape}")

        # Apply scaling for discretization (broadcasting handles the batch)
        scaled = state * np.array(self.discretize_state_weight)
        
        # Vectorized rounding and converting to int
        discretized_matrix = np.round(scaled).astype(int)
        
        # Convert each row into a tuple so it can be used as a dictionary key
        discretized_states = [tuple(row) for row in discretized_matrix]
        return discretized_states

    def get_discretize_action(self, obs_dis_list: list) -> np.ndarray:
        """
        Select actions using an epsilon-greedy policy for a batch of states.
        """
        actions = []
        for obs_dis in obs_dis_list:
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.num_of_action))
            else:
                if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                    q_sum = self.qa_values[obs_dis] + self.qb_values[obs_dis]
                    actions.append(int(np.argmax(q_sum)))
                else:
                    actions.append(int(np.argmax(self.q_values[obs_dis])))
                    
        return np.array(actions)

    def mapping_action(self, actions: np.ndarray, device) -> torch.Tensor:
        """
        Maps an array of discrete actions to continuous values.
        """
        action_min, action_max = self.action_range
        
        if self.num_of_action <= 1:
            scaled_values = np.full((len(actions), 1), action_min)
        else:
            # Vectorized mapping of discrete actions to continuous range
            scaled_values = action_min + (actions / (self.num_of_action - 1)) * (action_max - action_min)
            scaled_values = scaled_values.reshape(-1, 1) # Reshape to (num_envs, 1)
            
        return torch.tensor(scaled_values, dtype=torch.float32, device=device)

    def get_action(self, obs) -> tuple:
        """
        Get actions based on epsilon-greedy policy for a batch of environments.
        """
        # Determine the device (Isaac Sim usually expects tensors back on the same device)
        device = obs["policy"].device if isinstance(obs["policy"], torch.Tensor) else torch.device("cpu")
        
        obs_dis_list = self.discretize_state(obs)
        action_indices = self.get_discretize_action(obs_dis_list)
        action_tensor = self.mapping_action(action_indices, device)
        
        return action_tensor, action_indices

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_value(self, path, filename):
        q_values_str_keys = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.q_values.items()}
        model_params = {'q_values': q_values_str_keys}
        
        if self.control_type == ControlType.MONTE_CARLO:
            n_values_str_keys = {str(k): (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.n_values.items()}
            model_params['n_values'] = n_values_str_keys

        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        with open(full_path, 'w') as f:
            json.dump(model_params, f, indent=4)

    def load_q_value(self, path, filename):
        full_path = os.path.join(path, filename)        
        with open(full_path, 'r') as file:
            data = json.load(file)
            
            data_q_values = data['q_values']
            for state_str, action_values in data_q_values.items():
                tuple_state = tuple(int(x) for x in ast.literal_eval(state_str))
                self.q_values[tuple_state] = np.array(action_values)
                
                if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                    self.qa_values[tuple_state] = np.array(action_values)
                    self.qb_values[tuple_state] = np.array(action_values)
                    
            if self.control_type == ControlType.MONTE_CARLO and 'n_values' in data:
                data_n_values = data['n_values']
                for state_str, n_values in data_n_values.items():
                    tuple_state = tuple(int(x) for x in ast.literal_eval(state_str))
                    self.n_values[tuple_state] = np.array(n_values)
                    
            return self.q_values