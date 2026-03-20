from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning with function approximation.
    
    Uses linear weights to approximate Q-values: Q(s,a) = w^T * φ(s,a)
    where φ(s,a) = s (state features are independent of action choice)
    
    Weight matrix shape: (state_dim, num_actions)
    - Each column w[:, a] represents the weights for action a
    - Q(s,a) = s @ w[:, a] = sum(s_i * w_i,a)

    Args:
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] continuous action range.
        learning_rate (float): TD weight-update step size.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # Linear weight matrix: (state_dim, num_actions)
        # Initialize with small random values for better initial exploration
        self.w = np.random.randn(4, num_of_action) * 0.01
        
        # Track training statistics
        self.update_count = 0

    # ------------------------------------------------------------------ #
    # Linear Q-value estimation                                           #
    # ------------------------------------------------------------------ #

    def q_value(self, state: np.ndarray, action: int = None) -> np.ndarray | float:
        """
        Compute Q-value(s) using linear function approximation.
        
        Q(s,a) = w[:, a]^T @ s = sum_i(w[i,a] * s[i])
        
        Args:
            state (ndarray): State vector φ(s), shape (state_dim,).
            action (int | None): Specific action index. 
                                If None, returns Q-values for all actions.

        Returns:
            ndarray | float: 
                - If action is None: array of shape (num_actions,)
                - If action is int: scalar Q-value
        """
        # Compute Q-values for all actions: q = s @ W
        # state: (4,), w: (4, num_actions) -> q: (num_actions,)
        all_q_values = state @ self.w
        
        if action is None:
            return all_q_values
        else:
            return all_q_values[action]

    def greedy_action(self, state: np.ndarray) -> int:
        """
        Select greedy action: argmax_a Q(s,a)
        
        Args:
            state (ndarray): State vector.
            
        Returns:
            int: Action index with highest Q-value.
        """
        q_values = self.q_value(state)
        return int(np.argmax(q_values))

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state: np.ndarray) -> tuple[torch.Tensor, int]:
        """
        Epsilon-greedy action selection.

        Args:
            state (ndarray): Current state features.

        Returns:
            Tuple[Tensor, int]: 
                - Scaled continuous action tensor
                - Action index
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            # Exploit: greedy action
            action_idx = self.greedy_action(state)
        
        # Scale discrete action to continuous range
        action_scaled = self.scale_action(action_idx)
        
        return action_scaled, action_idx

    def td_update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        Temporal Difference update for linear Q-Learning.
        
        Update rule:
            δ = r + γ * max_a' Q(s', a') - Q(s, a)
            w[:, a] ← w[:, a] + α * δ * s
        
        Args:
            state (ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (ndarray): Next state.
            done (bool): Whether episode terminated.
            
        Returns:
            float: TD error magnitude (for monitoring).
        """
        # Current Q-value: Q(s, a)
        current_q = self.q_value(state, action)
        
        # TD target: r + γ * max_a' Q(s', a')
        if done:
            # Terminal state: target = reward only
            td_target = reward
        else:
            # Non-terminal: target = r + γ * max Q(s', a')
            next_q_values = self.q_value(next_state)
            max_next_q = np.max(next_q_values)
            td_target = reward + self.discount_factor * max_next_q
        
        # TD error: δ = target - current
        td_error = td_target - current_q
        
        # Gradient descent update: w[:, a] += α * δ * ∇Q = α * δ * s
        # Since Q(s,a) = w[:, a]^T @ s, we have ∇Q = s
        self.w[:, action] += self.lr * td_error * state
        
        self.update_count += 1
        
        return abs(td_error)

    # ------------------------------------------------------------------ #
    # Training loop with parallel environments                             #
    # ------------------------------------------------------------------ #

    def learn(self, env, max_steps: int, num_agents: int = 1) -> tuple[float, int]:
        """
        Train Linear Q-Learning agent with parallel environments.
        
        Collects experience from multiple parallel environments and performs
        TD updates for each transition. Episodes may finish at different times
        across environments.

        Args:
            env: Isaac Lab vectorized environment.
            max_steps (int): Total number of environment steps to collect.
            num_agents (int): Number of parallel environments.

        Returns:
            Tuple[float, int]: 
                - Average episode return across completed episodes
                - Total steps taken
        """
        # Reset all environments
        obs, _ = env.reset()
        
        # Track episode rewards per environment
        episode_rewards = np.zeros(num_agents)
        total_return = 0.0
        episodes_completed = 0
        
        # Training loop
        for step in range(max_steps):
            # Select actions for all parallel environments
            actions_scaled = []
            action_indices = []
            
            for env_idx in range(num_agents):
                # Extract state for this environment
                state = obs["policy"][env_idx].cpu().numpy()
                
                # Select action using epsilon-greedy
                action_scaled, action_idx = self.select_action(state)
                
                actions_scaled.append(action_scaled)
                action_indices.append(action_idx)
            
            # Stack actions into tensor for environment
            action_tensor = torch.cat(actions_scaled, dim=0)
            
            # Execute actions in all environments
            next_obs, rewards, terminated, truncated, _ = env.step(action_tensor)
            dones = terminated | truncated
            
            # Process transitions for each environment
            for env_idx in range(num_agents):
                # Extract transition components
                state = obs["policy"][env_idx].cpu().numpy()
                next_state = next_obs["policy"][env_idx].cpu().numpy()
                reward = float(rewards[env_idx].item())
                done = bool(dones[env_idx].item())
                action_idx = action_indices[env_idx]
                
                # TD update for this transition
                td_error = self.td_update(state, action_idx, reward, next_state, done)
                
                # Track episode rewards
                episode_rewards[env_idx] += reward
                
                # Handle episode completion
                if done:
                    # Record completed episode
                    total_return += episode_rewards[env_idx]
                    episodes_completed += 1
                    
                    # Reset this environment's episode reward
                    episode_rewards[env_idx] = 0.0
            
            # Decay epsilon after each step
            self.decay_epsilon()
            
            # Update observations for next iteration
            obs = next_obs
        
        # Calculate average return across completed episodes
        if episodes_completed > 0:
            avg_return = total_return / episodes_completed
        else:
            # If no episodes completed, use current accumulated rewards
            avg_return = np.mean(episode_rewards)
        
        # Store for plotting
        self.episode_durations.append(avg_return)
        
        return avg_return, max_steps

    # ------------------------------------------------------------------ #
    # Persistence — save/load linear weights                               #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save linear weight matrix to disk.

        Args:
            path (str): Directory path.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        # Save weights as numpy array
        np.save(filepath, self.w)
        
        print(f"✅ Linear Q-Learning model saved to {filepath}")
        print(f"   Weight matrix shape: {self.w.shape}")
        print(f"   Total updates: {self.update_count}")

    def load_model(self, path: str, filename: str) -> None:
        """
        Load linear weight matrix from disk.

        Args:
            path (str): Directory path.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        filepath = os.path.join(path, filename)
        
        # Load weights
        self.w = np.load(filepath)
        
        print(f"✅ Linear Q-Learning model loaded from {filepath}")
        print(f"   Weight matrix shape: {self.w.shape}")
        
        # Set epsilon to final value for evaluation
        self.epsilon = self.final_epsilon

    # ------------------------------------------------------------------ #
    # Additional utility methods                                           #
    # ------------------------------------------------------------------ #

    def get_q_table_visualization(self) -> dict:
        """
        Get Q-values for visualization purposes.
        
        Returns:
            dict: Information about the Q-function approximation.
        """
        return {
            'weights': self.w.copy(),
            'weight_norm': np.linalg.norm(self.w),
            'max_weight': np.max(np.abs(self.w)),
            'num_updates': self.update_count,
        }

    def reset_weights(self) -> None:
        """Reset weights to small random values."""
        self.w = np.random.randn(4, self.num_of_action) * 0.01
        self.update_count = 0
        print("⚠️ Weights have been reset")