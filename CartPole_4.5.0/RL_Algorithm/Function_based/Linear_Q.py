from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning with function approximation.

    Simple and standard: Q(s,a) = φ(s)^T · w_a
    Uses only env[0] for learning (single-env Q-learning).
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

        # Linear weight matrix: (obs_dim, num_of_action)
        self.w = np.zeros((4, num_of_action))

    def q(self, obs, a=None):
        """Q(s,a) = obs^T · w_a"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        obs = np.asarray(obs, dtype=np.float32).flatten()
        q_values = obs @ self.w
        if a is None:
            return q_values
        return float(q_values[a])

    def calculate_loss(self, obs, action, reward, next_obs, next_action, terminated):
        """TD error: δ = r + γ·Q(s',a')·(1-done) - Q(s,a)"""
        q_current = self.q(obs, action)
        q_next = self.q(next_obs, next_action)
        return reward + self.discount_factor * q_next * (1 - int(terminated)) - q_current

    def update(self, obs, action, reward, next_obs, next_action, terminated):
        """Weight update: w_a += α · δ · φ(s)"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()
        obs = np.asarray(obs, dtype=np.float32).flatten()
        next_obs = np.asarray(next_obs, dtype=np.float32).flatten()

        td_error = self.calculate_loss(obs, action, reward, next_obs, next_action, terminated)
        self.w[:, action] += self.lr * td_error * obs

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        state = np.asarray(state, dtype=np.float32).flatten()

        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            action_idx = int(np.argmax(self.q(state)))

        return self.scale_action(action_idx), action_idx

    def learn(self, env, max_steps: int):
        """Train for one episode using env[0] only."""
        obs, _ = env.reset()
        episode_return = 0.0
        num_envs = obs["policy"].shape[0]

        state = obs["policy"][0].cpu().numpy()
        action_tensor, action_idx = self.select_action(state)

        for step in range(max_steps):
            action_batched = action_tensor.unsqueeze(0).unsqueeze(0).repeat(num_envs, 1)

            next_obs, reward, terminated, truncated, _ = env.step(action_batched)

            next_state = next_obs["policy"][0].cpu().numpy()
            r = float(reward[0].item())
            t = bool(terminated[0].item())
            done = bool((terminated[0] | truncated[0]).item())

            # SARSA: choose next action before updating
            next_action_tensor, next_action_idx = self.select_action(next_state)

            self.update(state, action_idx, r, next_state, next_action_idx, t)
            self.decay_epsilon()

            episode_return += r
            state = next_state
            action_tensor = next_action_tensor
            action_idx = next_action_idx

            if done:
                break

        self.episode_durations.append(episode_return)
        return episode_return, step + 1

    def save_model(self, path: str, filename: str) -> None:
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        np.save(filepath, self.w)
        print(f"✅ Linear_Q model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        self.w = np.load(filepath)
        print(f"✅ Linear_Q model loaded from {filepath}")
