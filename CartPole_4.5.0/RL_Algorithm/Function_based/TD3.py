from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm
from RL_Algorithm.storage.buffers import TensorReplayBuffer
from RL_Algorithm.networks.mlp import MLP


class TD3_Discrete_Actor(nn.Module):
    """
    Actor network for discrete TD3 - outputs action probabilities.

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Number of discrete actions.
    """

    def __init__(self, n_observations: int, hidden_dim: int, n_actions: int):
        super(TD3_Discrete_Actor, self).__init__()
        self.net = MLP(
            n_observations, n_actions,
            [hidden_dim, hidden_dim],
            activation="relu",
            last_activation=None,  # No activation - we'll use softmax
        )
        self.n_actions = n_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - outputs action logits.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tensor: Action logits (not probabilities yet).
        """
        return self.net(state)
    
    def get_action_probs(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probabilities with temperature scaling.

        Args:
            state (Tensor): State tensor.
            temperature (float): Temperature for softmax (lower = more deterministic).

        Returns:
            Tensor: Action probabilities.
        """
        logits = self.forward(state)
        return F.softmax(logits / temperature, dim=-1)


class TD3_Discrete_Critic(nn.Module):
    """
    Twin Q-networks for discrete TD3 - outputs Q-values for each action.

    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Number of discrete actions.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(TD3_Discrete_Critic, self).__init__()
        self.q1 = MLP(n_observations, n_actions, [hidden_dim, hidden_dim], activation="relu")
        self.q2 = MLP(n_observations, n_actions, [hidden_dim, hidden_dim], activation="relu")

    def forward(self, state: torch.Tensor):
        """
        Compute Q1 and Q2 values for all actions.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both of shape (batch, n_actions).
        """
        return self.q1(state), self.q2(state)

    def Q1(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return only Q1 values - used for the actor update.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tensor: Q1 values of shape (batch, n_actions).
        """
        return self.q1(state)


class TD3_Discrete(OffPolicyAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) for Discrete Actions.

    Args:
        device: Torch device.
        num_of_action (int): Number of discrete actions.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for actor and critics.
        tau (float): Polyak soft-update coefficient.
        epsilon_start (float): Initial exploration rate.
        epsilon_end (float): Final exploration rate.
        epsilon_decay (float): Epsilon decay rate per step.
        target_smoothing_temperature (float): Temperature for target policy smoothing.
        policy_update_freq (int): Delayed policy updates (update actor every N critic updates).
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = 4,  # Number of discrete actions
            action_range: list = None,  # Not used for discrete, kept for compatibility
            n_observations: int = 4,
            hidden_dim: int = 256,
            learning_rate: float = 3e-4,
            tau: float = 0.005,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995,
            target_smoothing_temperature: float = 1.0,
            policy_update_freq: int = 2,
            discount_factor: float = 0.99,
            buffer_size: int = 100_000,
            batch_size: int = 256,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Networks
        self.actor = TD3_Discrete_Actor(n_observations, hidden_dim, num_of_action).to(self.device)
        self.actor_target = TD3_Discrete_Actor(n_observations, hidden_dim, num_of_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3_Discrete_Critic(n_observations, num_of_action, hidden_dim).to(self.device)
        self.critic_target = TD3_Discrete_Critic(n_observations, num_of_action, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Persistent env state
        self._last_obs = None
        self._episode_rewards = None

        super(TD3_Discrete, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range if action_range is not None else [0, num_of_action - 1],
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # TD3 hyperparameters — MUST be after super().__init__() because
        # BaseAlgorithm sets self.epsilon and self.epsilon_decay to defaults
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_smoothing_temperature = target_smoothing_temperature
        self.policy_update_freq = policy_update_freq
        self.total_steps = 0
        self.num_of_action = num_of_action

        # Override the CPU namedtuple buffer with a GPU tensor buffer
        self.memory = TensorReplayBuffer(
            buffer_size=buffer_size,
            obs_dim=n_observations,
            device=self.device,
            action_dim=1,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state, add_noise: bool = False):
        """
        Select a discrete action using epsilon-greedy exploration.

        Args:
            state: Current state (Tensor or ndarray).
            add_noise (bool): Use epsilon-greedy exploration during training.

        Returns:
            Tuple[int, None]: (action_index, None) - None for compatibility.
        """
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            action_probs = self.actor.get_action_probs(state_tensor)
            action_idx = int(action_probs.argmax(dim=-1).item())

        return self.scale_action(action_idx), action_idx

    def calculate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Compute TD3 critic and actor losses for discrete actions.

        Args:
            states (Tensor): State batch, shape (B, obs_dim).
            actions (Tensor): Action batch (indices), shape (B,) or (B, 1).
            rewards (Tensor): Reward batch, shape (B, 1).
            next_states (Tensor): Next-state batch, shape (B, obs_dim).
            dones (Tensor): Done flags, shape (B, 1).

        Returns:
            dict: {"critic_loss", "actor_loss"} — both Tensors.
        """
        # Ensure actions are long indices
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        actions = actions.long()

        with torch.no_grad():
            # Get target action probabilities with temperature smoothing
            next_action_probs = self.actor_target.get_action_probs(
                next_states, temperature=self.target_smoothing_temperature
            )
            
            # Get Q-values for next states
            q1_next, q2_next = self.critic_target(next_states)
            
            # Compute expected Q-value using probability distribution (policy smoothing)
            q_next = torch.min(q1_next, q2_next)
            expected_q_next = (next_action_probs * q_next).sum(dim=-1, keepdim=True)
            
            q_target = rewards + self.discount_factor * (1 - dones) * expected_q_next

        # Get current Q-values for taken actions
        q1_all, q2_all = self.critic(states)
        q1 = q1_all.gather(1, actions.unsqueeze(-1))
        q2 = q2_all.gather(1, actions.unsqueeze(-1))
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Actor loss: maximize expected Q-value under current policy
        # Q1 detached — actor gradient flows only through the policy, not the critic
        action_probs = self.actor.get_action_probs(states)
        with torch.no_grad():
            q1_values = self.critic.Q1(states)
        actor_loss = -(action_probs * q1_values).sum(dim=-1).mean()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        }

    def update_policy(self):
        """Perform one gradient step on critics and optionally on actor."""
        result = self.memory.sample(self.batch_size)
        if result is None:
            return

        states, actions, rewards, next_states, dones = result
        losses = self.calculate_loss(states, actions, rewards, next_states, dones)

        # ===== Critic Update ===== #
        self.critic_optimizer.zero_grad()
        losses["critic_loss"].backward()
        self.critic_optimizer.step()
        self._step_critic_losses.append(losses["critic_loss"].item())

        self.total_steps += 1

        # ===== Delayed Actor Update ===== #
        if self.total_steps % self.policy_update_freq == 0:
            self.actor_optimizer.zero_grad()
            losses["actor_loss"].backward()
            self.actor_optimizer.step()
            self._step_actor_losses.append(losses["actor_loss"].item())

            # Soft update target networks
            self.update_target_networks()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_networks(self):
        """Soft update of target actor and critic networks."""
        # Update target actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # Update target critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def learn(self, env, max_steps: int, num_agents: int = None):
        """
        Train TD3 with parallel environments.

        Args:
            env: Vectorized environment.
            max_steps (int): Total steps to collect.
            num_agents (int | None): Number of parallel environments (auto-detected if None).

        Returns:
            Tuple[float, int]: (average_return, steps_taken)
        """
        self.actor.train()
        self.critic.train()

        if self._last_obs is None:
            obs, _ = env.reset()
            num_agents = obs["policy"].shape[0] if num_agents is None else num_agents
            self._episode_rewards = torch.zeros(num_agents, device=self.device)
        else:
            obs = self._last_obs
            num_agents = obs["policy"].shape[0] if num_agents is None else num_agents

        self._step_critic_losses = []
        self._step_actor_losses = []

        for step in range(max_steps):
            obs_tensor = obs["policy"]

            # Vectorized epsilon-greedy action selection
            with torch.no_grad():
                random_mask    = torch.rand(num_agents, device=self.device) < self.epsilon
                greedy_actions = self.actor(obs_tensor).argmax(dim=-1)
                random_actions = torch.randint(0, self.num_of_action, (num_agents,), device=self.device)
                action_indices = torch.where(random_mask, random_actions, greedy_actions)

            # Scale action indices to continuous range for the environment
            min_a, max_a   = self.action_range
            actions_scaled = (
                min_a + (max_a - min_a) * action_indices.float() / (self.num_of_action - 1)
            ).unsqueeze(-1)  # (N, 1)

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(actions_scaled)
            done_flags = terminated | truncated

            # Vectorized batch write to GPU tensor buffer — no Python loop
            rewards_2d     = reward.unsqueeze(-1) if reward.dim() == 1 else reward
            terminated_2d  = terminated.float().unsqueeze(-1)
            actions_2d     = action_indices.unsqueeze(-1)
            self.memory.add_batch(obs_tensor, actions_2d, rewards_2d,
                                  next_obs["policy"], terminated_2d)

            # Vectorized episode return tracking
            self._episode_rewards += reward.squeeze(-1) if reward.dim() > 1 else reward
            done_mask = done_flags.bool().squeeze(-1) if done_flags.dim() > 1 else done_flags.bool()
            if done_mask.any():
                self.episode_durations.extend(self._episode_rewards[done_mask].tolist())
                self._episode_rewards[done_mask] = 0.0

            # Update policy
            self.update_policy()

            obs = next_obs

        self._last_obs = obs

        if len(self.episode_durations) > 0:
            recent = min(100, len(self.episode_durations))
            avg_return = sum(self.episode_durations[-recent:]) / recent
        else:
            avg_return = 0.0

        mean_vloss = (sum(self._step_critic_losses) / len(self._step_critic_losses)
                      if self._step_critic_losses else 0.0)
        mean_aloss = (sum(self._step_actor_losses) / len(self._step_actor_losses)
                      if self._step_actor_losses else 0.0)
        self._episode_stats = {
            'value_loss': mean_vloss,
            'policy_gradient_loss': mean_aloss,
            'ep_rew_mean': avg_return,
            'ep_len_mean': avg_return,
            'epsilon': self.epsilon,
        }

        return avg_return, max_steps

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """Save actor and critic weights."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'episode_durations': self.episode_durations,
            'epsilon': self.epsilon,
        }, filepath)
        
        print(f"✅ TD3_Discrete model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """Load actor and critic weights."""
        filepath = os.path.join(path, filename)
        
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        self.actor.eval()
        self.critic.eval()
        self._last_obs = None
        self._episode_rewards = None
        print(f"✅ TD3_Discrete model loaded from {filepath}")