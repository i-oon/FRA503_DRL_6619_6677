from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm
from RL_Algorithm.storage.buffers import TensorReplayBuffer
from RL_Algorithm.networks.mlp import MLP


class SAC_Discrete_Actor(nn.Module):
    """
    Stochastic actor network for discrete SAC (Categorical policy).

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Number of discrete actions.
    """

    def __init__(self, n_observations: int, hidden_dim: int, n_actions: int):
        super(SAC_Discrete_Actor, self).__init__()

        # Shared trunk: n_obs → hidden → hidden
        self.trunk = MLP(
            n_observations, hidden_dim,
            [hidden_dim],
            activation="relu",
            last_activation="relu",
        )

        # Output head for action logits
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.n_actions = n_actions

    def forward(self, state: torch.Tensor):
        """
        Forward pass.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tensor: Action logits, shape (batch, n_actions).
        """
        x = self.trunk(state)
        logits = self.action_head(x)
        return logits

    def get_action_probs(self, state: torch.Tensor):
        """
        Get action probabilities.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tensor: Action probabilities, shape (batch, n_actions).
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)

    def sample(self, state: torch.Tensor):
        """
        Sample action from categorical distribution and compute log-prob + entropy.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (action_indices, log_probs, entropy)
                - action_indices: shape (batch,)
                - log_probs: shape (batch, 1)
                - entropy: shape (batch, 1)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Create categorical distribution
        dist = Categorical(probs)
        
        # Sample action
        action = dist.sample()
        
        # Compute log probability of sampled action
        log_prob = dist.log_prob(action).unsqueeze(-1)
        
        # Compute entropy: -sum(p * log(p))
        entropy = dist.entropy().unsqueeze(-1)
        
        return action, log_prob, entropy


class SAC_Discrete_Critic(nn.Module):
    """
    Twin Q-networks for discrete SAC (Q1 and Q2).

    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Number of discrete actions.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(SAC_Discrete_Critic, self).__init__()
        self.q1 = MLP(n_observations, n_actions, [hidden_dim, hidden_dim], activation="relu")
        self.q2 = MLP(n_observations, n_actions, [hidden_dim, hidden_dim], activation="relu")

    def forward(self, state: torch.Tensor):
        """
        Compute both Q-values for all actions.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both shape (batch, n_actions).
        """
        return self.q1(state), self.q2(state)


class SAC_Discrete(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) for Discrete Action Spaces.

    Args:
        device: Torch device.
        num_of_action (int): Number of discrete actions.
        action_range (list): Not used for discrete, kept for compatibility.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for actor and critics.
        alpha_lr (float): Learning rate for automatic temperature tuning.
        tau (float): Polyak soft-update coefficient.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
        init_alpha (float): Initial temperature α.
        auto_alpha (bool): Enable automatic α tuning.
        target_entropy (float | None): Target entropy for auto-tuning.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = 4,  # Number of discrete actions
            action_range: list = None,  # Not used for discrete
            n_observations: int = 4,
            hidden_dim: int = 256,
            learning_rate: float = 3e-4,
            alpha_lr: float = 3e-4,
            tau: float = 0.005,
            discount_factor: float = 0.99,
            buffer_size: int = 100_000,
            batch_size: int = 256,
            init_alpha: float = 0.2,
            auto_alpha: bool = True,
            target_entropy: float | None = None,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_of_action = num_of_action

        # Networks
        self.actor = SAC_Discrete_Actor(n_observations, hidden_dim, num_of_action).to(self.device)
        
        self.critic = SAC_Discrete_Critic(n_observations, num_of_action, hidden_dim).to(self.device)
        self.critic_target = SAC_Discrete_Critic(n_observations, num_of_action, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Temperature (entropy coefficient)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = target_entropy if target_entropy is not None else 0.98 * np.log(num_of_action)
            self.log_alpha = torch.tensor([np.log(init_alpha)], dtype=torch.float32,
                                          requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.log_alpha = torch.log(torch.tensor([init_alpha], device=self.device))
        
        self.tau = tau
        self.update_count = 0

        # Persistent env state
        self._last_obs = None
        self._episode_rewards = None

        super(SAC_Discrete, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range if action_range is not None else [0, num_of_action - 1],
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        # Override the CPU namedtuple buffer with a GPU tensor buffer
        self.memory = TensorReplayBuffer(
            buffer_size=buffer_size,
            obs_dim=n_observations,
            device=self.device,
            action_dim=1,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state, deterministic: bool = False):
        """
        Select action (stochastic during training, deterministic for eval).

        Args:
            state: Current state (Tensor or ndarray).
            deterministic (bool): If True, use greedy action (for evaluation).

        Returns:
            Tuple[Tensor, int]: (scaled_action_tensor, action_index)
        """
        if isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            probs = self.actor.get_action_probs(state_tensor)
            if deterministic:
                action_idx = int(probs.argmax(dim=-1).item())
            else:
                action_idx = int(torch.multinomial(probs, 1).item())

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
        Compute SAC critic, actor, and temperature losses for discrete actions.

        Args:
            states (Tensor): State batch, shape (B, obs_dim).
            actions (Tensor): Action batch (indices), shape (B,) or (B, 1).
            rewards (Tensor): Reward batch, shape (B, 1).
            next_states (Tensor): Next-state batch, shape (B, obs_dim).
            dones (Tensor): Done flags, shape (B, 1).

        Returns:
            dict: {"critic_loss", "actor_loss", "alpha_loss", "log_probs", "entropy"}
        """
        # Ensure actions are long indices
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        actions = actions.long()

        # ===== Compute Target Q-value ===== #
        with torch.no_grad():
            next_probs = self.actor.get_action_probs(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            q1_next, q2_next = self.critic_target(next_states)
            q_next = torch.min(q1_next, q2_next)
            # V(s') = Σ_a π(a|s') * (Q(s',a) - α * log π(a|s'))
            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            q_target = rewards + self.discount_factor * (1 - dones) * v_next

        # ===== Critic Loss ===== #
        q1_all, q2_all = self.critic(states)
        q1 = q1_all.gather(1, actions.unsqueeze(-1))
        q2 = q2_all.gather(1, actions.unsqueeze(-1))
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # ===== Actor Loss ===== #
        # Q-values are detached — actor gradient flows only through the policy, not the critic
        current_probs = self.actor.get_action_probs(states)
        with torch.no_grad():
            q1_new, q2_new = self.critic(states)
            q_new = torch.min(q1_new, q2_new)

        current_log_probs = torch.log(current_probs + 1e-8)
        entropy = -(current_probs * current_log_probs).sum(dim=-1, keepdim=True)

        # L = Σ_a π(a|s) * (α * log π(a|s) - Q(s,a))
        actor_loss = (current_probs * (self.alpha * current_log_probs - q_new)).sum(dim=-1).mean()

        # ===== Temperature Loss ===== #
        alpha_loss = None
        if self.auto_alpha:
            # L_α = log_α * (H(π) - H_target): increases α when entropy < target, decreases when > target
            alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "log_probs": current_log_probs,
            "entropy": entropy.mean(),
        }

    def update_policy(self):
        """Perform one gradient step on actor and critics."""
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

        # ===== Actor Update ===== #
        self.actor_optimizer.zero_grad()
        losses["actor_loss"].backward()
        self.actor_optimizer.step()
        self._step_actor_losses.append(losses["actor_loss"].item())

        # ===== Temperature Update ===== #
        if self.auto_alpha and losses["alpha_loss"] is not None:
            self.alpha_optimizer.zero_grad()
            losses["alpha_loss"].backward()
            self.alpha_optimizer.step()
        
        # ===== Soft Update Target Networks ===== #
        self.update_target_networks()
        self.update_count += 1

    def update_target_networks(self):
        """Soft update of target critic network."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def learn(self, env, max_steps: int, num_agents: int = None):
        """
        Train SAC with parallel environments.

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

            # Vectorized action selection — single forward pass for all envs
            with torch.no_grad():
                probs = self.actor.get_action_probs(obs_tensor)
                action_indices = torch.multinomial(probs, 1).squeeze(-1)

            # Scale action indices to continuous range for the environment
            min_a, max_a = self.action_range
            actions_scaled = (
                min_a + (max_a - min_a) * action_indices.float() / (self.num_of_action - 1)
            ).unsqueeze(-1)  # (N, 1)

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(actions_scaled)
            done_flags = terminated | truncated

            # Vectorized batch write to GPU tensor buffer — no Python loop
            rewards_2d = reward.unsqueeze(-1) if reward.dim() == 1 else reward
            terminated_2d = terminated.float().unsqueeze(-1)
            actions_2d = action_indices.unsqueeze(-1)
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
            'alpha': self.alpha.item(),
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
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'episode_durations': self.episode_durations,
        }, filepath)
        
        print(f"✅ SAC_Discrete model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """Load actor and critic weights."""
        filepath = os.path.join(path, filename)
        
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        
        self.actor.eval()
        self.critic.eval()
        self._last_obs = None
        self._episode_rewards = None
        print(f"✅ SAC_Discrete model loaded from {filepath}")