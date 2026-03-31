from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from ..storage.on_policy import OnPolicyAlgorithm
from ..storage.buffers import RolloutBuffer
from ..networks.mlp import MLP

# ============================================================ #
# =================== Actor-Critic Network =================== #
# ============================================================ #

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network supporting continuous and discrete actions.

    Args:
        state_dim (int): Observation space dimension.
        action_dim (int): Action vector dim (continuous) or # choices (discrete).
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        action_type: str = "continuous",
        init_noise_std: float = 1.0,
    ):
        super().__init__()

        assert action_type in ("continuous", "discrete"), \
            f"action_type must be 'continuous' or 'discrete', got '{action_type}'"

        self.action_type = action_type
        self.action_dim = action_dim

        self.actor = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1, hidden_dims, activation)

        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        self.distribution: Normal | Categorical | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.mean
        return self.distribution.probs

    @property
    def action_std(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.stddev
        return torch.ones_like(self.distribution.probs)

    @property
    def entropy(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use act() or evaluate().")

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """Build the action distribution from current observations."""
        if self.action_type == "continuous":
            mean = self.actor(obs)
            std = torch.exp(self.std)
            self.distribution = Normal(mean, std)
        else:
            logits = self.actor(obs)
            self.distribution = Categorical(logits=logits)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample an action."""
        self._update_distribution(obs)
        action = self.distribution.sample()
        
        if self.action_type == "discrete":
            action = action.unsqueeze(-1)
        
        return action

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: actor mean (continuous) or argmax (discrete)."""
        if self.action_type == "continuous":
            return self.actor(obs)
        else:
            logits = self.actor(obs)
            return torch.argmax(logits, dim=-1, keepdim=True)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimate V(s), shape (batch, 1)."""
        return self.critic(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of given actions under the current distribution."""
        if self.action_type == "continuous":
            return self.distribution.log_prob(actions).sum(dim=-1)
        else:
            return self.distribution.log_prob(actions.squeeze(-1))


# ============================================================ #
# ====================== AC Agent ============================ #
# ============================================================ #

class AC(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C-style) for PARALLEL environments.
    
    Simpler than PPO: no clipping, no multi-epoch updates, uses TD advantages.
    Works with parallel environments by collecting fixed-length rollouts.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or # choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        learning_rate (float): Adam learning rate.
        discount_factor (float): Discount factor γ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        max_grad_norm (float): Gradient clipping norm.
    """

    def __init__(
        self,
        device=None,
        num_of_action: int = 2,
        action_range: list = [-2.5, 2.5],
        n_observations: int = 4,
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        action_type: str = "discrete",
        init_noise_std: float = 1.0,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        super(AC, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Trajectory Collection (Parallel)                                     #
    # ------------------------------------------------------------------ #

    def collect_rollout(self, env, num_agents: int, rollout_length: int):
        """
        Collect a fixed-length rollout from parallel environments.
        
        Args:
            env: Vectorized environment
            num_agents (int): Number of parallel environments
            rollout_length (int): Number of steps to collect per env
            
        Returns:
            dict: Rollout data (states, actions, rewards, values, log_probs, dones)
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs_tensor = self._last_obs

        for step in range(rollout_length):
            with torch.no_grad():
                action = self.policy.act(obs_tensor)
                value = self.policy.evaluate(obs_tensor)
                log_prob = self.policy.get_actions_log_prob(action)

            if self.action_type == "discrete":
                action_idx     = action.squeeze(-1).long()
                min_a, max_a   = self.action_range
                actions_scaled = (
                    min_a + (max_a - min_a) * action_idx.float() / (self.num_of_action - 1)
                ).unsqueeze(-1)
            else:
                actions_scaled = torch.clamp(
                    action, min=self.action_range[0], max=self.action_range[1]
                )
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(actions_scaled)
            done = terminated | truncated
            
            # Store transition
            states.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            # Track episode returns
            r = reward.squeeze(-1) if reward.dim() > 1 else reward
            self._episode_returns += r
            done_mask = done.bool().squeeze(-1) if done.dim() > 1 else done.bool()
            if done_mask.any():
                self.episode_durations.extend(self._episode_returns[done_mask].tolist())
                self._episode_returns[done_mask] = 0.0

            obs_tensor = next_obs["policy"].to(self.device)

        self._last_obs = obs_tensor

        # Get final value for bootstrapping
        with torch.no_grad():
            final_value = self.policy.evaluate(obs_tensor)
        
        return {
            'states': torch.stack(states),        # (T, N, obs_dim)
            'actions': torch.stack(actions),      # (T, N, action_dim)
            'rewards': torch.stack(rewards),      # (T, N, 1)
            'values': torch.stack(values),        # (T, N, 1)
            'log_probs': torch.stack(log_probs),  # (T, N)
            'dones': torch.stack(dones),          # (T, N, 1)
            'final_value': final_value,           # (N, 1)
        }

    # ------------------------------------------------------------------ #
    # Advantage Computation (TD-style)                                     #
    # ------------------------------------------------------------------ #

    def compute_advantages(self, rollout: dict) -> tuple:
        """
        Compute TD advantages and returns.
        
        A(s,a) = r + γ*V(s') - V(s)
        
        Args:
            rollout: Dictionary containing rollout data
            
        Returns:
            Tuple: (advantages, returns)
        """
        T, N = rollout['rewards'].shape[0], rollout['rewards'].shape[1]

        # Normalize all tensors to (T, N) — values/dones may be (T, N, 1)
        rewards     = rollout['rewards'].view(T, N)
        dones       = rollout['dones'].view(T, N)
        values      = rollout['values'].view(T, N)
        final_value = rollout['final_value'].view(N)

        advantages = torch.zeros(T, N, device=values.device)
        returns    = torch.zeros(T, N, device=values.device)

        for t in range(T):
            next_value = final_value if t == T - 1 else values[t + 1]

            # TD error: δ = r + γ*V(s')*(1-done) - V(s)
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.discount_factor * next_value * next_non_terminal - values[t]

            advantages[t] = delta
            returns[t]    = rewards[t] + self.discount_factor * next_value * next_non_terminal

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def calculate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict:
        """
        Compute actor, critic, and total losses for one AC update.

        Args:
            states (Tensor): Flattened state batch, shape (T*N, obs_dim).
            actions (Tensor): Flattened action batch, shape (T*N, action_dim).
            advantages (Tensor): Normalized advantages, shape (T*N,).
            returns (Tensor): Target returns, shape (T*N, 1).

        Returns:
            dict: {"total_loss", "actor_loss", "critic_loss", "entropy"} — all Tensors.
        """
        self.policy._update_distribution(states)
        log_probs = self.policy.get_actions_log_prob(actions)
        values    = self.policy.evaluate(states)
        entropy   = self.policy.entropy.mean()

        actor_loss  = -(log_probs * advantages).mean()
        critic_loss = ((values - returns) ** 2).mean()
        total_loss  = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        return {
            "total_loss":  total_loss,
            "actor_loss":  actor_loss,
            "critic_loss": critic_loss,
            "entropy":     entropy,
        }

    def update_policy(self, rollout: dict) -> dict:
        """
        Update policy using collected rollout.
        
        Args:
            rollout: Dictionary containing rollout data
            
        Returns:
            dict: Loss statistics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rollout)
        
        # Flatten batch dimensions (T, N) -> (T*N,)
        states = rollout['states'].view(-1, rollout['states'].shape[-1])
        actions = rollout['actions'].view(-1, rollout['actions'].shape[-1])
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1, 1)
        
        losses = self.calculate_loss(states, actions, advantages_flat, returns_flat)

        # Backpropagation
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss":  losses["total_loss"].item(),
            "actor_loss":  losses["actor_loss"].item(),
            "critic_loss": losses["critic_loss"].item(),
            "entropy":     losses["entropy"].item(),
        }

    # ------------------------------------------------------------------ #
    # Training Loop                                                        #
    # ------------------------------------------------------------------ #

    def learn(self, env, num_envs: int, num_transitions_per_env: int) -> tuple:
        """
        Collect ONE rollout and perform ONE AC update.

        Call this repeatedly from an external loop (train.py).

        Returns:
            Tuple[float, int]: (avg_return_last_100, steps_collected)
        """
        self.policy.train()

        # One-time init on first call
        if not hasattr(self, '_last_obs') or self._last_obs is None:
            obs, _ = env.reset()
            self._last_obs        = obs["policy"].to(self.device)
            self._episode_returns = torch.zeros(num_envs, device=self.device)
            if not hasattr(self, '_loss_history'):
                self._loss_history = []

        rollout = self.collect_rollout(env, num_envs, num_transitions_per_env)

        losses = self.update_policy(rollout)
        if not hasattr(self, '_loss_history'):
            self._loss_history = []
        self._loss_history.append(losses)

        avg_return = (
            sum(self.episode_durations[-100:]) / min(100, len(self.episode_durations))
            if self.episode_durations else 0.0
        )

        self._episode_stats = {
            'value_loss':           losses.get('critic_loss', 0.0),
            'policy_gradient_loss': losses.get('actor_loss',  0.0),
            'ep_rew_mean':          avg_return,
            'ep_len_mean':          avg_return,
        }

        return avg_return, num_envs * num_transitions_per_env

    # ------------------------------------------------------------------ #
    # Interface methods                                                    #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions for parallel envs."""
        return self.policy.act(obs)

    def process_env_step(self, rewards, dones) -> None:
        """Not used in this rollout-based implementation."""
        pass

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        self.policy.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs = obs.to(self.device)
            action = self.policy.act_inference(obs)
            
            if self.action_type == "discrete":
                action_idx = int(action.item())
                return self.scale_action(action_idx)
            else:
                return torch.clamp(
                    action.squeeze(0),
                    min=self.action_range[0],
                    max=self.action_range[1]
                )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """Save actor-critic weights."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_durations': self.episode_durations,
        }, filepath)
        
        print(f"✅ AC model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """Load actor-critic weights."""
        filepath = os.path.join(path, filename)
        
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.policy.eval()
        
        print(f"✅ AC model loaded from {filepath}")