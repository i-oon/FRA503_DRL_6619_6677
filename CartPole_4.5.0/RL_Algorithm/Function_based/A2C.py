from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from ..storage.on_policy import OnPolicyAlgorithm
from ..storage.buffers import RolloutBuffer
from ..networks.mlp import MLP

class ActorCritic_A2C(nn.Module):
    """
    Shared Actor-Critic network for A2C.

    Args:
        state_dim (int): Observation space dimension.
        action_dim (int): Action space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
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
        super(ActorCritic_A2C, self).__init__()

        assert action_type in ("continuous", "discrete"), \
            f"action_type must be 'continuous' or 'discrete', got '{action_type}'"

        # 🔧 FIXED: Was "self.action_type = action_dim" (WRONG!)
        self.action_type = action_type  # ✅ Now correctly stores the action type string
        self.action_dim = action_dim

        # Actor and Critic networks
        self.actor = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1, hidden_dims, activation)

        # Learnable log_std for continuous actions
        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        self.distribution = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use act() or evaluate().")

    @property
    def action_mean(self):
        if self.action_type == "continuous":
            return self.distribution.mean
        return self.distribution.probs

    @property
    def action_std(self):
        if self.action_type == "continuous":
            return self.distribution.stddev
        return torch.ones_like(self.distribution.probs)

    @property
    def entropy(self):
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

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
        """Sample an action from the current distribution."""
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


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) — synchronous on-policy for parallel training.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
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
        action_range: list = [-3.0, 3.0],
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

        self.policy = ActorCritic_A2C(
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

        super(A2C, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection                                                   #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions for all parallel envs and populate self.transition."""
        actions = self.policy.act(obs)
        values = self.policy.evaluate(obs)
        log_probs = self.policy.get_actions_log_prob(actions)
        
        self.transition.observations = obs
        self.transition.actions = actions
        self.transition.values = values
        self.transition.actions_log_prob = log_probs
        # mu/sigma must match actions_shape=(1,) for discrete; probs has shape (N, num_actions)
        if self.action_type == "discrete":
            self.transition.action_mean  = torch.zeros(obs.shape[0], 1, device=self.device)
            self.transition.action_sigma = torch.ones(obs.shape[0], 1, device=self.device)
        else:
            self.transition.action_mean  = self.policy.action_mean
            self.transition.action_sigma = self.policy.action_std
        
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Record rewards and dones into self.transition, then flush to storage."""
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        self.transition.rewards = rewards
        self.transition.dones = dones
        
        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation (TD-based)                            #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """Compute one-step TD advantages and returns over the rollout."""
        with torch.no_grad():
            last_values = self.policy.evaluate(last_obs)
        
        advantages = torch.zeros_like(self.storage.rewards)
        T = self.storage.num_transitions_per_env
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[t + 1]
            
            # TD delta: δ = r + γ·V(s')·(1-done) - V(s)
            next_non_terminal = 1.0 - self.storage.dones[t].float()
            delta = (
                self.storage.rewards[t]
                + self.discount_factor * next_values * next_non_terminal
                - self.storage.values[t]
            )
            
            # A2C: advantage = delta (no lambda accumulation)
            advantages[t] = delta
        
        # Returns: R = A + V(s)
        returns = advantages + self.storage.values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.storage.advantages[:] = advantages
        self.storage.returns[:] = returns

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def calculate_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple:
        """
        Compute A2C actor, critic, and total losses.

        Args:
            obs (Tensor): Flattened observations, shape (T*N, obs_dim).
            actions (Tensor): Flattened actions, shape (T*N, action_dim).
            advantages (Tensor): Normalized advantages, shape (T*N,).
            returns (Tensor): Target returns, shape (T*N, 1).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: (total_loss, actor_loss, critic_loss, entropy).
        """
        self.policy._update_distribution(obs)
        log_probs = self.policy.get_actions_log_prob(actions)
        values    = self.policy.evaluate(obs)
        entropy   = self.policy.entropy.mean()

        actor_loss  = -(log_probs * advantages).mean()
        critic_loss = ((values - returns) ** 2).mean()
        total_loss  = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        return total_loss, actor_loss, critic_loss, entropy

    def update(self) -> dict:
        """Perform a single A2C update over the collected rollout."""
        # Flatten rollout tensors (T, N, ...) -> (T*N, ...)
        T, N = self.storage.observations.shape[:2]
        obs_flat = self.storage.observations.view(T * N, -1)
        actions_flat = self.storage.actions.view(T * N, -1)
        advantages_flat = self.storage.advantages.view(T * N)
        returns_flat = self.storage.returns.view(T * N, 1)
        
        loss, actor_loss, critic_loss, entropy = self.calculate_loss(
            obs_flat, actions_flat, advantages_flat, returns_flat
        )

        # Gradient step with clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.storage.clear()
        
        return {
            "value": critic_loss.item(),
            "actor": actor_loss.item(),
            "entropy": entropy.item(),
        }

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(self, env, num_envs: int, num_transitions_per_env: int) -> tuple:
        """
        Collect ONE rollout and perform ONE A2C update.

        Call this repeatedly from an external loop (train.py).

        Returns:
            Tuple[float, int]: (avg_return_last_100, steps_collected)
        """
        self.policy.train()

        # One-time storage initialisation on first call
        if self.storage is None:
            obs, _ = env.reset()
            obs_shape = (obs["policy"].shape[-1],)
            actions_shape = (self.num_of_action,) if self.action_type == "continuous" else (1,)
            self._init_storage(
                num_envs=num_envs,
                num_transitions_per_env=num_transitions_per_env,
                obs_shape=obs_shape,
                actions_shape=actions_shape,
                device=self.device,
            )
            self._last_obs        = obs["policy"].to(self.device)
            self._episode_returns = torch.zeros(num_envs, device=self.device)
            self._loss_history    = []

        obs_tensor = self._last_obs

        # ── Collect rollout ───────────────────────────────────────────── #
        for _ in range(num_transitions_per_env):
            with torch.no_grad():
                actions = self.act(obs_tensor)

            if self.action_type == "discrete":
                action_idx    = actions.squeeze(-1).long()
                min_a, max_a  = self.action_range
                actions_scaled = (
                    min_a + (max_a - min_a) * action_idx.float() / (self.num_of_action - 1)
                ).unsqueeze(-1)
            else:
                actions_scaled = torch.clamp(
                    actions, min=self.action_range[0], max=self.action_range[1]
                )

            next_obs, rewards, terminated, truncated, _ = env.step(actions_scaled)
            dones = terminated | truncated

            self.process_env_step(rewards, dones)

            self._episode_returns += rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            done_mask = dones.bool().squeeze(-1) if dones.dim() > 1 else dones.bool()
            if done_mask.any():
                self.episode_durations.extend(self._episode_returns[done_mask].tolist())
                self._episode_returns[done_mask] = 0.0

            obs_tensor = next_obs["policy"].to(self.device)

        self._last_obs = obs_tensor

        # ── Update ────────────────────────────────────────────────────── #
        self.compute_returns(obs_tensor)
        losses = self.update()
        self._loss_history.append(losses)

        avg_return = (
            sum(self.episode_durations[-100:]) / min(100, len(self.episode_durations))
            if self.episode_durations else 0.0
        )

        self._episode_stats = {
            'value_loss':           losses['value'],
            'policy_gradient_loss': losses['actor'],
            'ep_rew_mean':          avg_return,
            'ep_len_mean':          avg_return,
        }

        return avg_return, num_envs * num_transitions_per_env

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        self.policy.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs = obs.to(self.device)
            action = self.policy.act_inference(obs)
            if self.action_type == "discrete":
                return self.scale_action(int(action.item()))
            else:
                return torch.clamp(
                    action.squeeze(0),
                    min=self.action_range[0],
                    max=self.action_range[1],
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
        
        print(f"✅ A2C model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """Load actor-critic weights."""
        filepath = os.path.join(path, filename)
        
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.policy.eval()
        
        print(f"✅ A2C model loaded from {filepath}")