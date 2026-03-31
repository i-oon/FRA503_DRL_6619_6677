from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
from RL_Algorithm.Function_based.AC import ActorCritic
from ..storage.on_policy import OnPolicyAlgorithm

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) — on-policy, clipped surrogate.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        num_learning_epochs (int): Epochs per PPO update.
        num_mini_batches (int): Mini-batches per epoch.
        clip_param (float): PPO clipping ε.
        gamma (float): Discount factor γ.
        lam (float): GAE lambda λ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        learning_rate (float): Adam learning rate.
        max_grad_norm (float): Gradient clipping norm.
        desired_kl (float): KL target for adaptive LR (0 to disable; use 0 for discrete).
        normalize_advantage_per_mini_batch (bool): Normalise advantages per mini-batch.
        use_clipped_value_loss (bool): Apply clipped value loss.
    """

    def __init__(
        self,
        device=None,
        num_of_action: int = 2,
        action_range: list = [-3.0, 3.0],
        n_observations: int = 4,
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        action_type: str = "continuous",
        init_noise_std: float = 1.0,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 0.5,
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_clipped_value_loss: bool = True,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build ActorCritic network
        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.action_type                        = action_type
        self.clip_param                         = clip_param
        self.num_learning_epochs                = num_learning_epochs
        self.num_mini_batches                   = num_mini_batches
        self.value_loss_coef                    = value_loss_coef
        self.entropy_coef                       = entropy_coef
        self.gamma                              = gamma
        self.lam                                = lam
        self.max_grad_norm                      = max_grad_norm
        self.desired_kl                         = desired_kl
        self.learning_rate                      = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_clipped_value_loss             = use_clipped_value_loss

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection                                                   #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for all parallel envs and populate self.transition.

        Continuous: actions shape (num_envs, action_dim).
        Discrete  : actions shape (num_envs, 1).

        Args:
            obs (Tensor): shape (num_envs, obs_dim).

        Returns:
            Tensor: Sampled actions.
        """
        # Sample actions
        actions = self.policy.act(obs)
        
        # Get values and distribution stats
        values = self.policy.evaluate(obs)
        log_probs = self.policy.get_actions_log_prob(actions)
        
        # Store in transition container
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
        """
        Write rewards and dones into self.transition, then flush to storage.

        Args:
            rewards (Tensor): shape (num_envs,) or (num_envs, 1).
            dones (Tensor): shape (num_envs,) or (num_envs, 1).
        """
        # Ensure correct shape
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        self.transition.rewards = rewards
        self.transition.dones = dones
        
        # Flush transition into RolloutBuffer
        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation (GAE)                                 #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute GAE returns and advantages over the collected rollout.

        GAE formula:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...

        Args:
            last_obs (Tensor): Observation after the final rollout step.
                               Shape: (num_envs, obs_dim).
        """
        with torch.no_grad():
            # Get value of final observation
            last_values = self.policy.evaluate(last_obs)  # (num_envs, 1)
        
        # GAE computation
        advantages = torch.zeros_like(self.storage.rewards)  # (T, N, 1)
        last_advantage = 0.0
        
        T = self.storage.num_transitions_per_env
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            next_non_terminal = 1.0 - self.storage.dones[t].float()
            delta = (
                self.storage.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.storage.values[t]
            )
            
            # GAE: A_t = δ_t + γλ * A_{t+1} * (1 - done_t)
            advantages[t] = delta + self.gamma * self.lam * last_advantage * next_non_terminal
            last_advantage = advantages[t]
        
        # Returns: R_t = A_t + V(s_t)
        returns = advantages + self.storage.values
        
        # Store in buffer
        self.storage.advantages[:] = advantages
        self.storage.returns[:] = returns

    # ------------------------------------------------------------------ #
    # Policy Update (PPO Clipped Surrogate Objective)                     #
    # ------------------------------------------------------------------ #

    def calculate_loss(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_log_probs_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
    ) -> tuple:
        """
        Compute PPO clipped surrogate, value, and entropy losses for one mini-batch.

        Args:
            obs_batch (Tensor): Observations, shape (mini_batch_size, obs_dim).
            actions_batch (Tensor): Actions, shape (mini_batch_size, action_dim).
            advantages_batch (Tensor): Advantages, shape (mini_batch_size, 1).
            returns_batch (Tensor): Returns, shape (mini_batch_size, 1).
            old_log_probs_batch (Tensor): Old log-probs, shape (mini_batch_size, 1).
            target_values_batch (Tensor): Old value estimates for clipped value loss.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: (total_loss, surrogate_loss, value_loss, entropy).
        """
        self.policy._update_distribution(obs_batch)
        log_probs = self.policy.get_actions_log_prob(actions_batch)
        values    = self.policy.evaluate(obs_batch)
        entropy   = self.policy.entropy

        log_ratio = log_probs - old_log_probs_batch.squeeze(-1)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages_batch.squeeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch.squeeze(-1)
        surrogate_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = target_values_batch + torch.clamp(
                values - target_values_batch, -self.clip_param, self.clip_param
            )
            value_losses         = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss           = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = ((values - returns_batch) ** 2).mean()

        total_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        return total_loss, surrogate_loss, value_loss, entropy

    def update(self) -> dict:
        """
        Perform PPO updates over the collected rollout.

        Returns:
            dict: Mean losses {'value', 'surrogate', 'entropy'}.
        """
        mean_value_loss     = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy        = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            
            # Normalize advantages per mini-batch (optional)
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            loss, surrogate_loss, value_loss, entropy_batch = self.calculate_loss(
                obs_batch,
                actions_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                target_values_batch,
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            mean_value_loss     += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy        += entropy_batch.mean().item()

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates

        self.storage.clear()   # On-policy: discard rollout after update

        return {
            "value":     mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy":   mean_entropy,
        }

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(self, env, num_envs: int, num_transitions_per_env: int) -> tuple:
        """
        Collect ONE rollout and perform ONE PPO update.

        Call this repeatedly from an external loop (train.py) to get
        live progress, TensorBoard logging, and checkpoints per iteration.

        Args:
            env: Isaac Lab vectorised environment.
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon per env.

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
            'policy_gradient_loss': losses['surrogate'],
            'ep_rew_mean':          avg_return,
            'ep_len_mean':          avg_return,
        }

        return avg_return, num_envs * num_transitions_per_env

    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        Args:
            obs (Tensor): shape (1, obs_dim) or (obs_dim,).
        """
        self.policy.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
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

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_durations': self.episode_durations,
        }, filepath)
        
        print(f"✅ PPO model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        filepath = os.path.join(path, filename)
        
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.policy.eval()
        
        print(f"✅ PPO model loaded from {filepath}")