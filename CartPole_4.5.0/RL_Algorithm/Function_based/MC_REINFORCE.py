from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from RL_Algorithm.RL_base_function import BaseAlgorithm


# ============================================================ #
# ==================== Policy Network ======================== #
# ============================================================ #

class MC_REINFORCE_network(nn.Module):
    """
    Policy network for the MC REINFORCE algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons per layer.
        n_actions (int): Number of output values.
        dropout (float): Dropout rate for regularization.
        action_type (str): ``'discrete'`` or ``'continuous'``.
    """

    def __init__(
        self,
        n_observations: int,
        hidden_size: int,
        n_actions: int,
        dropout: float,
        action_type: str = "discrete",
    ):
        super(MC_REINFORCE_network, self).__init__()

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        self.action_type = action_type

        # Shared MLP body
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, n_actions)

        # Learnable log_std (continuous only)
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        if self.action_type == "discrete":
            logits = self.fc3(x)
            return logits
        else:
            mean = self.fc3(x)
            return mean


# ============================================================ #
# =================== MC REINFORCE Agent ===================== #
# ============================================================ #

class MC_REINFORCE(BaseAlgorithm):
    """
    Monte-Carlo REINFORCE for PARALLEL environments.
    
    Collects complete episodes from multiple parallel environments,
    handling episodes that finish at different times.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        action_type (str): ``'discrete'`` or ``'continuous'``.
        learning_rate (float): AdamW learning rate.
        discount_factor (float): Discount factor γ.
        max_episode_length (int): Maximum steps per episode (for memory allocation).
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 128,
            dropout: float = 0.1,
            action_type: str = "discrete",
            learning_rate: float = 1e-3,
            discount_factor: float = 0.99,
            max_episode_length: int = 500,
    ) -> None:

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        self.action_type = action_type
        self.LR = learning_rate
        self.max_episode_length = max_episode_length

        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Distribution helpers                                                 #
    # ------------------------------------------------------------------ #

    def _get_distribution(self, obs: torch.Tensor):
        """Build the action distribution from the current observation."""
        if self.action_type == "discrete":
            logits = self.policy_net(obs)
            dist = Categorical(logits=logits)
            return dist
        else:
            mean = self.policy_net(obs)
            std = torch.exp(self.policy_net.log_std)
            dist = Normal(mean, std)
            return dist

    def _sample_action(self, dist):
        """Sample an action from the distribution and compute its log-probability."""
        if self.action_type == "discrete":
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

    # ------------------------------------------------------------------ #
    # Trajectory Collection (Parallel)                                     #
    # ------------------------------------------------------------------ #

    def collect_trajectories(self, env, num_agents: int, max_steps: int):
        """
        Collect complete episodes from parallel environments.
        
        Args:
            env: Vectorized environment
            num_agents (int): Number of parallel environments
            max_steps (int): Maximum steps to collect
            
        Returns:
            List of completed episodes, each containing:
            (states, actions, rewards, log_probs)
        """
        # Storage for each parallel environment
        env_states = [[] for _ in range(num_agents)]
        env_actions = [[] for _ in range(num_agents)]
        env_rewards = [[] for _ in range(num_agents)]
        env_log_probs = [[] for _ in range(num_agents)]
        
        completed_episodes = []
        episode_returns = np.zeros(num_agents)
        
        obs, _ = env.reset()
        obs_tensor = obs["policy"].to(self.device)
        
        for step in range(max_steps):
            # Get actions for all envs
            with torch.no_grad():
                dist = self._get_distribution(obs_tensor)
                actions, log_probs = self._sample_action(dist)
            
            # Convert actions to environment format
            if self.action_type == "discrete":
                actions_scaled = torch.stack([
                    self.scale_action(int(a.item()))
                    for a in actions
                ]).unsqueeze(-1).to(self.device)
            else:
                actions_scaled = torch.clamp(
                    actions,
                    min=self.action_range[0],
                    max=self.action_range[1]
                )
            
            # Step all environments
            next_obs, rewards, terminated, truncated, _ = env.step(actions_scaled)
            dones = terminated | truncated
            
            # Store transitions for each env
            for i in range(num_agents):
                env_states[i].append(obs_tensor[i].cpu())
                env_actions[i].append(actions[i].cpu())
                env_rewards[i].append(float(rewards[i].item()))
                env_log_probs[i].append(log_probs[i].cpu())
                episode_returns[i] += rewards[i].item()
                
                # Check if episode completed
                if dones[i]:
                    # Save completed episode
                    completed_episodes.append({
                        'states': env_states[i],
                        'actions': env_actions[i],
                        'rewards': env_rewards[i],
                        'log_probs': env_log_probs[i],
                        'return': episode_returns[i],
                    })
                    
                    # Reset storage for this env
                    env_states[i] = []
                    env_actions[i] = []
                    env_rewards[i] = []
                    env_log_probs[i] = []
                    episode_returns[i] = 0.0
            
            obs_tensor = next_obs["policy"].to(self.device)
        
        return completed_episodes

    # ------------------------------------------------------------------ #
    # Returns Calculation                                                  #
    # ------------------------------------------------------------------ #

    def calculate_returns(self, rewards: list) -> torch.Tensor:
        """
        Compute Monte Carlo returns G_t for each timestep (unnormalized).

        Args:
            rewards (list): Rewards collected in the episode.

        Returns:
            Tensor: Returns of shape (T,).
        """
        T = len(rewards)
        returns = torch.zeros(T, device=self.device)

        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.discount_factor * G
            returns[t] = G

        return returns

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def calculate_loss(
        self,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the REINFORCE policy gradient loss.

        Args:
            log_probs (Tensor): Log-probabilities of selected actions, shape (N,).
            returns (Tensor): Normalized Monte Carlo returns, shape (N,).

        Returns:
            Tensor: Scalar loss = -E[log π(a|s) · G_t].
        """
        return -(log_probs * returns).mean()

    def update_policy(self, episodes: list) -> dict:
        """
        Update policy using collected episodes.

        Normalizes returns across the full batch so the agent can distinguish
        between short and long episodes (cross-episode signal preserved).

        Args:
            episodes: List of completed episodes

        Returns:
            dict: Loss statistics
        """
        if len(episodes) == 0:
            return {'loss': 0.0, 'num_episodes': 0}

        all_log_probs = []
        all_returns   = []

        for episode in episodes:
            states  = torch.stack(episode['states']).to(self.device)
            actions = torch.stack(episode['actions']).to(self.device)

            # Unnormalized returns — normalization happens across the full batch
            returns = self.calculate_returns(episode['rewards'])

            # Recompute log-probs with current policy (for gradient)
            dist = self._get_distribution(states)
            if self.action_type == "discrete":
                log_probs = dist.log_prob(actions)
            else:
                log_probs = dist.log_prob(actions).sum(dim=-1)

            all_log_probs.append(log_probs)
            all_returns.append(returns)

        # Concatenate across all episodes and normalize once
        all_log_probs = torch.cat(all_log_probs)
        all_returns   = torch.cat(all_returns)
        all_returns   = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

        # Single backward pass over the full batch
        loss = self.calculate_loss(all_log_probs, all_returns)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'num_episodes': len(episodes),
        }

    # ------------------------------------------------------------------ #
    # Training Loop                                                        #
    # ------------------------------------------------------------------ #

    def learn(self, env, max_steps: int, num_agents: int = 1):
        """
        Train the agent with parallel environments.
        
        Args:
            env: Vectorized Isaac Lab environment
            max_steps (int): Number of steps to collect
            num_agents (int): Number of parallel environments
            
        Returns:
            Tuple: (average_return, loss, num_episodes)
        """
        self.policy_net.train()
        
        # Collect complete episodes from parallel envs
        episodes = self.collect_trajectories(env, num_agents, max_steps)
        
        # Update policy if we have any completed episodes
        stats = self.update_policy(episodes)
        
        # Calculate statistics
        if len(episodes) > 0:
            avg_return = np.mean([ep['return'] for ep in episodes])
            self.episode_durations.append(avg_return)
        else:
            avg_return = 0.0
        
        return avg_return, stats['loss'], stats['num_episodes']

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation (greedy / mean)."""
        self.policy_net.eval()
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs = obs.to(self.device)
            if self.action_type == "discrete":
                logits = self.policy_net(obs)
                action_idx = int(logits.argmax(dim=-1).item())
                return self.scale_action(action_idx)
            else:
                mean = self.policy_net(obs)
                return torch.clamp(
                    mean.squeeze(0),
                    min=self.action_range[0],
                    max=self.action_range[1],
                )

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """Save policy network weights to disk."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_durations': self.episode_durations,
        }, filepath)
        
        print(f"✅ MC_REINFORCE model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """Load policy network weights from disk."""
        filepath = os.path.join(path, filename)
        
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.policy_net.eval()
        
        print(f"✅ MC_REINFORCE model loaded from {filepath}")