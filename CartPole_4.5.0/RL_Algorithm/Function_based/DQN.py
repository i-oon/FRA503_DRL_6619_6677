from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from storage.off_policy import OffPolicyAlgorithm


class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state tensor.

        Returns:
            Tensor: Q-value estimates for each action.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN) — off-policy, value-based.

    Args:
        device: Torch device.
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        learning_rate (float): Adam learning rate.
        tau (float): Polyak soft-update coefficient for target network.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            learning_rate: float = None,
            tau: float = None,
            initial_epsilon: float = None,
            epsilon_decay: float = None,
            final_epsilon: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
    ) -> None:

        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device        = device
        self.steps_done    = 0
        self.num_of_action = num_of_action
        self.tau           = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state: Current state (numpy array).

        Returns:
            Tuple[Tensor, int]: Scaled action tensor and action index.
        """
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_of_action)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = int(q_values.argmax(dim=1).item())
        
        scaled_action = self.scale_action(action_idx)
        return scaled_action, action_idx

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Compute the Bellman loss for a sampled mini-batch.

        Args:
            non_final_mask (Tensor): True where next state is not terminal.
            non_final_next_states (Tensor): Non-terminal next states.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of action indices.
            reward_batch (Tensor): Batch of rewards.

        Returns:
            Tensor: Scalar Huber / MSE loss.
        """
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = reward_batch + (self.discount_factor * next_state_values)
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        return loss

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack it into DQN-ready tensors.

        Returns:
            Tuple or None:
                - non_final_mask (Tensor)
                - non_final_next_states (Tensor)
                - state_batch (Tensor)
                - action_batch (Tensor)
                - reward_batch (Tensor)
            Returns None if the buffer is not ready.
        """
        batch = super().generate_sample()
        if batch is None:
            return None

        states = torch.FloatTensor([t.state for t in batch]).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        
        non_final_mask = torch.BoolTensor([not t.done for t in batch]).to(self.device)
        non_final_next_states = torch.FloatTensor(
            [t.next_state for t in batch if not t.done]
        ).to(self.device)
        
        return non_final_mask, non_final_next_states, states, actions, rewards

    def update_policy(self):
        """Perform one gradient step on the policy network."""
        sample = self.generate_sample()
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_networks(self):
        """Soft update of target network."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                self.tau * policy_net_state_dict[key] + 
                (1 - self.tau) * target_net_state_dict[key]
            )
        
        self.target_net.load_state_dict(target_net_state_dict)

    def learn(self, env, max_steps: int, num_agents: int = 1):
        """
        Train the agent for fixed steps with parallel environments.

        Args:
            env: The Isaac Lab environment.
            num_agents (int): Number of parallel environments.
            max_steps (int): Total training steps.

        Returns:
            Tuple[float, int]: (average_return, timestep)
        """
        self.policy_net.train()
        
        obs, _ = env.reset()
        episode_rewards = np.zeros(num_agents)
        total_return = 0.0
        episodes_done = 0

        for step in range(max_steps):
            # Select actions for all environments
            actions = []
            action_indices = []
            for i in range(num_agents):
                obs_np = obs["policy"][i].cpu().numpy()
                scaled_action, action_idx = self.select_action(obs_np)
                actions.append(scaled_action)
                action_indices.append(action_idx)

            action_tensor = torch.cat(actions, dim=0)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done_flags = terminated | truncated

            # Store transitions and update for each environment
            for i in range(num_agents):
                obs_np = obs["policy"][i].cpu().numpy()
                next_obs_np = next_obs["policy"][i].cpu().numpy()
                r = float(reward[i].item())
                done = bool(done_flags[i].item())
                episode_rewards[i] += r

                # Store in replay buffer
                self.store_transition(obs_np, action_indices[i], r, next_obs_np, done)

                if done:
                    total_return += episode_rewards[i]
                    episodes_done += 1
                    episode_rewards[i] = 0.0

            # Update policy
            self.update_policy()
            self.update_target_networks()
            self.decay_epsilon()
            
            obs = next_obs

        avg_return = total_return / max(1, episodes_done)
        self.episode_durations.append(avg_return)
        return avg_return, max_steps

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """Save policy network weights."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"✅ DQN model saved to {filepath}")

    def load_model(self, path: str, filename: str) -> None:
        """Load policy network weights and sync to target network."""
        filepath = os.path.join(path, filename)
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        print(f"✅ DQN model loaded from {filepath}")