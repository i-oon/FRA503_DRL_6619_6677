import torch
from collections.abc import Generator
from tensordict import TensorDict

class RolloutStorage:

    class Transition:
        """Storage for a single state transition."""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states = (None, None)

        def clear(self) -> None:
            self.__init__()

        
    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape,
        actions_shape,
        device = "cpu"
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For reinforcement learning
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        
        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition):
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        
        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        self.step = 0 

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Create the mini-batch
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                )