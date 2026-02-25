import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

# Example: Initialize the agent
agent = BaseAlgorithm(
    control_type=ControlType.Q_LEARNING,
    num_of_action=5,
    action_range=[-10, 10],
    discretize_state_weight=[1, 10, 1, 10],
    learning_rate=0.1,
    initial_epsilon=1.0,
    epsilon_decay=0.99,
    final_epsilon=0.1,
    discount_factor=0.9
)

# Test 1: Test `discretize_state()`
obs = {'policy': np.array([0.1, 0.2, -0.3, 0.4])}  # Sample observation
discretized_state = agent.discretize_state(obs)
print("Discretized State:", discretized_state)
assert isinstance(discretized_state, tuple), "discretize_state failed"
assert all(isinstance(i, int) for i in discretized_state), "discretize_state contains non-integer values"

# Test 2: Test `get_discretize_action()`
obs_dis = (0, 2, -1, 4)  # Example discretized state
agent.q_values[obs_dis] = np.array([1, 2, 3, 4, 5])  # Fake Q-values
agent.epsilon = 0.0  # Pure exploitation
action_idx = agent.get_discretize_action(obs_dis)
print("Action Index (Exploitation):", action_idx)
assert action_idx == 4, "Exploitation failed"

# Test 3: Test `mapping_action()`
action_idx = 2  # Choose middle action
action = agent.mapping_action(action_idx)
print("Mapped Action (Multiple Actions):", action.item())

# Test 4: Test `decay_epsilon()`
agent.epsilon = agent.initial_epsilon
for _ in range(10):
    agent.decay_epsilon()
print(f"Epsilon after 10 decays: {agent.epsilon}")
# Assert that epsilon hasn't increased and is within expected bounds
assert agent.epsilon <= agent.initial_epsilon, f"Epsilon increased after decay: {agent.epsilon}"
assert agent.epsilon >= agent.final_epsilon, f"Epsilon dropped below final_epsilon: {agent.epsilon}"