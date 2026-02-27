# scripts/RL_Algorithm/config.py

ALGORITHM = "Double_Q_Learning"  # Options: "Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"
TASK = "Stabilize"  # Options: "Stabilize", "SwingUp"


# Action Space
NUM_OF_ACTION = 5
ACTION_RANGE = [-10.0, 10.0]

# State Discretization (critical for performance!)
# Format: [cart_pos_weight, pole_angle_weight, cart_vel_weight, pole_vel_weight]
DISCRETIZE_STATE_WEIGHT = [1, 8, 1, 8]

# Learning Parameters
LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.99

# Exploration Parameters
START_EPSILON = 1.0
EPSILON_DECAY = 0.9995
FINAL_EPSILON = 0.1  

# Training Parameters
N_EPISODES = 5000

# Testing Parameters
N_TEST_EPISODES = 10  # How many episodes to run in play.py



ALGORITHM_CONFIGS = {
    "Q_Learning": {
        # Uses defaults above
    },
    
    "SARSA": {
        # Uses defaults above
    },
    
    "Double_Q_Learning": {
        # Uses defaults above
    },
    
    "Monte_Carlo": {
        # MC needs coarser discretization and more episodes
    }
}



def get_algorithm():
    """Get the currently selected algorithm name."""
    return ALGORITHM

def get_task():
    """Get the currently selected task name."""
    return TASK

def get_config(algorithm_name: str = None) -> dict:
    """
    Get configuration for a specific algorithm.
    If algorithm_name is None, uses the global ALGORITHM setting.
    Returns defaults + algorithm-specific overrides.
    """
    if algorithm_name is None:
        algorithm_name = ALGORITHM
    
    config = {
        "algorithm_name": algorithm_name,
        "task_name": TASK,
        "num_of_action": NUM_OF_ACTION,
        "action_range": ACTION_RANGE,
        "discretize_state_weight": DISCRETIZE_STATE_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "discount_factor": DISCOUNT_FACTOR,
        "start_epsilon": START_EPSILON,
        "epsilon_decay": EPSILON_DECAY,
        "final_epsilon": FINAL_EPSILON,
        "n_episodes": N_EPISODES,
        "n_test_episodes": N_TEST_EPISODES,
    }
    
    # Apply algorithm-specific overrides
    if algorithm_name in ALGORITHM_CONFIGS:
        config.update(ALGORITHM_CONFIGS[algorithm_name])
    
    return config

def print_config(algorithm_name: str = None):
    """Print configuration for an algorithm."""
    if algorithm_name is None:
        algorithm_name = ALGORITHM
        
    config = get_config(algorithm_name)
    
    print("\n" + "="*70)
    print(f"CONFIGURATION")
    print("="*70)
    print(f"Algorithm:       {config['algorithm_name']}")
    print(f"Task:            {config['task_name']}")
    print(f"Discretization:  {config['discretize_state_weight']}")
    print(f"Episodes:        {config['n_episodes']}")
    print(f"Learning Rate:   {config['learning_rate']}")
    print(f"Epsilon:         {config['start_epsilon']} → {config['final_epsilon']} (decay: {config['epsilon_decay']})")
    print(f"Actions:         {config['num_of_action']} (range: {config['action_range']})")
    print(f"Test Episodes:   {config['n_test_episodes']}")
    print("="*70 + "\n")

def get_agent_class(algorithm_name: str = None):
    """
    Get the agent class for the specified algorithm.
    Returns the class itself, not an instance.
    """
    if algorithm_name is None:
        algorithm_name = ALGORITHM

    from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
    from RL_Algorithm.Algorithm.SARSA import SARSA
    from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
    from RL_Algorithm.Algorithm.MC import MC
    
    algorithm_map = {
        "Q_Learning": Q_Learning,
        "SARSA": SARSA,
        "Double_Q_Learning": Double_Q_Learning,
        "Monte_Carlo": MC,
    }
    
    if algorithm_name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Choose from: {list(algorithm_map.keys())}")
    
    return algorithm_map[algorithm_name]

def create_agent(algorithm_name: str = None, testing: bool = False):
    """
    Create an agent with the correct configuration.
    
    Args:
        algorithm_name: Name of algorithm (uses ALGORITHM if None)
        testing: If True, sets epsilon=0 (no exploration)
    
    Returns:
        Configured agent instance
    """
    if algorithm_name is None:
        algorithm_name = ALGORITHM
    
    config = get_config(algorithm_name)
    AgentClass = get_agent_class(algorithm_name)
    
    # Override epsilon for testing
    if testing:
        start_epsilon = 0.0
        epsilon_decay = 1.0
        final_epsilon = 0.0
    else:
        start_epsilon = config['start_epsilon']
        epsilon_decay = config['epsilon_decay']
        final_epsilon = config['final_epsilon']
    
    agent = AgentClass(
        num_of_action=config['num_of_action'],
        action_range=config['action_range'],
        discretize_state_weight=config['discretize_state_weight'],
        learning_rate=config['learning_rate'],
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=config['discount_factor']
    )
    
    return agent