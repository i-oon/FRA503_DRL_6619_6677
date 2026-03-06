# scripts/RL_Algorithm/config.py
# Enhanced version with 4-value Sensitivity Analysis
ALGORITHM = "SARSA"
TASK = "Stabilize"

EXPERIMENTAL_MODE = False  
TEST_HYPERPARAMETER = "epsilon_decay"
EXPERIMENT_ID = "exp3"

NUM_OF_ACTION = 4
ACTION_RANGE = [-2.5, 2.5]
DISCRETIZE_STATE_WEIGHT = [2, 10, 2, 10]
N_TEST_EPISODES = 10

ALGORITHM_CONFIGS = {
    "Q_Learning": {
        "LEARNING_RATE": 0.08,
        "DISCOUNT_FACTOR": 0.99,
        "START_EPSILON": 1.0,
        "EPSILON_DECAY": 0.9995,
        "FINAL_EPSILON": 0.15,
        "N_EPISODES": 40000,
    },
    "SARSA": {
        "LEARNING_RATE": 0.08,
        "DISCOUNT_FACTOR": 0.99,
        "START_EPSILON": 1.0,
        "EPSILON_DECAY": 0.9997,
        "FINAL_EPSILON": 0.15,
        "N_EPISODES": 40000,
    },
    "Double_Q_Learning": {
        "LEARNING_RATE": 0.08,
        "DISCOUNT_FACTOR": 0.99,
        "START_EPSILON": 1.0,
        "EPSILON_DECAY": 0.9997,
        "FINAL_EPSILON": 0.15,
        "N_EPISODES": 40000,
    },
    "Monte_Carlo": {
        "LEARNING_RATE": 0.08,
        "DISCOUNT_FACTOR": 0.99,
        "START_EPSILON": 1.0,
        "EPSILON_DECAY": 0.9995,
        "FINAL_EPSILON": 0.15,
        "N_EPISODES": 40000,
    }
}

EXPERIMENTAL_EPISODES = 25000

HYPERPARAMETER_RANGES = {
    "learning_rate": {
        "values": [0.03, 0.08, 0.15, 0.25],
        "description": "Learning rate (α) - controls update step size",
        "test_algorithms": ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"],
    },
    "discount_factor": {
        "values": [0.5, 0.8, 0.95, 0.99],
        "description": "Discount factor (γ) - planning horizon",
        "test_algorithms": ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"],
    },
    "epsilon_decay": {
        "values": [0.995, 0.998, 0.9995, 0.9999],
        "description": "Epsilon decay rate - exploration strategy",
        "test_algorithms": ["SARSA"],
    },
}

CURRENT_TEST = HYPERPARAMETER_RANGES.get(TEST_HYPERPARAMETER, {})
TEST_VALUES = CURRENT_TEST.get("values", [])

def get_experiment_config(algorithm_name, test_param, test_value):
    config = ALGORITHM_CONFIGS[algorithm_name].copy()  # ✅ แก้แล้ว
    config["N_EPISODES"] = EXPERIMENTAL_EPISODES
    param_map = {
        "learning_rate": "LEARNING_RATE",
        "discount_factor": "DISCOUNT_FACTOR",
        "epsilon_decay": "EPSILON_DECAY",
    }
    config_key = param_map.get(test_param)
    if config_key:
        config[config_key] = test_value
    return config

if EXPERIMENTAL_MODE:
    import os
    test_value_str = os.environ.get('TEST_VALUE', None)
    if test_value_str is not None:
        try:
            TEST_VALUE = float(test_value_str)
        except ValueError:
            TEST_VALUE = TEST_VALUES[0] if TEST_VALUES else 0.08
    else:
        TEST_VALUE = TEST_VALUES[0] if TEST_VALUES else 0.08
    
    EXPERIMENT_CONFIGS = {}
    for algo in ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"]:
        EXPERIMENT_CONFIGS[algo] = get_experiment_config(algo, TEST_HYPERPARAMETER, TEST_VALUE)
    ALGORITHM_CONFIGS = EXPERIMENT_CONFIGS  
    
    print(f"\n🔬 EXPERIMENTAL MODE")
    print(f"   Testing: {TEST_HYPERPARAMETER} = {TEST_VALUE}")
    print(f"   Episodes: {EXPERIMENTAL_EPISODES}")
    print(f"   Experiment ID: {EXPERIMENT_ID}")
else:
    print(f"\n✅ BASELINE MODE (Optimized)")

if ALGORITHM in ALGORITHM_CONFIGS:
    config = ALGORITHM_CONFIGS[ALGORITHM]
    LEARNING_RATE = config["LEARNING_RATE"]
    DISCOUNT_FACTOR = config["DISCOUNT_FACTOR"]
    START_EPSILON = config["START_EPSILON"]
    EPSILON_DECAY = config["EPSILON_DECAY"]
    FINAL_EPSILON = config["FINAL_EPSILON"]
    N_EPISODES = config["N_EPISODES"]
    
    if EXPERIMENTAL_MODE:
        print(f"   Algorithm: {ALGORITHM}")
        print(f"   α={LEARNING_RATE}, γ={DISCOUNT_FACTOR}, ε_decay={EPSILON_DECAY}")
    else:
        print(f"   Loaded config for {ALGORITHM}")
else:
    LEARNING_RATE = 0.08
    DISCOUNT_FACTOR = 0.99
    START_EPSILON = 1.0
    EPSILON_DECAY = 0.9997
    FINAL_EPSILON = 0.05
    N_EPISODES = 50000

def get_algorithm():
    return ALGORITHM

def get_task():
    return TASK

def get_experiment_suffix():
    if not EXPERIMENTAL_MODE:
        return ""
    param_short = {
        "learning_rate": "lr",
        "discount_factor": "gamma",
        "epsilon_decay": "eps",
    }
    short_name = param_short.get(TEST_HYPERPARAMETER, "test")
    value_str = str(TEST_VALUE).replace(".", "p")
    return f"_{EXPERIMENT_ID}_{short_name}_{value_str}"

def get_config(algorithm_name: str = None) -> dict:
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
        "experimental_mode": EXPERIMENTAL_MODE,
        "experiment_suffix": get_experiment_suffix(),
    }
    if algorithm_name in ALGORITHM_CONFIGS:
        config.update(ALGORITHM_CONFIGS[algorithm_name])
    return config

def print_config(algorithm_name: str = None):
    if algorithm_name is None:
        algorithm_name = ALGORITHM
    config = get_config(algorithm_name)
    print("\n" + "="*70)
    if EXPERIMENTAL_MODE:
        print(f"🔬 EXPERIMENTAL CONFIGURATION")
        print(f"   Experiment: {EXPERIMENT_ID}")
        print(f"   Testing: {TEST_HYPERPARAMETER} = {TEST_VALUE}")
    else:
        print(f"✅ BASELINE CONFIGURATION")
    print("="*70)
    print(f"Algorithm:       {config['algorithm_name']}")
    print(f"Task:            {config['task_name']}")
    print(f"Episodes:        {config['n_episodes']}")
    print(f"Learning Rate:   {config['learning_rate']}")
    print(f"Discount Factor: {config['discount_factor']}")
    print(f"Epsilon Decay:   {config['epsilon_decay']}")
    print(f"Final Epsilon:   {config['final_epsilon']}")
    print(f"Actions:         {config['num_of_action']} (range: {config['action_range']})")
    print(f"Discretization:  {config['discretize_state_weight']}")
    if EXPERIMENTAL_MODE:
        print(f"Output Suffix:   {get_experiment_suffix()}")
    print("="*70 + "\n")

def get_agent_class(algorithm_name: str = None):
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
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return algorithm_map[algorithm_name]

def create_agent(algorithm_name: str = None, testing: bool = False):
    if algorithm_name is None:
        algorithm_name = ALGORITHM
    config = get_config(algorithm_name)
    AgentClass = get_agent_class(algorithm_name)
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
