# scripts/Function_based/config.py
# HW3 Configuration - Function Approximation Based Algorithms
# Supports: Linear_Q, DQN, MC_REINFORCE, A2C, PPO, SAC, TD3

import torch

# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================

# Algorithm selection - Change this to train different algorithms
ALGORITHM = "DQN"  # Options: Linear_Q, DQN, MC_REINFORCE, A2C, PPO, SAC, TD3

# Task configuration
TASK = "Stabilize-Isaac-Cartpole-v0"

# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Environment configuration
NUM_ENVS = 256  # Number of parallel environments (CRITICAL for HW3!)
NUM_OF_ACTION = 2  # Discrete action space for CartPole
ACTION_RANGE = [-2.5, 2.5]  # Continuous action range
N_OBSERVATIONS = 4  # CartPole observation space (position, velocity, angle, angular velocity)

# Training configuration
HEADLESS = True  # Run without GUI for faster training
SAVE_INTERVAL = 100  # Save model every N iterations
LOG_INTERVAL = 10  # Print stats every N iterations

# Testing configuration
N_TEST_EPISODES = 10

# Model save/load pathsฟ
MODEL_DIR = "models"
LOG_DIR = "logs"


# ===========================================================================
# ALGORITHM CONFIGURATIONS
# ===========================================================================

## ============================================================ ##
## FAIR COMPARISON BUDGET: ~10M total env interactions         ##
## Formula: 256 envs × steps_per_iter × num_iters             ##
## X-axis for TensorBoard: use "total_env_steps" not episodes  ##
## ============================================================ ##

ALGORITHM_CONFIGS = {
    # ========================================================================
    # VALUE-BASED ALGORITHMS (Discrete Actions)
    # ========================================================================
    "Linear_Q": {
        "description": "Linear Q-Learning with function approximation",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: env[0] only × 1000 steps × 10000 ep ≈ 10M
        "n_episodes": 10000,
        "max_steps": 1000,

        "learning_rate": 0.01,
        "initial_epsilon": 1.0,
        "epsilon_decay": 2e-4,
        "final_epsilon": 0.01,
        "discount_factor": 0.99,

        "expected_episodes_to_solve": 3000,
        "expected_final_return": 80,
        "expected_time_minutes": 15,
    },
    "DQN": {
        "description": "Deep Q-Network with experience replay",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: 256 × 200 steps × 200 ep = 10.24M
        "n_episodes": 200,
        "max_steps": 200,

        # Network architecture
        "hidden_dim": 256,
        "dropout": 0.1,

        # Algorithm hyperparameters
        "learning_rate": 3e-4,
        "tau": 0.005,
        "initial_epsilon": 1.0,
        "epsilon_decay": 2.5e-4,  # reaches 0.01 by ~step 4000 (ep 20)
        "final_epsilon": 0.01,
        "discount_factor": 0.99,

        # Replay buffer
        "buffer_size": 1_000_000,
        "batch_size": 256,

        # Update frequencies
        "update_freq": 4,
        "target_update_freq": 200,

        "expected_episodes_to_solve": 60,
        "expected_final_return": 490,
        "expected_time_minutes": 8,
    },

    # ========================================================================
    # POLICY-BASED ALGORITHMS
    # ========================================================================

    "MC_REINFORCE": {
        "description": "Monte Carlo REINFORCE policy gradient",
        "action_type": "discrete",
        "num_of_action": 5,

        # max_steps MUST exceed env max episode (1000) so complete episodes
        # can be collected. Budget: 256 × 1100 × 36 ≈ 10.1M
        "n_iterations": 36,
        "max_steps": 1100,

        # Network architecture (matched to other discrete algos)
        "hidden_dim": 256,
        "dropout": 0.1,

        # Algorithm hyperparameters — higher LR because only 36 gradient updates total
        "learning_rate": 3e-3,
        "discount_factor": 0.99,
        "max_episode_length": 1000,

        "expected_iterations_to_solve": 150,
        "expected_final_return": 445,
        "expected_time_minutes": 10,
    },

    # ========================================================================
    # ACTOR-CRITIC ALGORITHMS
    # ========================================================================

    "AC": {
        "description": "Actor-Critic (basic rollout-based, no GAE)",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: 256 × 20 × 2000 = 10.24M
        "max_iterations": 2000,
        "num_transitions_per_env": 20,

        # Network architecture
        "hidden_dims": [256, 256],
        "activation": "elu",
        "init_noise_std": 1.0,

        # Algorithm hyperparameters
        "learning_rate": 3e-4,
        "discount_factor": 0.99,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,

        "expected_iterations_to_solve": 500,
        "expected_final_return": 450,
        "expected_time_minutes": 12,
    },

    "A2C": {
        "description": "Advantage Actor-Critic (synchronous, no PPO clipping)",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: 256 × 20 × 2000 = 10.24M
        "max_iterations": 2000,
        "num_transitions_per_env": 20,

        # Network architecture
        "hidden_dims": [256, 256],
        "activation": "elu",
        "init_noise_std": 1.0,

        # Algorithm hyperparameters
        "learning_rate": 3e-4,
        "discount_factor": 0.99,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,

        "expected_iterations_to_solve": 400,
        "expected_final_return": 465,
        "expected_time_minutes": 10,
    },

    "PPO": {
        "description": "Proximal Policy Optimization",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: 256 × 20 × 2000 = 10.24M
        "max_iterations": 2000,
        "num_transitions_per_env": 20,

        # Network architecture
        "hidden_dims": [256, 256], #128,128
        "activation": "elu",
        "init_noise_std": 1.0,

        # PPO-specific hyperparameters
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,

        # Loss coefficients
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "learning_rate":3e-4,
        "max_grad_norm": 0.5,

        "expected_iterations_to_solve": 200,
        "expected_final_return": 490,
        "expected_time_minutes": 6,
    },

    "SAC": {
        "description": "Soft Actor-Critic for discrete actions (Christodoulou 2019)",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: 256 × 200 × 200 = 10.24M
        "n_episodes": 200,
        "max_steps": 200,

        # Network architecture
        "hidden_dim": 256,

        # SAC-specific hyperparameters
        "learning_rate": 1e-4,        # lower LR to prevent Q-value explosion
        "alpha_lr": 1e-4,
        "tau": 0.005,
        "discount_factor": 0.99,

        # Replay buffer
        "buffer_size": 1_000_000,
        "batch_size": 512,            # larger batch for more stable Q estimates

        "init_alpha": 0.2,
        "auto_alpha": True,
        "target_entropy": None,

        "expected_episodes_to_solve": 100,
        "expected_final_return": 480,
        "expected_time_minutes": 15,
    },

    "TD3": {
        "description": "Twin Delayed DDPG (discrete actions via expected Q-value)",
        "action_type": "discrete",
        "num_of_action": 5,

        # Budget: 256 × 200 × 200 = 10.24M
        "n_episodes": 200,
        "max_steps": 200,

        # Network architecture
        "hidden_dim": 256,

        # TD3-specific hyperparameters
        "learning_rate": 3e-4,
        "tau": 0.005,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "target_smoothing_temperature": 1.0,
        "policy_update_freq": 2,
        "discount_factor": 0.99,

        # Replay buffer
        "buffer_size": 1_000_000,
        "batch_size": 256,

        "expected_episodes_to_solve": 100,
        "expected_final_return": 485,
        "expected_time_minutes": 15,
    },
}


# ===========================================================================
# CONFIGURATION FUNCTIONS
# ===========================================================================

def get_config(algorithm=None):
    """
    Get the configuration for the specified algorithm.
    
    Args:
        algorithm (str | None): Algorithm name. If None, uses global ALGORITHM.
    
    Returns:
        dict: Complete configuration dictionary.
    """
    algo = algorithm if algorithm else ALGORITHM
    
    if algo not in ALGORITHM_CONFIGS:
        raise ValueError(
            f"Unknown algorithm: {algo}. "
            f"Available: {list(ALGORITHM_CONFIGS.keys())}"
        )
    
    # Build complete config
    config = {
        'algorithm_name': algo,
        'task': TASK,
        'device': DEVICE,
        'num_envs': NUM_ENVS,
        'num_of_action': NUM_OF_ACTION,
        'action_range': ACTION_RANGE,
        'n_observations': N_OBSERVATIONS,
        'headless': HEADLESS,
        'save_interval': SAVE_INTERVAL,
        'log_interval': LOG_INTERVAL,
        'n_test_episodes': N_TEST_EPISODES,
        'model_dir': MODEL_DIR,
        'log_dir': LOG_DIR,
    }
    
    # Add algorithm-specific parameters
    algo_config = ALGORITHM_CONFIGS[algo].copy()
    config.update(algo_config)
    
    return config


def get_algorithm():
    """Get the current algorithm name."""
    return ALGORITHM


def get_task():
    """Get the current task name."""
    return TASK


def get_device():
    """Get the torch device."""
    return DEVICE


def print_config():
    """Print a summary of the current configuration."""
    config = get_config()
    
    print(f"\n{'='*80}")
    print(f"🚀 HW3 Configuration - Function Approximation Algorithms")
    print(f"{'='*80}")
    print(f"Algorithm: {config['algorithm_name']}")
    print(f"Description: {config['description']}")
    print(f"Task: {config['task']}")
    print(f"Device: {config['device']}")
    
    print(f"\n🔧 Environment Settings:")
    print(f"  • Parallel Environments: {config['num_envs']} (CRITICAL for HW3!)")
    print(f"  • Action Space: {config['action_type']}")
    print(f"  • Number of Actions: {config['num_of_action']}")
    print(f"  • Action Range: {config['action_range']}")
    print(f"  • Observation Dim: {config['n_observations']}")
    print(f"  • Headless Mode: {config['headless']}")
    
    print(f"\n📊 Training Parameters:")
    
    # Different training parameters for different algorithm types
    if 'n_episodes' in config:
        print(f"  • Episodes: {config['n_episodes']:,}")
        print(f"  • Steps per Episode: {config['max_steps']:,}")
    elif 'max_iterations' in config:
        print(f"  • Max Iterations: {config['max_iterations']:,}")
        print(f"  • Transitions per Env: {config['num_transitions_per_env']}")
    elif 'n_iterations' in config:
        print(f"  • Iterations: {config['n_iterations']:,}")
        print(f"  • Steps per Iteration: {config['max_steps']:,}")
    
    print(f"  • Learning Rate: {config['learning_rate']}")
    print(f"  • Discount Factor: {config.get('discount_factor', config.get('gamma', 'N/A'))}")
    
    # Algorithm-specific parameters
    if config['action_type'] == 'discrete' and 'initial_epsilon' in config:
        print(f"  • Epsilon: {config['initial_epsilon']} → {config['final_epsilon']} (decay: {config['epsilon_decay']})")
    
    if 'buffer_size' in config:
        print(f"  • Replay Buffer: {config['buffer_size']:,} (batch: {config['batch_size']})")
    
    if 'clip_param' in config:
        print(f"  • PPO Clip: {config['clip_param']}, GAE λ: {config['lam']}")
    
    if 'tau' in config:
        print(f"  • Target Network τ: {config['tau']}")
    
    print(f"\n🎯 Expected Performance (with {config['num_envs']} parallel envs):")
    if 'expected_episodes_to_solve' in config:
        print(f"  • Episodes to Solve: ~{config['expected_episodes_to_solve']}")
    elif 'expected_iterations_to_solve' in config:
        print(f"  • Iterations to Solve: ~{config['expected_iterations_to_solve']}")
    print(f"  • Final Average Return: ~{config['expected_final_return']}")
    print(f"  • Training Time: ~{config['expected_time_minutes']} min")
    
    print(f"\n💾 Model I/O:")
    print(f"  • Save Directory: {config['model_dir']}")
    print(f"  • Log Directory: {config['log_dir']}")
    print(f"  • Save Interval: Every {config['save_interval']} iterations")
    
    print(f"{'='*80}\n")


def create_agent(algorithm=None, testing=False):
    """
    Create an agent instance based on configuration.
    
    Args:
        algorithm (str | None): Algorithm name. If None, uses global ALGORITHM.
        testing (bool): If True, disables exploration (epsilon=0).
    
    Returns:
        Agent instance configured with hyperparameters.
    """
    from RL_Algorithm.Function_based.Linear_Q import Linear_QN
    from RL_Algorithm.Function_based.DQN import DQN
    from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
    from RL_Algorithm.Function_based.AC import AC
    from RL_Algorithm.Function_based.A2C import A2C
    from RL_Algorithm.Function_based.PPO import PPO
    from RL_Algorithm.Function_based.SAC import SAC_Discrete as SAC
    from RL_Algorithm.Function_based.TD3 import TD3_Discrete as TD3
    
    config = get_config(algorithm)
    algo_name = config['algorithm_name']
    
    # Agent class mapping
    agent_classes = {
        'Linear_Q': Linear_QN,
        'DQN': DQN,
        'MC_REINFORCE': MC_REINFORCE,
        'AC': AC,
        'A2C': A2C,
        'PPO': PPO,
        'SAC': SAC,
        'TD3': TD3,
    }
    
    agent_class = agent_classes.get(algo_name)
    if agent_class is None:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Common parameters
    common_params = {
        'num_of_action': config['num_of_action'],
        'action_range': config['action_range'],
    }
    
    # Override exploration for testing
    if testing and 'initial_epsilon' in config:
        exploration_override = {
            'initial_epsilon': 0.0,
            'epsilon_decay': 0.0,
            'final_epsilon': 0.0,
        }
    else:
        exploration_override = {}
    
    # Algorithm-specific parameters
    if algo_name == 'Linear_Q':
        agent = agent_class(
            **common_params,
            learning_rate=config['learning_rate'],
            initial_epsilon=exploration_override.get('initial_epsilon', config['initial_epsilon']),
            epsilon_decay=exploration_override.get('epsilon_decay', config['epsilon_decay']),
            final_epsilon=exploration_override.get('final_epsilon', config['final_epsilon']),
            discount_factor=config['discount_factor'],
        )
    
    elif algo_name == 'DQN':
        agent = agent_class(
            device=config['device'],
            **common_params,
            n_observations=config['n_observations'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            tau=config['tau'],
            initial_epsilon=exploration_override.get('initial_epsilon', config['initial_epsilon']),
            epsilon_decay=exploration_override.get('epsilon_decay', config['epsilon_decay']),
            final_epsilon=exploration_override.get('final_epsilon', config['final_epsilon']),
            discount_factor=config['discount_factor'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            update_freq=config.get('update_freq', 4),
            target_update_freq=config.get('target_update_freq', 200),
        )
    
    elif algo_name == 'MC_REINFORCE':
        agent = agent_class(
            device=config['device'],
            **common_params,
            n_observations=config['n_observations'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout'],
            action_type=config['action_type'],
            learning_rate=config['learning_rate'],
            discount_factor=config['discount_factor'],
            max_episode_length=config['max_episode_length'],
        )
    
    elif algo_name in ('AC', 'A2C'):
        agent = agent_class(
            device=config['device'],
            **common_params,
            n_observations=config['n_observations'],
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            action_type=config['action_type'],
            init_noise_std=config['init_noise_std'],
            learning_rate=config['learning_rate'],
            discount_factor=config['discount_factor'],
            value_loss_coef=config['value_loss_coef'],
            entropy_coef=config['entropy_coef'],
            max_grad_norm=config['max_grad_norm'],
        )
    
    elif algo_name == 'PPO':
        agent = agent_class(
            device=config['device'],
            **common_params,
            n_observations=config['n_observations'],
            hidden_dims=config['hidden_dims'],
            activation=config['activation'],
            action_type=config['action_type'],
            init_noise_std=config['init_noise_std'],
            num_learning_epochs=config['num_learning_epochs'],
            num_mini_batches=config['num_mini_batches'],
            clip_param=config['clip_param'],
            gamma=config['gamma'],
            lam=config['lam'],
            value_loss_coef=config['value_loss_coef'],
            entropy_coef=config['entropy_coef'],
            learning_rate=config['learning_rate'],
            max_grad_norm=config['max_grad_norm'],
        )
    
    elif algo_name == 'SAC':
        agent = agent_class(
            device=config['device'],
            **common_params,
            n_observations=config['n_observations'],
            hidden_dim=config['hidden_dim'],
            learning_rate=config['learning_rate'],
            alpha_lr=config['alpha_lr'],
            tau=config['tau'],
            discount_factor=config['discount_factor'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            init_alpha=config['init_alpha'],
            auto_alpha=config['auto_alpha'],
            target_entropy=config['target_entropy'],
        )
    
    elif algo_name == 'TD3':
        epsilon_start = 0.0 if testing else config['epsilon_start']
        agent = agent_class(
            device=config['device'],
            **common_params,
            n_observations=config['n_observations'],
            hidden_dim=config['hidden_dim'],
            learning_rate=config['learning_rate'],
            tau=config['tau'],
            epsilon_start=epsilon_start,
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            target_smoothing_temperature=config['target_smoothing_temperature'],
            policy_update_freq=config['policy_update_freq'],
            discount_factor=config['discount_factor'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
        )
    
    print(f"✅ Created {algo_name} agent (testing={testing})")
    
    return agent


def validate_config():
    """Validate the current configuration."""
    config = get_config()
    
    print(f"\n{'='*80}")
    print(f"🔍 Configuration Validation")
    print(f"{'='*80}")
    
    # Check parallel environments
    if config['num_envs'] < 256:
        print(f"⚠️  WARNING: num_envs={config['num_envs']} is less than recommended 256")
        print(f"   This may result in slower training.")
    else:
        print(f"✅ Parallel environments: {config['num_envs']} (optimal)")
    
    # Check action type compatibility
    if config['action_type'] == 'continuous' and config['num_of_action'] > 1:
        print(f"⚠️  WARNING: Continuous action with num_of_action={config['num_of_action']}")
        print(f"   For continuous, num_of_action should be action dimension (typically 1 for CartPole)")
    
    # Check device
    if str(config['device']) == 'cpu':
        print(f"⚠️  WARNING: Using CPU - training will be slower")
        print(f"   Consider using GPU if available")
    else:
        print(f"✅ Device: {config['device']} (accelerated)")
    
    # Check algorithm-specific issues
    algo = config['algorithm_name']
    
    if algo == 'DQN' and config['action_type'] == 'continuous':
        print(f"⚠️  WARNING: DQN works best with discrete actions")
    
    print(f"\n✅ Configuration is valid for {algo}")
    print(f"{'='*80}\n")
    
    return True


# ===========================================================================
# COMPARISON UTILITIES
# ===========================================================================

def compare_algorithms():
    """Print a comparison table of all algorithms."""
    print(f"\n{'='*100}")
    print(f"📊 Algorithm Comparison (with {NUM_ENVS} parallel environments)")
    print(f"{'='*100}")
    
    print(f"\n{'Algorithm':<15} {'Type':<12} {'Action':<12} {'Sample Eff':<12} {'Speed':<12} {'Stability':<10}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    comparisons = [
        ('Linear_Q', 'Value', 'Discrete', '⭐⭐☆☆☆', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐☆'),
        ('DQN', 'Value', 'Discrete', '⭐⭐⭐⭐☆', '⭐⭐⭐⭐☆', '⭐⭐⭐⭐☆'),
        ('MC_REINFORCE', 'Policy', 'Both', '⭐⭐☆☆☆', '⭐⭐⭐☆☆', '⭐⭐☆☆☆'),
        ('A2C', 'Actor-Critic', 'Both', '⭐⭐⭐☆☆', '⭐⭐⭐⭐☆', '⭐⭐⭐☆☆'),
        ('PPO', 'Actor-Critic', 'Both', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐☆', '⭐⭐⭐⭐⭐'),
        ('SAC', 'Actor-Critic', 'Discrete', '⭐⭐⭐⭐⭐', '⭐⭐⭐☆☆', '⭐⭐⭐⭐⭐'),
        ('TD3', 'Actor-Critic', 'Discrete', '⭐⭐⭐⭐☆', '⭐⭐⭐☆☆', '⭐⭐⭐⭐☆'),
    ]
    
    for algo, algo_type, action, sample_eff, speed, stability in comparisons:
        print(f"{algo:<15} {algo_type:<12} {action:<12} {sample_eff:<12} {speed:<12} {stability:<10}")
    
    print(f"\n{'='*100}")
    print(f"📝 Recommendation for HW3:")
    print(f"  Option 1 (Classic Mix): Linear_Q + DQN + MC_REINFORCE + PPO")
    print(f"  Option 2 (All Types):   DQN + A2C + SAC + TD3")
    print(f"  Option 3 (Best 4):      DQN + PPO + SAC + TD3")
    print(f"{'='*100}\n")


# ===========================================================================
# MAIN - FOR TESTING
# ===========================================================================

if __name__ == "__main__":
    print_config()
    validate_config()
    compare_algorithms()
    
    print("\n🧪 Testing agent creation...")
    try:
        agent = create_agent()
        print(f"✅ Successfully created {ALGORITHM} agent")
        
        # Test agent in testing mode
        test_agent = create_agent(testing=True)
        print(f"✅ Successfully created test agent")
        
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()