"""Script to plot training results from CSV files.

# Compare all algorithms at once: python plot_results.py --all
plot_results.py --algorithm SARSA

python scripts/RL_Algorithm/plot_results.py --algorithm SARSA

"""



import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import from config
from config import get_config, get_algorithm, get_task, ALGORITHM_CONFIGS


def load_csv_data(task, algorithm, project_root):
    """Load training metrics from CSV file."""
    csv_path = os.path.join(project_root, "logs", task, algorithm, "training_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} episodes from {algorithm}")
        return df
    except Exception as e:
        print(f"❌ Error loading {csv_path}: {e}")
        return None


def plot_training_curves(df, algorithm, save_dir):
    """Plot training curves for a single algorithm."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    episodes = df['episode'].values
    
    # Plot 1: Episode Rewards
    axes[0].plot(episodes, df['reward'].values, alpha=0.3, color='blue', 
                label='Raw Rewards', linewidth=0.8)
    axes[0].plot(episodes, df['avg_reward_100'].values, linewidth=2.5, color='darkblue', 
                 label='100-Episode Moving Average')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title(f'{algorithm}: Episode Rewards over Training', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    axes[1].plot(episodes, df['length'].values, alpha=0.3, color='green', 
                label='Raw Lengths', linewidth=0.8)
    axes[1].plot(episodes, df['avg_length_100'].values, linewidth=2.5, color='darkgreen', 
                 label='100-Episode Moving Average')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Episode Length (steps)', fontsize=12)
    axes[1].set_title(f'{algorithm}: Episode Lengths over Training', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    axes[2].plot(episodes, df['epsilon'].values, linewidth=2, color='orange')
    axes[2].set_xlabel('Episode', fontsize=12)
    axes[2].set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    axes[2].set_title(f'{algorithm}: Epsilon Decay', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, f'{algorithm}_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_detailed_analysis(df, algorithm, save_dir):
    """Plot detailed analysis showing catastrophic forgetting."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    episodes = df['episode'].values
    rewards = df['reward'].values
    lengths = df['length'].values
    avg_rewards = df['avg_reward_100'].values
    epsilon = df['epsilon'].values
    
    # Plot 1: Reward vs Epsilon (exploration-exploitation tradeoff)
    scatter = axes[0, 0].scatter(epsilon, rewards, c=episodes, cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Epsilon (Exploration Rate)', fontsize=11)
    axes[0, 0].set_ylabel('Episode Reward', fontsize=11)
    axes[0, 0].set_title('Reward vs Exploration Rate\n(Color = Episode Number)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Episode')
    
    # Plot 2: Catastrophic Forgetting Detection
    rolling_max = pd.Series(avg_rewards).expanding().max()
    axes[0, 1].plot(episodes, avg_rewards, linewidth=2, color='blue', label='Current Avg Reward')
    axes[0, 1].plot(episodes, rolling_max, linewidth=2, color='red', linestyle='--', 
                   label='Best So Far')
    axes[0, 1].fill_between(episodes, avg_rewards, rolling_max, alpha=0.2, color='red', 
                            label='Performance Gap')
    axes[0, 1].set_xlabel('Episode', fontsize=11)
    axes[0, 1].set_ylabel('Avg Reward (100-ep MA)', fontsize=11)
    axes[0, 1].set_title('Catastrophic Forgetting Detection\n(Gap shows forgotten performance)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Episode length histogram by training phase
    early = lengths[:100] if len(lengths) >= 100 else lengths
    mid = lengths[200:300] if len(lengths) >= 300 else []
    late = lengths[-100:] if len(lengths) >= 100 else []
    
    bins = np.linspace(0, max(lengths) + 1, 30)
    axes[1, 0].hist(early, bins=bins, alpha=0.6, label='Episodes 0-100', color='green')
    if len(mid) > 0:
        axes[1, 0].hist(mid, bins=bins, alpha=0.6, label='Episodes 200-300', color='orange')
    if len(late) > 0:
        axes[1, 0].hist(late, bins=bins, alpha=0.6, label='Last 100 Episodes', color='red')
    axes[1, 0].set_xlabel('Episode Length (steps)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Episode Length Distribution by Training Phase', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Efficiency (reward per step)
    reward_per_step = rewards / np.maximum(lengths, 1)
    axes[1, 1].plot(episodes, reward_per_step, alpha=0.4, color='purple', linewidth=0.8)
    
    # Moving average
    window = min(50, len(reward_per_step) // 2)
    if len(reward_per_step) >= window:
        rps_ma = np.convolve(reward_per_step, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(episodes[window-1:], rps_ma, linewidth=2, color='darkviolet', 
                       label=f'{window}-Episode MA')
    
    axes[1, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Optimal (1.0)')
    axes[1, 1].set_xlabel('Episode', fontsize=11)
    axes[1, 1].set_ylabel('Reward per Step', fontsize=11)
    axes[1, 1].set_title('Learning Efficiency\n(Higher = Better balance control)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, f'{algorithm}_detailed_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_performance_summary(df, algorithm, save_dir):
    """Create a summary report of training performance."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    episodes = df['episode'].values
    rewards = df['reward'].values
    lengths = df['length'].values
    avg_rewards = df['avg_reward_100'].values
    
    # Statistics
    best_episode = int(np.argmax(rewards))
    best_reward = float(rewards[best_episode])
    final_avg_reward = float(avg_rewards[-1])
    peak_avg_reward = float(np.max(avg_rewards))
    peak_episode = int(np.argmax(avg_rewards))
    
    # Main plot - Training curve with annotations
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(episodes, avg_rewards, linewidth=2.5, color='blue', label='Avg Reward (100-ep MA)')
    ax_main.scatter([best_episode], [rewards[best_episode]], color='gold', s=200, 
                   marker='*', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Best Episode: {best_episode} (Reward: {best_reward:.1f})')
    ax_main.scatter([peak_episode], [peak_avg_reward], color='red', s=150,
                   marker='D', edgecolors='black', linewidths=2, zorder=5,
                   label=f'Peak Avg: Episode {peak_episode} ({peak_avg_reward:.1f})')
    ax_main.axhline(y=peak_avg_reward, color='red', linestyle='--', alpha=0.3)
    ax_main.axhline(y=final_avg_reward, color='blue', linestyle='--', alpha=0.3)
    ax_main.set_xlabel('Episode', fontsize=12)
    ax_main.set_ylabel('Average Reward', fontsize=12)
    ax_main.set_title(f'{algorithm}: Training Performance Summary', fontsize=14, fontweight='bold')
    ax_main.legend(loc='best', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    # Stats boxes
    performance_drop = peak_avg_reward - final_avg_reward
    drop_percentage = (performance_drop / peak_avg_reward * 100) if peak_avg_reward > 0 else 0
    
    stats_data = [
        ('Best Single Episode', f'Episode {best_episode}\nReward: {best_reward:.1f}'),
        ('Peak Performance', f'Episode {peak_episode}\nAvg: {peak_avg_reward:.1f}'),
        ('Final Performance', f'Episode {len(episodes)-1}\nAvg: {final_avg_reward:.1f}'),
    ]
    
    for idx, (title, value) in enumerate(stats_data):
        row = 1
        col = idx
        ax = fig.add_subplot(gs[row, col])
        ax.text(0.5, 0.5, value, ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.9, title, ha='center', va='top', fontsize=10, style='italic')
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                                   edgecolor='black', linewidth=2))
    
    output_path = os.path.join(save_dir, f'{algorithm}_performance_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_algorithm_comparison(data_dict, task, save_dir):
    """Compare multiple algorithms on the same plot."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (algorithm, df) in enumerate(data_dict.items()):
        color = colors[idx % len(colors)]
        episodes = df['episode'].values
        
        # Plot rewards
        axes[0].plot(episodes, df['avg_reward_100'].values, linewidth=2.5, 
                    color=color, label=algorithm)
        
        # Plot lengths
        axes[1].plot(episodes, df['avg_length_100'].values, linewidth=2.5,
                    color=color, label=algorithm)
    
    # Rewards plot
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward (100-ep MA)', fontsize=12)
    axes[0].set_title(f'{task}: Algorithm Comparison - Rewards', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Lengths plot
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Average Length (100-ep MA)', fontsize=12)
    axes[1].set_title(f'{task}: Algorithm Comparison - Episode Lengths', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, f'{task}_algorithm_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()

def compare_algorithm_variance(task, save_dir):
    """Compare variance across all algorithms."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    algorithms = ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"]
    colors = ['steelblue', 'coral', 'mediumseagreen', 'orange']
    
    stats_list = []
    
    # Collect statistics for each algorithm
    for algo in algorithms:
        csv_path = f"logs/{task}/{algo}/training_metrics.csv"
        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {algo}: CSV not found")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Late training variance (last 500 episodes)
        late = df.tail(500)
        stats_list.append({
            'algorithm': algo,
            'mean': late['reward'].mean(),
            'std': late['reward'].std(),
            'var': late['reward'].var(),
            'cv': late['reward'].std() / abs(late['reward'].mean())
        })
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    algos = [s['algorithm'] for s in stats_list]
    means = [s['mean'] for s in stats_list]
    stds = [s['std'] for s in stats_list]
    vars = [s['var'] for s in stats_list]
    cvs = [s['cv'] for s in stats_list]
    
    # Plot 1: Mean Reward
    ax1 = axes[0, 0]
    bars1 = ax1.bar(algos, means, color=colors[:len(algos)], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Reward', fontsize=11)
    ax1.set_title('Mean Reward (Last 500 Episodes)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Standard Deviation
    ax2 = axes[0, 1]
    bars2 = ax2.bar(algos, stds, color=colors[:len(algos)], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Standard Deviation', fontsize=11)
    ax2.set_title('Reward Std Dev (Last 500 Episodes)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars2, stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Variance
    ax3 = axes[1, 0]
    bars3 = ax3.bar(algos, vars, color=colors[:len(algos)], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Variance', fontsize=11)
    ax3.set_title('Reward Variance (Last 500 Episodes)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars3, vars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Coefficient of Variation (Normalized)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(algos, cvs, color=colors[:len(algos)], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Coefficient of Variation', fontsize=11)
    ax4.set_title('Normalized Variance (CV = σ/μ)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    # Add value labels
    for bar, val in zip(bars4, cvs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(save_dir, f'{task}_algorithm_variance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved variance comparison: {output_path}")
    plt.close()
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"Variance Comparison: All Algorithms")
    print(f"{'='*80}")
    print(f"{'Algorithm':<20} {'Mean':<10} {'Std Dev':<10} {'Variance':<12} {'CV':<10}")
    print(f"{'-'*80}")
    for s in stats_list:
        print(f"{s['algorithm']:<20} {s['mean']:<10.2f} {s['std']:<10.2f} "
              f"{s['var']:<12.1f} {s['cv']:<10.4f}")
    print(f"{'='*80}")
    
    # Highlight MC vs TD difference
    mc_stats = next((s for s in stats_list if 'Monte_Carlo' in s['algorithm']), None)
    td_stats = [s for s in stats_list if 'Monte_Carlo' not in s['algorithm']]
    
    if mc_stats and td_stats:
        avg_td_var = np.mean([s['var'] for s in td_stats])
        avg_td_cv = np.mean([s['cv'] for s in td_stats])
        
        print(f"\n📊 Key Findings:")
        print(f"  Monte Carlo Variance: {mc_stats['var']:.1f}")
        print(f"  TD Methods Avg Variance: {avg_td_var:.1f}")
        print(f"  Variance Ratio (MC/TD): {mc_stats['var']/avg_td_var:.2f}x")
        print(f"\n  Monte Carlo CV: {mc_stats['cv']:.4f}")
        print(f"  TD Methods Avg CV: {avg_td_cv:.4f}")
        print(f"  CV Ratio (MC/TD): {mc_stats['cv']/avg_td_cv:.2f}x")
        
        if mc_stats['var'] > avg_td_var * 1.5:
            print(f"\n✅ Monte Carlo shows {mc_stats['var']/avg_td_var:.1f}x HIGHER variance than TD methods!")
            print(f"   This confirms theoretical prediction: MC has high variance, TD has low variance.")
        else:
            print(f"\n⚠️  Variance difference smaller than expected. Possible reasons:")
            print(f"   - Environment stochasticity dominates")
            print(f"   - Episode lengths similar")
            print(f"   - Need more episodes for convergence")

def plot_episode_variability(task, save_dir):
    """Plot reward variability over consecutive episodes."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    algorithms = ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"]
    colors = {'Q_Learning': 'steelblue', 'SARSA': 'coral', 
              'Double_Q_Learning': 'mediumseagreen', 'Monte_Carlo': 'orange'}
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for algo in algorithms:
        csv_path = f"logs/{task}/{algo}/training_metrics.csv"
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path)
        
        # Calculate consecutive differences (how much reward changes episode-to-episode)
        df['reward_diff'] = df['reward'].diff().abs()
        
        # Plot rolling average of absolute differences
        window = 100
        df['reward_diff_ma'] = df['reward_diff'].rolling(window=window).mean()
        
        ax.plot(df['episode'], df['reward_diff_ma'], 
               label=algo, linewidth=2, alpha=0.8, color=colors[algo])
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average |Reward Change| per Episode', fontsize=12)
    ax.set_title('Episode-to-Episode Reward Variability (100-ep MA)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, f'{task}_episode_variability.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved variability plot: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot RL training results from CSV files.")
    parser.add_argument("--task", type=str, default=None, 
                       help="Task name (if not specified, uses config.py)")
    parser.add_argument("--algorithm", type=str, default=None,
                       help="Algorithm to plot (if not specified, uses config.py)")
    parser.add_argument("--all", action="store_true",
                       help="Plot all available algorithms for comparison")
    args = parser.parse_args()
    
    # Get from config if not specified
    task = args.task if args.task else get_task()
    algorithm = args.algorithm if args.algorithm else get_algorithm()
    
    # Setup paths
    project_root = os.path.join(script_dir, "..", "..")  # Up to CartPole_4.5.0/
    plots_dir = os.path.join(project_root, "plots", task)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📊 Plotting Results")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Algorithm: {algorithm}")
    print(f"Project root: {project_root}")
    print(f"Plots directory: {plots_dir}")
    print(f"{'='*70}\n")
    
    if args.all:
        # Plot all available algorithms
        print("🔍 Searching for all trained algorithms...\n")
        
        available_algorithms = ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"]
        data_dict = {}
        for algo in available_algorithms:
            df = load_csv_data(task, algo, project_root)
            if df is not None:
                data_dict[algo] = df
                # Generate individual plots
                print(f"\n📈 Generating plots for {algo}...")
                plot_training_curves(df, algo, plots_dir)
                plot_detailed_analysis(df, algo, plots_dir)
                plot_performance_summary(df, algo, plots_dir)
        
        if len(data_dict) > 1:
            print(f"\n📊 Generating comparison plot...")
            plot_algorithm_comparison(data_dict, task, plots_dir)
        elif len(data_dict) == 1:
            print(f"\n⚠️  Only one algorithm found. Need at least 2 for comparison.")
        else:
            print(f"\n❌ No trained algorithms found!")
            return
        compare_algorithm_variance(task, plots_dir)
        plot_episode_variability(task, plots_dir)
    else:
        # Plot single algorithm
        df = load_csv_data(task, algorithm, project_root)
        
        if df is None:
            print(f"\n❌ Error: No data loaded for {algorithm}")
            print(f"Make sure you have run training first.")
            print(f"Expected CSV at: {project_root}/logs/{task}/{algorithm}/training_metrics.csv")
            return
        
        print(f"\n📈 Generating plots for {algorithm}...")
        plot_training_curves(df, algorithm, plots_dir)
        plot_detailed_analysis(df, algorithm, plots_dir)
        plot_performance_summary(df, algorithm, plots_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ All plots saved to: {plots_dir}")
    print(f"{'='*70}\n")
    
    if args.all:
        print("Generated comparison:")
        print(f"  • {task}_algorithm_comparison.png")
        print("\nGenerated for each algorithm:")
    
    print("  • {algorithm}_training_curves.png")
    print("  • {algorithm}_detailed_analysis.png")
    print("  • {algorithm}_performance_summary.png")


if __name__ == "__main__":
    main()