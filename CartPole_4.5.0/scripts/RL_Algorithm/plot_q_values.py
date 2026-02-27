"""Script to visualize Q-values: Cart Position vs Pole Angle (averaged over velocities)."""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import from config
from config import get_config, get_algorithm, get_task


def load_q_values(q_value_dir, algorithm, episode=None):
    """Load Q-values from JSON file."""
    
    # Get config to build filename pattern
    config = get_config(algorithm)
    num_of_action = config['num_of_action']
    action_range = config['action_range']
    discretize_state_weight = config['discretize_state_weight']
    
    # Build pattern
    if episode is not None:
        # Load specific episode
        filename = f"{algorithm}_{episode}_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
    else:
        # Find latest episode
        import glob
        pattern = f"{algorithm}_*_{num_of_action}_{int(action_range[1])}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
        files = glob.glob(os.path.join(q_value_dir, pattern))
        
        if not files:
            print(f"❌ No Q-value files found matching: {pattern}")
            return None, None
        
        # Get latest by episode number
        def extract_episode(filepath):
            basename = os.path.basename(filepath)
            # Remove algorithm name prefix (handles underscores in algorithm name)
            without_algo = basename.replace(f"{algorithm}_", "", 1)
            # Now first part is episode number
            episode_str = without_algo.split('_')[0]
            return int(episode_str)
        
        latest_file = max(files, key=extract_episode)
        filename = os.path.basename(latest_file)
    
    filepath = os.path.join(q_value_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None, None
    
    print(f"📂 Loading Q-values from: {filename}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to tuples
    q_values = {}
    for key_str, values in data['q_values'].items():
        # Parse "(0, 1, 0, 5)" -> (0, 1, 0, 5)
        key_tuple = tuple(map(int, key_str.strip('()').split(', ')))
        q_values[key_tuple] = np.array(values)
    
    print(f"✅ Loaded {len(q_values)} states")
    
    return q_values, config


def plot_cart_pole_surface(q_values, config, algorithm, save_dir):
    """
    Create 3D surface: Cart Position vs Pole Angle (averaged over velocities).
    This is the MOST IMPORTANT plot for CartPole!
    """
    
    weights = config['discretize_state_weight']
    
    print("\n📊 Creating Cart Position vs Pole Angle 3D Surface")
    print("   (Averaging over all cart and pole velocities)")
    
    # Group by position dimensions, average over velocity dimensions
    averaged_q = {}
    counts = {}
    
    for state, q_vals in q_values.items():
        # state = (cart_pos_bin, pole_angle_bin, cart_vel_bin, pole_vel_bin)
        cart_pos_bin = state[0]
        pole_angle_bin = state[1]
        
        # Convert bin indices to continuous values
        cart_pos = cart_pos_bin / weights[0]  # meters
        pole_angle = pole_angle_bin / weights[1]  # radians
        
        pos_key = (cart_pos, pole_angle)
        max_q = np.max(q_vals)
        
        if pos_key not in averaged_q:
            averaged_q[pos_key] = 0.0
            counts[pos_key] = 0
        
        averaged_q[pos_key] += max_q
        counts[pos_key] += 1
    
    # Average
    for key in averaged_q:
        averaged_q[key] /= counts[key]
    
    print(f"   Found {len(averaged_q)} unique (cart_pos, pole_angle) combinations")
    print(f"   Average velocity states per position: {np.mean(list(counts.values())):.1f}")
    
    # Convert to arrays
    data_points = np.array([[k[0], k[1], v] for k, v in averaged_q.items()])
    
    # Create grid
    x_unique = np.sort(np.unique(data_points[:, 0]))
    y_unique = np.sort(np.unique(data_points[:, 1]))
    
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = np.zeros_like(X, dtype=float)
    
    # Fill Z values
    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            mask = (np.abs(data_points[:, 0] - x_val) < 1e-6) & (np.abs(data_points[:, 1] - y_val) < 1e-6)
            if np.any(mask):
                Z[i, j] = data_points[mask, 2][0]
            else:
                Z[i, j] = np.nan
    
    # Print ranges
    print(f"   Cart Position range: [{x_unique[0]:.2f}, {x_unique[-1]:.2f}] m")
    print(f"   Pole Angle range: [{y_unique[0]:.2f}, {y_unique[-1]:.2f}] rad = [{np.rad2deg(y_unique[0]):.1f}°, {np.rad2deg(y_unique[-1]):.1f}°]")
    print(f"   Q-value range: [{np.nanmin(Z):.2f}, {np.nanmax(Z):.2f}]")
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8,
                          linewidth=0, antialiased=True, edgecolor='none')
    
    # Wireframe for structure
    ax.plot_wireframe(X, Y, Z, color='black', alpha=0.15, linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Cart Position (m)', fontsize=12, labelpad=12)
    ax.set_ylabel('Pole Angle (rad)', fontsize=12, labelpad=12)
    ax.set_zlabel('Average Max Q-Value', fontsize=12, labelpad=12)
    
    # Title
    title = f'3D Surface Plot of Q-Values\n{algorithm}\n'
    title += '(Averaged over Cart Velocity and Pole Velocity)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
    
    # Color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, pad=0.1)
    cbar.set_label('Average Max Q-Value', fontsize=11)
    
    # Better viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filename = f'{algorithm}_q_surface_Cart_Position_vs_Pole_Angle.png'
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Q-values: Cart Position vs Pole Angle")
    parser.add_argument("--task", type=str, default=None,
                       help="Task name (default: from config.py)")
    parser.add_argument("--algorithm", type=str, default=None,
                       help="Algorithm to visualize (default: from config.py)")
    parser.add_argument("--episode", type=int, default=None,
                       help="Specific episode to load (default: latest)")
    args = parser.parse_args()
    
    # Get from config if not specified
    task = args.task if args.task else get_task()
    algorithm = args.algorithm if args.algorithm else get_algorithm()
    
    # Setup paths
    project_root = os.path.join(script_dir, "..", "..")
    q_value_dir = os.path.join(project_root, "q_value", task, algorithm)
    plots_dir = os.path.join(project_root, "plots", task, "q_values_3d")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("📊 Q-Value Visualization: Cart Position vs Pole Angle")
    print("="*70)
    print(f"Task: {task}")
    print(f"Algorithm: {algorithm}")
    print(f"Q-value directory: {q_value_dir}")
    print(f"Output directory: {plots_dir}")
    print("="*70)
    
    # Check if directory exists
    if not os.path.exists(q_value_dir):
        print(f"\n❌ Q-value directory not found: {q_value_dir}")
        print("Make sure you have trained this algorithm first!")
        return
    
    # Load Q-values
    q_values, config = load_q_values(q_value_dir, algorithm, args.episode)
    
    if q_values is None:
        print("\n❌ Failed to load Q-values!")
        return
    
    # Generate THE plot
    plot_cart_pole_surface(q_values, config, algorithm, plots_dir)
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print("="*70)


if __name__ == "__main__":
    main()