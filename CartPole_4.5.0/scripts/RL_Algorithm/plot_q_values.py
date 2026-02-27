"""Script to visualize Q-values: Cart Position vs Pole Angle (averaged over velocities)."""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

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
            return None, None, None
        
        # Get latest by episode number
        def extract_episode(filepath):
            basename = os.path.basename(filepath)
            without_algo = basename.replace(f"{algorithm}_", "", 1)
            episode_str = without_algo.split('_')[0]
            try:
                return int(episode_str)
            except ValueError:
                return 0
        
        latest_file = max(files, key=extract_episode)
        filename = os.path.basename(latest_file)
        episode = extract_episode(latest_file)
    
    filepath = os.path.join(q_value_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None, None, None
    
    print(f"📂 Loading Q-values from: {filename}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, None, None
    
    # Convert string keys back to tuples
    q_values = {}
    for key_str, values in data['q_values'].items():
        try:
            key_tuple = tuple(map(int, key_str.strip('()').split(', ')))
            q_values[key_tuple] = np.array(values)
        except Exception as e:
            print(f"⚠️  Warning: Skipping invalid state key: {key_str}")
            continue
    
    print(f"✅ Loaded {len(q_values)} states from episode {episode}")
    
    return q_values, config, episode


def plot_cart_pole_surface(q_values, config, algorithm, save_dir, episode=None, 
                           interpolation_resolution=100):
    """
    Create smooth 3D surface: Cart Position vs Pole Angle (averaged over velocities).
    
    Args:
        interpolation_resolution: Number of points in each dimension for smooth surface (default: 200)
    """
    
    weights = config['discretize_state_weight']
    
    print("\n📊 Creating Cart Position vs Pole Angle 3D Surface")
    print("   (Averaging over all cart and pole velocities)")
    print(f"   Interpolation resolution: {interpolation_resolution}x{interpolation_resolution}")
    
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
    print(f"   Max velocity states per position: {np.max(list(counts.values()))}")
    print(f"   Min velocity states per position: {np.min(list(counts.values()))}")
    
    # Convert to arrays for interpolation
    points = np.array([[k[0], k[1]] for k in averaged_q.keys()])  # (N, 2)
    values = np.array([v for v in averaged_q.values()])  # (N,)
    
    # Get data ranges
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    print(f"   Cart Position range: [{x_min:.2f}, {x_max:.2f}] m")
    print(f"   Pole Angle range: [{y_min:.2f}, {y_max:.2f}] rad = [{np.rad2deg(y_min):.1f}°, {np.rad2deg(y_max):.1f}°]")
    print(f"   Q-value range: [{values.min():.2f}, {values.max():.2f}]")
    print(f"   Q-value mean: {values.mean():.2f}")
    print(f"   Q-value std: {values.std():.2f}")
    
    # Create fine mesh grid for smooth interpolation
    x_fine = np.linspace(x_min, x_max, interpolation_resolution)
    y_fine = np.linspace(y_min, y_max, interpolation_resolution)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Interpolate Q-values onto fine grid
    print(f"   Interpolating Q-values onto {interpolation_resolution}x{interpolation_resolution} grid...")
    Z_fine = griddata(points, values, (X_fine, Y_fine), method='cubic', fill_value=np.nan)
    
    # Fill any remaining NaN values at edges with nearest neighbor
    mask_nan = np.isnan(Z_fine)
    if mask_nan.any():
        Z_nearest = griddata(points, values, (X_fine, Y_fine), method='nearest')
        Z_fine[mask_nan] = Z_nearest[mask_nan]
    
    print(f"   Interpolation complete!")
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Smooth surface plot
    surf = ax.plot_surface(X_fine, Y_fine, Z_fine, cmap=cm.viridis, alpha=0.9,
                          linewidth=0, antialiased=True, edgecolor='none',
                          rcount=100, ccount=100)  # Smooth rendering
    
    # Optional: Add original data points as scatter
    # ax.scatter(points[:, 0], points[:, 1], values, c='red', marker='o', s=20, alpha=0.5)
    
    # Labels
    ax.set_xlabel('Cart Position (m)', fontsize=12, labelpad=12)
    ax.set_ylabel('Pole Angle (rad)', fontsize=12, labelpad=12)
    ax.set_zlabel('Average Max Q-Value', fontsize=12, labelpad=12)
    
    # Title
    title = f'3D Surface Plot of Q-Values\n{algorithm}'
    if episode is not None:
        title += f' (Episode {episode})'
    title += '\n(Averaged over Cart Velocity and Pole Velocity)'
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
    episode_str = f"_ep{episode}" if episode is not None else ""
    filename = f'{algorithm}_q_surface_Cart_Position_vs_Pole_Angle{episode_str}.png'
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return points, X_fine, Y_fine, Z_fine


def plot_cart_pole_heatmap(q_values, config, algorithm, save_dir, episode=None,
                           interpolation_resolution=100):
    """Create smooth 2D heatmap as alternative visualization."""
    
    weights = config['discretize_state_weight']
    
    print("\n📊 Creating Cart Position vs Pole Angle Heatmap")
    
    # Group by position dimensions
    averaged_q = {}
    counts = {}
    
    for state, q_vals in q_values.items():
        cart_pos_bin = state[0]
        pole_angle_bin = state[1]
        
        cart_pos = cart_pos_bin / weights[0]
        pole_angle = pole_angle_bin / weights[1]
        
        pos_key = (cart_pos, pole_angle)
        max_q = np.max(q_vals)
        
        if pos_key not in averaged_q:
            averaged_q[pos_key] = 0.0
            counts[pos_key] = 0
        
        averaged_q[pos_key] += max_q
        counts[pos_key] += 1
    
    for key in averaged_q:
        averaged_q[key] /= counts[key]
    
    # Convert to arrays
    points = np.array([[k[0], k[1]] for k in averaged_q.keys()])
    values = np.array([v for v in averaged_q.values()])
    
    # Get ranges
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Create fine grid
    x_fine = np.linspace(x_min, x_max, interpolation_resolution)
    y_fine = np.linspace(y_min, y_max, interpolation_resolution)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Interpolate
    print(f"   Interpolating onto {interpolation_resolution}x{interpolation_resolution} grid...")
    Z_fine = griddata(points, values, (X_fine, Y_fine), method='cubic', fill_value=np.nan)
    
    # Fill NaN with nearest
    mask_nan = np.isnan(Z_fine)
    if mask_nan.any():
        Z_nearest = griddata(points, values, (X_fine, Y_fine), method='nearest')
        Z_fine[mask_nan] = Z_nearest[mask_nan]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    
    im = ax.imshow(Z_fine, cmap='viridis', aspect='auto', origin='lower',
                   extent=[x_min, x_max, y_min, y_max], interpolation='bilinear')
    
    ax.set_xlabel('Cart Position (m)', fontsize=12)
    ax.set_ylabel('Pole Angle (rad)', fontsize=12)
    
    title = f'Q-Value Heatmap: {algorithm}'
    if episode is not None:
        title += f' (Episode {episode})'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Max Q-Value', fontsize=11)
    
    # Add contour lines for better visualization
    contours = ax.contour(X_fine, Y_fine, Z_fine, levels=10, colors='white', 
                          alpha=0.3, linewidths=0.5)
    
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    episode_str = f"_ep{episode}" if episode is not None else ""
    filename = f'{algorithm}_q_heatmap_Cart_Position_vs_Pole_Angle{episode_str}.png'
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_comparison(all_data, save_dir):
    """Create comparison plot of all algorithms."""
    
    print("\n📊 Creating Comparison Plot")
    
    fig = plt.figure(figsize=(18, 12))
    
    algorithms = list(all_data.keys())
    n_algs = len(algorithms)
    
    for idx, algorithm in enumerate(algorithms, 1):
        points, X, Y, Z = all_data[algorithm]
        
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9,
                              linewidth=0, antialiased=True, edgecolor='none',
                              rcount=80, ccount=80)
        
        ax.set_xlabel('Cart Pos (m)', fontsize=10)
        ax.set_ylabel('Pole Angle (rad)', fontsize=10)
        ax.set_zlabel('Avg Max Q', fontsize=10)
        ax.set_title(algorithm, fontsize=12, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.suptitle('Q-Value Comparison: All Algorithms', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename = 'ALL_ALGORITHMS_comparison.png'
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Q-values: Cart Position vs Pole Angle")
    parser.add_argument("--task", type=str, default=None,
                       help="Task name (default: from config.py)")
    parser.add_argument("--algorithm", type=str, default=None,
                       help="Algorithm to visualize (default: from config.py)")
    parser.add_argument("--episode", type=int, default=None,
                       help="Specific episode to load (default: latest)")
    parser.add_argument("--all", action="store_true",
                       help="Generate plots for all algorithms")
    parser.add_argument("--heatmap", action="store_true",
                       help="Also generate 2D heatmap")
    parser.add_argument("--compare", action="store_true",
                       help="Create comparison plot (requires --all)")
    parser.add_argument("--resolution", type=int, default=200,
                       help="Interpolation resolution (default: 200)")
    args = parser.parse_args()
    
    # Get from config if not specified
    task = args.task if args.task else get_task()
    
    # Setup paths
    project_root = os.path.join(script_dir, "..", "..")
    plots_dir = os.path.join(project_root, "plots", task, "q_values_3d")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("📊 Q-Value Visualization: Cart Position vs Pole Angle")
    print("="*70)
    print(f"Task: {task}")
    print(f"Output directory: {plots_dir}")
    print(f"Interpolation resolution: {args.resolution}x{args.resolution}")
    print("="*70)
    
    # Determine which algorithms to process
    if args.all:
        algorithms = ["Q_Learning", "SARSA", "Double_Q_Learning", "Monte_Carlo"]
    else:
        algorithm = args.algorithm if args.algorithm else get_algorithm()
        algorithms = [algorithm]
    
    all_data = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"Processing: {algorithm}")
        print('='*70)
        
        q_value_dir = os.path.join(project_root, "q_value", task, algorithm)
        
        if not os.path.exists(q_value_dir):
            print(f"⚠️  Q-value directory not found: {q_value_dir}")
            print(f"   Skipping {algorithm}")
            continue
        
        # Load Q-values
        q_values, config, episode = load_q_values(q_value_dir, algorithm, args.episode)
        
        if q_values is None:
            print(f"⚠️  Failed to load Q-values for {algorithm}")
            continue
        
        # Generate smooth 3D surface plot
        points, X, Y, Z = plot_cart_pole_surface(q_values, config, algorithm, plots_dir, 
                                                  episode, args.resolution)
        all_data[algorithm] = (points, X, Y, Z)
        
        # Generate smooth heatmap if requested
        if args.heatmap:
            plot_cart_pole_heatmap(q_values, config, algorithm, plots_dir, episode, args.resolution)
    
    # Generate comparison plot if requested
    if args.compare and len(all_data) > 1:
        plot_comparison(all_data, plots_dir)
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print("="*70)
    print(f"\nGenerated files in: {plots_dir}")


if __name__ == "__main__":
    main()