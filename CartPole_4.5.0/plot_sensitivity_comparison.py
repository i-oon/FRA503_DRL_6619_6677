#!/usr/bin/env python3
"""
Generate 12 plots comparing parameter values for each algorithm
Style: Similar to the reference image with shaded std deviation
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExperimentPlotter:
    def __init__(self, base_dir="logs/Stabilize"):
        self.base_dir = base_dir
        self.output_dir = "plots/sensitivity_comparison"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_algorithm_experiment(self, algorithm, experiment, param_name, values, param_label):
        """Plot learning curves for one algorithm across different parameter values."""
        
        # Find experiment data
        data = {}
        for value in values:
            value_str = str(value).replace('.', 'p')
            
            if experiment == 'exp1':
                pattern = f"{self.base_dir}/{algorithm}_exp1_gamma_{value_str}/training_metrics.csv"
            elif experiment == 'exp2':
                pattern = f"{self.base_dir}/{algorithm}_exp2_lr_{value_str}/training_metrics.csv"
            elif experiment == 'exp3':
                pattern = f"{self.base_dir}/{algorithm}_exp3_eps_{value_str}/training_metrics.csv"
            
            if os.path.exists(pattern):
                df = pd.read_csv(pattern)
                
                # ✅ FIX: Convert to numeric explicitly (handle any string values)
                df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
                df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
                df['length'] = pd.to_numeric(df['length'], errors='coerce')
                df['avg_reward_100'] = pd.to_numeric(df['avg_reward_100'], errors='coerce')
                df['avg_length_100'] = pd.to_numeric(df['avg_length_100'], errors='coerce')
                
                # Drop any NaN rows
                df = df.dropna(subset=['episode', 'reward', 'length'])
                
                data[value] = df
                print(f"✅ Loaded: {algorithm} {param_name}={value} ({len(df)} episodes)")
            else:
                print(f"⚠️  Missing: {pattern}")
        
        if not data:
            print(f"❌ No data for {algorithm} - {experiment}")
            return
        
        # Create figure with 2 subplots (Rewards + Episode Lengths)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Calculate rolling statistics
        window = 100
        
        for idx, (value, df) in enumerate(sorted(data.items())):
            color = colors[idx % len(colors)]
            episodes = df['episode'].values
            
            # Rewards statistics
            rewards = df['reward'].values
            avg_reward = df['avg_reward_100'].values
            
            # Calculate rolling std for rewards
            reward_series = pd.Series(rewards)
            reward_std = reward_series.rolling(window=window, min_periods=1).std().fillna(0).values
            
            # Episode length statistics
            lengths = df['length'].values
            avg_length = df['avg_length_100'].values
            
            # Calculate rolling std for lengths
            length_series = pd.Series(lengths)
            length_std = length_series.rolling(window=window, min_periods=1).std().fillna(0).values
            
            # --- PLOT 1: REWARDS ---
            ax = axes[0]
            
            # Raw rewards (faded background)
            ax.plot(episodes, rewards, alpha=0.15, color=color, linewidth=0.5)
            
            # Average line (bold)
            ax.plot(episodes, avg_reward, linewidth=2.5, color=color,
                   label=f'{param_name}={value}')
            
            # Shaded std deviation area
            ax.fill_between(episodes,
                           avg_reward - reward_std,
                           avg_reward + reward_std,
                           color=color, alpha=0.2)
            
            # --- PLOT 2: EPISODE LENGTHS ---
            ax = axes[1]
            
            # Raw lengths (faded background)
            ax.plot(episodes, lengths, alpha=0.15, color=color, linewidth=0.5)
            
            # Average line (bold)
            ax.plot(episodes, avg_length, linewidth=2.5, color=color,
                   label=f'{param_name}={value}')
            
            # Shaded std deviation area
            ax.fill_between(episodes,
                           avg_length - length_std,
                           avg_length + length_std,
                           color=color, alpha=0.2)
        
        # --- STYLING: REWARDS PLOT ---
        axes[0].set_ylabel('Total Reward', fontsize=12)
        axes[0].set_title(f'{algorithm.replace("_", " ")}: {param_label} Comparison - Rewards\n(Shaded area = ±1 Standard Deviation)',
                         fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # --- STYLING: EPISODE LENGTHS PLOT ---
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Episode Length (steps)', fontsize=12)
        axes[1].set_title(f'{algorithm.replace("_", " ")}: {param_label} Comparison - Episode Lengths\n(Shaded area = ±1 Standard Deviation)',
                         fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = f"{algorithm}_{experiment}_{param_name}_comparison.png"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all 12 plots (4 algorithms × 3 experiments)."""
        
        print("\n" + "="*80)
        print("🎨 GENERATING SENSITIVITY COMPARISON PLOTS")
        print("="*80)
        
        algorithms = ['Q_Learning', 'SARSA', 'Double_Q_Learning', 'Monte_Carlo']
        
        # Experiment 1: Discount Factor (4 plots)
        print("\n📊 Experiment 1: Discount Factor (4 plots)")
        print("-" * 80)
        for algo in algorithms:
            self.plot_algorithm_experiment(
                algorithm=algo,
                experiment='exp1',
                param_name='γ',
                values=[0.5, 0.8, 0.95, 0.99],
                param_label='Discount Factor'
            )
        
        # Experiment 2: Learning Rate (4 plots)
        print("\n📊 Experiment 2: Learning Rate (4 plots)")
        print("-" * 80)
        for algo in algorithms:
            self.plot_algorithm_experiment(
                algorithm=algo,
                experiment='exp2',
                param_name='α',
                values=[0.03, 0.08, 0.15, 0.25],
                param_label='Learning Rate'
            )
        
        # Experiment 3: Epsilon Decay (1 plot - SARSA only)
        print("\n📊 Experiment 3: Epsilon Decay (SARSA only)")
        print("-" * 80)
        self.plot_algorithm_experiment(
            algorithm='SARSA',
            experiment='exp3',
            param_name='ε_decay',
            values=[0.995, 0.998, 0.9995, 0.9999],
            param_label='Epsilon Decay'
        )
        
        print("\n" + "="*80)
        print("✅ ALL PLOTS GENERATED!")
        print("="*80)
        print(f"📁 Output directory: {self.output_dir}")
        print("\nGenerated plots:")
        
        # List all generated plots
        plots = sorted(os.listdir(self.output_dir))
        for idx, plot in enumerate(plots, 1):
            print(f"  {idx:2d}. {plot}")
        
        print(f"\nTotal: {len(plots)} plots")
        print("="*80 + "\n")

def main():
    plotter = ExperimentPlotter()
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()
