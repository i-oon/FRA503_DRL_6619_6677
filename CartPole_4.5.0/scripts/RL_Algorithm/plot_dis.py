#!/usr/bin/env python3
"""
Sensitivity Analysis Plotting Script
Compares algorithm performance across different hyperparameters
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SensitivityPlotter:
    def __init__(self, base_dir="logs/Stabilize"):
        self.base_dir = base_dir
        self.experiments = self._load_all_experiments()
        self.output_dir = "plots/sensitivity_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_all_experiments(self):
        """Load all experiment results."""
        experiments = {
            'exp1_gamma': {},      # Discount Factor
            'exp2_lr': {},         # Learning Rate
            'exp3_eps': {}         # Epsilon Decay
        }
        
        # Find all experiment directories
        exp_dirs = glob.glob(f"{self.base_dir}/*_exp*")
        
        for exp_dir in exp_dirs:
            csv_file = os.path.join(exp_dir, "training_metrics.csv")
            if not os.path.exists(csv_file):
                continue
                
            # Parse directory name
            # Format: Algorithm_expID_param_value
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split('_')
            
            try:
                # Extract info
                if 'exp1' in dir_name and 'gamma' in dir_name:
                    algo = parts[0]
                    value = float(parts[-1].replace('p', '.'))
                    key = 'exp1_gamma'
                elif 'exp2' in dir_name and 'lr' in dir_name:
                    algo = parts[0]
                    value = float(parts[-1].replace('p', '.'))
                    key = 'exp2_lr'
                elif 'exp3' in dir_name and 'eps' in dir_name:
                    algo = parts[0]
                    value = float(parts[-1].replace('p', '.'))
                    key = 'exp3_eps'
                else:
                    continue
                
                # Load data
                df = pd.read_csv(csv_file)
                
                if algo not in experiments[key]:
                    experiments[key][algo] = {}
                
                experiments[key][algo][value] = df
                
            except Exception as e:
                print(f"⚠️  Skipping {dir_name}: {e}")
                continue
        
        return experiments
    
    def plot_learning_curves(self):
        """Plot learning curves for each experiment."""
        print("\n📈 Plotting learning curves...")
        
        # Experiment 1: Discount Factor
        if self.experiments['exp1_gamma']:
            self._plot_experiment_curves(
                self.experiments['exp1_gamma'],
                'Discount Factor (γ)',
                'exp1_discount_factor_curves.png'
            )
        
        # Experiment 2: Learning Rate
        if self.experiments['exp2_lr']:
            self._plot_experiment_curves(
                self.experiments['exp2_lr'],
                'Learning Rate (α)',
                'exp2_learning_rate_curves.png'
            )
        
        # Experiment 3: Epsilon Decay
        if self.experiments['exp3_eps']:
            self._plot_experiment_curves(
                self.experiments['exp3_eps'],
                'Epsilon Decay',
                'exp3_epsilon_decay_curves.png',
                single_algo=True
            )
    
    def _plot_experiment_curves(self, data, param_name, filename, single_algo=False):
        """Plot learning curves for one experiment."""
        if single_algo:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            axes = [ax]
            algos = list(data.keys())[:1]  # Just SARSA for epsilon
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            algos = ['Q_Learning', 'SARSA', 'Double_Q_Learning', 'Monte_Carlo']
        
        for idx, algo in enumerate(algos):
            if algo not in data:
                continue
                
            ax = axes[idx]
            
            # Plot each parameter value
            for value, df in sorted(data[algo].items()):
                ax.plot(df['episode'], df['avg_reward_100'], 
                       label=f'{param_name}={value}', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Avg Reward (100 episodes)', fontsize=12)
            ax.set_title(f'{algo.replace("_", " ")}', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_sensitivity_curves(self):
        """Plot sensitivity curves (parameter vs final performance)."""
        print("\n📊 Plotting sensitivity curves...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Experiment 1: Discount Factor
        if self.experiments['exp1_gamma']:
            self._plot_sensitivity_subplot(
                axes[0], self.experiments['exp1_gamma'],
                'Discount Factor (γ)', 'Discount Factor Sensitivity'
            )
        
        # Experiment 2: Learning Rate
        if self.experiments['exp2_lr']:
            self._plot_sensitivity_subplot(
                axes[1], self.experiments['exp2_lr'],
                'Learning Rate (α)', 'Learning Rate Sensitivity'
            )
        
        # Experiment 3: Epsilon Decay
        if self.experiments['exp3_eps']:
            self._plot_sensitivity_subplot(
                axes[2], self.experiments['exp3_eps'],
                'Epsilon Decay', 'Epsilon Decay Sensitivity (SARSA)',
                single_algo=True
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'sensitivity_curves_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def _plot_sensitivity_subplot(self, ax, data, xlabel, title, single_algo=False):
        """Plot sensitivity curve on a subplot."""
        algos = ['Q_Learning', 'SARSA', 'Double_Q_Learning', 'Monte_Carlo']
        if single_algo:
            algos = ['SARSA']
        
        for algo in algos:
            if algo not in data:
                continue
            
            # Extract final performance for each parameter value
            values = []
            performances = []
            
            for value, df in sorted(data[algo].items()):
                values.append(value)
                # Take last 100 episodes average
                final_perf = df['avg_reward_100'].iloc[-1]
                performances.append(final_perf)
            
            ax.plot(values, performances, marker='o', linewidth=2, 
                   markersize=8, label=algo.replace('_', ' '))
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Final Avg Reward', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def plot_comparison_bars(self):
        """Plot bar chart comparison of final performance."""
        print("\n📊 Plotting comparison bar charts...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Experiment 1: Discount Factor
        if self.experiments['exp1_gamma']:
            self._plot_comparison_bars_subplot(
                axes[0], self.experiments['exp1_gamma'],
                'Discount Factor (γ)', 'Discount Factor Comparison'
            )
        
        # Experiment 2: Learning Rate
        if self.experiments['exp2_lr']:
            self._plot_comparison_bars_subplot(
                axes[1], self.experiments['exp2_lr'],
                'Learning Rate (α)', 'Learning Rate Comparison'
            )
        
        # Experiment 3: Epsilon Decay
        if self.experiments['exp3_eps']:
            self._plot_comparison_bars_subplot(
                axes[2], self.experiments['exp3_eps'],
                'Epsilon Decay', 'Epsilon Decay Comparison (SARSA)',
                single_algo=True
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'comparison_bars.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def _plot_comparison_bars_subplot(self, ax, data, param_label, title, single_algo=False):
        """Plot bar comparison on a subplot."""
        algos = ['Q_Learning', 'SARSA', 'Double_Q_Learning', 'Monte_Carlo']
        if single_algo:
            algos = ['SARSA']
        
        # Prepare data
        param_values = sorted(list(next(iter(data.values())).keys()))
        x = np.arange(len(param_values))
        width = 0.2 if not single_algo else 0.6
        
        for idx, algo in enumerate(algos):
            if algo not in data:
                continue
            
            performances = []
            for value in param_values:
                if value in data[algo]:
                    perf = data[algo][value]['avg_reward_100'].iloc[-1]
                    performances.append(perf)
                else:
                    performances.append(0)
            
            offset = (idx - len(algos)/2) * width
            ax.bar(x + offset, performances, width, 
                  label=algo.replace('_', ' '), alpha=0.8)
        
        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel('Final Avg Reward', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_values])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    def create_summary_table(self):
        """Create summary table of all results."""
        print("\n📋 Creating summary table...")
        
        summary = []
        
        # Process all experiments
        for exp_key, exp_data in self.experiments.items():
            exp_name = {
                'exp1_gamma': 'Discount Factor',
                'exp2_lr': 'Learning Rate',
                'exp3_eps': 'Epsilon Decay'
            }[exp_key]
            
            for algo, values_dict in exp_data.items():
                for value, df in values_dict.items():
                    final_reward = df['avg_reward_100'].iloc[-1]
                    max_reward = df['avg_reward_100'].max()
                    episodes = df['episode'].iloc[-1]
                    
                    summary.append({
                        'Experiment': exp_name,
                        'Algorithm': algo.replace('_', ' '),
                        'Parameter Value': value,
                        'Final Reward': f'{final_reward:.2f}',
                        'Max Reward': f'{max_reward:.2f}',
                        'Episodes': episodes
                    })
        
        if summary:
            df_summary = pd.DataFrame(summary)
            
            # Save as CSV
            csv_path = os.path.join(self.output_dir, 'summary_results.csv')
            df_summary.to_csv(csv_path, index=False)
            print(f"✅ Saved: {csv_path}")
            
            # Print table
            print("\n" + "="*80)
            print("SUMMARY RESULTS")
            print("="*80)
            print(df_summary.to_string(index=False))
            print("="*80)
    
    def plot_heatmap(self):
        """Create heatmap of final performance."""
        print("\n🔥 Creating performance heatmaps...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Discount Factor
        if self.experiments['exp1_gamma']:
            self._plot_heatmap_subplot(
                axes[0], self.experiments['exp1_gamma'],
                'Discount Factor (γ)', 'Discount Factor Impact'
            )
        
        # Heatmap 2: Learning Rate
        if self.experiments['exp2_lr']:
            self._plot_heatmap_subplot(
                axes[1], self.experiments['exp2_lr'],
                'Learning Rate (α)', 'Learning Rate Impact'
            )
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'performance_heatmaps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def _plot_heatmap_subplot(self, ax, data, param_label, title):
        """Plot heatmap on subplot."""
        algos = ['Q_Learning', 'SARSA', 'Double_Q_Learning', 'Monte_Carlo']
        param_values = sorted(list(next(iter(data.values())).keys()))
        
        # Create matrix
        matrix = []
        for algo in algos:
            if algo not in data:
                matrix.append([0] * len(param_values))
                continue
            
            row = []
            for value in param_values:
                if value in data[algo]:
                    perf = data[algo][value]['avg_reward_100'].iloc[-1]
                    row.append(perf)
                else:
                    row.append(0)
            matrix.append(row)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # Labels
        ax.set_xticks(np.arange(len(param_values)))
        ax.set_yticks(np.arange(len(algos)))
        ax.set_xticklabels([str(v) for v in param_values])
        ax.set_yticklabels([a.replace('_', ' ') for a in algos])
        
        ax.set_xlabel(param_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add values
        for i in range(len(algos)):
            for j in range(len(param_values)):
                text = ax.text(j, i, f'{matrix[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Final Avg Reward')
    
    def generate_all_plots(self):
        """Generate all plots and summaries."""
        print("\n" + "="*80)
        print("🎨 SENSITIVITY ANALYSIS PLOTTING")
        print("="*80)
        
        self.plot_learning_curves()
        self.plot_sensitivity_curves()
        self.plot_comparison_bars()
        self.plot_heatmap()
        self.create_summary_table()
        
        print("\n" + "="*80)
        print("✅ ALL PLOTS COMPLETE!")
        print("="*80)
        print(f"📁 Output directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(os.listdir(self.output_dir)):
            print(f"  • {file}")
        print("="*80 + "\n")

def main():
    plotter = SensitivityPlotter()
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()
