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
        
        print(f"\n🔍 Searching for experiments in: {self.base_dir}")
        print(f"Found {len(exp_dirs)} experiment directories")
        
        for exp_dir in exp_dirs:
            csv_file = os.path.join(exp_dir, "training_metrics.csv")
            if not os.path.exists(csv_file):
                print(f"⚠️  No CSV in: {os.path.basename(exp_dir)}")
                continue
                
            # Parse directory name
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split('_')
            
            try:
                # Extract info
                if 'exp1' in dir_name and 'gamma' in dir_name:
                    algo = parts[0]
                    value = float(parts[-1].replace('p', '.'))
                    key = 'exp1_gamma'
                    param_type = 'γ'
                elif 'exp2' in dir_name and 'lr' in dir_name:
                    algo = parts[0]
                    value = float(parts[-1].replace('p', '.'))
                    key = 'exp2_lr'
                    param_type = 'α'
                elif 'exp3' in dir_name and 'eps' in dir_name:
                    algo = parts[0]
                    value = float(parts[-1].replace('p', '.'))
                    key = 'exp3_eps'
                    param_type = 'ε'
                else:
                    continue
                
                # Load data
                df = pd.read_csv(csv_file)
                
                if algo not in experiments[key]:
                    experiments[key][algo] = {}
                
                experiments[key][algo][value] = df
                print(f"✅ Loaded: {algo} {param_type}={value}")
                
            except Exception as e:
                print(f"⚠️  Error loading {dir_name}: {e}")
                continue
        
        return experiments
    
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
        save_path = os.path.join(self.output_dir, 'sensitivity_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def _plot_sensitivity_subplot(self, ax, data, xlabel, title, single_algo=False):
        """Plot sensitivity curve on a subplot."""
        algos = ['Q_Learning', 'SARSA', 'Double_Q_Learning', 'Monte_Carlo']
        if single_algo:
            algos = ['SARSA']
        
        colors = {'Q_Learning': 'blue', 'SARSA': 'red', 
                  'Double_Q_Learning': 'green', 'Monte_Carlo': 'orange'}
        
        for algo in algos:
            if algo not in data:
                continue
            
            # Extract final performance for each parameter value
            values = []
            performances = []
            
            for value, df in sorted(data[algo].items()):
                values.append(value)
                final_perf = df['avg_reward_100'].iloc[-1]
                performances.append(final_perf)
            
            if values:
                ax.plot(values, performances, marker='o', linewidth=2.5, 
                       markersize=10, label=algo.replace('_', ' '),
                       color=colors.get(algo, 'black'))
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Avg Reward', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def create_summary_table(self):
        """Create summary table of all results."""
        print("\n📋 Creating summary table...")
        
        summary = []
        
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
                        'Param Value': value,
                        'Final Reward': round(final_reward, 2),
                        'Max Reward': round(max_reward, 2),
                        'Episodes': int(episodes)
                    })
        
        if summary:
            df_summary = pd.DataFrame(summary)
            
            # Save as CSV
            csv_path = os.path.join(self.output_dir, 'summary_results.csv')
            df_summary.to_csv(csv_path, index=False)
            print(f"✅ Saved: {csv_path}")
            
            # Print table
            print("\n" + "="*80)
            print("📊 SUMMARY RESULTS")
            print("="*80)
            print(df_summary.to_string(index=False))
            print("="*80)
    
    def generate_all_plots(self):
        """Generate all plots and summaries."""
        print("\n" + "="*80)
        print("🎨 SENSITIVITY ANALYSIS PLOTTING")
        print("="*80)
        
        # Check if we have data
        total_experiments = sum(len(exp_data) for exp_data in self.experiments.values())
        if total_experiments == 0:
            print("\n❌ No experiment data found!")
            print(f"   Please check that experiments have completed and CSV files exist in:")
            print(f"   {self.base_dir}")
            return
        
        self.plot_sensitivity_curves()
        self.create_summary_table()
        
        print("\n" + "="*80)
        print("✅ PLOTTING COMPLETE!")
        print("="*80)
        print(f"📁 Output directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(os.listdir(self.output_dir)):
            print(f"  • {file}")
        print("="*80 + "\n")

def main():
    print("="*80)
    print("🎨 SENSITIVITY ANALYSIS PLOTTER")
    print("="*80)
    
    plotter = SensitivityPlotter()
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()
