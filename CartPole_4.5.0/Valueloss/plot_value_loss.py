"""Plot Loss/value_loss for all algorithms from TensorBoard JSON exports."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

ALGORITHMS = {
    "DQN":  {"color": "#1f77b4", "label": "DQN"},
    "AC":   {"color": "#d62728", "label": "AC"},
    "A2C":  {"color": "#ff7f0e", "label": "A2C"},
    "PPO":  {"color": "#9467bd", "label": "PPO"},
    "SAC":  {"color": "#e377c2", "label": "SAC"},
    "TD3":  {"color": "#17becf", "label": "TD3"},
}

fig, ax = plt.subplots(figsize=(12, 6))

for algo_key, meta in ALGORITHMS.items():
    matches = [f for f in os.listdir(PLOT_DIR) if f.startswith(algo_key) and f.endswith(".json")]
    if not matches:
        print(f"Warning: No JSON file found for {algo_key}")
        continue

    filepath = os.path.join(PLOT_DIR, matches[0])
    with open(filepath) as f:
        data = json.load(f)

    steps = np.array([d[1] for d in data])
    losses = np.array([d[2] for d in data])

    ax.plot(steps, losses,
            color=meta["color"], label=meta["label"], linewidth=1.5, alpha=0.85)

ax.set_xlabel("Episode / Iteration", fontsize=12)
ax.set_ylabel("Value Loss", fontsize=12)
ax.set_title("Value Loss During Training", fontsize=13)
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(PLOT_DIR, "value_loss.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.show()
