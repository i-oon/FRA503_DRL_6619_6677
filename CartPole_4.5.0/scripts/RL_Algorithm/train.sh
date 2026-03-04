#!/bin/bash

cd ~/FRA503_DRL_6619_6677/CartPole_4.5.0
cp scripts/RL_Algorithm/config.py scripts/RL_Algorithm/config.py.backup

ALGORITHMS=("Q_Learning" "SARSA" "Double_Q_Learning" "Monte_Carlo")

# Train each algorithm
for ALGO in "${ALGORITHMS[@]}"; do
    echo "=========================================="
    echo "Training: $ALGO"
    echo "=========================================="
    
    # Update config.py
    sed -i "s/^ALGORITHM = \".*\"/ALGORITHM = \"$ALGO\"/" scripts/RL_Algorithm/config.py
    
    # Run training
    python scripts/RL_Algorithm/train.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 256 --headless
    
    echo ""
done

# Restore config
cp scripts/RL_Algorithm/config.py.backup scripts/RL_Algorithm/config.py

# Generate plots
echo "=========================================="
echo "Generating plots..."
echo "=========================================="
python3 scripts/RL_Algorithm/plot_results.py --all
python3 scripts/RL_Algorithm/plot_q_values.py --all
echo ""
echo "Done!"