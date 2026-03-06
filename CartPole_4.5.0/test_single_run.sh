#!/bin/bash
echo "🧪 Testing single run with headless..."

export TEST_VALUE=0.99

sed -i 's/EXPERIMENTAL_MODE = False/EXPERIMENTAL_MODE = True/' scripts/RL_Algorithm/config.py
sed -i 's/TEST_HYPERPARAMETER = .*/TEST_HYPERPARAMETER = "discount_factor"/' scripts/RL_Algorithm/config.py
sed -i 's/ALGORITHM = .*/ALGORITHM = "Q_Learning"/' scripts/RL_Algorithm/config.py

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running: Q_Learning with γ=0.99 (Headless)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts/RL_Algorithm/train.py \
    --headless \
    --num_envs 1 \
    --task Isaac-Cartpole-Stabilize-v0 \
    2>&1 | tee test_run.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Test complete!"
echo "Check: tail -50 test_run.log"
