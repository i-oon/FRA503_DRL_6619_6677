#!/bin/bash
echo "======================================================================"
echo "🔬 SENSITIVITY ANALYSIS - Headless Mode"
echo "======================================================================"

PROJECT_DIR="$HOME/FRA503_DRL_6619_6677/CartPole_4.5.0"
SCRIPT_DIR="$PROJECT_DIR/scripts/RL_Algorithm"
CONFIG_FILE="$SCRIPT_DIR/config.py"
LOG_DIR="$PROJECT_DIR/logs/experiments"
TASK_NAME="Stabilize-Isaac-Cartpole-v0"

mkdir -p "$LOG_DIR"

if [ ! -f "$CONFIG_FILE.autobackup" ]; then
    cp "$CONFIG_FILE" "$CONFIG_FILE.autobackup"
fi

sed -i 's/EXPERIMENTAL_MODE = False/EXPERIMENTAL_MODE = True/' "$CONFIG_FILE"
echo "✅ EXPERIMENTAL_MODE = True"
echo "✅ Task: $TASK_NAME"
echo ""

# =============================================================================
# EXPERIMENT 1: Discount Factor (16 runs)
# =============================================================================
echo "======================================================================"
echo "📊 EXPERIMENT 1: Discount Factor"
echo "======================================================================"
sed -i 's/TEST_HYPERPARAMETER = .*/TEST_HYPERPARAMETER = "discount_factor"/' "$CONFIG_FILE"
sed -i 's/EXPERIMENT_ID = .*/EXPERIMENT_ID = "exp1"/' "$CONFIG_FILE"

ALGORITHMS=("Q_Learning" "SARSA" "Double_Q_Learning" "Monte_Carlo")
GAMMA_VALUES=(0.5 0.8 0.95 0.99)

run_count=0
for algo in "${ALGORITHMS[@]}"; do
    for gamma in "${GAMMA_VALUES[@]}"; do
        run_count=$((run_count + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶️  Run $run_count/16: $algo with γ=$gamma"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        export TEST_VALUE=$gamma
        sed -i "s/ALGORITHM = .*/ALGORITHM = \"$algo\"/" "$CONFIG_FILE"
        cd "$PROJECT_DIR"
        
        python "$SCRIPT_DIR/train.py" \
            --headless \
            --num_envs 1 \
            --task "$TASK_NAME" \
            > "$LOG_DIR/${algo}_gamma_${gamma}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed successfully"
        else
            echo "❌ Failed - check $LOG_DIR/${algo}_gamma_${gamma}.log"
        fi
    done
done

echo ""
echo "======================================================================"
echo "✅ EXPERIMENT 1 COMPLETE (16/16 runs)"
echo "======================================================================"
echo ""

# =============================================================================
# EXPERIMENT 2: Learning Rate (16 runs)
# =============================================================================
echo "======================================================================"
echo "📊 EXPERIMENT 2: Learning Rate"
echo "======================================================================"
sed -i 's/TEST_HYPERPARAMETER = .*/TEST_HYPERPARAMETER = "learning_rate"/' "$CONFIG_FILE"
sed -i 's/EXPERIMENT_ID = .*/EXPERIMENT_ID = "exp2"/' "$CONFIG_FILE"

LR_VALUES=(0.03 0.08 0.15 0.25)

run_count=0
for algo in "${ALGORITHMS[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
        run_count=$((run_count + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶️  Run $run_count/16: $algo with α=$lr"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        export TEST_VALUE=$lr
        sed -i "s/ALGORITHM = .*/ALGORITHM = \"$algo\"/" "$CONFIG_FILE"
        cd "$PROJECT_DIR"
        
        python "$SCRIPT_DIR/train.py" \
            --headless \
            --num_envs 1 \
            --task "$TASK_NAME" \
            > "$LOG_DIR/${algo}_lr_${lr}.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed successfully"
        else
            echo "❌ Failed - check $LOG_DIR/${algo}_lr_${lr}.log"
        fi
    done
done

echo ""
echo "======================================================================"
echo "✅ EXPERIMENT 2 COMPLETE (16/16 runs)"
echo "======================================================================"
echo ""

# =============================================================================
# EXPERIMENT 3: Epsilon Decay (4 runs)
# =============================================================================
echo "======================================================================"
echo "📊 EXPERIMENT 3: Epsilon Decay (SARSA only)"
echo "======================================================================"
sed -i 's/TEST_HYPERPARAMETER = .*/TEST_HYPERPARAMETER = "epsilon_decay"/' "$CONFIG_FILE"
sed -i 's/EXPERIMENT_ID = .*/EXPERIMENT_ID = "exp3"/' "$CONFIG_FILE"
sed -i 's/ALGORITHM = .*/ALGORITHM = "SARSA"/' "$CONFIG_FILE"

EPSILON_VALUES=(0.995 0.998 0.9995 0.9999)

run_count=0
for eps in "${EPSILON_VALUES[@]}"; do
    run_count=$((run_count + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶️  Run $run_count/4: SARSA with ε_decay=$eps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    export TEST_VALUE=$eps
    cd "$PROJECT_DIR"
    
    python "$SCRIPT_DIR/train.py" \
        --headless \
        --num_envs 1 \
        --task "$TASK_NAME" \
        > "$LOG_DIR/SARSA_eps_${eps}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed successfully"
    else
        echo "❌ Failed - check $LOG_DIR/SARSA_eps_${eps}.log"
    fi
done

echo ""
echo "======================================================================"
echo "✅ EXPERIMENT 3 COMPLETE (4/4 runs)"
echo "======================================================================"
echo ""

# =============================================================================
# CLEANUP & SUMMARY
# =============================================================================
sed -i 's/EXPERIMENTAL_MODE = True/EXPERIMENTAL_MODE = False/' "$CONFIG_FILE"

echo "======================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  • Experiment 1 (Discount Factor): 16 runs ✅"
echo "  • Experiment 2 (Learning Rate):   16 runs ✅"
echo "  • Experiment 3 (Epsilon Decay):   4 runs ✅"
echo "  • Total: 36 runs"
echo ""
echo "Results location:"
echo "  • Logs: $LOG_DIR"
echo "  • Training data: $PROJECT_DIR/logs/Stabilize/"
echo ""
echo "View results:"
echo "  ls -la logs/Stabilize/*_exp*/"
echo ""
echo "Config restored to baseline mode (EXPERIMENTAL_MODE = False)"
echo "======================================================================"
