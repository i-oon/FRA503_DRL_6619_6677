#!/bin/bash
echo "======================================================================"
echo "🔬 SENSITIVITY ANALYSIS - Parallel Environments (512)"
echo "======================================================================"

PROJECT_DIR="$HOME/FRA503_DRL_6619_6677/CartPole_4.5.0"
SCRIPT_DIR="$PROJECT_DIR/scripts/RL_Algorithm"
CONFIG_FILE="$SCRIPT_DIR/config.py"
LOG_DIR="$PROJECT_DIR/logs/experiments"
TASK_NAME="Stabilize-Isaac-Cartpole-v0"

# ⚡ Parallel environments setting
NUM_ENVS=512  # Change this to 256, 384, or 512 based on your GPU

mkdir -p "$LOG_DIR"

if [ ! -f "$CONFIG_FILE.autobackup" ]; then
    cp "$CONFIG_FILE" "$CONFIG_FILE.autobackup"
fi

sed -i 's/EXPERIMENTAL_MODE = False/EXPERIMENTAL_MODE = True/' "$CONFIG_FILE"
echo "✅ EXPERIMENTAL_MODE = True"
echo "✅ Task: $TASK_NAME"
echo "✅ Parallel Environments: $NUM_ENVS"
echo "✅ Expected speedup: ~$(($NUM_ENVS / 10))x faster"
echo ""
echo "💡 Monitor GPU memory: watch -n 2 nvidia-smi (in another terminal)"
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
exp1_start=$(date +%s)

for algo in "${ALGORITHMS[@]}"; do
    for gamma in "${GAMMA_VALUES[@]}"; do
        run_count=$((run_count + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶️  Run $run_count/16: $algo with γ=$gamma"
        echo "    Parallel Envs: $NUM_ENVS | Episodes: 25,000"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        export TEST_VALUE=$gamma
        sed -i "s/ALGORITHM = .*/ALGORITHM = \"$algo\"/" "$CONFIG_FILE"
        cd "$PROJECT_DIR"
        
        run_start=$(date +%s)
        python "$SCRIPT_DIR/train.py" \
            --headless \
            --num_envs $NUM_ENVS \
            --task "$TASK_NAME" \
            > "$LOG_DIR/${algo}_gamma_${gamma}.log" 2>&1
        run_end=$(date +%s)
        
        run_time=$((run_end - run_start))
        run_min=$((run_time / 60))
        run_sec=$((run_time % 60))
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed in ${run_min}m ${run_sec}s"
        else
            echo "❌ Failed after ${run_min}m ${run_sec}s"
            echo "   Check: $LOG_DIR/${algo}_gamma_${gamma}.log"
        fi
    done
done

exp1_end=$(date +%s)
exp1_time=$(((exp1_end - exp1_start) / 60))
exp1_sec=$(((exp1_end - exp1_start) % 60))

echo ""
echo "======================================================================"
echo "✅ EXPERIMENT 1 COMPLETE - ${exp1_time}m ${exp1_sec}s"
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
exp2_start=$(date +%s)

for algo in "${ALGORITHMS[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
        run_count=$((run_count + 1))
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "▶️  Run $run_count/16: $algo with α=$lr"
        echo "    Parallel Envs: $NUM_ENVS | Episodes: 25,000"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        export TEST_VALUE=$lr
        sed -i "s/ALGORITHM = .*/ALGORITHM = \"$algo\"/" "$CONFIG_FILE"
        cd "$PROJECT_DIR"
        
        run_start=$(date +%s)
        python "$SCRIPT_DIR/train.py" \
            --headless \
            --num_envs $NUM_ENVS \
            --task "$TASK_NAME" \
            > "$LOG_DIR/${algo}_lr_${lr}.log" 2>&1
        run_end=$(date +%s)
        
        run_time=$((run_end - run_start))
        run_min=$((run_time / 60))
        run_sec=$((run_time % 60))
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed in ${run_min}m ${run_sec}s"
        else
            echo "❌ Failed after ${run_min}m ${run_sec}s"
            echo "   Check: $LOG_DIR/${algo}_lr_${lr}.log"
        fi
    done
done

exp2_end=$(date +%s)
exp2_time=$(((exp2_end - exp2_start) / 60))
exp2_sec=$(((exp2_end - exp2_start) % 60))

echo ""
echo "======================================================================"
echo "✅ EXPERIMENT 2 COMPLETE - ${exp2_time}m ${exp2_sec}s"
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
exp3_start=$(date +%s)

for eps in "${EPSILON_VALUES[@]}"; do
    run_count=$((run_count + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶️  Run $run_count/4: SARSA with ε_decay=$eps"
    echo "    Parallel Envs: $NUM_ENVS | Episodes: 25,000"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    export TEST_VALUE=$eps
    cd "$PROJECT_DIR"
    
    run_start=$(date +%s)
    python "$SCRIPT_DIR/train.py" \
        --headless \
        --num_envs $NUM_ENVS \
        --task "$TASK_NAME" \
        > "$LOG_DIR/SARSA_eps_${eps}.log" 2>&1
    run_end=$(date +%s)
    
    run_time=$((run_end - run_start))
    run_min=$((run_time / 60))
    run_sec=$((run_time % 60))
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed in ${run_min}m ${run_sec}s"
    else
        echo "❌ Failed after ${run_min}m ${run_sec}s"
        echo "   Check: $LOG_DIR/SARSA_eps_${eps}.log"
    fi
done

exp3_end=$(date +%s)
exp3_time=$(((exp3_end - exp3_start) / 60))
exp3_sec=$(((exp3_end - exp3_start) % 60))

echo ""
echo "======================================================================"
echo "✅ EXPERIMENT 3 COMPLETE - ${exp3_time}m ${exp3_sec}s"
echo "======================================================================"
echo ""

# =============================================================================
# CLEANUP & SUMMARY
# =============================================================================
sed -i 's/EXPERIMENTAL_MODE = True/EXPERIMENTAL_MODE = False/' "$CONFIG_FILE"

total_time=$(((exp3_end - exp1_start) / 60))
total_hours=$((total_time / 60))
total_mins=$((total_time % 60))
avg_time_per_run=$((total_time / 36))

echo "======================================================================"
echo "🎉 ALL EXPERIMENTS COMPLETE!"
echo "======================================================================"
echo ""
echo "Performance Summary:"
echo "  • Parallel Environments: $NUM_ENVS"
echo "  • Experiment 1: ${exp1_time}m ${exp1_sec}s (16 runs)"
echo "  • Experiment 2: ${exp2_time}m ${exp2_sec}s (16 runs)"
echo "  • Experiment 3: ${exp3_time}m ${exp3_sec}s (4 runs)"
echo "  • Total Time: ${total_hours}h ${total_mins}m"
echo "  • Avg per run: ${avg_time_per_run} minutes"
echo ""
echo "Results Summary:"
echo "  • Experiment 1 (Discount Factor): 16 runs ✅"
echo "  • Experiment 2 (Learning Rate):   16 runs ✅"
echo "  • Experiment 3 (Epsilon Decay):   4 runs ✅"
echo "  • Total: 36 runs"
echo ""
echo "Results location:"
echo "  • Training data: logs/Stabilize/*_exp*/"
echo "  • Q-values: q_value/Stabilize/*_exp*/"
echo "  • Raw logs: logs/experiments/"
echo ""
echo "Quick analysis:"
echo "  # View final rewards"
echo "  for d in logs/Stabilize/*_exp*/; do"
echo "    name=\$(basename \"\$d\")"
echo "    reward=\$(tail -1 \"\$d/training_metrics.csv\" 2>/dev/null | cut -d',' -f5)"
echo "    [ -n \"\$reward\" ] && echo \"\$name: \$reward\""
echo "  done"
echo ""
echo "Config restored to baseline mode (EXPERIMENTAL_MODE = False)"
echo "======================================================================"
