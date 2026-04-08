#!/usr/bin/env bash
#
# run_experiments.sh — Run all 4 SimGrid decentralized learning experiments
#
# Usage: ./run_experiments.sh [results_dir]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${1:-${SCRIPT_DIR}/results}"
mkdir -p "$RESULTS_DIR"

DSGD_SIM="${SCRIPT_DIR}/dsgd_sim"
PLATFORM_GEN="${SCRIPT_DIR}/platform_gen.py"
PLATFORMS_DIR="${SCRIPT_DIR}/platforms"
mkdir -p "$PLATFORMS_DIR"

N=64
D=4
ROUNDS=500
MSG_SIZE=1000000
COMPUTE_FLOPS=1e9
SEED=42

TOPOLOGIES="ours base ring random expgraph equitopo torus"

echo "============================================"
echo " SimGrid Decentralized Learning Experiments"
echo "============================================"
echo ""

# --- Generate platform files ---
echo "[1/5] Generating platform files..."
python3 "$PLATFORM_GEN" 3tier   $N > "$PLATFORMS_DIR/3tier_${N}.xml"
python3 "$PLATFORM_GEN" fattree $N --k 4 > "$PLATFORMS_DIR/fattree_${N}.xml"
python3 "$PLATFORM_GEN" homo    $N > "$PLATFORMS_DIR/homo_${N}.xml"
echo "  Platforms generated in $PLATFORMS_DIR/"

# --- Build simulator ---
echo "[2/5] Building dsgd_sim..."
make -C "$SCRIPT_DIR" -j
echo ""

# ============================================================
# Experiment 1: Heterogeneous bandwidth (3-tier)
# ============================================================
echo "[3/5] Experiment 1: Heterogeneous bandwidth"
mkdir -p "$RESULTS_DIR/exp1_hetero"

for topo in $TOPOLOGIES; do
    echo "  Running $topo on 3-tier platform..."
    "$DSGD_SIM" \
        --platform "$PLATFORMS_DIR/3tier_${N}.xml" \
        --topology "$topo" \
        --n $N --d $D \
        --rounds $ROUNDS \
        --msg-size $MSG_SIZE \
        --compute-flops $COMPUTE_FLOPS \
        --seed $SEED \
        --output "$RESULTS_DIR/exp1_hetero/${topo}.json"
done
echo "  Experiment 1 complete."
echo ""

# ============================================================
# Experiment 2: Link failure resilience
# ============================================================
echo "[4/5] Experiment 2: Link failure resilience"
mkdir -p "$RESULTS_DIR/exp2_failure"

for fail_rate in 0.0 0.01 0.05 0.10; do
    for topo in $TOPOLOGIES; do
        echo "  Running $topo with fail_rate=$fail_rate..."
        "$DSGD_SIM" \
            --platform "$PLATFORMS_DIR/homo_${N}.xml" \
            --topology "$topo" \
            --n $N --d $D \
            --rounds $ROUNDS \
            --msg-size $MSG_SIZE \
            --compute-flops $COMPUTE_FLOPS \
            --fail-rate "$fail_rate" \
            --seed $SEED \
            --output "$RESULTS_DIR/exp2_failure/${topo}_fr${fail_rate}.json"
    done
done
echo "  Experiment 2 complete."
echo ""

# ============================================================
# Experiment 3: Straggler robustness
# ============================================================
echo "[5/5] Experiment 3: Straggler robustness"
mkdir -p "$RESULTS_DIR/exp3_straggler"

for strag_frac in 0.0 0.05 0.10 0.20; do
    for topo in $TOPOLOGIES; do
        echo "  Running $topo with straggler_frac=$strag_frac..."
        "$DSGD_SIM" \
            --platform "$PLATFORMS_DIR/homo_${N}.xml" \
            --topology "$topo" \
            --n $N --d $D \
            --rounds $ROUNDS \
            --msg-size $MSG_SIZE \
            --compute-flops $COMPUTE_FLOPS \
            --straggler-frac "$strag_frac" \
            --straggler-slowdown 10 \
            --seed $SEED \
            --output "$RESULTS_DIR/exp3_straggler/${topo}_sf${strag_frac}.json"
    done
done
echo "  Experiment 3 complete."
echo ""

# ============================================================
# Experiment 4: Fat-tree placement
# ============================================================
echo "Experiment 4: Fat-tree placement"
mkdir -p "$RESULTS_DIR/exp4_fattree"

for topo in $TOPOLOGIES; do
    echo "  Running $topo on fat-tree platform..."
    "$DSGD_SIM" \
        --platform "$PLATFORMS_DIR/fattree_${N}.xml" \
        --topology "$topo" \
        --n $N --d $D \
        --rounds $ROUNDS \
        --msg-size $MSG_SIZE \
        --compute-flops $COMPUTE_FLOPS \
        --seed $SEED \
        --output "$RESULTS_DIR/exp4_fattree/${topo}.json"
done
echo "  Experiment 4 complete."
echo ""

echo "============================================"
echo " All experiments complete!"
echo " Results in: $RESULTS_DIR/"
echo "============================================"
echo ""
echo "To generate plots: python3 ${SCRIPT_DIR}/plot_results.py $RESULTS_DIR"
