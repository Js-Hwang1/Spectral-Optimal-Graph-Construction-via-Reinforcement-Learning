#!/bin/bash
#SBATCH --job-name=deeprl-prof
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/profile_%j.out
#SBATCH --error=logs/profile_%j.err

# ============================================
# Deep RL Profiling (5 iterations)
# ============================================

set -e

DEPTH=${DEPTH:-256}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_ENVS=${NUM_ENVS:-64}
PROFILE_ITERS=${PROFILE_ITERS:-5}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-256}

USE_COMPILE=${USE_COMPILE:-0}

echo "=============================================="
echo "Deep RL - Timing Profile (d=$DEPTH)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: b40x4 (8-hour limit)"
echo "GPU: 1 x RTX PRO 6000"
echo "Profile iterations: $PROFILE_ITERS"
echo "Minibatch size: $MINIBATCH_SIZE"
echo "torch.compile: $USE_COMPILE"
echo "Started: $(date)"
echo "Memory limit: $SLURM_MEM_PER_NODE MB"
echo "=============================================="

# Log memory usage periodically in background
(
  while true; do
    echo "[$(date +%H:%M:%S)] RAM: $(free -h | awk '/^Mem:/{print $3"/"$2}')" >> logs/profile_${SLURM_JOB_ID}_mem.log
    sleep 30
  done
) &
MEM_MONITOR_PID=$!

cd /lustre/nvwulf/scratch/jungshwang/deeprl-graph

if [ ! -f "deeprl-graph.sif" ]; then
    echo "ERROR: Container not found!"
    exit 1
fi

# Build compile flag if enabled
COMPILE_FLAG=""
if [ "$USE_COMPILE" = "1" ]; then
    COMPILE_FLAG="--compile"
fi

singularity exec --nv \
    --bind $(pwd):/app \
    --pwd /app \
    deeprl-graph.sif \
    python train.py \
        --num-layers $DEPTH \
        --hidden-dim $HIDDEN_DIM \
        --num-envs $NUM_ENVS \
        --minibatch-size $MINIBATCH_SIZE \
        --no-progressive \
        --log-dir logs/profile_${DEPTH}_mb${MINIBATCH_SIZE} \
        --device cuda \
        --profile $PROFILE_ITERS \
        $COMPILE_FLAG

EXIT_CODE=$?
echo "Python exit code: $EXIT_CODE"

# Stop memory monitor
kill $MEM_MONITOR_PID 2>/dev/null

# Check dmesg for OOM killer (may require permissions)
echo ""
echo "=== Checking for OOM events ==="
dmesg 2>/dev/null | tail -50 | grep -i -E "oom|kill|memory" || echo "No OOM messages found (or no dmesg access)"

echo ""
echo "=== Memory log tail ==="
tail -20 logs/profile_${SLURM_JOB_ID}_mem.log 2>/dev/null || echo "No memory log"

echo ""
echo "Finished: $(date)"
