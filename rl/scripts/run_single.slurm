#!/bin/bash
#SBATCH --job-name=deeprl
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/deeprl_%j.out
#SBATCH --error=logs/deeprl_%j.err

# ============================================
# Deep RL Training - Single GPU (Default)
# ============================================
# Usage:
#   sbatch scripts/run_single.slurm                    # 256 layers, 1 GPU
#   sbatch --export=DEPTH=512 scripts/run_single.slurm # 512 layers
#   sbatch --export=DEBUG=1 scripts/run_single.slurm   # Quick test
#
# For 1024 layers, use run_large.slurm instead
# ============================================

set -e

# Configuration (override via --export)
DEPTH=${DEPTH:-256}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_ENVS=${NUM_ENVS:-64}
TOTAL_STEPS=${TOTAL_STEPS:-100000000}
DEBUG=${DEBUG:-0}

echo "=============================================="
echo "Deep RL Graph Construction"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 1"
echo "CPUs: 16"
echo "Memory: 64G"
echo "=============================================="
echo "Configuration:"
echo "  Depth: $DEPTH layers"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Num Envs: $NUM_ENVS"
echo "  Total Steps: $TOTAL_STEPS"
echo "=============================================="
echo "Started: $(date)"
echo "=============================================="

cd /lustre/nvwulf/scratch/jungshwang/deeprl-graph

if [ ! -f "deeprl-graph.sif" ]; then
    echo "ERROR: Container not found!"
    exit 1
fi

# Build command
if [ "$DEBUG" -eq 1 ]; then
    CMD="python train.py --debug"
else
    CMD="python train.py \
        --num-layers $DEPTH \
        --hidden-dim $HIDDEN_DIM \
        --num-envs $NUM_ENVS \
        --total-timesteps $TOTAL_STEPS \
        --no-progressive \
        --checkpoint-dir checkpoints/depth_${DEPTH} \
        --log-dir logs/depth_${DEPTH} \
        --device cuda"
fi

echo "Command: $CMD"
echo ""

singularity exec --nv \
    --bind $(pwd):/app \
    --pwd /app \
    deeprl-graph.sif \
    $CMD

echo ""
echo "Finished: $(date)"
