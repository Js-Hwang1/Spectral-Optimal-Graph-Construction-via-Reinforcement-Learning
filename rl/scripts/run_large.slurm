#!/bin/bash
#SBATCH --job-name=deeprl-large
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/deeprl_large_%j.out
#SBATCH --error=logs/deeprl_large_%j.err

# ============================================
# Deep RL Training - Large Config (2 GPUs)
# ============================================
# Use this for:
#   - 1024 layers
#   - hidden_dim >= 256
#   - num_envs >= 128
#
# Usage:
#   sbatch --export=DEPTH=1024 scripts/run_large.slurm
# ============================================

set -e

DEPTH=${DEPTH:-1024}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_ENVS=${NUM_ENVS:-64}
TOTAL_STEPS=${TOTAL_STEPS:-100000000}

echo "=============================================="
echo "Deep RL Graph Construction (Large)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: 2"
echo "CPUs: 32"
echo "Memory: 128G"
echo "=============================================="
echo "Configuration:"
echo "  Depth: $DEPTH layers"
echo "  Hidden Dim: $HIDDEN_DIM"
echo "  Num Envs: $NUM_ENVS"
echo "=============================================="
echo "Started: $(date)"
echo "=============================================="

cd /lustre/nvwulf/scratch/jungshwang/deeprl-graph

if [ ! -f "deeprl-graph.sif" ]; then
    echo "ERROR: Container not found!"
    exit 1
fi

singularity exec --nv \
    --bind $(pwd):/app \
    --pwd /app \
    deeprl-graph.sif \
    python train.py \
        --num-layers $DEPTH \
        --hidden-dim $HIDDEN_DIM \
        --num-envs $NUM_ENVS \
        --total-timesteps $TOTAL_STEPS \
        --no-progressive \
        --checkpoint-dir checkpoints/depth_${DEPTH} \
        --log-dir logs/depth_${DEPTH} \
        --device cuda

echo ""
echo "Finished: $(date)"
