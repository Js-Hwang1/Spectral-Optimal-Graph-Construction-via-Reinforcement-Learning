#!/bin/bash
#SBATCH --job-name=deeprl-d${DEPTH:-256}
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_d%j.out
#SBATCH --error=logs/train_d%j.err

# ============================================
# Deep RL Training - Depth Ablation
# ============================================
# Single GPU per job (efficient for fairshare)
# Usage: sbatch --export=DEPTH=256 scripts/train_depth.slurm
# ============================================

set -e

DEPTH=${DEPTH:-256}
HIDDEN_DIM=${HIDDEN_DIM:-128}
NUM_ENVS=${NUM_ENVS:-64}
TOTAL_STEPS=${TOTAL_STEPS:-100000000}

echo "=============================================="
echo "Deep RL - Depth Ablation (d=$DEPTH)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: 1 x RTX PRO 6000"
echo "Depth: $DEPTH layers"
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

echo "Finished: $(date)"
