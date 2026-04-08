#!/bin/bash
#SBATCH --job-name=dfl_ours
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/dfl_ours_%j.log

# Run ONLY ours.
# APPLE-TO-APPLE with Base k=1 n=32:
#   Base k=1: 5 rounds/cycle, stacked degree=5, one-peer
#   Ours d=5: 5 matchings/cycle, stacked degree=5, one-peer
# Same n, same alpha, same seed, same everything else.

module load slurm

WORKDIR=/lustre/nvwulf/scratch/jungshwang/Spectral-Optimal-Graph-Construction-via-Reinforcement-Learning
cd $WORKDIR

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

mkdir -p logs benchmarks/dfl/results

echo "=== Ours ONLY (d=5, apple-to-apple with Base k=1) ==="

singularity exec --nv dfl.sif python benchmarks/dfl/train.py \
    --topo ours \
    --n 32 \
    --d 5 \
    --alpha 0.1 \
    --seed 0 \
    --dataset cifar100 \
    --tau 5 \
    --rounds 2000 \
    --eval-freq 10 \
    --data-dir ${WORKDIR}/data \
    --output-dir benchmarks/dfl/results

echo "=== Done ==="
